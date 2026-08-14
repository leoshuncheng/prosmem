"""Zep / Graphiti baseline agent (arXiv:2501.13956; getzep/graphiti @ 413b9b2, graphiti-core 0.29.2).

Faithful reproduction wrapper around the OFFICIAL Graphiti temporal-knowledge-graph
engine (the open-source engine that powers Zep; the repo's own README carries the
arXiv:2501.13956 badge, so cite Zep / reproduce Graphiti). Graphiti is a real-time
knowledge-graph memory: `add_episode` runs LLM entity/edge extraction + temporal
dedup into a graph; `search` does hybrid (vector + BM25 + graph) retrieval with RRF
reranking. Backend = Neo4j 5.26 (the official/recommended driver; quickstart default).

Config touch-points (all OFFICIAL mechanisms; ZERO algorithm change):
  • graph backend  -> Neo4j 5.26.14 LTS, local bolt://127.0.0.1:7687 (standard driver)
  • LLM client     -> OpenAIGenericClient (Graphiti's OpenAI-compatible client), model =
    our backbone, base_url -> the configured OpenAI-compatible endpoint, with
    structured_output_mode='json_object' — Graphiti's OWN documented setting for
    providers without native json_schema (the OpenAIGenericClient docstring names
    DeepSeek explicitly).
  • embedder       -> OpenAIEmbedder with the official default model text-embedding-3-small,
    served from the same endpoint's embeddings API; same model across all
    backbones (the embedder is backbone-independent, kept at Graphiti's default).
  • cross_encoder  -> OpenAIRerankerClient routed the same way; NOT invoked by the basic
    search() path (its recipe EDGE_HYBRID_SEARCH_RRF uses algorithmic RRF, not the LLM
    cross-encoder) — provided only so the default client isn't a broken OpenAI-direct stub.
Graphiti's own algorithm params (extraction prompts, RRF search recipe, temporal
invalidation, dedup) are all at official defaults.

Online per-turn adaptation (identical to Mem0 / LightMem / EverOS in this harness):
seed the PM intention as an episode at turn 0, then per turn: search -> stitch facts into
the SHARED baseline prompt -> backbone LLM responds -> add the turn as an episode. Uses
only official public APIs Graphiti.add_episode / Graphiti.search. The harness interface,
the intention SEED text, and the per-turn prompt are IDENTICAL to the other baselines.
Per-sample isolation via Graphiti's `group_id` graph partition.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from prosmem.agent.llm import LLMClient
from prosmem.agent.loop import StepResult
from prosmem.core.config import ProsMemConfig

_EMBED_MODEL = "text-embedding-3-small"  # Graphiti's official DEFAULT_EMBEDDING_MODEL


def _graphiti_llm_route(config: ProsMemConfig) -> tuple[str, str, str, str]:
    """Graphiti's internal extraction LLM calls the same OpenAI-compatible endpoint
    as the main agent, in 'json_object' structured-output mode (Graphiti's documented
    mode for providers lacking native json_schema; the OpenAIGenericClient docstring
    names DeepSeek). Returns (model, api_key, base_url, structured_output_mode)."""
    return config.main_model, config.llm_api_key, config.llm_base_url, "json_object"


class GraphitiAgent:
    """Graphiti (Zep engine) knowledge-graph baseline, plugged into the shared harness."""

    def __init__(self, config: ProsMemConfig, system_prompt: str = "", collection_id: str = "default") -> None:
        # Deferred imports so the main env (no graphiti_core) is unaffected.
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        self.config = config
        self.llm = LLMClient(config)  # response-generation LLM (shared backbone, == Mem0Agent)

        mm_model, mm_key, mm_base, so_mode = _graphiti_llm_route(config)
        # Graphiti internal extraction LLM (OpenAI-compatible client).
        llm_cfg = LLMConfig(api_key=mm_key, model=mm_model, base_url=mm_base, temperature=0.0)
        llm_client = OpenAIGenericClient(config=llm_cfg, structured_output_mode=so_mode)
        # Official default embedder, served from the same OpenAI-compatible endpoint
        # (the endpoint must expose the embeddings API).
        emb_cfg = OpenAIEmbedderConfig(
            embedding_model=_EMBED_MODEL,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
        embedder = OpenAIEmbedder(config=emb_cfg)
        # cross_encoder routed the same as the LLM; not exercised by basic search() (RRF).
        cross_encoder = OpenAIRerankerClient(config=llm_cfg)
        # Keep refs so cleanup() can gracefully aclose their AsyncOpenAI httpx pools
        # BEFORE closing the event loop (otherwise Windows asyncio logs harmless
        # httpx/anyio teardown tracebacks at GC — cosmetic noise, no result impact).
        self._async_clients = [llm_client, embedder, cross_encoder]

        # Dedicated event loop (Graphiti APIs are async; harness step() is sync).
        self._loop = asyncio.new_event_loop()

        driver = Neo4jDriver(
            uri="bolt://127.0.0.1:7687", user="neo4j", password="prosmem-graphiti",
        )
        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
        # Build indices/constraints once (idempotent); required before ingestion.
        self._loop.run_until_complete(self._graphiti.build_indices_and_constraints())

        safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(collection_id))
        self._group_id = f"prosmem_{safe}"        # per-sample graph partition
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.conversation_history: list[dict] = []
        self.step_count = 0
        # bench turns carry no real dates; synthetic increasing tz-aware reference_time.
        self._ref_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _next_time(self) -> datetime:
        self._ref_time += timedelta(days=1)
        return self._ref_time

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def register_intention(self, description: str, trigger: str) -> None:
        """Seed Graphiti with the PM intention BEFORE the scenario starts.
        SEED text is IDENTICAL to Mem0Agent (fairness across baselines)."""
        seed = (
            f"Remember this pending task for later: when {trigger}, "
            f"I want you to {description}. Please watch for the right "
            f"moment and act on it when the condition is met."
        )
        try:
            self._run(self._graphiti.add_episode(
                name="pending_intention",
                episode_body=seed,
                source_description="user message",
                reference_time=self._next_time(),
                group_id=self._group_id,
            ))
        except Exception:
            pass  # non-fatal seeding; retrieval just has less context

    def _search(self, query: str, num_results: int = 20) -> list[str]:
        """Hybrid (vector+BM25+graph, RRF) search; return the relevant fact strings."""
        try:
            edges = self._run(self._graphiti.search(
                query=query, group_ids=[self._group_id], num_results=num_results,
            ))
        except Exception:
            return []
        return [e.fact for e in edges if getattr(e, "fact", None)]

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1

        retrieved = self._search(user_input, num_results=20)

        # Build system prompt — SAME format/wording as Mem0Agent (fairness).
        sys_msg = self.system_prompt
        if retrieved:
            sys_msg += "\n\n=== RELEVANT MEMORIES (from your long-term memory) ===\n"
            for i, fact in enumerate(retrieved, 1):
                sys_msg += f"{i}. {fact}\n"
            sys_msg += (
                "\nIf any memory describes an intention whose trigger condition "
                "is satisfied by the current conversation, execute that action "
                "explicitly in your response."
            )

        self.conversation_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(self.conversation_history)

        response = self.llm.chat(messages)
        self.conversation_history.append({"role": "assistant", "content": response})

        # Persist this turn as an episode (official add_episode; user+assistant pair).
        ts = self._next_time()
        try:
            self._run(self._graphiti.add_episode(
                name=f"turn_{self.step_count}",
                episode_body=f"user: {user_input}\nassistant: {response}",
                source_description="conversation turn",
                reference_time=ts,
                group_id=self._group_id,
            ))
        except Exception:
            pass

        return StepResult(step=self.step_count, user_input=user_input, agent_response=response)

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        """Close Graphiti's Neo4j driver + internal async clients and the event loop.

        Do NOT close self.llm: the shared harness reuses agent.llm AFTER cleanup() for the
        D3-content judge (_wrap_d3_content), exactly like Mem0Agent / LightMemAgent /
        EverOSAgent. self.llm's HTTP pool is closed by the evaluator's terminal _release_llm.
        """
        async def _aclose() -> None:
            # Close Graphiti (Neo4j driver) first, then each internal AsyncOpenAI
            # httpx pool, all WHILE the loop is still running — this drains the
            # connections cleanly and removes the teardown-traceback noise.
            try:
                await self._graphiti.close()
            except Exception:
                pass
            for c in self._async_clients:
                client = getattr(c, "client", None)
                close = getattr(client, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass
        try:
            self._run(_aclose())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
        self._graphiti = None
