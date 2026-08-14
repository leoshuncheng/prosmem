"""LightMem baseline agent (ICLR 2026, arXiv:2510.18866; zjunlp/LightMem @ 579ee76).

Faithful reproduction wrapper. Config is copied VERBATIM from the official
reproduction script `experiments/longmemeval/run_lightmem_gpt.py`; ONLY the
following integration touch-points are changed:
  ① llmlingua-2 model path  -> local HF cache snapshot
  ② memory_manager LLM model -> our backbone (DeepSeek-V3 / Llama / GPT-4o / DSV4-Pro)
  ③ api_key / openai_base_url -> the configured OpenAI-compatible endpoint
  ④ qdrant path -> local writable per-collection dir
  ⑤ device "cuda" -> "cpu"  (torch in this env is 2.8.0+cpu; device only affects
     compute location, NOT model outputs/results)
ALL LightMem-own algorithm params (pre_compress, topic_segment, extract_threshold,
retrieve_strategy, update mode, etc.) are kept at official defaults. ZERO algorithm change.

Online per-turn adaptation: the official usage is offline batch-QA
(add all -> retrieve once by final question). PM bench is per-turn, so we add+retrieve
each turn, exactly like the Mem0/A-MEM baselines in this harness (fairness). Uses only
official public APIs LightMemory.from_config / add_memory / retrieve. No offline
consolidation call (matches the official main eval script, which also omits it).

The harness interface (register_intention / step / run_scenario / cleanup), the
intention SEED text, and the per-turn response prompt are IDENTICAL to Mem0Agent so
all baselines are scored by the same evaluator on equal footing.
"""

from __future__ import annotations

import os
import shutil

from prosmem.agent.llm import LLMClient
from prosmem.agent.loop import StepResult
from prosmem.core.config import ProsMemConfig

_LLMLINGUA_REPO = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_EMBED_MODEL = "all-MiniLM-L6-v2"  # official text_embedder model (faithful default)


def _lightmem_llm_route(config: ProsMemConfig) -> tuple[str, str, str]:
    """LightMem's internal memory_manager LLM calls the same OpenAI-compatible
    endpoint as the main agent. Returns (model, api_key, openai_base_url)."""
    return config.main_model, config.llm_api_key, config.llm_base_url


def _resolve_llmlingua_path() -> str:
    """Resolve the local HF-cache snapshot path for the llmlingua-2 model (offline)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(_LLMLINGUA_REPO)


class LightMemAgent:
    """LightMem retrospective-memory baseline, plugged into the shared harness."""

    def __init__(self, config: ProsMemConfig, system_prompt: str = "", collection_id: str = "default") -> None:
        # Deferred import so the main env (which lacks lightmem) is unaffected.
        from lightmem.memory.lightmem import LightMemory

        self.config = config
        # LightMem's internal OpenaiManager builds httpx.Client(trust_env=True), so it
        # honors HTTP(S)_PROXY from the env; if a proxy is configured on our side
        # (--proxy → config.http_proxy), export it for that client too.
        # Transport-only; no effect on model outputs.
        if getattr(config, "http_proxy", ""):
            os.environ["HTTP_PROXY"] = config.http_proxy
            os.environ["HTTPS_PROXY"] = config.http_proxy
        self.llm = LLMClient(config)  # response-generation LLM (shared backbone, same as Mem0Agent)

        mm_model, mm_key, mm_base = _lightmem_llm_route(config)
        llmlingua_path = _resolve_llmlingua_path()
        # per-sample qdrant store dir (touch-point ④), isolated like other baselines
        self._qdrant_dir = os.path.join(
            os.environ.get("TEMP", os.path.join(os.getcwd(), ".lightmem_cache")),
            f"lightmem_{collection_id}",
        )

        # ===== Official config (run_lightmem_gpt.py), verbatim except the 5 touch-points =====
        config_dict = {
            "pre_compress": True,
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": llmlingua_path,          # ① local cache path
                        "device_map": "cpu",                   # ⑤ cuda->cpu (env torch is +cpu)
                        "use_llmlingua2": True,
                    },
                },
            },
            "topic_segment": True,
            "precomp_topic_shared": True,
            "topic_segmenter": {"model_name": "llmlingua-2"},
            "messages_use": "user_only",
            "metadata_generate": True,
            "text_summary": True,
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": mm_model,                          # ② backbone
                    "api_key": mm_key,                          # ③ key
                    "max_tokens": 16000,
                    "openai_base_url": mm_base,                 # ③ base_url
                },
            },
            "extract_threshold": 0.1,
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": _EMBED_MODEL,
                    "embedding_dims": 384,
                    "model_kwargs": {"device": "cpu"},          # ⑤ cuda->cpu
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": collection_id,
                    "embedding_model_dims": 384,
                    "path": self._qdrant_dir,                   # ④ local writable path
                },
            },
            "update": "offline",
        }
        # LightMem's OpenaiManager (factory/memory_manager/openai.py) takes a different
        # config branch when OPENROUTER_API_KEY happens to be present in the env, and on
        # that branch reads a field BaseMemoryManagerConfig does not define →
        # AttributeError (mis-reported by the factory as "class not found"). Temporarily
        # hide that variable and point OPENAI_* at our endpoint around from_config so the
        # manager honors the explicit openai_base_url; env is restored in finally.
        _saved: dict[str, str | None] = {}
        try:
            for _k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
                _saved[_k] = os.environ.get(_k)
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ["OPENAI_API_KEY"] = mm_key
            os.environ["OPENAI_BASE_URL"] = mm_base
            self._mem = LightMemory.from_config(config_dict)
        finally:
            for _k, _v in _saved.items():
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v

        self._user_id = f"bench_{collection_id}"
        self.conversation_history: list[dict] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.step_count = 0
        # synthetic monotonically-increasing date (bench turns carry no timestamps;
        # LightMem.add_memory requires a time_stamp field for its normalization).
        self._day = 0

    def _next_ts(self) -> str:
        # bench has no dates; produce a stable increasing date string per turn.
        self._day += 1
        return f"2025-01-{self._day:02d}" if self._day <= 28 else f"2025-02-{(self._day - 28):02d}"

    def register_intention(self, description: str, trigger: str) -> None:
        """Seed LightMem with the PM intention BEFORE the scenario starts.
        SEED text is IDENTICAL to Mem0Agent (fairness across baselines)."""
        seed = (
            f"Remember this pending task for later: when {trigger}, "
            f"I want you to {description}. Please watch for the right "
            f"moment and act on it when the condition is met."
        )
        ts = self._next_ts()
        try:
            self._mem.add_memory(
                messages=[{"role": "user", "content": seed, "time_stamp": ts}],
                force_segment=True,
                force_extract=True,
            )
        except Exception:
            pass  # non-fatal seeding; retrieval just has less context

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1
        ts = self._next_ts()

        # Retrieve relevant memories for the current turn (official retrieve API).
        try:
            retrieved = self._mem.retrieve(user_input, limit=20)  # list[str]
        except Exception:
            retrieved = []

        # Build system prompt — SAME format/wording as Mem0Agent (fairness).
        sys_msg = self.system_prompt
        if retrieved:
            sys_msg += "\n\n=== RELEVANT MEMORIES (from your long-term memory) ===\n"
            for i, text in enumerate(retrieved, 1):
                sys_msg += f"{i}. {text}\n"
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

        # Add this turn to LightMem memory (official add_memory; user+assistant pair).
        try:
            self._mem.add_memory(
                messages=[
                    {"role": "user", "content": user_input, "time_stamp": ts},
                    {"role": "assistant", "content": response, "time_stamp": ts},
                ],
                force_segment=True,
                force_extract=True,
            )
        except Exception:
            pass

        return StepResult(step=self.step_count, user_input=user_input, agent_response=response)

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        """Release the response LLM HTTP pool and the per-sample qdrant
        store so subsequent samples start clean and FDs/disk don't accumulate."""
        # NOTE: do NOT close self.llm here. The shared harness reuses agent.llm AFTER
        # cleanup() for the D3-content judge (_wrap_d3_content → _judge_content_correctness),
        # exactly like Mem0Agent/AMemAgent whose cleanup also leaves self.llm open. Closing
        # it here broke ONLY the firing D3-content samples (D3-24..30) — their post-cleanup
        # judge call hit a closed client (surfaced as APIConnectionError after SDK retries),
        # while non-content samples (no judge) were unaffected. self.llm's own pool is GC'd
        # like the other baselines; the LightMem-specific leak is the internal manager client
        # closed below.
        # close LightMem's INTERNAL memory_manager OpenAI/httpx client — OpenaiManager
        # builds its own httpx.Client(verify=False) + OpenAI() which we did not own; not
        # closing it leaks one socket per sample → FD/connection exhaustion on the tail of
        # a 185-sample run (observed: D3-24..30 APIConnectionError after 178 samples; the
        # same samples run cleanly in isolation). Same nature as the LLMClient close() fix.
        try:
            mgr = getattr(self._mem, "manager", None)
            mcl = getattr(mgr, "client", None) if mgr else None
            if mcl is not None and hasattr(mcl, "close"):
                mcl.close()
        except Exception:
            pass
        # best-effort: close LightMem's retriever client if exposed, then drop the dir
        for obj_attr in ("embedding_retriever", "summary_retriever"):
            try:
                comp = getattr(self._mem, obj_attr, None)
                client = getattr(comp, "client", None) if comp else None
                if client and hasattr(client, "close"):
                    client.close()
            except Exception:
                pass
        self._mem = None
        try:
            if os.path.isdir(self._qdrant_dir):
                shutil.rmtree(self._qdrant_dir, ignore_errors=True)
        except Exception:
            pass
