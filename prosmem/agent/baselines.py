"""SOTA memory baselines for ProsMem-Bench.

Baselines:
- Mem0Agent: vector + LLM fact extraction (pip mem0ai)
- AMemAgent: Zettelkasten linked-note graph (NeurIPS 2025, Xu et al.)
- GenAgentsAgent: recency × importance × relevance memory stream (UIST 2023, Park et al.)
- MemoryBankAgent: Ebbinghaus forgetting curve (AAAI 2024, Zhong et al.)


Each baseline wraps an external memory system behind the same
`run_scenario(turns)` -> list[StepResult] interface that our evaluator uses.

The core claim being tested: retrospective memory systems (Mem0, A-MEM,
Letta, EverMemOS) CAN store the PM intention but CANNOT autonomously trigger
it at the right moment, because their retrieval is pull-based (queried at
each turn) rather than condition-driven (fired when a trigger is detected).

We give each baseline the best possible chance by:
1. Registering the intention as memory at turn 0 (before scenario starts)
2. At each turn, letting the baseline retrieve memories given the user input
3. Injecting retrieved memories into the LLM system prompt
4. Letting the LLM decide whether to act

This is essentially the "RAG for intentions" pattern — the strongest version
of "just store the intention and retrieve it when semantically relevant."
"""
from __future__ import annotations

import os
from typing import Any

from prosmem.agent.llm import LLMClient
from prosmem.agent.loop import StepResult
from prosmem.core.config import ProsMemConfig


def _mem0_config(config: ProsMemConfig, collection: str) -> dict:
    """Build a Mem0 config dict. Mem0's internal extraction/evolution LLM calls
    the same OpenAI-compatible endpoint as the main agent; embeddings use local
    BGE via HuggingFace.
    """
    _llm_model, _llm_key, _llm_base = (
        config.main_model, config.llm_api_key, config.llm_base_url)
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": _llm_model,
                "api_key": _llm_key,
                "openai_base_url": _llm_base,
                "temperature": 0.1,
                "max_tokens": 1024,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": config.embedding_model,
                "embedding_dims": 384,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection,
                "path": os.path.join(
                    os.environ.get("TEMP", ".mem0_cache"),
                    f"mem0_{collection}",
                ),
            },
        },
        "version": "v1.1",
    }


class Mem0Agent:
    """Baseline: Mem0 retrospective memory system.

    At turn 0, we inject the intention as a user-authored memory
    ("remember that when X happens I want Y done"). At each subsequent turn,
    Mem0 searches its memory store given the new user input, surfaces
    relevant memories, and we stitch them into the system prompt. The LLM
    then decides whether to enact the PM action.
    """

    def __init__(self, config: ProsMemConfig, system_prompt: str = "", collection_id: str = "default") -> None:
        # Deferred import so importing this module doesn't require mem0 unless used
        from mem0 import Memory

        self.config = config
        self.llm = LLMClient(config)  # response-generation LLM (shared backbone)
        # Each scenario gets its own collection to avoid cross-sample leakage.
        self._mem = Memory.from_config(_mem0_config(config, collection_id))
        self._user_id = f"bench_{collection_id}"
        self.conversation_history: list[dict] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.step_count = 0

    def register_intention(self, description: str, trigger: str) -> None:
        """Seed Mem0 with the PM intention BEFORE the scenario starts.

        Framed as a first-person preference message, which is Mem0's
        intended input format.
        """
        seed = (
            f"Remember this pending task for later: when {trigger}, "
            f"I want you to {description}. Please watch for the right "
            f"moment and act on it when the condition is met."
        )
        self._mem.add(seed, user_id=self._user_id)

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1

        # Retrieve top-K memories relevant to the current user input
        try:
            retrieved = self._mem.search(
                query=user_input,
                top_k=5,
                filters={"user_id": self._user_id},
            )
            memories = retrieved.get("results", []) if isinstance(retrieved, dict) else retrieved
        except Exception:
            memories = []

        # Build system prompt with retrieved memories
        sys_msg = self.system_prompt
        if memories:
            sys_msg += "\n\n=== RELEVANT MEMORIES (from your long-term memory) ===\n"
            for i, m in enumerate(memories, 1):
                text = m.get("memory") if isinstance(m, dict) else str(m)
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

        # Add this turn to memory for continued retrieval over long scenarios
        try:
            self._mem.add(
                [{"role": "user", "content": user_input},
                 {"role": "assistant", "content": response}],
                user_id=self._user_id,
            )
        except Exception:
            pass  # non-fatal; retrieval in later turns just has less context

        return StepResult(
            step=self.step_count,
            user_input=user_input,
            agent_response=response,
        )

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        """Wipe collection so subsequent samples start clean, AND release the
        underlying ChromaDB client / file handles.

        Without the release step the per-sample `Memory.from_config(...)` clients
        accumulate persistent SQLite + Parquet file descriptors, exhausting the
        soft `RLIMIT_NOFILE` after ~128 samples on Windows MSYS (observed on
        long runs). delete_all() alone clears row data but leaves the handles open.
        """
        try:
            self._mem.delete_all(user_id=self._user_id)
        except Exception:
            pass
        # Best-effort close of the underlying vector store + DB client. Mem0
        # versions differ in the exact attribute name, so we try the common
        # ones; any failure is non-fatal because gc.collect() below will
        # finalize anything we missed.
        for attr in ("reset", "close"):
            try:
                fn = getattr(self._mem, attr, None)
                if callable(fn):
                    fn()
                    break
            except Exception:
                pass
        try:
            vs = getattr(self._mem, "vector_store", None)
            client = getattr(vs, "client", None) if vs is not None else None
            if client is not None and hasattr(client, "reset"):
                client.reset()
        except Exception:
            pass
        self._mem = None
        import gc
        gc.collect()


class AMemAgent:
    """Baseline: A-MEM (Xu et al., NeurIPS 2025) Zettelkasten-style linked-note memory.

    A-MEM builds a ChromaDB-backed note system where each add_note call triggers
    an LLM-driven evolution step that links the new note to neighbors and may
    rewrite neighbor tags/context. Retrieval is a pull-based k-NN search plus
    neighbor expansion — so this is a strong "graph/link" baseline for PM.

    We seed the intention as the first note; each subsequent turn we search(k=5)
    on user input, inject results into the system prompt, let the LLM respond,
    then add the exchange as a new note (which will evolve via A-MEM's own logic).
    """

    def __init__(self, config: ProsMemConfig, system_prompt: str = "",
                 collection_id: str = "default") -> None:
        # A-MEM's OpenAIController reads OPENAI_* from the env; point it at the
        # same OpenAI-compatible endpoint as the main agent.
        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

        # Deferred import so importing this module doesn't require A-MEM unless used
        from agentic_memory.memory_system import AgenticMemorySystem

        self.config = config
        self.llm = LLMClient(config)
        # A-MEM defaults to all-MiniLM-L6-v2 (384-dim, matches paper).
        # Each instance creates a fresh ChromaDB client+collection (reset on init).
        self._mem = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openai",
            llm_model=config.main_model,
            api_key=config.llm_api_key,
            evo_threshold=100,  # far above per-sample note count, so no consolidation
        )
        self._collection_id = collection_id
        self.conversation_history: list[dict] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.step_count = 0

    def register_intention(self, description: str, trigger: str) -> None:
        """Seed A-MEM with the PM intention as the first note."""
        seed = (
            f"Pending task to remember for later: when {trigger}, "
            f"I should {description}. Watch for the right moment and act on it."
        )
        try:
            self._mem.add_note(content=seed)
        except Exception:
            pass  # non-fatal; search will just return nothing

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1

        # Retrieve top-K related notes (vector + link-expansion via search_agentic)
        try:
            retrieved = self._mem.search_agentic(query=user_input, k=5)
        except Exception:
            retrieved = []

        sys_msg = self.system_prompt
        if retrieved:
            sys_msg += "\n\n=== RELEVANT MEMORIES (from your long-term memory) ===\n"
            for i, m in enumerate(retrieved, 1):
                content = m.get("content", "") if isinstance(m, dict) else str(m)
                sys_msg += f"{i}. {content}\n"
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

        # Persist this exchange so link evolution over turns reflects the dialogue
        try:
            self._mem.add_note(content=f"User said: {user_input}\nAgent replied: {response}")
        except Exception:
            pass

        return StepResult(
            step=self.step_count,
            user_input=user_input,
            agent_response=response,
        )

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        """Drop in-memory notes and the ChromaDB collection."""
        try:
            # Delete each note (also removes from ChromaDB via AgenticMemorySystem.delete)
            for mid in list(self._mem.memories.keys()):
                self._mem.delete(mid)
        except Exception:
            pass


# ============================================================================
# Generative Agents (Park et al., UIST 2023)
# ============================================================================

import math
from dataclasses import dataclass, field

import numpy as np

from prosmem.retrieval.embedder import embed_text


@dataclass
class _MemNode:
    content: str
    created_step: int
    last_accessed_step: int
    importance: float               # 0-1 (LLM score / 10)
    embedding: np.ndarray = field(repr=False)


class GenAgentsAgent:
    """Baseline: Generative Agents memory stream (Park et al., UIST 2023).

    Retrieval score = α_recency*recency + α_importance*importance + α_relevance*relevance
      - recency  = decay ** steps_since_last_access     (decay = 0.99)
      - importance = LLM-rated 1-10 at add time, normalized to [0,1]
      - relevance = cosine similarity to query embedding (BGE, pre-normalized)

    Top-K retrieved memories are injected into the system prompt each turn.
    The PM intention is seeded at turn 0 with importance forced to 1.0.
    """

    DECAY = 0.99
    TOP_K = 5
    ALPHA_RECENCY = 1.0
    ALPHA_IMPORTANCE = 1.0
    ALPHA_RELEVANCE = 1.0

    def __init__(self, config: ProsMemConfig, system_prompt: str = "",
                 collection_id: str = "default") -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.conversation_history: list[dict] = []
        self.step_count = 0
        self._stream: list[_MemNode] = []

    def _score_importance(self, content: str) -> float:
        """LLM rates importance 1-10; return normalized to [0,1]."""
        prompt = (
            "On the scale of 1 to 10, where 1 is purely mundane "
            "(e.g., brushing teeth) and 10 is extremely poignant "
            "(e.g., a break-up, a promise to keep), rate the likely "
            "importance of the following piece of memory. "
            "Respond with only a single integer.\n\n"
            f"Memory: {content}\nRating: "
        )
        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=4,
            ).strip()
            score = int("".join(c for c in resp if c.isdigit())[:2] or "5")
            return max(1, min(10, score)) / 10.0
        except Exception:
            return 0.5

    def _add(self, content: str, importance: float | None = None) -> None:
        imp = importance if importance is not None else self._score_importance(content)
        emb = embed_text(content)
        self._stream.append(_MemNode(
            content=content,
            created_step=self.step_count,
            last_accessed_step=self.step_count,
            importance=imp,
            embedding=emb,
        ))

    def _retrieve(self, query: str) -> list[_MemNode]:
        if not self._stream:
            return []
        q = embed_text(query)
        scored = []
        for m in self._stream:
            recency = self.DECAY ** max(0, self.step_count - m.last_accessed_step)
            relevance = float(np.dot(q, m.embedding))  # BGE is normalized
            relevance = (relevance + 1) / 2  # map [-1,1] to [0,1]
            score = (self.ALPHA_RECENCY * recency
                     + self.ALPHA_IMPORTANCE * m.importance
                     + self.ALPHA_RELEVANCE * relevance)
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        top = [m for _, m in scored[:self.TOP_K]]
        for m in top:
            m.last_accessed_step = self.step_count
        return top

    def register_intention(self, description: str, trigger: str) -> None:
        seed = (
            f"Pending task: when {trigger}, I should {description}. "
            f"This is a promise to keep."
        )
        # Force max importance so the intention ranks above mundane turn notes
        self._add(seed, importance=1.0)

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1
        retrieved = self._retrieve(user_input)

        sys_msg = self.system_prompt
        if retrieved:
            sys_msg += "\n\n=== RELEVANT MEMORIES (from your long-term memory) ===\n"
            for i, m in enumerate(retrieved, 1):
                sys_msg += f"{i}. {m.content}\n"
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

        # Store the turn (without re-scoring importance on every single turn to
        # save tokens — dialogue turns get a fixed mid-importance)
        self._stream.append(_MemNode(
            content=f"User said: {user_input}\nI replied: {response}",
            created_step=self.step_count,
            last_accessed_step=self.step_count,
            importance=0.3,
            embedding=embed_text(user_input),
        ))

        return StepResult(
            step=self.step_count,
            user_input=user_input,
            agent_response=response,
        )

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        self._stream.clear()


# ============================================================================
# MemoryBank (Zhong et al., AAAI 2024) — Ebbinghaus forgetting curve
# ============================================================================


@dataclass
class _BankNode:
    content: str
    created_step: int
    strength: float                 # S in R = exp(-t/S); grows with reviews
    last_accessed_step: int
    review_count: int
    embedding: np.ndarray = field(repr=False)


class MemoryBankAgent:
    """Baseline: MemoryBank (Zhong et al., AAAI 2024).

    Each memory has a strength S. Retention at time t since last access:
        R(t) = exp(-t / S)
    Strength grows on each review: S := S * STRENGTH_MULT.
    Accessible memories are those with R >= RETENTION_THRESHOLD.
    Final retrieval score = R * relevance; take top-K accessible.
    """

    STRENGTH_INIT = 5.0
    STRENGTH_MULT = 1.5
    RETENTION_THRESHOLD = 0.1
    TOP_K = 5

    def __init__(self, config: ProsMemConfig, system_prompt: str = "",
                 collection_id: str = "default") -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.conversation_history: list[dict] = []
        self.step_count = 0
        self._bank: list[_BankNode] = []

    def _add(self, content: str, strength: float = STRENGTH_INIT) -> None:
        self._bank.append(_BankNode(
            content=content,
            created_step=self.step_count,
            strength=strength,
            last_accessed_step=self.step_count,
            review_count=0,
            embedding=embed_text(content),
        ))

    def _retrieve(self, query: str) -> list[_BankNode]:
        if not self._bank:
            return []
        q = embed_text(query)
        scored = []
        for m in self._bank:
            t = max(0, self.step_count - m.last_accessed_step)
            retention = math.exp(-t / max(m.strength, 0.1))
            if retention < self.RETENTION_THRESHOLD:
                continue  # forgotten
            relevance = (float(np.dot(q, m.embedding)) + 1) / 2
            scored.append((retention * relevance, m))
        scored.sort(key=lambda x: -x[0])
        top = [m for _, m in scored[:self.TOP_K]]
        for m in top:
            m.last_accessed_step = self.step_count
            m.review_count += 1
            m.strength *= self.STRENGTH_MULT
        return top

    def register_intention(self, description: str, trigger: str) -> None:
        seed = (
            f"Pending task: when {trigger}, I should {description}. "
            f"This is a promise to keep."
        )
        # Give the intention a high initial strength so it resists forgetting
        self._add(seed, strength=self.STRENGTH_INIT * 4)

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1
        retrieved = self._retrieve(user_input)

        sys_msg = self.system_prompt
        if retrieved:
            sys_msg += "\n\n=== RELEVANT MEMORIES (from your long-term memory) ===\n"
            for i, m in enumerate(retrieved, 1):
                sys_msg += f"{i}. {m.content}\n"
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

        self._add(f"User said: {user_input}\nI replied: {response}")
        return StepResult(
            step=self.step_count,
            user_input=user_input,
            agent_response=response,
        )

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        self._bank.clear()
