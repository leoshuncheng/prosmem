"""EverOS / EverMemOS baseline agent (arXiv:2601.02163; EverMind-AI/EverOS @ dbfe348).

Faithful reproduction wrapper around the OFFICIAL EverOS server. EverOS is a
server-side, md-first memory-extraction framework (Markdown = truth + SQLite =
state + LanceDB = vector/BM25 index). Unlike the in-process baselines (Mem0 /
A-MEM / LightMem), EverOS is a REST service: this agent is a thin HTTP client
that talks to a locally-running `everos server` over its official v1 API
(`/api/v1/memory/{add,flush,search}`). The server is configured entirely via
the official `.env` / EVEROS_* settings:
  • LLM   = the configured OpenAI-compatible endpoint; model = our backbone, set
            per run via EVEROS_LLM__MODEL at server start.
  • EMBED = official model Qwen/Qwen3-Embedding-4B via the same endpoint's
            embeddings API (live-verified dim=2560).
  • RERANK= intentionally unset → official graceful no-rerank (episode path
            ignores rerank by contract).
ZERO EverOS algorithm parameters are changed; defaults (HYBRID search, server-
side boundary detection + extraction) are used verbatim. The fcntl shim in the
everos venv is a Windows-only no-op port of POSIX advisory locking (EverOS is
Unix-first); it does not touch extraction or retrieval behavior.

Online per-turn adaptation (Option A, identical to the Mem0 / A-MEM / LightMem
baselines in this harness): add the turn → flush (force extraction) → search.
The harness interface (register_intention / step / run_scenario / cleanup), the
intention SEED text, and the per-turn response prompt are IDENTICAL to Mem0Agent
so every baseline is scored by the same evaluator on equal footing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from prosmem.agent.llm import LLMClient
from prosmem.agent.loop import StepResult
from prosmem.core.config import ProsMemConfig

# Local EverOS server (started separately with config-B .env). Localhost calls
# MUST bypass any HTTP(S)_PROXY (the proxy returns 502 for 127.0.0.1).
_SERVER_URL = os.environ.get("EVEROS_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
_API = _SERVER_URL + "/api/v1/memory"

# Dedicated no-proxy opener so localhost requests are never intercepted by a
# system-level HTTP(S) proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(path: str, body: dict, timeout: float = 120.0) -> dict:
    """POST JSON to the EverOS server, bypassing the proxy. Returns parsed JSON."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _API + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class EverOSAgent:
    """EverOS server-backed retrospective-memory baseline, plugged into the harness.

    Per-sample isolation is by a unique ``user_id`` (memory owner) + ``session_id``
    under the shared server's memory root, so concurrent samples never cross-read.
    """

    def __init__(self, config: ProsMemConfig, system_prompt: str = "", collection_id: str = "default") -> None:
        self.config = config
        self.llm = LLMClient(config)  # response-generation LLM (shared backbone, == Mem0Agent)
        # Owner + session namespace — unique per sample (memory partition key).
        safe = "".join(c if (c.isalnum() or c in "_.@+-") else "_" for c in str(collection_id))
        self._user_id = f"prosmem_{safe}"
        self._session_id = f"sess_{safe}"
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.conversation_history: list[dict] = []
        self.step_count = 0
        # bench turns carry no timestamps; EverOS requires a strictly-positive ms
        # epoch per message → synthetic monotonically-increasing ms clock.
        self._ts_ms = 1_735_689_600_000  # 2025-01-01T00:00:00Z in ms

    def _next_ts(self) -> int:
        self._ts_ms += 86_400_000  # +1 day per message (monotonic, ms)
        return self._ts_ms

    def _add(self, messages: list[tuple[str, str]]) -> None:
        """POST a batch of (role, content) messages into the session buffer, then
        /flush to force boundary detection + extraction. A turn's user+assistant
        pair is sent as ONE batch + ONE flush — mirrors the single per-turn
        add_memory call of the LightMem/Mem0 baselines (fairness + halves flushes)."""
        try:
            _post("/add", {
                "session_id": self._session_id,
                "messages": [
                    {"sender_id": self._user_id, "role": role,
                     "timestamp": self._next_ts(), "content": content}
                    for role, content in messages
                ],
            })
            # /flush forces extraction so the turn becomes searchable (OSS-only).
            _post("/flush", {"session_id": self._session_id})
        except Exception:
            pass  # non-fatal; retrieval just has less context (faithful degradation)

    def _search(self, query: str, top_k: int = 20) -> list[str]:
        """Hybrid search for relevant episodes; return formatted memory snippets.

        EverOS writes markdown synchronously but rebuilds the LanceDB index in an
        async cascade coroutine, so a search right after a flush may miss the new
        row (official api.md: sub-second typical, up to ~10-15s under load — "retry
        with backoff" for read-your-write). We do exactly that: bounded backoff on
        an empty result so EverOS gets its full faithful recall, not a race-loss."""
        resp = None
        for delay in (0.0, 1.0, 2.0, 4.0):
            if delay:
                time.sleep(delay)
            try:
                resp = _post("/search", {
                    "user_id": self._user_id,
                    "query": query,
                    "method": "hybrid",       # config-B default
                    "top_k": top_k,
                })
            except Exception:
                resp = None
                continue
            if resp.get("data", {}).get("episodes"):
                break
        if resp is None:
            return []
        out: list[str] = []
        for ep in resp.get("data", {}).get("episodes", []):
            # Episode payload: summary / subject / episode (narrative) + atomic_facts.
            text = ep.get("episode") or ep.get("summary") or ep.get("subject") or ""
            facts = ep.get("atomic_facts") or []
            if facts:
                joined = "; ".join(f.get("content", "") for f in facts if f.get("content"))
                if joined:
                    text = (text + " | " + joined) if text else joined
            if text:
                out.append(text)
        return out

    def register_intention(self, description: str, trigger: str) -> None:
        """Seed EverOS with the PM intention BEFORE the scenario starts.
        SEED text is IDENTICAL to Mem0Agent (fairness across baselines)."""
        seed = (
            f"Remember this pending task for later: when {trigger}, "
            f"I want you to {description}. Please watch for the right "
            f"moment and act on it when the condition is met."
        )
        self._add([("user", seed)])

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1

        # Retrieve relevant memories for the current turn (official /search).
        retrieved = self._search(user_input, top_k=20)

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

        # Persist this turn into EverOS memory as ONE batch (official /add + /flush).
        self._add([("user", user_input), ("assistant", response)])

        return StepResult(step=self.step_count, user_input=user_input, agent_response=response)

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]

    def cleanup(self) -> None:
        """Release nothing that the harness still needs.

        Do NOT close self.llm here: the shared harness reuses agent.llm AFTER
        cleanup() for the D3-content judge (_wrap_d3_content), exactly like
        Mem0Agent / LightMemAgent. The EverOS server is a shared persistent
        process (not per-sample), and per-sample isolation is by user_id /
        session_id, so there is no server-side resource to tear down here.
        self.llm's HTTP pool is closed by the evaluator's terminal _release_llm.
        """
        return
