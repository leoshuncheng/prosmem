"""A-MEM + ProsMem composition agent (composability experiment).

Demonstrates that ProsMem is an ORTHOGONAL, PLUGGABLE prospective-memory layer
on top of an unmodified retrospective host. This agent bolts the ProsMem dual-
pathway PM engine on top of A-MEM's (Xu et al., NeurIPS 2025) retrieve→augment→
generate loop:

  * PROSPECTIVE (ProsMem): the intention is registered in the PM engine, which
    checks triggers each turn and fires the deferred action (context switch) —
    inherited verbatim from ProsMemAgent.
  * RETROSPECTIVE (A-MEM): response generation uses A-MEM's linked-note memory
    (search_agentic → prompt augmentation → add_note), IDENTICAL to AMemAgent.

Two claims this supports:
  ① Gain (ProsMem-Bench): A-MEM alone cannot fire PM (pull-based retrieval, ~34%);
     A-MEM+PM fires via the PM engine (~95%). Bolting on PM upgrades the host.
  ② No degradation (dormancy): with NO intention registered the PM engine's
     process_step is a literal no-op (empty registry → associative/time/monitor
     all iterate zero armed intentions → return None, zero extra LLM calls), so
     the response path is byte-for-byte A-MEM's. Recall cannot drop by construction.

Non-invasive: AMemAgent (the frozen baseline) is NOT modified; the A-MEM memory
mechanics are re-used here via the same public AgenticMemorySystem API.
"""

from __future__ import annotations

import os

from prosmem.agent.llm import LLMClient
from prosmem.agent.loop import ProsMemAgent, StepResult
from prosmem.core.config import ProsMemConfig


class AMemPMAgent(ProsMemAgent):
    """ProsMem PM engine (inherited) + A-MEM retrospective response generation."""

    def __init__(self, config: ProsMemConfig, system_prompt: str = "",
                 collection_id: str = "default") -> None:
        # Inherit the full ProsMem stack: llm, registry, DualPathwayEngine,
        # register_intention, _execute_pm_action, state. (db_path=None → in-memory.)
        super().__init__(config, system_prompt=system_prompt)
        # A-MEM's OpenAIController reads OPENAI_* from env (same as AMemAgent).
        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url
        from agentic_memory.memory_system import AgenticMemorySystem

        self._amem = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openai",
            llm_model=config.main_model,
            api_key=config.llm_api_key,
            evo_threshold=100,  # above per-sample note count → no consolidation
        )
        self._collection_id = collection_id

    # ---- A-MEM memory ops (mirror AMemAgent; frozen baseline untouched) ----
    def _amem_search(self, query: str) -> list:
        try:
            return self._amem.search_agentic(query=query, k=5)
        except Exception:
            return []

    def _amem_add(self, user_input: str, response: str) -> None:
        try:
            self._amem.add_note(content=f"User said: {user_input}\nAgent replied: {response}")
        except Exception:
            pass

    def step(self, user_input: str) -> StepResult:
        """PM engine trigger check (ProsMem) + A-MEM retrieval-augmented response."""
        self.state.step += 1
        current_step = self.state.step
        result = StepResult(step=current_step, user_input=user_input)

        # === Prospective layer: check PM triggers (no-op when registry empty) ===
        triggered = self.engine.process_step(user_input, current_step)
        if triggered:
            result.pm_triggered = triggered
            result.pm_action_response = self._execute_pm_action(triggered)
            triggered.complete(current_step)

        # === Retrospective layer: A-MEM retrieve → augment → generate ===
        retrieved = self._amem_search(user_input)
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

        self.state.conversation_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(self.state.conversation_history)
        if triggered:
            messages.append({
                "role": "system",
                "content": (
                    f"[PROSPECTIVE MEMORY EXECUTED] You just completed a deferred task: "
                    f'"{triggered.action_description}". '
                    f"Result: {result.pm_action_response}. "
                    f"Now continue with the user's current request."
                ),
            })

        response = self.llm.chat(messages)
        result.agent_response = response
        self.state.conversation_history.append({"role": "assistant", "content": response})
        self._amem_add(user_input, response)
        self.step_results.append(result)
        return result

    def cleanup(self) -> None:
        """Drop A-MEM notes + collection (mirror AMemAgent). Does NOT close llm —
        the evaluator reuses agent.llm for the D3 content judge, then releases it."""
        try:
            for mid in list(self._amem.memories.keys()):
                self._amem.delete(mid)
        except Exception:
            pass
