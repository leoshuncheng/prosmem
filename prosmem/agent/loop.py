"""Minimal agent loop with ProsMem integration.

Implements the simplest possible agent that can:
1. Process a sequence of user messages (simulating multi-turn interaction)
2. Check for PM triggers at each step via DualPathwayEngine
3. Execute context switch when a trigger fires
4. Resume the ongoing task after PM execution
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prosmem.agent.llm import LLMClient
from prosmem.core.config import ProsMemConfig
from prosmem.core.intention import Intention, IntentionStatus, TriggerCondition, TriggerType
from prosmem.core.registry import IntentionRegistry
from prosmem.retrieval.embedder import embed_text, embed_texts
from prosmem.retrieval.engine import DualPathwayEngine


@dataclass
class StepResult:
    """Result of a single agent step."""
    step: int
    user_input: str
    agent_response: str = ""
    pm_triggered: Intention | None = None
    pm_action_response: str = ""


@dataclass
class AgentState:
    """Tracks the agent's working state for context save/restore."""
    conversation_history: list[dict] = field(default_factory=list)
    current_task: str = ""
    step: int = 0


class ProsMemAgent:
    """Agent with ProsMem prospective memory layer."""

    def __init__(self, config: ProsMemConfig, system_prompt: str = "",
                 db_path: str | None = None) -> None:
        self.config = config
        self.llm = LLMClient(config)
        # db_path=None → in-memory (default, used by all bench runs). A path turns
        # on SQLite persistence for cross-session PM (Case Study 3).
        self.registry = IntentionRegistry(db_path=db_path)
        self.engine = DualPathwayEngine(
            self.registry,
            llm=self.llm,
            monitor_model=config.monitor_model,
            enable_associative=config.enable_associative,
            enable_monitor=config.enable_monitor,
            enable_gating=config.enable_gating,
            focal_fallback=config.enable_focal_fallback,
            monitor_multihop=config.enable_monitor_multihop,
        )
        self.state = AgentState()
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Complete the user's requests. "
            "When you receive a [PROSPECTIVE MEMORY TRIGGERED] message, "
            "execute the described action immediately."
        )
        self.step_results: list[StepResult] = []

    def register_intention(
        self,
        trigger_type: TriggerType,
        action_description: str,
        event_cues: list[str] | None = None,
        focality: float = 0.5,
        semantic_threshold: float = 0.7,
        target_step: int | None = None,
        step_interval: int | None = None,
        activity_completion: str | None = None,
        # Composite-category fields:
        activity_completions: list[str] | None = None,  # C2 multi-activity AND
        outcome: str | None = None,                     # C3 success/fail filter
        anchor_event: str | None = None,                # B3 anchor description
        delay_steps: int | None = None,                 # B3 fire delay after anchor
        priority: int = 4,
        importance: float = 0.5,
        expiry_step: int | None = None,
        implementation_intention: str = "",
    ) -> str:
        """Create and register an intention. Returns intention_id."""
        tc = TriggerCondition(
            event_cues=event_cues or [],
            focality=focality,
            semantic_threshold=semantic_threshold,
            target_step=target_step,
            step_interval=step_interval,
            activity_completion=activity_completion,
            activity_completions=activity_completions or [],
            outcome=outcome,
            anchor_event=anchor_event,
            delay_steps=delay_steps,
        )

        # Compute embeddings for event cues
        if event_cues:
            tc.event_cue_embeddings = embed_texts(event_cues)

        if activity_completion:
            tc.activity_embedding = embed_text(activity_completion)

        # C2: per-activity embeddings (used by strategic monitor for gating)
        if activity_completions:
            tc.activity_completion_embeddings = embed_texts(activity_completions)

        # B3: anchor event embedding for strategic-monitor gating
        if anchor_event:
            tc.anchor_event_embedding = embed_text(anchor_event)

        # Ablation (c): when enable_ii=False, strip implementation intention
        ii_stored = implementation_intention if self.config.enable_ii else ""

        intention = Intention(
            trigger_type=trigger_type,
            trigger_condition=tc,
            action_description=action_description,
            priority=priority,
            importance=importance,
            expiry_step=expiry_step,
            implementation_intention=ii_stored,
        )

        # Compute trigger embedding from all cues + action for general matching.
        # For B3 anchor, fold the anchor description in too so gating works pre-anchor.
        combined_parts = list(event_cues or [])
        if anchor_event:
            combined_parts.append(anchor_event)
        combined_parts.append(action_description)
        intention.trigger_embedding = embed_text(" ".join(combined_parts))

        return self.registry.register(intention)

    def step(self, user_input: str) -> StepResult:
        """Process one turn: check PM triggers, handle context switch if needed, respond."""
        self.state.step += 1
        current_step = self.state.step
        result = StepResult(step=current_step, user_input=user_input)

        # === ProsMem Layer: check triggers BEFORE reasoning ===
        triggered = self.engine.process_step(user_input, current_step)

        if triggered:
            result.pm_triggered = triggered
            # Context switch: save state, execute PM action, restore
            result.pm_action_response = self._execute_pm_action(triggered)
            triggered.complete(current_step)

        # === Normal agent reasoning ===
        self.state.conversation_history.append({"role": "user", "content": user_input})
        messages = self._build_messages()

        # If PM was triggered, inject notification into context
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
        self.step_results.append(result)

        return result

    def _execute_pm_action(self, intention: Intention) -> str:
        """Execute a PM action via LLM call (the ISR)."""
        intention.begin_execution()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"[PROSPECTIVE MEMORY TRIGGERED]\n"
                    f"Intention: {intention.action_description}\n"
                    f"Implementation plan: {intention.implementation_intention}\n"
                    f"Execute this action now and report what you did."
                ),
            },
        ]
        return self.llm.chat(messages)

    def _build_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.state.conversation_history)
        return messages

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        """Run a full multi-turn scenario. Returns all step results."""
        results = []
        for turn in turns:
            results.append(self.step(turn))
        return results


class VanillaAgent:
    """Baseline: plain LLM with no memory system. For comparison."""

    def __init__(self, config: ProsMemConfig, system_prompt: str = "") -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.conversation_history: list[dict] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.step_count = 0

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1
        self.conversation_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        response = self.llm.chat(messages)
        self.conversation_history.append({"role": "assistant", "content": response})
        return StepResult(
            step=self.step_count,
            user_input=user_input,
            agent_response=response,
        )

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]


class NaiveReminderAgent:
    """Baseline: all pending intentions stuffed into system prompt as TODO list."""

    def __init__(self, config: ProsMemConfig, system_prompt: str = "") -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.conversation_history: list[dict] = []
        self.reminders: list[dict] = []  # {"description": ..., "trigger": ...}
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.step_count = 0

    def add_reminder(self, description: str, trigger: str) -> None:
        self.reminders.append({"description": description, "trigger": trigger})

    def step(self, user_input: str) -> StepResult:
        self.step_count += 1
        self.conversation_history.append({"role": "user", "content": user_input})

        # Build system prompt with TODO list
        sys_msg = self.system_prompt
        if self.reminders:
            sys_msg += "\n\n=== PENDING REMINDERS (check each turn if conditions are met) ===\n"
            for i, r in enumerate(self.reminders, 1):
                sys_msg += f"{i}. WHEN: {r['trigger']} → DO: {r['description']}\n"
            sys_msg += "\nIf any reminder's condition is met in the current conversation, "
            sys_msg += "execute the action and mention it explicitly in your response."

        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(self.conversation_history)
        response = self.llm.chat(messages)
        self.conversation_history.append({"role": "assistant", "content": response})
        return StepResult(
            step=self.step_count,
            user_input=user_input,
            agent_response=response,
        )

    def run_scenario(self, turns: list[str]) -> list[StepResult]:
        return [self.step(turn) for turn in turns]
