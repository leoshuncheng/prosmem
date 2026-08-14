"""Core data structures for ProsMem: Intention, TriggerCondition, and related enums."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TriggerType(str, Enum):
    EVENT_BASED = "event_based"
    TIME_BASED = "time_based"
    TIME_BASED_DYNAMIC = "time_based_dynamic"  # B3: target_step set after anchor detected
    ACTIVITY_BASED = "activity_based"


class IntentionStatus(str, Enum):
    CREATED = "created"
    ARMED = "armed"
    TRIGGERED = "triggered"
    EXECUTING = "executing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SNOOZED = "snoozed"


@dataclass
class TriggerCondition:
    """Defines when an intention should fire."""

    # Event-based fields
    event_cues: list[str] = field(default_factory=list)
    event_cue_embeddings: list[np.ndarray] = field(default_factory=list)
    focality: float = 0.5          # [0, 1] — high = focal cue, low = non-focal
    semantic_threshold: float = 0.7

    # Time-based fields
    target_step: int | None = None      # Absolute step number to trigger
    step_interval: int | None = None    # B2: every N steps (rearm path in complete())

    # Time-based dynamic (B3): anchor + delay
    anchor_event: str | None = None             # Anchor description (LLM-judged)
    anchor_event_embedding: np.ndarray | None = None
    delay_steps: int | None = None              # Fire delay_steps after anchor detected
    anchor_detected_step: int | None = None     # Runtime: step when anchor was detected

    # Activity-based fields
    # Legacy single-activity (C1, backward-compatible)
    activity_completion: str | None = None
    activity_embedding: np.ndarray | None = None
    # C2: multi-activity convergence (AND of activities — only logic plan calls for)
    activity_completions: list[str] = field(default_factory=list)
    activity_completion_embeddings: list[np.ndarray] = field(default_factory=list)
    completed_activities: set[int] = field(default_factory=set)  # Runtime: indices observed done (C2 AND tracking across turns)
    # C3: conditional activity (fire only if outcome matches)
    outcome: str | None = None                   # "success" | "fail" | None

    def to_dict(self) -> dict:
        return {
            "event_cues": self.event_cues,
            "focality": self.focality,
            "semantic_threshold": self.semantic_threshold,
            "target_step": self.target_step,
            "step_interval": self.step_interval,
            "anchor_event": self.anchor_event,
            "delay_steps": self.delay_steps,
            "anchor_detected_step": self.anchor_detected_step,
            "activity_completion": self.activity_completion,
            "activity_completions": self.activity_completions,
            "outcome": self.outcome,
        }


@dataclass
class Intention:
    """A prospective memory intention: remember to do X when Y happens."""

    # Identity
    intention_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    source_context: str = ""

    # Prospective component — WHEN to act
    trigger_type: TriggerType = TriggerType.EVENT_BASED
    trigger_condition: TriggerCondition = field(default_factory=TriggerCondition)

    # Retrospective component — WHAT to do
    action_description: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)

    # Scheduling metadata
    priority: int = 4              # 0 (highest) to 7 (lowest), maps to ARM NVIC
    importance: float = 0.5        # [0, 1]
    expiry_step: int | None = None # Auto-cancel after this step

    # State
    status: IntentionStatus = IntentionStatus.CREATED

    # Embeddings (populated by Registry on arm())
    trigger_embedding: np.ndarray | None = None

    # Implementation intention (Gollwitzer 1999)
    implementation_intention: str = ""  # "IF <situation>, THEN <action>"

    # Tracking
    triggered_at_step: int | None = None
    completed_at_step: int | None = None

    def arm(self) -> None:
        if self.status == IntentionStatus.CREATED:
            self.status = IntentionStatus.ARMED

    def trigger(self, step: int) -> None:
        if self.status == IntentionStatus.ARMED:
            self.status = IntentionStatus.TRIGGERED
            self.triggered_at_step = step

    def begin_execution(self) -> None:
        if self.status == IntentionStatus.TRIGGERED:
            self.status = IntentionStatus.EXECUTING

    def complete(self, step: int) -> None:
        """Finalize a fired intention.

        For non-recurring intentions (default): set COMPLETED + raise threshold to
        2.0 to suppress commission errors (the acknowledge-and-clear step).

        For B2 recurring intentions (step_interval set): re-arm with the next
        target_step instead of completing — adds the Executing→Armed edge to the
        lifecycle state machine. The associative threshold and activity matching
        machinery are left untouched so the intention remains eligible for
        future fires.
        """
        self.completed_at_step = step
        if self.trigger_condition.step_interval is not None:
            # Recurring time-based intention (B2): rearm rather than complete
            self.status = IntentionStatus.ARMED
            # TimeChecker tracks _last_interval_fire by intention_id internally,
            # so we only need to keep status ARMED for the next interval to fire.
            return
        # Default (non-recurring): commit Acknowledged-and-Cleared step
        self.status = IntentionStatus.COMPLETED
        self.trigger_condition.semantic_threshold = 2.0

    def cancel(self) -> None:
        self.status = IntentionStatus.CANCELLED

    def snooze(self) -> None:
        if self.status == IntentionStatus.TRIGGERED:
            self.status = IntentionStatus.SNOOZED

    def rearm(self) -> None:
        if self.status == IntentionStatus.SNOOZED:
            self.status = IntentionStatus.ARMED

    def is_active(self) -> bool:
        return self.status in (IntentionStatus.ARMED, IntentionStatus.SNOOZED)

    def is_expired(self, current_step: int) -> bool:
        return self.expiry_step is not None and current_step > self.expiry_step

    def to_dict(self) -> dict:
        return {
            "intention_id": self.intention_id,
            "trigger_type": self.trigger_type.value,
            "trigger_condition": self.trigger_condition.to_dict(),
            "action_description": self.action_description,
            "implementation_intention": self.implementation_intention,
            "priority": self.priority,
            "importance": self.importance,
            "status": self.status.value,
        }
