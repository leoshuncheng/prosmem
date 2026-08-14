"""Dual-Pathway Retrieval Engine: coordinates associative + time + strategic monitor.

All three pathways: fast associative + time checker + slow strategic monitor.
"""

from __future__ import annotations

from prosmem.agent.llm import LLMClient
from prosmem.core.intention import Intention
from prosmem.core.registry import IntentionRegistry
from prosmem.retrieval.associative import AssociativeRetriever, TimeChecker
from prosmem.retrieval.strategic import StrategicMonitor


class DualPathwayEngine:
    """Coordinate all retrieval pathways and return highest-priority triggered intention."""

    def __init__(
        self,
        registry: IntentionRegistry,
        llm: LLMClient | None = None,
        monitor_model: str | None = None,
        enable_associative: bool = True,
        enable_monitor: bool = True,
        enable_gating: bool = True,
        focal_fallback: bool = False,
        monitor_multihop: bool = False,
    ) -> None:
        self.registry = registry
        self.enable_associative = enable_associative
        self.associative = AssociativeRetriever(registry)
        self.time_checker = TimeChecker(registry)
        self.monitor: StrategicMonitor | None = None
        if enable_monitor and llm is not None and monitor_model:
            self.monitor = StrategicMonitor(
                registry, llm, monitor_model, enable_gating=enable_gating,
                focal_fallback=focal_fallback,
                multihop=monitor_multihop,
            )

    def process_step(
        self,
        input_text: str,
        current_step: int,
    ) -> Intention | None:
        """Run all enabled retrieval pathways. Returns highest-priority triggered intention."""
        # Expire overdue intentions first
        self.registry.expire_overdue(current_step)

        # Path 1: Associative retrieval (fast path, event-based focal)
        assoc_results: list[Intention] = []
        if self.enable_associative:
            assoc_results = self.associative.retrieve(input_text, current_step)

        # Path 2: Time-based checks
        time_results = self.time_checker.check(current_step)

        # Path 3: Strategic monitor (slow path, non-focal + activity-based)
        monitor_results: list[Intention] = []
        if self.monitor is not None:
            monitor_results = self.monitor.check(input_text, current_step)

        # Merge and deduplicate
        seen_ids: set[str] = set()
        all_triggered: list[Intention] = []
        for i in assoc_results + time_results + monitor_results:
            if i.intention_id not in seen_ids:
                seen_ids.add(i.intention_id)
                all_triggered.append(i)

        if not all_triggered:
            return None

        # Priority arbitration (like NVIC): lower priority number = higher priority,
        # ties broken by importance
        all_triggered.sort(key=lambda i: (i.priority, -i.importance))
        return all_triggered[0]
