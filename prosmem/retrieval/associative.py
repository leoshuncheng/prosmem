"""Associative Retrieval Path (fast path): embedding-based trigger matching.

Maps to hardware interrupt / spontaneous retrieval in multiprocess theory.
Runs every agent step at near-zero LLM cost (embedding only).
"""

from __future__ import annotations

from prosmem.core.intention import Intention, TriggerType
from prosmem.core.registry import IntentionRegistry
from prosmem.retrieval.embedder import cosine_similarity, embed_text


class AssociativeRetriever:
    """Fast path: match current input against armed intention trigger embeddings."""

    def __init__(self, registry: IntentionRegistry) -> None:
        self.registry = registry

    def retrieve(self, input_text: str, current_step: int) -> list[Intention]:
        """Check all armed event-based intentions against input.

        Returns triggered intentions sorted by (priority, similarity) descending.
        """
        input_emb = embed_text(input_text)
        triggered: list[tuple[Intention, float]] = []

        for intention in self.registry.get_armed():
            if intention.trigger_type != TriggerType.EVENT_BASED:
                continue
            if not intention.trigger_condition.event_cue_embeddings:
                continue

            # Max similarity across all cue embeddings
            max_sim = max(
                cosine_similarity(input_emb, cue_emb)
                for cue_emb in intention.trigger_condition.event_cue_embeddings
            )

            # Focal discount removed: focal cues already match high (0.8+);
            # a discount only helps decoy mentions cross the threshold.
            threshold = intention.trigger_condition.semantic_threshold

            if max_sim >= threshold:
                triggered.append((intention, max_sim))

        # Sort: higher priority (lower number) first, then higher similarity
        triggered.sort(key=lambda x: (x[0].priority, -x[1]))

        # Actually trigger the intentions
        result = []
        for intention, _sim in triggered:
            intention.trigger(current_step)
            result.append(intention)

        return result


class TimeChecker:
    """Check time-based (step-based) triggers. Runs every step, O(n) scan."""

    def __init__(self, registry: IntentionRegistry) -> None:
        self.registry = registry
        self._last_interval_fire: dict[str, int] = {}

    def check(self, current_step: int) -> list[Intention]:
        triggered: list[Intention] = []

        for intention in self.registry.get_armed():
            # Accept both TIME_BASED (absolute / interval) and TIME_BASED_DYNAMIC
            # (B3 — only fires once strategic monitor has set target_step after
            # detecting the anchor).
            if intention.trigger_type not in (
                TriggerType.TIME_BASED,
                TriggerType.TIME_BASED_DYNAMIC,
            ):
                continue

            tc = intention.trigger_condition
            fire = False

            # Absolute step trigger (covers B1 and B3-after-anchor)
            if tc.target_step is not None and current_step >= tc.target_step:
                fire = True

            # Interval trigger (B2): rearm path in Intention.complete() keeps the
            # intention ARMED, so subsequent intervals re-fire here. _last_interval_fire
            # bookkeeping prevents same-step double-fires.
            if tc.step_interval is not None:
                last = self._last_interval_fire.get(intention.intention_id, 0)
                if current_step - last >= tc.step_interval:
                    fire = True
                    self._last_interval_fire[intention.intention_id] = current_step

            if fire:
                intention.trigger(current_step)
                triggered.append(intention)

        triggered.sort(key=lambda i: i.priority)
        return triggered
