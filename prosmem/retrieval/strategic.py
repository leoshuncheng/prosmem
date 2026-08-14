"""Strategic Monitor (slow path): LLM-based deep semantic check with dynamic gating.

Maps to strategic monitoring / polling interrupt in multiprocess theory.
Runs conditionally — only when gating threshold is exceeded — to save LLM cost.
Handles non-focal EVENT_BASED and ACTIVITY_BASED triggers.
"""

from __future__ import annotations

from prosmem.agent.llm import LLMClient
from prosmem.core.intention import Intention, TriggerType
from prosmem.core.registry import IntentionRegistry
from prosmem.retrieval.embedder import cosine_similarity, embed_text


class StrategicMonitor:
    """Slow path: LLM-based deep semantic matching for non-focal and activity triggers."""

    GATING_THRESHOLD = 0.4      # Below this, skip LLM call entirely
    NON_FOCAL_MAX = 0.7         # At/above this focality, leave to Associative fast path
    MAX_CHECKS_PER_STEP = 3     # Budget cap: never check more than N intentions per step

    def __init__(
        self,
        registry: IntentionRegistry,
        llm: LLMClient,
        monitor_model: str,
        enable_gating: bool = True,
        focal_fallback: bool = False,
        multihop: bool = False,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.monitor_model = monitor_model
        self.enable_gating = enable_gating
        # When True, focal event intentions are NOT hard-skipped — kept as an LLM
        # fallback for mis-inferred focality (encoder/end-to-end path).
        self.focal_fallback = focal_fallback
        # When True, the EVENT_BASED non-focal check uses a one-hop "does the
        # situation INVOLVE / LEAD TO the cue" reframe instead of surface match,
        # so an auto-encoded cue that surfaces indirectly still bridges. Default
        # False = surface-match prompt used by the structured main table.
        self.multihop = multihop
        # Tracked separately for efficiency analysis (monitor-only, isolated from main LLM)
        self.monitor_llm_calls = 0
        self.monitor_tokens_used = 0

    def check(self, input_text: str, current_step: int) -> list[Intention]:
        """Run slow-path LLM check over gated candidates. Returns triggered intentions.

        Handles four kinds of armed intentions:
          - EVENT_BASED non-focal (focality < NON_FOCAL_MAX)
          - ACTIVITY_BASED (single activity_completion, or multi via activity_completions
            with AND logic, or single + outcome for C3)
          - TIME_BASED_DYNAMIC pre-anchor (run anchor-detection LLM judge; on match,
            set anchor_detected_step + target_step and let TimeChecker fire next step)

        TIME_BASED (absolute / interval) is handled exclusively by TimeChecker.
        """
        input_emb = embed_text(input_text)
        candidates: list[tuple[Intention, float]] = []

        for intention in self.registry.get_armed():
            tc = intention.trigger_condition

            # Plain TIME_BASED is handled by TimeChecker; skip here
            if intention.trigger_type == TriggerType.TIME_BASED:
                continue

            # TIME_BASED_DYNAMIC: only adjudicate anchor if not yet detected;
            # once detected, TimeChecker will fire when current_step ≥ target_step
            if intention.trigger_type == TriggerType.TIME_BASED_DYNAMIC:
                if tc.anchor_detected_step is not None:
                    continue
                ref_emb = tc.anchor_event_embedding
                if self.enable_gating and ref_emb is not None:
                    sim = cosine_similarity(input_emb, ref_emb)
                    if sim < self.GATING_THRESHOLD:
                        continue
                    candidates.append((intention, sim))
                else:
                    candidates.append((intention, 0.0))
                continue

            # EVENT_BASED: non-focal go through the monitor; focal are normally left
            # to the Assoc fast path. With focal_fallback ON, focal intentions ALSO
            # stay eligible (still gated + budget-capped), so a mis-inferred focality
            # does not silently miss.
            if intention.trigger_type == TriggerType.EVENT_BASED:
                if tc.focality >= self.NON_FOCAL_MAX and not self.focal_fallback:
                    continue

            # C2 multi-activity (AND): gate on the MAX similarity over ALL activity
            # embeddings, so a turn that completes ANY one of the watched activities
            # passes the gate (completions arrive across different turns).
            if tc.activity_completions and tc.activity_completion_embeddings and tc.outcome is None:
                sim = max(cosine_similarity(input_emb, e) for e in tc.activity_completion_embeddings)
                if self.enable_gating and sim < self.GATING_THRESHOLD:
                    continue
                candidates.append((intention, sim))
                continue

            # Dynamic gating: use trigger_embedding (or activity_embedding) as relevance proxy
            ref_emb = intention.trigger_embedding
            if ref_emb is None and tc.activity_embedding is not None:
                ref_emb = tc.activity_embedding

            if self.enable_gating and ref_emb is not None:
                sim = cosine_similarity(input_emb, ref_emb)
                if sim < self.GATING_THRESHOLD:
                    continue
                candidates.append((intention, sim))
            else:
                # Gating disabled: always include
                candidates.append((intention, 0.0))

        # Budget cap: check top-K most relevant first
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[: self.MAX_CHECKS_PER_STEP]

        triggered: list[Intention] = []
        for intention, _sim in candidates:
            tc = intention.trigger_condition
            if intention.trigger_type == TriggerType.TIME_BASED_DYNAMIC:
                # Anchor-detection branch: update state, do NOT fire this step
                if self._llm_check_anchor(intention, input_text):
                    tc.anchor_detected_step = current_step
                    delay = tc.delay_steps or 0
                    tc.target_step = current_step + delay
                continue
            # C2 multi-activity AND: stateful — mark each activity done as its turn
            # passes; fire only once ALL are marked complete (single-turn checks
            # accumulate across turns, since completions arrive in different turns).
            if (intention.trigger_type == TriggerType.ACTIVITY_BASED
                    and tc.activity_completions and tc.outcome is None):
                if self._check_c2_and(intention, input_text):
                    intention.trigger(current_step)
                    triggered.append(intention)
                continue
            if self._llm_check(intention, input_text):
                intention.trigger(current_step)
                triggered.append(intention)

        triggered.sort(key=lambda i: i.priority)
        return triggered

    def _check_c2_and(self, intention: Intention, input_text: str) -> bool:
        """C2: check each not-yet-completed activity against the current turn, mark
        the ones that completed, and return True only when ALL are complete. State
        (completed_activities) accumulates across turns on the intention."""
        tc = intention.trigger_condition
        for idx, activity in enumerate(tc.activity_completions):
            if idx in tc.completed_activities:
                continue
            prompt = (
                "Question: Has the following activity been completed in the current turn?\n\n"
                f"Activity: {activity}\n\n"
                f"Current turn: {input_text}\n\n"
                "Answer strictly with YES or NO."
            )
            if "YES" in self._ask(prompt).strip().upper():
                tc.completed_activities.add(idx)
        return len(tc.completed_activities) == len(tc.activity_completions)

    def _llm_check(self, intention: Intention, input_text: str) -> bool:
        """Ask the monitor LLM whether the trigger condition is satisfied."""
        tc = intention.trigger_condition

        if intention.trigger_type == TriggerType.ACTIVITY_BASED:
            # C3: outcome-conditional activity (fire only if the watched activity reaches
            # the configured outcome IN THE CURRENT TURN). The prompt names the activity
            # and the outcome-to-watch separately, and forces PENDING when the activity is
            # merely starting / in-progress — this prevents firing at turn 1 on samples
            # whose activity description merely mentions a possible failure.
            if tc.outcome is not None:
                activity = tc.activity_completion or (
                    tc.activity_completions[0] if tc.activity_completions else "the activity"
                )
                prompt = (
                    "Question: In the CURRENT turn, has the watched activity reached a "
                    "definitive outcome yet?\n\n"
                    f"Watched activity: {activity}\n"
                    f"Outcome we are waiting for: {tc.outcome}\n\n"
                    f"Current turn: {input_text}\n\n"
                    "Reply strictly with one of: SUCCESS, FAIL, PENDING. Answer PENDING if "
                    "the activity is only starting or still in progress and THIS turn shows "
                    "no definitive success or failure."
                )
                resp = self._ask(prompt)
                observed = resp.strip().upper()
                # Only fire when the observed outcome matches the configured one
                if tc.outcome.upper() == "FAIL" and "FAIL" in observed:
                    return True
                if tc.outcome.upper() == "SUCCESS" and (
                    "SUCCESS" in observed and "FAIL" not in observed
                ):
                    return True
                return False

            # C2 multi-activity AND is handled statefully in _check_c2_and (not here).

            # C1: single activity_completion string
            prompt = (
                "Question: Has the following activity been completed in the current context?\n\n"
                f"Activity to watch for: {tc.activity_completion}\n\n"
                f"Current context: {input_text}\n\n"
                "Answer strictly with YES or NO."
            )
            return "YES" in self._ask(prompt).strip().upper()

        # EVENT_BASED non-focal
        cues = ", ".join(tc.event_cues) if tc.event_cues else "(no explicit cues)"
        if self.multihop:
            # One-hop reframe: an auto-encoded cue (e.g. "conference/talk") may surface
            # indirectly in a later turn (e.g. "my seminar this afternoon"). Surface
            # match misses it; asking whether the situation INVOLVES / LEADS TO the cue
            # bridges the gap, while the NO-guard keeps merely-related decoys out.
            prompt = (
                "Question: Does the current situation involve, lead to, or naturally "
                "include the trigger cue?\n\n"
                f"Trigger cue(s): {cues}\n"
                f"Planned action if triggered: {intention.action_description}\n\n"
                f"Current situation: {input_text}\n\n"
                "Answer strictly with YES or NO. Answer YES if the situation clearly "
                "involves or leads to the cue, even when it is not named literally; "
                "answer NO for unrelated or only coincidentally similar situations."
            )
            return "YES" in self._ask(prompt).strip().upper()
        prompt = (
            "Question: Does the current context semantically match this trigger?\n\n"
            f"Trigger cues: {cues}\n"
            f"Action to execute if matched: {intention.action_description}\n\n"
            f"Current context: {input_text}\n\n"
            "Answer strictly with YES or NO. Only answer YES if the match is "
            "clear and unambiguous."
        )
        return "YES" in self._ask(prompt).strip().upper()

    def _llm_check_anchor(self, intention: Intention, input_text: str) -> bool:
        """B3: ask whether the dynamic-time anchor event has just occurred or been
        announced in the current turn. Accepts announce/plan/confirm/execute forms
        (e.g., "I'm planning to merge X today") because B3 anchors are often
        future-tense announcements per the cognitive paradigm."""
        anchor = intention.trigger_condition.anchor_event or "(no anchor described)"
        prompt = (
            "Question: In the CURRENT turn, does the user announce, plan, "
            "confirm, or execute the following anchor event?\n\n"
            f"Anchor event: {anchor}\n\n"
            f"Current turn: {input_text}\n\n"
            "Answer strictly with YES or NO. Answer YES if the user explicitly "
            "announces an intention to do X, plans to do X in the immediate "
            "timeframe (e.g. today/now/soon), confirms X has just happened, "
            "or executes X in this turn. Answer NO if X is merely a distant "
            "future possibility, a past recollection unrelated to current action, "
            "or absent from this turn."
        )
        return "YES" in self._ask(prompt).strip().upper()

    def _ask(self, prompt: str) -> str:
        """Centralized monitor-LLM call with token accounting."""
        before = self.llm.total_tokens_used
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.monitor_model,
            temperature=0.0,
            max_tokens=10,
        )
        self.monitor_llm_calls += 1
        self.monitor_tokens_used += self.llm.total_tokens_used - before
        return resp
