"""Evaluation harness for ProsMem-Bench.

Runs all samples across ProsMem, Vanilla, and NaiveReminder agents.
Computes Tier 1 metrics: PM Hit Rate, Precision, F1, Latency.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from prosmem.agent.llm import LLMClient
from prosmem.agent.loop import NaiveReminderAgent, ProsMemAgent, StepResult, VanillaAgent
from prosmem.core.config import ProsMemConfig
from prosmem.core.intention import TriggerType


@dataclass
class IntentionResult:
    """Per-intention scoring inside a multi-intention (D1) sample."""
    intention_id: str
    trigger_turn: int
    actual_trigger_turns: list[int]
    response_mentions_turns: list[int]
    hit: bool = False
    precise: bool = False
    latency: int = 0


@dataclass
class SampleResult:
    sample_id: str
    category: str
    agent_name: str
    # Did the agent trigger PM at the correct turn(s)?
    expected_trigger_turns: list[int]
    actual_trigger_turns: list[int]
    # For Vanilla/Naive: did the response mention the expected action?
    response_mentions_action: list[int]
    # Metrics
    hit: bool = False            # True if triggered at any expected turn
    precise: bool = False        # True if no false positive triggers
    latency: int = 0             # Steps between expected trigger and actual (0 = perfect)
    tokens_used: int = 0
    wall_time: float = 0.0
    # Composite-category extensions:
    intention_results: list[IntentionResult] = field(default_factory=list)  # D1
    ongoing_task_accuracy: float | None = None  # D2 (None if not applicable)
    interference_score: float | None = None     # D2 = pm_f1 × ongoing_accuracy
    completeness: float | None = None           # B2 = |actual ∩ target| / |target|
    content_correct: bool | None = None         # D3-content: did response include required format/content (None if N/A)


@dataclass
class EvalReport:
    results: list[SampleResult] = field(default_factory=list)

    def add(self, r: SampleResult) -> None:
        self.results.append(r)

    def summary_by_agent(self) -> dict[str, dict]:
        agents: dict[str, list[SampleResult]] = {}
        for r in self.results:
            agents.setdefault(r.agent_name, []).append(r)

        summary = {}
        for agent_name, results in agents.items():
            n = len(results)
            hits = sum(1 for r in results if r.hit)
            # Precision: of all triggers fired, how many were at expected turns
            # For PM-aware systems use actual_trigger_turns; for baselines that only
            # surface the action in their response text, fall back to response_mentions_action.
            # This keeps precision meaningful across agent types.
            def _fires(r: SampleResult) -> list[int]:
                return r.actual_trigger_turns if r.actual_trigger_turns else r.response_mentions_action
            total_triggers = sum(len(_fires(r)) for r in results)
            correct_triggers = sum(
                len(set(_fires(r)) & set(r.expected_trigger_turns))
                for r in results
            )
            precision = correct_triggers / total_triggers if total_triggers > 0 else 0.0
            hit_rate = hits / n if n > 0 else 0.0
            f1 = 2 * hit_rate * precision / (hit_rate + precision) if (hit_rate + precision) > 0 else 0.0
            avg_latency = (
                sum(r.latency for r in results if r.hit) / hits if hits > 0 else float("inf")
            )
            total_tokens = sum(r.tokens_used for r in results)

            summary[agent_name] = {
                "n_samples": n,
                "hits": hits,
                "hit_rate": hit_rate,
                "precision": precision,
                "f1": f1,
                "avg_latency": avg_latency,
                "total_tokens": total_tokens,
            }
        return summary

    def summary_by_category(self, agent_name: str) -> dict[str, dict]:
        cats: dict[str, list[SampleResult]] = {}
        for r in self.results:
            if r.agent_name == agent_name:
                cats.setdefault(r.category, []).append(r)

        summary = {}
        for cat, results in cats.items():
            n = len(results)
            hits = sum(1 for r in results if r.hit)
            summary[cat] = {
                "n_samples": n,
                "hits": hits,
                "hit_rate": hits / n if n > 0 else 0.0,
            }
        return summary


def load_samples(path: str | Path | None = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent / "golden_samples.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_expected_turns(sample: dict) -> list[int]:
    """Extract expected trigger turns from sample (handles both single and multi)."""
    if "trigger_turn" in sample:
        return [sample["trigger_turn"]]
    if "trigger_turns" in sample:
        return sample["trigger_turns"]
    return []


def _check_response_mentions_action(results: list[StepResult], expected_action: str) -> list[int]:
    """Heuristic: check if any response mentions keywords from expected action."""
    keywords = [w.lower() for w in expected_action.split() if len(w) > 4]
    if not keywords:
        return []
    mentioned_turns = []
    for r in results:
        text = r.agent_response.lower()
        # At least 2 keywords must appear (or 1 if only 1 keyword)
        threshold = min(2, len(keywords))
        matches = sum(1 for kw in keywords if kw in text)
        if matches >= threshold:
            mentioned_turns.append(r.step)
    return mentioned_turns


def _register_intention_from_cfg(agent: ProsMemAgent, intention_cfg: dict) -> str:
    """Translate a sample's intention dict into a ProsMem registration call.
    Centralized so every category (A/B/C/D) uses the same field mapping."""
    trigger_type = TriggerType(intention_cfg["trigger_type"])
    return agent.register_intention(
        trigger_type=trigger_type,
        action_description=intention_cfg["action_description"],
        event_cues=intention_cfg.get("event_cues"),
        focality=intention_cfg.get("focality", 0.5),
        semantic_threshold=intention_cfg.get("semantic_threshold", 0.7),
        target_step=intention_cfg.get("target_step"),
        step_interval=intention_cfg.get("step_interval"),
        activity_completion=intention_cfg.get("activity_completion"),
        activity_completions=intention_cfg.get("activity_completions"),
        outcome=intention_cfg.get("outcome"),
        anchor_event=intention_cfg.get("anchor_event"),
        delay_steps=intention_cfg.get("delay_steps"),
        implementation_intention=intention_cfg.get("implementation_intention", ""),
    )


# Registration is factored behind an optional `register_fn(agent, sample) -> list[str]`
# so the SAME ProsMem eval + scoring paths serve two feeds:
#   * default (structured): read sample["intention"]/["intentions"] — the main table.
#   * encoder (end-to-end probe): run_e2e_probe.py injects a register_fn that reads
#     sample["encoder_utterance"] through the IntentionEncoder.
# Default preserves the exact prior behavior → frozen 185×9×4 results UNAFFECTED.
def _default_single_register(agent: ProsMemAgent, sample: dict) -> list[str]:
    return [_register_intention_from_cfg(agent, sample["intention"])]


def _default_d1_register(agent: ProsMemAgent, sample: dict) -> list[str]:
    return [_register_intention_from_cfg(agent, ic) for ic in sample["intentions"]]


def _default_agent(config: ProsMemConfig, sample: dict) -> ProsMemAgent:
    return ProsMemAgent(config)


def eval_prosmem(sample: dict, config: ProsMemConfig, register_fn=None,
                 agent_factory=None) -> SampleResult:
    """Run ProsMem agent on a single-intention sample (A1/A2/A3/B1/B2/B3/C1/C2/C3/D3).

    `agent_factory(config, sample)` lets the composability runner substitute the
    A-MEM+PM agent (same PM interface). Default = ProsMemAgent → frozen results
    unaffected."""
    t0 = time.time()
    agent = (agent_factory or _default_agent)(config, sample)
    intention_cfg = sample["intention"]
    (register_fn or _default_single_register)(agent, sample)

    results = agent.run_scenario(sample["turns"])
    elapsed = time.time() - t0

    expected = _get_expected_turns(sample)
    actual = [r.step for r in results if r.pm_triggered]

    # C3 semantics: "expected to NOT fire" is derived from the intention's `outcome`
    # not matching the sample's `ground_truth_outcome`. No separate flag needed.
    intention_cfg = sample.get("intention", {})
    expect_no_fire = (
        sample.get("category") == "C3"
        and intention_cfg.get("outcome") is not None
        and sample.get("ground_truth_outcome") is not None
        and intention_cfg.get("outcome") != sample.get("ground_truth_outcome")
    )
    if expect_no_fire:
        hit = (len(actual) == 0)              # success = correctly suppressed
        precise = (len(actual) == 0)
        latency = 0
    elif sample["category"] == "C2":
        # C2 multi-activity AND: the exact turn at which "all activities are done"
        # carries inherent ±1 ambiguity (the last completion can be recognized on the
        # turn it is stated or the immediately following confirmation). Score with a
        # ±1 convergence tolerance: a fire within one turn of the labeled convergence
        # point counts as a hit and is not penalized as a false positive.
        tol = 1
        hit = any(abs(a - e) <= tol for a in actual for e in expected)
        false_positives = [a for a in actual if all(abs(a - e) > tol for e in expected)]
        precise = len(false_positives) == 0
        latency = 0
        if hit:
            latency = min(abs(a - e) for a in actual for e in expected)
    else:
        hit = bool(set(actual) & set(expected))
        false_positives = set(actual) - set(expected)
        precise = len(false_positives) == 0
        latency = 0
        if hit:
            latency = min(
                min(abs(a - e) for e in expected) for a in actual if a in expected
            )

    # B2 completeness: fraction of expected interval-firings actually hit
    completeness = None
    if intention_cfg.get("step_interval") is not None and expected:
        completeness = len(set(actual) & set(expected)) / len(expected)

    base = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name="ProsMem",
        expected_trigger_turns=expected,
        actual_trigger_turns=actual,
        response_mentions_action=[],
        hit=hit,
        precise=precise,
        latency=latency,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
        completeness=completeness,
    )
    # Composability agent (A-MEM+PM) holds an external note store → drop it.
    # No-op for plain ProsMemAgent (no cleanup attr). Does not touch agent.llm.
    if hasattr(agent, "cleanup"):
        agent.cleanup()
    # D3 content-error judge: only fires if sample is D3-content
    return _wrap_d3_content(base, sample, results, agent.llm, config.monitor_model)


def eval_vanilla(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run Vanilla agent on a single sample."""
    t0 = time.time()
    agent = VanillaAgent(config)
    results = agent.run_scenario(sample["turns"])
    elapsed = time.time() - t0

    expected = _get_expected_turns(sample)
    mentioned = _check_response_mentions_action(results, sample["expected_action"])

    # For vanilla, a "hit" means it spontaneously mentioned the action at the right turn
    hit = bool(set(mentioned) & set(expected))

    base = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name="Vanilla",
        expected_trigger_turns=expected,
        actual_trigger_turns=[],
        response_mentions_action=mentioned,
        hit=hit,
        precise=True,  # No PM mechanism, so no false triggers
        latency=0,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
    )
    return _wrap_d3_content(base, sample, results, agent.llm, config.monitor_model)


def eval_mem0(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run Mem0 baseline agent on a single sample."""
    from prosmem.agent.baselines import Mem0Agent
    t0 = time.time()
    agent = Mem0Agent(config, collection_id=sample["id"])
    intention_cfg = sample["intention"]
    trigger_desc = intention_cfg.get("implementation_intention") or \
                   " or ".join(intention_cfg.get("event_cues", []) or []) or \
                   intention_cfg.get("activity_completion", "the condition is met")
    try:
        agent.register_intention(
            description=intention_cfg["action_description"],
            trigger=trigger_desc,
        )
        results = agent.run_scenario(sample["turns"])
    finally:
        agent.cleanup()
    elapsed = time.time() - t0

    expected = _get_expected_turns(sample)
    mentioned = _check_response_mentions_action(results, sample["expected_action"])
    hit = bool(set(mentioned) & set(expected))
    # For baselines without explicit trigger mechanism, "precise" means no spurious mentions
    false_mentions = set(mentioned) - set(expected)
    precise = len(false_mentions) == 0

    base = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name="Mem0",
        expected_trigger_turns=expected,
        actual_trigger_turns=[],
        response_mentions_action=mentioned,
        hit=hit,
        precise=precise,
        latency=0,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
    )
    return _wrap_d3_content(base, sample, results, agent.llm, config.monitor_model)


def eval_amem(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run A-MEM baseline agent on a single sample."""
    from prosmem.agent.baselines import AMemAgent
    t0 = time.time()
    agent = AMemAgent(config, collection_id=sample["id"])
    intention_cfg = sample["intention"]
    trigger_desc = intention_cfg.get("implementation_intention") or \
                   " or ".join(intention_cfg.get("event_cues", []) or []) or \
                   intention_cfg.get("activity_completion", "the condition is met")
    try:
        agent.register_intention(
            description=intention_cfg["action_description"],
            trigger=trigger_desc,
        )
        results = agent.run_scenario(sample["turns"])
    finally:
        agent.cleanup()
    elapsed = time.time() - t0

    expected = _get_expected_turns(sample)
    mentioned = _check_response_mentions_action(results, sample["expected_action"])
    hit = bool(set(mentioned) & set(expected))
    false_mentions = set(mentioned) - set(expected)
    precise = len(false_mentions) == 0

    base = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name="A-MEM",
        expected_trigger_turns=expected,
        actual_trigger_turns=[],
        response_mentions_action=mentioned,
        hit=hit,
        precise=precise,
        latency=0,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
    )
    return _wrap_d3_content(base, sample, results, agent.llm, config.monitor_model)


def _eval_retrospective(
    sample: dict, config: ProsMemConfig, agent_cls: type, agent_label: str,
) -> SampleResult:
    """Shared harness for pull-based retrospective memory baselines
    (GenAgents, MemoryBank) — they share the same register/run/cleanup API."""
    t0 = time.time()
    agent = agent_cls(config, collection_id=sample["id"])
    intention_cfg = sample["intention"]
    trigger_desc = intention_cfg.get("implementation_intention") or \
                   " or ".join(intention_cfg.get("event_cues", []) or []) or \
                   intention_cfg.get("activity_completion", "the condition is met")
    try:
        agent.register_intention(
            description=intention_cfg["action_description"],
            trigger=trigger_desc,
        )
        results = agent.run_scenario(sample["turns"])
    finally:
        agent.cleanup()
    elapsed = time.time() - t0
    expected = _get_expected_turns(sample)
    mentioned = _check_response_mentions_action(results, sample["expected_action"])
    hit = bool(set(mentioned) & set(expected))
    precise = len(set(mentioned) - set(expected)) == 0
    base = SampleResult(
        sample_id=sample["id"], category=sample["category"],
        agent_name=agent_label,
        expected_trigger_turns=expected, actual_trigger_turns=[],
        response_mentions_action=mentioned,
        hit=hit, precise=precise, latency=0,
        tokens_used=agent.llm.total_tokens_used, wall_time=elapsed,
    )
    return _wrap_d3_content(base, sample, results, agent.llm, config.monitor_model)


def eval_genagents(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run Generative Agents (Park et al. UIST 2023) baseline on a single sample."""
    from prosmem.agent.baselines import GenAgentsAgent
    return _eval_retrospective(sample, config, GenAgentsAgent, "GenAgents")


def eval_memorybank(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run MemoryBank (Zhong et al. AAAI 2024) baseline on a single sample."""
    from prosmem.agent.baselines import MemoryBankAgent
    return _eval_retrospective(sample, config, MemoryBankAgent, "MemoryBank")


def eval_lightmem(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run LightMem (ICLR 2026, arXiv:2510.18866) baseline on a single sample."""
    from prosmem.agent.lightmem_agent import LightMemAgent
    return _eval_retrospective(sample, config, LightMemAgent, "LightMem")


def eval_everos(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run EverOS / EverMemOS (arXiv:2601.02163) baseline on a single sample."""
    from prosmem.agent.everos_agent import EverOSAgent
    return _eval_retrospective(sample, config, EverOSAgent, "EverOS")


def eval_graphiti(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run Zep / Graphiti (arXiv:2501.13956) baseline on a single sample."""
    from prosmem.agent.graphiti_agent import GraphitiAgent
    return _eval_retrospective(sample, config, GraphitiAgent, "Graphiti")


def eval_naive(sample: dict, config: ProsMemConfig) -> SampleResult:
    """Run NaiveReminder agent on a single sample."""
    t0 = time.time()
    agent = NaiveReminderAgent(config)
    intention_cfg = sample["intention"]
    agent.add_reminder(
        description=intention_cfg["action_description"],
        trigger=intention_cfg.get("implementation_intention", "Check every turn"),
    )

    results = agent.run_scenario(sample["turns"])
    elapsed = time.time() - t0

    expected = _get_expected_turns(sample)
    mentioned = _check_response_mentions_action(results, sample["expected_action"])

    hit = bool(set(mentioned) & set(expected))

    base = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name="NaiveReminder",
        expected_trigger_turns=expected,
        actual_trigger_turns=[],
        response_mentions_action=mentioned,
        hit=hit,
        precise=True,
        latency=0,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
    )
    return _wrap_d3_content(base, sample, results, agent.llm, config.monitor_model)


# ============================================================================
# D1 (multi-intention) evaluators
# ============================================================================

def _score_intentions_against_actual(
    intentions_cfg: list[dict],
    actual_per_intention: dict[str, list[int]],   # intention_id (or idx str) -> turns fired
    response_mentions: dict[str, list[int]],      # intention_id -> turns response mentioned action
) -> tuple[list[IntentionResult], dict]:
    """Compute per-intention hit/precise/latency + sample-level micro/macro/set-F1."""
    per_results: list[IntentionResult] = []
    sample_hits = 0
    sample_precise = 0
    for idx, icfg in enumerate(intentions_cfg):
        iid = icfg.get("intention_id") or f"i{idx}"
        expected = icfg["trigger_turn"]
        actual = actual_per_intention.get(iid, [])
        mentioned = response_mentions.get(iid, [])
        fires = actual if actual else mentioned
        hit = expected in fires
        precise = (not fires) or (set(fires) == {expected})
        latency = min((abs(f - expected) for f in fires if f == expected), default=0) if hit else 0
        per_results.append(
            IntentionResult(
                intention_id=iid,
                trigger_turn=expected,
                actual_trigger_turns=actual,
                response_mentions_turns=mentioned,
                hit=hit,
                precise=precise,
                latency=latency,
            )
        )
        if hit:
            sample_hits += 1
        if precise:
            sample_precise += 1
    n = len(intentions_cfg)
    macro_hit_rate = sample_hits / n if n else 0.0
    macro_precision = sample_precise / n if n else 0.0
    if (macro_hit_rate + macro_precision) > 0:
        macro_f1 = 2 * macro_hit_rate * macro_precision / (macro_hit_rate + macro_precision)
    else:
        macro_f1 = 0.0
    set_hit = 1 if (sample_hits == n and sample_precise == n) else 0
    aggregates = {
        "macro_hit_rate": macro_hit_rate,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "set_hit": set_hit,
    }
    return per_results, aggregates


def eval_d1_prosmem(sample: dict, config: ProsMemConfig, register_fn=None,
                    agent_factory=None) -> SampleResult:
    """ProsMem on D1 multi-intention sample. Registers all listed intentions,
    matches actual fires to expected intentions by intention_id (or position)."""
    t0 = time.time()
    agent = (agent_factory or _default_agent)(config, sample)
    intentions_cfg = sample["intentions"]
    # register_fn returns the registered intention_ids IN THE SAME ORDER as
    # intentions_cfg, so the position-based label mapping below is preserved for
    # both the structured feed and the encoder feed (end-to-end probe).
    registered_ids: list[str] = (register_fn or _default_d1_register)(agent, sample)

    results = agent.run_scenario(sample["turns"])
    elapsed = time.time() - t0

    # Map registered intentions back to sample's intention_id labels
    actual_per_intention: dict[str, list[int]] = {
        (icfg.get("intention_id") or f"i{idx}"): []
        for idx, icfg in enumerate(intentions_cfg)
    }
    rid_to_label = {
        rid: (intentions_cfg[idx].get("intention_id") or f"i{idx}")
        for idx, rid in enumerate(registered_ids)
    }
    for r in results:
        if r.pm_triggered is not None:
            label = rid_to_label.get(r.pm_triggered.intention_id)
            if label is not None:
                actual_per_intention[label].append(r.step)

    per_results, agg = _score_intentions_against_actual(
        intentions_cfg, actual_per_intention, response_mentions={}
    )
    result = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name="ProsMem",
        expected_trigger_turns=[i["trigger_turn"] for i in intentions_cfg],
        actual_trigger_turns=sorted({s for fl in actual_per_intention.values() for s in fl}),
        response_mentions_action=[],
        hit=agg["macro_hit_rate"] > 0,
        precise=agg["macro_precision"] == 1.0,
        latency=0,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
        intention_results=per_results,
    )
    if hasattr(agent, "cleanup"):
        agent.cleanup()
    _release_llm(agent.llm)
    return result


def _eval_d1_baseline(
    sample: dict,
    config: ProsMemConfig,
    agent_factory,
    agent_label: str,
    is_naive: bool = False,
) -> SampleResult:
    """Shared harness for D1 evaluation on retrospective baselines (Vanilla/Naive/
    Mem0/A-MEM/GenAgents/MemoryBank). Each intention is registered separately;
    per-intention "hit" derives from keyword match on agent responses."""
    t0 = time.time()
    agent = agent_factory()
    intentions_cfg = sample["intentions"]
    try:
        for icfg in intentions_cfg:
            if is_naive:
                agent.add_reminder(
                    description=icfg["action_description"],
                    trigger=icfg.get("implementation_intention", "Check every turn"),
                )
            elif hasattr(agent, "register_intention"):
                # Mem0/A-MEM/GenAgents/MemoryBank shared interface
                trigger_desc = icfg.get("implementation_intention") or \
                               " or ".join(icfg.get("event_cues", []) or []) or \
                               icfg.get("activity_completion", "the condition is met")
                agent.register_intention(
                    description=icfg["action_description"],
                    trigger=trigger_desc,
                )
            # Vanilla has no registration interface → it only sees turns
        results = agent.run_scenario(sample["turns"])
    finally:
        if hasattr(agent, "cleanup"):
            agent.cleanup()
    elapsed = time.time() - t0

    # Per-intention keyword match on responses
    response_mentions: dict[str, list[int]] = {}
    for idx, icfg in enumerate(intentions_cfg):
        label = icfg.get("intention_id") or f"i{idx}"
        response_mentions[label] = _check_response_mentions_action(
            results, icfg["action_description"]
        )

    per_results, agg = _score_intentions_against_actual(
        intentions_cfg, actual_per_intention={}, response_mentions=response_mentions
    )
    result = SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        agent_name=agent_label,
        expected_trigger_turns=[i["trigger_turn"] for i in intentions_cfg],
        actual_trigger_turns=[],
        response_mentions_action=sorted({s for fl in response_mentions.values() for s in fl}),
        hit=agg["macro_hit_rate"] > 0,
        precise=agg["macro_precision"] == 1.0,
        latency=0,
        tokens_used=agent.llm.total_tokens_used,
        wall_time=elapsed,
        intention_results=per_results,
    )
    _release_llm(agent.llm)
    return result


def eval_d1_vanilla(sample, config):
    return _eval_d1_baseline(sample, config, lambda: VanillaAgent(config), "Vanilla")


def eval_d1_naive(sample, config):
    return _eval_d1_baseline(sample, config, lambda: NaiveReminderAgent(config), "NaiveReminder", is_naive=True)


def eval_d1_mem0(sample, config):
    from prosmem.agent.baselines import Mem0Agent
    return _eval_d1_baseline(sample, config, lambda: Mem0Agent(config, collection_id=sample["id"]), "Mem0")


def eval_d1_amem(sample, config):
    from prosmem.agent.baselines import AMemAgent
    return _eval_d1_baseline(sample, config, lambda: AMemAgent(config, collection_id=sample["id"]), "A-MEM")


def eval_d1_genagents(sample, config):
    from prosmem.agent.baselines import GenAgentsAgent
    return _eval_d1_baseline(sample, config, lambda: GenAgentsAgent(config, collection_id=sample["id"]), "GenAgents")


def eval_d1_memorybank(sample, config):
    from prosmem.agent.baselines import MemoryBankAgent
    return _eval_d1_baseline(sample, config, lambda: MemoryBankAgent(config, collection_id=sample["id"]), "MemoryBank")


def eval_d1_lightmem(sample, config):
    from prosmem.agent.lightmem_agent import LightMemAgent
    return _eval_d1_baseline(sample, config, lambda: LightMemAgent(config, collection_id=sample["id"]), "LightMem")


def eval_d1_everos(sample, config):
    from prosmem.agent.everos_agent import EverOSAgent
    return _eval_d1_baseline(sample, config, lambda: EverOSAgent(config, collection_id=sample["id"]), "EverOS")


def eval_d1_graphiti(sample, config):
    from prosmem.agent.graphiti_agent import GraphitiAgent
    return _eval_d1_baseline(sample, config, lambda: GraphitiAgent(config, collection_id=sample["id"]), "Graphiti")


# ============================================================================
# D2 (interference robustness) post-hoc ongoing-task judge
# ============================================================================

def _judge_ongoing_task(
    step_results: list[StepResult],
    ongoing_task: dict,
    llm: LLMClient,
    judge_model: str,
) -> float:
    """LLM judge: for each turn with an expected ongoing-task answer, score 0/1.
    Returns accuracy = correct / total."""
    expected = ongoing_task.get("expected_answers", {})
    if not expected:
        return float("nan")
    by_turn: dict[int, str] = {}
    for k, v in expected.items():
        try:
            t = int(k.replace("turn_", "")) if isinstance(k, str) else int(k)
        except ValueError:
            continue
        by_turn[t] = str(v)
    if not by_turn:
        return float("nan")
    n_correct = 0
    n_total = 0
    for r in step_results:
        if r.step not in by_turn:
            continue
        n_total += 1
        prompt = (
            "Judge: does the assistant's response correctly address the ongoing-task "
            "expected answer?\n\n"
            f"Expected answer: {by_turn[r.step]}\n\n"
            f"Assistant response: {r.agent_response}\n\n"
            "Reply with strictly one token: YES or NO."
        )
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            model=judge_model, temperature=0.0, max_tokens=5,
        )
        if "YES" in resp.strip().upper():
            n_correct += 1
    return n_correct / n_total if n_total else float("nan")


def _release_llm(llm: LLMClient) -> None:
    """Best-effort close of an agent's LLMClient HTTP pools at the end of an
    eval. Centralizes FD release so every eval path frees sockets (prevents
    Errno 24 'Too many open files' on long multi-agent runs)."""
    try:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _wrap_d2(
    base_result: SampleResult,
    sample: dict,
    step_results: list[StepResult],
    llm: LLMClient,
    judge_model: str,
) -> SampleResult:
    """Attach ongoing_task_accuracy + interference_score to a SampleResult."""
    try:
        ongoing = sample.get("ongoing_task")
        if ongoing is not None:
            acc = _judge_ongoing_task(step_results, ongoing, llm, judge_model)
            base_result.ongoing_task_accuracy = acc
            if not (acc != acc):  # not NaN
                base_result.interference_score = (1.0 if base_result.hit else 0.0) * acc
        return base_result
    finally:
        _release_llm(llm)


# ============================================================================
# D3 content-correctness LLM judge (only applies to D3 with failure_mode='content')
# ============================================================================

def _judge_content_correctness(
    step_results: list[StepResult],
    sample: dict,
    llm: LLMClient,
    judge_model: str,
) -> bool | None:
    """For D3 content samples: does the agent's response at the trigger turn
    satisfy the required action specification? Returns None if not applicable
    (not a content sample) or if no response found at trigger turn."""
    if sample.get("category") != "D3":
        return None
    if sample.get("failure_mode") != "content":
        return None
    trigger_turn = sample.get("trigger_turn")
    if trigger_turn is None:
        return None
    # Find response at trigger turn (1-indexed step). For ProsMem the action's
    # actual output (e.g. citation format) is in `pm_action_response` (separate
    # ISR-style execution LLM call); `agent_response` is the user-turn continuation
    # that may not repeat the spec. For baselines pm_action_response is empty and
    # everything lives in agent_response. Concatenate both for a fair, agent-
    # agnostic content view.
    sr = next((s for s in step_results if s.step == trigger_turn), None)
    if sr is None:
        return None
    parts = []
    pm_out = getattr(sr, "pm_action_response", "") or ""
    if pm_out:
        parts.append(pm_out)
    if sr.agent_response:
        parts.append(sr.agent_response)
    response = "\n\n".join(parts)
    if not response:
        return None
    required = (
        sample.get("intention", {}).get("action_description")
        or sample.get("expected_action", "")
    )
    if not required:
        return None
    prompt = (
        "Question: Does the agent's response include the SPECIFIC content/format "
        "required by the intention?\n\n"
        f"Required content/format: {required}\n\n"
        f"Agent response: {response}\n\n"
        "Answer strictly with YES or NO. Answer YES only if the response "
        "explicitly contains the required specific format, exact value, or "
        "concrete spec (e.g. citation pattern, file path, dosage, address). "
        "Answer NO if the response only paraphrases the topic without the "
        "concrete spec, or omits the specific content."
    )
    resp = llm.chat(
        [{"role": "user", "content": prompt}],
        model=judge_model, temperature=0.0, max_tokens=5,
    )
    return "YES" in resp.strip().upper()


def _wrap_d3_content(
    base_result: SampleResult,
    sample: dict,
    step_results: list[StepResult],
    llm: LLMClient,
    judge_model: str,
) -> SampleResult:
    """Attach content_correct to a SampleResult if it's a D3 content sample
    AND the agent actually fired (or response_mentions captured the action turn).

    This is the terminal call of every single-intention eval path (and of D2
    baselines via _eval_d2_baseline), so it closes the agent's LLMClient on the
    way out — both return branches — to release HTTP/FD resources."""
    try:
        if not (base_result.hit or base_result.response_mentions_action):
            return base_result  # no fire, no response to judge → leave None
        base_result.content_correct = _judge_content_correctness(
            step_results, sample, llm, judge_model
        )
        return base_result
    finally:
        _release_llm(llm)


def eval_d2_prosmem(sample: dict, config: ProsMemConfig, register_fn=None,
                    agent_factory=None) -> SampleResult:
    """ProsMem on D2 interference sample: standard single-intention eval +
    post-hoc ongoing-task accuracy via LLM judge."""
    t0 = time.time()
    agent = (agent_factory or _default_agent)(config, sample)
    (register_fn or _default_single_register)(agent, sample)
    step_results = agent.run_scenario(sample["turns"])
    elapsed = time.time() - t0
    expected = _get_expected_turns(sample)
    actual = [r.step for r in step_results if r.pm_triggered]
    hit = bool(set(actual) & set(expected))
    false_positives = set(actual) - set(expected)
    precise = len(false_positives) == 0
    latency = min(
        (min(abs(a - e) for e in expected) for a in actual if a in expected), default=0
    ) if hit else 0
    base = SampleResult(
        sample_id=sample["id"], category=sample["category"], agent_name="ProsMem",
        expected_trigger_turns=expected, actual_trigger_turns=actual,
        response_mentions_action=[],
        hit=hit, precise=precise, latency=latency,
        tokens_used=agent.llm.total_tokens_used, wall_time=elapsed,
    )
    if hasattr(agent, "cleanup"):
        agent.cleanup()
    return _wrap_d2(base, sample, step_results, agent.llm, config.monitor_model)


def _eval_d2_baseline(
    sample: dict,
    config: ProsMemConfig,
    base_eval_fn,
    judge_model: str | None = None,
) -> SampleResult:
    """Run a standard baseline evaluator, then attach ongoing-task accuracy.

    The base eval discards step_results, so we re-run only the judge layer
    against the agent's responses (recovered by re-running with the same
    config). For now we simply skip the judge for non-ProsMem agents — this
    keeps D2's interference signal focused on ProsMem vs. backbone (the
    ongoing-task-cost measurement)."""
    return base_eval_fn(sample, config)
    # NOTE: a full per-baseline ongoing-task judge would require step_response
    # capture to be wired for the baseline agents as well.


def eval_d2_vanilla(sample, config):
    return _eval_d2_baseline(sample, config, eval_vanilla)


def eval_d2_naive(sample, config):
    return _eval_d2_baseline(sample, config, eval_naive)


def eval_d2_mem0(sample, config):
    return _eval_d2_baseline(sample, config, eval_mem0)


def eval_d2_amem(sample, config):
    return _eval_d2_baseline(sample, config, eval_amem)


def eval_d2_genagents(sample, config):
    return _eval_d2_baseline(sample, config, eval_genagents)


def eval_d2_memorybank(sample, config):
    return _eval_d2_baseline(sample, config, eval_memorybank)


def eval_d2_lightmem(sample, config):
    return _eval_d2_baseline(sample, config, eval_lightmem)


def eval_d2_everos(sample, config):
    return _eval_d2_baseline(sample, config, eval_everos)


def eval_d2_graphiti(sample, config):
    return _eval_d2_baseline(sample, config, eval_graphiti)


# ============================================================================
# Dispatch table (category → per-agent eval functions)
# ============================================================================

# Standard single-intention categories (A1/A2/A3/B1/B2/B3/C1/C2/C3/D3) all use
# the same eval_* functions; intention-field passthrough is handled inside
# _register_intention_from_cfg.
_STANDARD_DISPATCH = {
    "ProsMem": eval_prosmem,
    "Vanilla": eval_vanilla,
    "NaiveReminder": eval_naive,
    "Mem0": eval_mem0,
    "A-MEM": eval_amem,
    "GenAgents": eval_genagents,
    "MemoryBank": eval_memorybank,
    "LightMem": eval_lightmem,
    "EverOS": eval_everos,
    "Graphiti": eval_graphiti,
}

_D1_DISPATCH = {
    "ProsMem": eval_d1_prosmem,
    "Vanilla": eval_d1_vanilla,
    "NaiveReminder": eval_d1_naive,
    "Mem0": eval_d1_mem0,
    "A-MEM": eval_d1_amem,
    "GenAgents": eval_d1_genagents,
    "MemoryBank": eval_d1_memorybank,
    "LightMem": eval_d1_lightmem,
    "EverOS": eval_d1_everos,
    "Graphiti": eval_d1_graphiti,
}

_D2_DISPATCH = {
    "ProsMem": eval_d2_prosmem,
    "Vanilla": eval_d2_vanilla,
    "NaiveReminder": eval_d2_naive,
    "Mem0": eval_d2_mem0,
    "A-MEM": eval_d2_amem,
    "GenAgents": eval_d2_genagents,
    "MemoryBank": eval_d2_memorybank,
    "LightMem": eval_d2_lightmem,
    "EverOS": eval_d2_everos,
    "Graphiti": eval_d2_graphiti,
}


def get_eval_fn(category: str, agent_name: str):
    """Return the correct eval function for (category, agent_name).
    Falls back to the standard single-intention dispatch."""
    if category == "D1":
        return _D1_DISPATCH.get(agent_name, _STANDARD_DISPATCH.get(agent_name))
    if category == "D2":
        return _D2_DISPATCH.get(agent_name, _STANDARD_DISPATCH.get(agent_name))
    return _STANDARD_DISPATCH.get(agent_name)


def run_full_eval(
    config: ProsMemConfig,
    samples: list[dict] | None = None,
    verbose: bool = True,
) -> EvalReport:
    """Run all samples through all 3 agents. Returns EvalReport."""
    if samples is None:
        samples = load_samples()

    report = EvalReport()

    for i, sample in enumerate(samples):
        sid = sample["id"]
        if verbose:
            print(f"\n[{i+1}/{len(samples)}] Running {sid}: {sample['description'][:60]}...")

        category = sample.get("category", "A1")
        for name in ("ProsMem", "Vanilla", "NaiveReminder"):
            eval_fn = get_eval_fn(category, name)
            if eval_fn is None:
                continue
            if verbose:
                print(f"  {name}...", end=" ", flush=True)
            result = eval_fn(sample, config)
            report.add(result)
            if verbose:
                status = "HIT" if result.hit else "MISS"
                print(f"{status} (triggers={result.actual_trigger_turns or result.response_mentions_action}, "
                      f"tokens={result.tokens_used}, {result.wall_time:.1f}s)")

    return report
