"""ProsMem-Bench generator: template-based LLM generation with structural validation.

Category-specific prompts produce candidate samples, which must clear the
semantic-similarity pre-validation gates before acceptance (failed candidates
are regenerated).

Usage:
    # Dry-run (1 sample per category, for prompt inspection):
    python scripts/generate_bench.py --dry-run

    # Full generation:
    python scripts/generate_bench.py --out results/generated_samples.json

    # Single category regen:
    python scripts/generate_bench.py --category A2 --count 17
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Load .env with UTF-8
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from prosmem.agent.llm import LLMClient  # noqa: E402
from prosmem.core.config import ProsMemConfig  # noqa: E402
from prosmem.retrieval.embedder import cosine_similarity, embed_text, embed_texts  # noqa: E402

# =============================================================================
# Category specifications
# =============================================================================

SEED_PATH = Path(__file__).parent.parent / "prosmem" / "bench" / "golden_samples.json"

DOMAINS = ["coding", "research", "personal_assistant", "customer_service",
           "education", "health", "data_analysis", "workflow"]

CATEGORY_SPECS: dict[str, dict] = {
    "A1": {
        "name": "Event-based Focal",
        "difficulty": "easy",
        "trigger_type": "event_based",
        "target_count": 11,          # 15 target - 4 seed
        "focality_range": (0.85, 0.95),
        "threshold_range": (0.72, 0.80),
        "n_turns_range": (5, 7),
        "n_cues": (3, 4),
        "desc": (
            "The trigger cue appears almost verbatim in the dialogue. "
            "The user mentions the cue directly (e.g., 'pharmacy', 'database migration'). "
            "Focal means the PM cue overlaps semantically with what the user is already discussing."
        ),
    },
    "A2": {
        "name": "Event-based Non-focal",
        "difficulty": "medium",
        "trigger_type": "event_based",
        "target_count": 17,          # 20 - 3
        "focality_range": (0.20, 0.40),
        "threshold_range": (0.60, 0.70),
        "n_turns_range": (5, 7),
        "n_cues": (3, 5),
        "desc": (
            "The trigger requires one-step semantic inference. The user describes a situation that "
            "MATCHES the cue class without using the cue word (e.g., says 'I have nothing to do' when "
            "cues are 'free time / bored / spare time'). Non-focal = cue is semantically distant from "
            "the ongoing task.\n\n"
            "CRITICAL: the stored cues must NOT appear as a literal substring in ANY turn — not at "
            "trigger_turn, not before. The agent must bridge from the turn's surface form to the "
            "cue concept via embedding similarity alone. If the user literally says 'free time', "
            "that's an A1 focal sample, not A2."
        ),
    },
    "A3": {
        "name": "Event-based Multi-hop Semantic",
        "difficulty": "hard",
        "trigger_type": "event_based",
        "target_count": 12,          # 15 - 3
        "focality_range": (0.15, 0.35),
        "threshold_range": (0.60, 0.72),
        "n_turns_range": (5, 7),
        "n_cues": (3, 4),
        "desc": (
            "The trigger requires 2+ reasoning hops. E.g., user mentions 'seminar at 10am' → agent "
            "infers 'note-taking context' → triggers 'bring new fountain pen'. The cue is an ABSTRACT "
            "situation class, not a concrete keyword.\n\n"
            "CRITICAL: stored cues must be at the CONCEPTUAL level (e.g., 'note-taking situation', "
            "'extended listening activity') NOT object-level ('pen', 'notebook', 'tablet'). And they "
            "must NOT appear as literal substring in any turn. The match happens via embedding bridge "
            "from the concrete scenario (seminar, lecture, conference) to the abstract concept."
        ),
    },
    "B1": {
        "name": "Time-based Absolute",
        "difficulty": "easy",
        "trigger_type": "time_based",
        "target_count": 9,           # 12 - 3
        "focality_range": None,      # N/A for time-based
        "threshold_range": None,
        "n_turns_range": (6, 8),
        "n_cues": None,
        "desc": (
            "Trigger fires at a specific turn number (e.g., turn 5). Use target_step field. "
            "The scenario should have the agent doing unrelated ongoing work, with PM firing on schedule."
        ),
    },
    "C1": {
        "name": "Activity-based Single Completion",
        "difficulty": "medium",
        "trigger_type": "activity_based",
        "target_count": 12,          # 15 - 3
        "focality_range": (0.20, 0.40),
        "threshold_range": None,
        "n_turns_range": (5, 7),
        "n_cues": None,
        "desc": (
            "Trigger fires when user signals an activity is FINISHED (e.g., 'bags are packed', "
            "'code is deployed'). Uses activity_completion field — a natural-language description "
            "of the completion state that the slow-path LLM monitor checks."
        ),
    },
    "DC": {
        "name": "Commission Error (Event-based)",
        "difficulty": "medium",
        "trigger_type": "event_based",
        "target_count": 12,          # 15 - 3
        "focality_range": (0.85, 0.95),
        "threshold_range": (0.72, 0.80),
        "n_turns_range": (6, 8),
        "n_cues": (3, 4),
        "desc": (
            "The cue phrase must appear LITERALLY in AT LEAST 3 DIFFERENT TURNS of the dialogue "
            "(e.g., turns 3, 5, 7), using the SAME concrete cue word or near-identical phrasing each "
            "time. The intention should trigger ONLY at the FIRST occurrence. Tests commission-error "
            "prevention: after completion the intention must not re-fire on later cue mentions.\n\n"
            "MANDATORY CONSTRUCTION: pick a concrete cue noun (e.g., 'pharmacy', 'database', 'the "
            "client call'). In turns[trigger_turn-1] the user mentions it for the first time. In AT "
            "LEAST 2 later turns the user mentions it again (e.g., 'actually let me hit the pharmacy "
            "again...', 'one more pharmacy stop...'). Without multiple cue mentions this sample is "
            "useless — you MUST make the cue repeat."
        ),
    },
    # ------------------------------------------------------------------
    # Composite / extended categories
    # ------------------------------------------------------------------
    "B2": {
        "name": "Time-based Relative Interval",
        "difficulty": "medium",
        "trigger_type": "time_based",
        "target_count": 12,
        "focality_range": None,
        "threshold_range": None,
        "n_turns_range": (10, 14),       # long scenario for ≥3 interval fires
        "n_cues": None,
        "desc": (
            "Trigger fires every K turns (step_interval). The agent has a single recurring intention "
            "(e.g., 'every 3 turns provide a checkpoint summary'). target_steps list enumerates ALL "
            "expected firing turns; trigger_turns mirror that list. Sample should have natural ongoing "
            "work where periodic checkpoints make sense (long planning session, batch processing, etc.). "
            "DO NOT include cue words in turns — the periodicity is purely time-based."
        ),
    },
    "B3": {
        "name": "Time-based Dynamic (Anchor + Delay)",
        "difficulty": "hard",
        "trigger_type": "time_based_dynamic",
        "target_count": 12,
        "focality_range": None,
        "threshold_range": None,
        "n_turns_range": (6, 9),
        "n_cues": None,
        "desc": (
            "Trigger fires delay_steps after an anchor event is mentioned. anchor_event is a "
            "natural-language description of the anchor (e.g., 'user announces upcoming deadline'). "
            "anchor_turn is the ground-truth turn where the anchor occurs; trigger_turn = anchor_turn + "
            "delay_steps. Earlier turns must NOT mention the anchor or any close synonym."
        ),
    },
    "C2": {
        "name": "Activity Multi-Convergence (AND)",
        "difficulty": "hard",
        "trigger_type": "activity_based",
        "target_count": 12,
        "focality_range": None,
        "threshold_range": None,
        "n_turns_range": (6, 9),
        "n_cues": None,
        "desc": (
            "Trigger fires when TWO OR THREE activities have ALL been completed (AND logic). "
            "activity_completions is a list of 2-3 short activity descriptions. The scenario must have "
            "each activity completed in a different turn (no earlier than turn 2, last completion at "
            "trigger_turn). Earlier turns describe in-progress work on any subset; the trigger_turn "
            "marks the LAST activity finishing such that ALL of them are now done."
        ),
    },
    "C3": {
        "name": "Activity Conditional (IF X fails THEN Y)",
        "difficulty": "hard",
        "trigger_type": "activity_based",
        "target_count": 12,
        "focality_range": None,
        "threshold_range": None,
        "n_turns_range": (5, 8),
        "n_cues": None,
        "desc": (
            "Trigger fires ONLY if a watched activity has the configured outcome (typically 'fail'). "
            "Roughly 1/3 of samples should be no-fire cases — the activity actually succeeds, "
            "so the agent must NOT fire. intention.outcome ∈ {'success','fail'} is what the "
            "intention watches for; ground_truth_outcome is what the scenario actually exhibits. "
            "When they match → trigger_turn is set; when they differ → trigger_turn is null "
            "(no-fire is derived from outcome != ground_truth_outcome)."
        ),
    },
    "D1": {
        "name": "Multi-intention Concurrent (3-5 intentions)",
        "difficulty": "hard",
        "trigger_type": "multi_intention",
        "target_count": 15,
        "focality_range": None,
        "threshold_range": None,
        "n_turns_range": (8, 12),
        "n_cues": None,
        "desc": (
            "Sample has 3-5 INDEPENDENT intentions registered together. Schema uses 'intentions' (LIST) "
            "instead of 'intention'. Each intention has its own trigger_type (mix event_based, "
            "time_based, activity_based), action_description, and trigger_turn. CRITICAL: "
            "trigger_turn values across intentions must be DISTINCT (no two intentions fire on the same turn) — the "
            "engine's single-firing-per-step rule otherwise introduces unavoidable latency. The scenario "
            "must give each intention a clear, natural trigger moment."
        ),
    },
    "D2": {
        "name": "Interference Robustness (PM × ongoing task)",
        "difficulty": "hard",
        "trigger_type": "event_based",   # most D2 samples use event-based PM
        "target_count": 15,
        "focality_range": (0.20, 0.40),  # non-focal so PM and ongoing don't share cues
        "threshold_range": (0.60, 0.70),
        "n_turns_range": (8, 12),
        "n_cues": (3, 4),
        "desc": (
            "Sample has (a) a single non-focal event-based PM intention AND (b) an ongoing_task block "
            "specifying a per-turn cognitive workload (math, coding, logic-QA) at low/medium/high "
            "complexity. The user's turns interleave PM-relevant content with the ongoing task. The "
            "evaluator measures BOTH PM hit and ongoing-task accuracy (LLM-judged against "
            "expected_answers). Tests Smith 2003 PAM prediction: does maintaining PM degrade "
            "ongoing performance?"
        ),
    },
    # D3: failure-mode probes. Single category. Sub-modes (commission / omission /
    # content) are picked at generation time via the --failure-mode CLI argument,
    # which is an implementation flag — it does NOT propagate to the generated
    # sample as a field. Sub-mode differentiation in the sample is carried by the
    # ID range alone (plan-locked):  D3-01..15 commission, D3-16..23 omission,
    # D3-24..30 content. The `category_name` string also tracks the sub-mode for
    # backward compatibility with the legacy DC samples ("Commission Error
    # (Event-based)" etc.).
    "D3": {
        "name": "Failure-mode probe",
        "difficulty": "hard",
        "trigger_type": "event_based",
        # Per-sub-mode parameter table (internal dispatch).
        "by_failure_mode": {
            "commission": {
                "target_count": 15,
                "focality_range": (0.85, 0.95),
                "threshold_range": (0.72, 0.80),
                "n_turns_range": (6, 8),
                "id_start": 1,
                "category_name": "Commission Error (Event-based)",
                "desc_suffix": (
                    "Commission probe. Cue phrase appears literally in at least 3 turns; "
                    "intention triggers only at the first occurrence and must not re-fire."
                ),
            },
            "omission": {
                "target_count": 8,
                "focality_range": (0.20, 0.45),
                "threshold_range": (0.60, 0.72),
                "n_turns_range": (8, 12),
                "id_start": 16,
                "category_name": "Omission Error (Event-based)",
                "desc_suffix": (
                    "Omission probe. Every turn before trigger_turn contains a DECOY phrase "
                    "near the cue (sim 0.3-0.55); trigger turn carries the genuine non-focal cue."
                ),
            },
            "content": {
                "target_count": 7,
                "focality_range": (0.70, 0.95),
                "threshold_range": (0.70, 0.80),
                "n_turns_range": (5, 7),
                "id_start": 24,
                "category_name": "Content Error (Event-based)",
                "desc_suffix": (
                    "Content-error probe. Trigger fires correctly; action_description is "
                    "specific enough that a generic execution misses it."
                ),
            },
        },
        # These four fields are populated at generation time from by_failure_mode[mode].
        "target_count": None,
        "focality_range": None,
        "threshold_range": None,
        "n_turns_range": None,
        "n_cues": (3, 4),
        "desc": "Failure-mode probes (commission / omission / content).",
    },
}


# =============================================================================
# Prompt builders
# =============================================================================

def load_seed_examples(category: str, n: int = 2) -> list[dict]:
    """Return up to N seed samples from this category as in-context examples."""
    with open(SEED_PATH, encoding="utf-8") as f:
        all_seed = json.load(f)
    matches = [s for s in all_seed if s["category"] == category]
    random.shuffle(matches)
    return matches[:n]


def build_generation_prompt(
    category: str,
    domain: str,
    idx: int,
    failure_mode: str | None = None,
) -> str:
    spec = dict(CATEGORY_SPECS[category])
    sub_category_name = spec["name"]
    # For D3, fold the sub-mode-specific params into the working spec copy.
    if category == "D3":
        if failure_mode not in spec.get("by_failure_mode", {}):
            raise ValueError(
                f"D3 requires failure_mode in {list(spec['by_failure_mode'])}; got {failure_mode!r}"
            )
        sub = spec["by_failure_mode"][failure_mode]
        spec["target_count"] = sub["target_count"]
        spec["focality_range"] = sub["focality_range"]
        spec["threshold_range"] = sub["threshold_range"]
        spec["n_turns_range"] = sub["n_turns_range"]
        spec["desc"] = spec["desc"] + " " + sub["desc_suffix"]
        sub_category_name = sub["category_name"]
        sample_id = f"D3-{sub['id_start'] + (idx - 1):02d}"
    else:
        sample_id = f"{category}-{idx:02d}"
    # The composite categories (B2/B3/C2/C3/D1/D2/D3) have no same-category seed examples.
    # To teach the LLM the required JSON NESTING (especially that trigger fields live
    # inside an `intention` sub-dict), we borrow one real sample from the structurally
    # closest EXISTING category as a schema reference; the structure_hint above already
    # documents how this category differs. D1 has no structural match (it uses an
    # `intentions` LIST), so it relies on the explicit skeleton in its structure_hint.
    examples = load_seed_examples(category, n=2)
    examples_block = ""
    if examples:
        examples_str = json.dumps(examples, indent=2, ensure_ascii=False)
        examples_block = (
            "TWO REFERENCE EXAMPLES (same category, matching schema):\n"
            f"```json\n{examples_str}\n```\n\n"
        )
    else:
        reference_category = {
            "B2": "B1", "B3": "B1",      # time-based
            "C2": "C1", "C3": "C1",      # activity-based
            "D2": "A2",                  # event-based non-focal
            "D3": "A2" if failure_mode == "omission" else "A1",  # event-based
        }.get(category)
        ref = load_seed_examples(reference_category, n=1) if reference_category else []
        if ref:
            ref_str = json.dumps(ref[0], indent=2, ensure_ascii=False)
            examples_block = (
                f"SCHEMA REFERENCE (a real {reference_category} sample — copy its JSON NESTING, "
                "especially that all trigger fields live inside the `intention` sub-dict; then "
                "apply the structural differences described above for this category):\n"
                f"```json\n{ref_str}\n```\n\n"
            )
        else:
            examples_block = (
                "REFERENCE SCHEMA: follow the explicit JSON skeleton in the structure hint "
                "above; place all trigger fields inside the `intention` sub-dict.\n\n"
            )

    focality_hint = ""
    if spec["focality_range"]:
        focality_hint = f"- focality: pick a value in [{spec['focality_range'][0]:.2f}, {spec['focality_range'][1]:.2f}]\n"

    threshold_hint = ""
    if spec["threshold_range"]:
        threshold_hint = f"- semantic_threshold: pick a value in [{spec['threshold_range'][0]:.2f}, {spec['threshold_range'][1]:.2f}]\n"

    cues_hint = ""
    if spec["n_cues"]:
        cues_hint = (
            f"- event_cues: {spec['n_cues'][0]}-{spec['n_cues'][1]} short phrases "
            "(not single ambiguous words like 'meeting' — prefer 2-4 word phrases "
            "like 'lunch with marketing team')\n"
        )

    # Structure hint: for non-focal categories, earlier turns MUST be topically disjoint
    if category in ("A2", "A3"):
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for non-focal):\n"
            f"  Turns 1 to {spec['n_turns_range'][0] - 2}: user discusses topic A, which is TOPICALLY "
            "UNRELATED to the PM cue concept. E.g., if the PM is about 'client meeting preparation', "
            "earlier turns should be about 'lunch plans' or 'weekend trip' — NOT about clients, meetings, "
            "or professional prep. Check: would a BGE embedding sim between these early turns and "
            "the cue be LOW (< 0.5)? If not, rewrite.\n"
            f"  Turn {spec['n_turns_range'][0] - 1} or {spec['n_turns_range'][0]} (trigger_turn): user "
            "PIVOTS to a new topic that semantically matches the cue (via inference for A2, via 2-hop "
            "for A3).\n"
        )
    elif category == "C1":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for activity-based):\n"
            "  Turns 1 to trigger_turn-1: user describes work IN PROGRESS. Use phrases like 'I'm "
            "working on', 'next step is', 'still need to', 'halfway through'. NEVER use phrases that "
            "could be read as completion: 'that seems done', 'almost ready', 'ready to submit', "
            "'finished with', 'all set' — these belong ONLY at trigger_turn.\n"
            "  Turn trigger_turn: user CLEARLY signals the activity is DONE. Use unambiguous "
            "completion markers: 'bags are packed and by the door', 'code is deployed', 'draft is "
            "finalized and submitted', 'everything is ready to go'.\n"
        )
    elif category == "DC":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for DC commission test):\n"
            "  Turns 1 to trigger_turn-1: user discusses OTHER errands/tasks, NOT mentioning the "
            "cue subject at all. No synonyms, no precursors.\n"
            "  Turn trigger_turn: user FIRST introduces the cue topic (e.g., 'going to the pharmacy').\n"
            "  At least 2 later turns: user re-mentions the SAME cue word/phrase (not paraphrased).\n"
        )
    elif category == "B2":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for B2 interval):\n"
            "  Pick step_interval K in {2,3,4}. Set target_steps = [K, 2K, 3K, ...] up to n_turns.\n"
            "  Top-level 'trigger_turns' (PLURAL list) MUST EQUAL intention.target_steps.\n"
            "  This category has NO singular 'trigger_turn' field.\n"
            "  Each turn is the user describing in-progress work that benefits from periodic\n"
            "  checkpoints (long planning, batch labeling, multi-iteration debugging). DO NOT\n"
            "  inject any explicit cue word ('checkpoint', 'summarize') — the schedule itself fires.\n"
            "  EXACT shape (note step_interval + target_steps live INSIDE intention):\n"
            "    {\n"
            '      "id": "...", "category": "B2", "category_name": "...", "difficulty": "medium",\n'
            '      "intention": {\n'
            '        "trigger_type": "time_based", "step_interval": 3, "target_steps": [3,6,9,12],\n'
            '        "action_description": "...", "implementation_intention": "EVERY 3 steps THEN ..."\n'
            "      },\n"
            '      "turns": [...12+ turns...], "trigger_turns": [3,6,9,12],\n'
            '      "description": "...", "expected_action": "...", "domain": "..."\n'
            "    }\n"
        )
    elif category == "B3":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for B3 dynamic time):\n"
            "  Pick a concrete anchor event (e.g., 'user announces submission deadline of Friday').\n"
            "  anchor_turn ∈ {2..n_turns-3}; delay_steps ∈ {2,3}; trigger_turn = anchor_turn + delay_steps.\n"
            "  Earlier turns: unrelated context. Anchor turn: user explicitly states the anchor.\n"
            "  Intermediate turns between anchor and trigger: user continues regular work.\n"
            "  intention.anchor_event MUST describe the anchor at the conceptual level, not as a\n"
            "  literal substring of the anchor turn (so the LLM monitor needs semantic judgment).\n"
        )
    elif category == "C2":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for C2 multi-activity AND):\n"
            "  Pick 2 or 3 ACTIVITIES. List them in intention.activity_completions as SHORT\n"
            "  noun phrases naming the FINISHED state (e.g., ['data cleaning', 'report draft']).\n"
            "  CRITICAL — SINGLE COMPLETION POINT PER ACTIVITY: each activity is declared DONE\n"
            "  in EXACTLY ONE turn, with an unambiguous marker ('X is done/complete/finalized').\n"
            "  That activity MUST NOT appear with any completion-sounding language in ANY other\n"
            "  turn — not earlier (no 'summarized/identified/wrapped up' that reads as done), and\n"
            "  not in a later recap. Before its completion turn, describe it ONLY as in-progress\n"
            "  ('working on', 'still need to', 'halfway through').\n"
            "  Order the completions so the LAST activity finishes at trigger_turn; trigger_turn\n"
            "  is the FINAL turn (no trailing recap turns after it). Do NOT add a closing turn\n"
            "  like 'both X and Y are now complete' — trigger_turn itself is the last completion.\n"
        )
    elif category == "C3":
        # Deterministically assign every 3rd sample to the no-fire case so the
        # batch lands ~1/3 CASE B (independent LLM calls otherwise always pick
        # CASE A). Uses the existing idx; no new identifiers.
        c3_no_fire = (idx % 3 == 0)
        if c3_no_fire:
            c3_case = (
                "  THIS SAMPLE MUST BE A NO-FIRE CASE (CASE B): intention.outcome MUST DIFFER from\n"
                "  top-level ground_truth_outcome. The watched activity actually resolves to the\n"
                "  OPPOSITE of what the intention watches for, so the agent must NOT fire.\n"
                "  Set trigger_turn to null. Example: intention.outcome='fail' but the activity\n"
                "  succeeds (ground_truth_outcome='success'); the fallback action must be suppressed.\n"
            )
        else:
            c3_case = (
                "  THIS SAMPLE MUST BE A FIRE CASE (CASE A): intention.outcome MUST EQUAL top-level\n"
                "  ground_truth_outcome. Set trigger_turn to the turn where that outcome becomes clear.\n"
                "  Example: intention.outcome='fail' and the activity indeed fails, so the fallback fires.\n"
            )
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for C3 conditional activity):\n"
            "  intention.activity_completion MUST be a NEUTRAL activity name with NO outcome\n"
            "  word in it (e.g., 'the database migration', 'the deployment', 'the API call') —\n"
            "  do NOT bake 'fail'/'error'/'success' into the activity description; the outcome\n"
            "  lives ONLY in intention.outcome.\n"
            "  intention.outcome ∈ {'fail','success'} = the outcome the intention watches for.\n"
            "  top-level ground_truth_outcome ∈ {'fail','success'} = what the scenario exhibits.\n"
            f"{c3_case}"
            "  Turn 1 and early turns must describe the activity STARTING / in progress with NO\n"
            "  definitive outcome. Only the resolution turn shows a CLEAR, explicit success or\n"
            "  failure signal (e.g., 'ERROR: ... aborting' for fail; 'completed successfully' for\n"
            "  success). The watched outcome signal must appear in exactly that turn, not earlier.\n"
        )
    elif category == "D1":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for D1 multi-intention):\n"
            "  Top-level field is 'intentions' (LIST), NOT 'intention'. 3-5 entries, each with\n"
            "  DISTINCT trigger_turn (no two intentions fire on the same turn — the engine's\n"
            "  single-firing-per-step rule would otherwise force latency). Mix trigger types:\n"
            "  at least one event_based, at least one time_based or activity_based.\n"
            "  Do NOT include a top-level 'trigger_turn' field.\n"
            "  EXACT shape:\n"
            "    {\n"
            '      "id": "...", "category": "D1", "category_name": "...", "difficulty": "hard",\n'
            '      "intentions": [\n'
            '        {"intention_id": "<id>-i1", "trigger_type": "event_based",\n'
            '         "event_cues": ["pharmacy","drugstore"], "focality": 0.85,\n'
            '         "semantic_threshold": 0.75, "action_description": "...",\n'
            '         "implementation_intention": "IF ... THEN ...", "trigger_turn": 3},\n'
            '        {"intention_id": "<id>-i2", "trigger_type": "time_based", "target_step": 5,\n'
            '         "action_description": "...", "implementation_intention": "AT step 5 THEN ...",\n'
            '         "trigger_turn": 5},\n'
            '        {"intention_id": "<id>-i3", "trigger_type": "activity_based",\n'
            '         "activity_completion": "report drafted", "action_description": "...",\n'
            '         "implementation_intention": "AFTER ... THEN ...", "trigger_turn": 7}\n'
            "      ],\n"
            '      "turns": [...8-12 turns...],\n'
            '      "description": "...", "expected_action": "...", "domain": "..."\n'
            "    }\n"
        )
    elif category == "D2":
        # Deterministically cycle complexity low/medium/high by idx so the 15-sample
        # batch spans the gradient the plan calls for ("increasing ongoing task
        # complexity"). Independent LLM calls otherwise all pick 'high'. Uses the
        # existing idx and the plan's existing complexity values; no new identifiers.
        d2_complexity = ("low", "medium", "high")[(idx - 1) % 3]
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for D2 interference):\n"
            "  Standard single-intention non-focal event-based PM (same structure as A2).\n"
            "  ADDITIONALLY, include a top-level 'ongoing_task' block:\n"
            "    {\n"
            "      'task_type': 'math' | 'coding' | 'logic_qa',\n"
            f"      'complexity': '{d2_complexity}',   (THIS SAMPLE MUST USE complexity='{d2_complexity}')\n"
            "      'expected_answers': { 'turn_2': '<expected>', 'turn_4': '<expected>', ... }\n"
            "    }\n"
            f"  Calibrate the per-turn ongoing-task difficulty to the '{d2_complexity}' level: "
            "low = single-step arithmetic / trivial lookups; medium = 2-3 step reasoning; "
            "high = multi-step derivations or nested logic.\n"
            "  The ongoing task should require genuine reasoning each turn (e.g., 'what is "
            "  17x23?', 'simplify this regex'). Include expected answers for at least 3-4 turns.\n"
            "  Mix in the PM cue at trigger_turn alongside the ongoing-task content.\n"
        )
    elif category == "D3" and failure_mode == "commission":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for D3 commission probe):\n"
            "  Focal event-based PM (A1-like cue). The cue phrase must appear LITERALLY in at\n"
            "  least 3 different turns (trigger_turn + 2 later). The intention triggers ONLY at\n"
            "  the FIRST occurrence; later cue mentions must not re-fire.\n"
        )
    elif category == "D3" and failure_mode == "omission":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for D3 omission probe):\n"
            "  Non-focal event-based PM (A2-like). EVERY turn before trigger_turn must contain at\n"
            "  least one DECOY — a phrase that is semantically NEAR the cue (BGE sim 0.3-0.55) but\n"
            "  is NOT the cue. Decoys exhaust monitor budget / tempt false positives. Trigger turn\n"
            "  carries the genuine cue.\n"
        )
    elif category == "D3" and failure_mode == "content":
        structure_hint = (
            "\nSCENARIO STRUCTURE (MANDATORY for D3 content-error probe):\n"
            "  Focal event-based PM (A1-like). The action_description must be SPECIFIC enough that\n"
            "  a generic execution misses it (named entity / quantity / conditional clause / phone\n"
            "  number / file path / cited spec section). expected_action mirrors the specificity.\n"
            "  Trigger fires correctly at trigger_turn.\n"
        )
    else:
        structure_hint = ""

    prompt = f"""You are generating a prospective-memory (PM) benchmark sample.

Category: {category} — {spec['name']}
Difficulty: {spec['difficulty']}
Domain: {domain}

PARADIGM DEFINITION:
{spec['desc']}
{structure_hint}

{examples_block}YOUR TASK:
Produce ONE new sample for category {category} in domain `{domain}`.
The sample must be substantively DIFFERENT from the reference examples (different action, different
trigger phrasing, different scenario setup), not a paraphrase. It must follow the same JSON schema.

Constraints:
- id: "{sample_id}"
- category: "{category}"
- category_name: "{sub_category_name}"
- difficulty: "{spec['difficulty']}"
- trigger_type: "{spec['trigger_type']}"
- turns: {spec['n_turns_range'][0]}-{spec['n_turns_range'][1]} natural user utterances, forming a coherent multi-turn scenario
- trigger_turn: 1-indexed turn where the PM should fire (see category paradigm for timing)
- description: one-sentence summary of what this sample tests
- domain: "{domain}"
- expected_action: one-sentence description of what the agent should do when triggered
{focality_hint}{threshold_hint}{cues_hint}

CRITICAL QUALITY RULES:
1. The trigger at `trigger_turn` MUST actually satisfy the PM cue according to the category paradigm.
2. For A1: the cue word/phrase SHOULD appear literally in `turns[trigger_turn-1]`, and MUST NOT appear (literally or as a close synonym) in any earlier turn.
3. For A2/A3: the cue should NOT appear literally anywhere. Earlier turns must be on a DIFFERENT TOPIC
   from the cue concept (e.g., if cue is about "note-taking", earlier turns should discuss unrelated
   setup like "checking weather", "coffee order" — not "studying", "classroom", "textbook"). Only at
   trigger_turn does the user mention the specific scenario that requires inference to reach the cue.
4. For DC: the cue must appear AT TRIGGER_TURN and at AT LEAST TWO later turns, but NOT in any turn
   before trigger_turn. The earlier turns set up unrelated context (e.g., other errands) so only
   trigger_turn introduces the cue for the first time.
5. For B1: target_step MUST equal trigger_turn; turns should not semantically imply the action at trigger_turn.
6. For C1: the completion cue appears at trigger_turn as user signaling task done. Earlier turns
   describe work IN PROGRESS (not completion). Avoid phrases like "that seems clear", "I'm done",
   "ready to submit" in turns before trigger_turn — those belong ONLY at trigger_turn.
7. Action description and expected_action should match semantically.
8. Turns should read as a real user-agent dialogue, not an artificial cue list.

OUTPUT FORMAT (STRICT): return ONLY valid JSON matching the reference schema. No markdown fences, no commentary, no thinking. Just the JSON object starting with `{{` and ending with `}}`.
"""
    return prompt


# =============================================================================
# Validation
# =============================================================================

def validate_sample(
    sample: dict,
    category: str,
    expected_id: str,
    failure_mode: str | None = None,
) -> tuple[bool, list[str]]:
    """Structural + category-specific validation. Returns (ok, errors).

    `category` is one of the 12 public category IDs (A1/A2/A3/B1/B2/B3/C1/C2/C3/D1/D2/D3)
    plus the legacy DC kept for the existing 15 samples awaiting data-rename. For D3,
    `failure_mode` must be passed to select the parameter sub-table.
    """
    errors: list[str] = []
    spec = dict(CATEGORY_SPECS[category])
    # D3 dispatch on failure_mode (single category, three sub-modes)
    if category == "D3":
        sub_table = spec.get("by_failure_mode", {})
        if failure_mode not in sub_table:
            return False, [f"D3 requires failure_mode in {list(sub_table)}; got {failure_mode!r}"]
        sub = sub_table[failure_mode]
        spec["target_count"] = sub["target_count"]
        spec["focality_range"] = sub["focality_range"]
        spec["threshold_range"] = sub["threshold_range"]
        spec["n_turns_range"] = sub["n_turns_range"]

    # Required top-level fields — vary by category schema:
    #   D1 uses 'intentions' (list) instead of 'intention' + 'trigger_turn'
    #   B2 fires at multiple turns: uses 'trigger_turns' (plural) instead of 'trigger_turn'
    #   D2 also requires 'ongoing_task' block
    #   C3 may omit 'trigger_turn' when intention.outcome != ground_truth_outcome (no-fire case)
    # D3 sub-mode is encoded by the ID range + the category_name string (per the
    # by_failure_mode["category_name"] override); no extra sample-level field.
    if category == "D1":
        required_top = ["id", "category", "category_name", "difficulty",
                        "intentions", "turns", "expected_action", "description", "domain"]
    else:
        required_top = ["id", "category", "category_name", "difficulty",
                        "intention", "turns", "expected_action", "description", "domain"]
        if category == "B2":
            required_top.append("trigger_turns")
        elif category != "C3":
            required_top.append("trigger_turn")
        if category == "D2":
            required_top.append("ongoing_task")
    for k in required_top:
        if k not in sample:
            errors.append(f"missing top-level field: {k}")
    if errors:
        return False, errors

    # ID / category match — sample fields must equal the public expected values
    if sample["id"] != expected_id:
        errors.append(f"id {sample['id']} != expected {expected_id}")
    if sample["category"] != category:
        errors.append(f"category {sample['category']} != expected {category}")
    if category == "D3":
        expected_name = spec["by_failure_mode"][failure_mode]["category_name"]
        if sample.get("category_name") != expected_name:
            errors.append(
                f"D3: category_name {sample.get('category_name')!r} != expected {expected_name!r}"
            )

    # Turns
    turns = sample.get("turns", [])
    nmin, nmax = spec["n_turns_range"]
    if not (nmin <= len(turns) <= nmax):
        errors.append(f"n_turns {len(turns)} outside [{nmin}, {nmax}]")
    if not all(isinstance(t, str) and len(t) > 10 for t in turns):
        errors.append("some turns are non-string or too short")

    # D1 uses an `intentions` LIST (no single intention / trigger_turn). Its
    # per-intention checks live in the D1-specific block below; skip all the
    # single-intention generic checks (trigger_turn range, intention sub-schema,
    # trigger-type-specific) for D1.
    tt = 0
    if category != "D1":
        # trigger_turn in range. B2 fires at multiple turns (trigger_turns list);
        # each must be in range. Other categories use a single trigger_turn (C3
        # may omit it in the no-fire case, handled below).
        if category == "B2":
            ttl = sample.get("trigger_turns", [])
            if not isinstance(ttl, list) or not ttl:
                errors.append("B2: trigger_turns must be a non-empty list")
            elif not all(isinstance(t, int) and 1 <= t <= len(turns) for t in ttl):
                errors.append(f"B2: some trigger_turns {ttl} outside [1, {len(turns)}]")
        else:
            tt = sample.get("trigger_turn", 0)
            if category != "C3" and not (1 <= tt <= len(turns)):
                errors.append(f"trigger_turn {tt} outside [1, {len(turns)}]")

        # Intention sub-schema
        intention = sample.get("intention", {})
        if intention.get("trigger_type") != spec["trigger_type"]:
            errors.append("intention.trigger_type mismatch")
        if not intention.get("action_description"):
            errors.append("missing intention.action_description")
        if not intention.get("implementation_intention"):
            errors.append("missing intention.implementation_intention")
    else:
        intention = {}

    # Category-specific
    if spec["trigger_type"] == "event_based":
        cues = intention.get("event_cues", [])
        if spec["n_cues"]:
            cmin, cmax = spec["n_cues"]
            if not (cmin <= len(cues) <= cmax):
                errors.append(f"n_cues {len(cues)} outside [{cmin}, {cmax}]")
        focality = intention.get("focality")
        if spec["focality_range"] and focality is not None:
            fmin, fmax = spec["focality_range"]
            if not (fmin - 0.05 <= focality <= fmax + 0.05):
                errors.append(f"focality {focality} outside [{fmin}, {fmax}]")
        if spec["threshold_range"]:
            th = intention.get("semantic_threshold", 0.7)
            tmin, tmax = spec["threshold_range"]
            if not (tmin - 0.02 <= th <= tmax + 0.02):
                errors.append(f"semantic_threshold {th} outside [{tmin}, {tmax}]")

        # A1: cue literal appearance
        if category == "A1":
            trigger_turn_text = turns[tt - 1].lower() if 1 <= tt <= len(turns) else ""
            cue_hit = any(any(word in trigger_turn_text for word in c.lower().split())
                          for c in cues if c)
            if not cue_hit:
                errors.append(f"A1: no cue word appears in trigger turn '{trigger_turn_text[:80]}'")

        # A2/A3: cues must NOT appear literally in any turn (non-focal requires semantic bridge)
        if category in ("A2", "A3"):
            all_text = " ".join(turns).lower()
            for c in cues:
                cl = c.lower().strip()
                if len(cl) < 6:
                    continue  # skip short generic words
                if cl in all_text:
                    errors.append(f"{category}: cue '{c}' appears literally in turns (non-focal must use abstract cues)")
                    break

        # DC: cue should appear in 2+ turns
        if category == "DC":
            cue_turns = []
            for i, t in enumerate(turns, 1):
                tl = t.lower()
                if any(c.lower() in tl for c in cues if c):
                    cue_turns.append(i)
            if len(cue_turns) < 2:
                errors.append(f"DC: cue appears in only {len(cue_turns)} turns (need 2+)")
            elif tt != min(cue_turns):
                errors.append(f"DC: trigger_turn {tt} is not first cue occurrence {cue_turns}")

    elif spec["trigger_type"] == "time_based":
        # B1 (single absolute step): target_step must equal the single trigger_turn.
        # B2 (recurring interval) uses target_steps/trigger_turns plural — checked in
        # its own block below, not here.
        if category == "B1":
            target = intention.get("target_step")
            if target != tt:
                errors.append(f"B1: target_step {target} != trigger_turn {tt}")

    elif spec["trigger_type"] == "activity_based":
        # C1 uses single activity_completion; C2 uses activity_completions list;
        # C3 uses single activity_completion + outcome
        if category == "C2":
            comps = intention.get("activity_completions", [])
            if not isinstance(comps, list) or not (2 <= len(comps) <= 3):
                errors.append(f"C2: activity_completions must be a list of 2-3 (got {len(comps)})")
        elif category == "C3":
            if not intention.get("activity_completion"):
                errors.append("C3: missing intention.activity_completion")
            out = intention.get("outcome")
            if out not in ("fail", "success"):
                errors.append(f"C3: intention.outcome must be fail/success (got {out!r})")
            gt = sample.get("ground_truth_outcome")
            if gt not in ("fail", "success"):
                errors.append(f"C3: missing/invalid top-level ground_truth_outcome (got {gt!r})")
            should_no_fire = (out != gt) if (out and gt) else False
            if not should_no_fire and (not isinstance(tt, int) or tt < 1 or tt > len(turns)):
                errors.append(
                    f"C3 fire-case (outcome==ground_truth) needs a valid trigger_turn; got {tt!r}"
                )
        else:
            if not intention.get("activity_completion"):
                errors.append("C1: missing activity_completion")

    elif spec["trigger_type"] == "time_based_dynamic":
        # B3
        if not intention.get("anchor_event"):
            errors.append("B3: missing intention.anchor_event")
        delay = intention.get("delay_steps")
        if not isinstance(delay, int) or delay < 1:
            errors.append(f"B3: delay_steps must be int >= 1 (got {delay!r})")
        anchor_turn = sample.get("anchor_turn")
        if not isinstance(anchor_turn, int) or anchor_turn < 1:
            errors.append(f"B3: missing/invalid top-level anchor_turn (got {anchor_turn!r})")
        if isinstance(anchor_turn, int) and isinstance(delay, int) and tt != anchor_turn + delay:
            errors.append(
                f"B3: trigger_turn {tt} != anchor_turn {anchor_turn} + delay {delay}"
            )

    # B2-specific (still trigger_type=time_based, but with step_interval + target_steps list)
    if category == "B2":
        interval = intention.get("step_interval")
        target_steps = intention.get("target_steps")
        if not isinstance(interval, int) or interval < 2:
            errors.append(f"B2: step_interval must be int >= 2 (got {interval!r})")
        if not isinstance(target_steps, list) or len(target_steps) < 3:
            errors.append(f"B2: target_steps must be a list of length >= 3 (got {target_steps!r})")
        elif isinstance(interval, int):
            expected_targets = list(range(interval, len(turns) + 1, interval))
            if target_steps != expected_targets:
                errors.append(
                    f"B2: target_steps {target_steps} != expected interval expansion {expected_targets}"
                )
        # The single 'trigger_turn' field semantically isn't a single value for B2; allow it
        # to equal target_steps[0] OR be replaced by 'trigger_turns' list parallel to target_steps.
        trigger_turns_list = sample.get("trigger_turns")
        if trigger_turns_list and target_steps and trigger_turns_list != target_steps:
            errors.append(
                f"B2: trigger_turns {trigger_turns_list} != target_steps {target_steps}"
            )

    # D1 multi-intention
    if category == "D1":
        intentions = sample.get("intentions", [])
        if not isinstance(intentions, list) or not (3 <= len(intentions) <= 5):
            errors.append(f"D1: intentions must be list of 3-5 (got {len(intentions)})")
        seen_turns: set[int] = set()
        for j, it in enumerate(intentions):
            if "trigger_type" not in it or "action_description" not in it:
                errors.append(f"D1: intention[{j}] missing trigger_type/action_description")
            etn = it.get("trigger_turn")
            if not isinstance(etn, int) or etn < 1 or etn > len(turns):
                errors.append(f"D1: intention[{j}] invalid trigger_turn={etn!r}")
            elif etn in seen_turns:
                errors.append(f"D1: intention[{j}] trigger_turn={etn} collides with another")
            else:
                seen_turns.add(etn)
            if not it.get("intention_id"):
                # We'll synthesize a deterministic id if missing
                it["intention_id"] = f"{expected_id}-i{j+1}"

    # D2 ongoing_task block
    if category == "D2":
        ot = sample.get("ongoing_task", {})
        if not isinstance(ot, dict) or "expected_answers" not in ot:
            errors.append("D2: ongoing_task must contain expected_answers dict")
        else:
            ans = ot["expected_answers"]
            if not isinstance(ans, dict) or len(ans) < 3:
                errors.append(f"D2: ongoing_task.expected_answers needs >=3 entries (got {len(ans)})")
        if ot.get("complexity") not in ("low", "medium", "high"):
            errors.append(f"D2: ongoing_task.complexity must be low/medium/high (got {ot.get('complexity')!r})")

    return (len(errors) == 0), errors


def semantic_validation(sample: dict, category: str) -> tuple[bool, list[str]]:
    """Embedding-based check: simulate ProsMem's associative retrieval on each turn.
    Reject if any turn earlier than trigger_turn has cue-similarity >= threshold,
    and require trigger_turn itself to reach threshold (for focal event-based)."""
    errors: list[str] = []
    spec = CATEGORY_SPECS[category]

    # D1 has no single `intention` (it uses an `intentions` list); per-intention
    # semantic checks are out of scope for the embedding pre-screen — structural
    # validation + the PM-engine validation pass cover D1.
    if category == "D1":
        return True, []

    # Time-based families don't use embedding-based semantic matching:
    #   B1/B2 (time_based) fire on the step counter; B3 (time_based_dynamic) is
    #   LLM-judged on the anchor. Return before touching trigger_turn — B2 has
    #   no singular trigger_turn (it uses the trigger_turns list).
    if spec["trigger_type"] in ("time_based", "time_based_dynamic"):
        return True, []

    intention = sample["intention"]
    # C3 no-fire samples (outcome != ground_truth_outcome) omit trigger_turn; the
    # activity_based branch below never uses it for C2/C3 anyway. Default to 0.
    trigger_turn = sample.get("trigger_turn", 0)
    turns = sample["turns"]

    if spec["trigger_type"] == "event_based":
        cues = intention.get("event_cues", [])
        if not cues:
            return False, ["no cues"]
        threshold = intention.get("semantic_threshold", 0.7)
        focality = intention.get("focality", 0.5)

        cue_embs = embed_texts(cues)
        turn_embs = embed_texts(turns)

        # For each turn, find max cue sim
        max_sims = []
        for turn_emb in turn_embs:
            max_sim = max(cosine_similarity(turn_emb, ce) for ce in cue_embs)
            max_sims.append(max_sim)

        # Non-focal samples are checked by both paths; focal only by assoc
        # For the SEMANTIC gate we simulate assoc: sim >= threshold triggers
        # (we ignore the monitor's LLM judgment here; it's a soft filter)

        # Each turn before trigger_turn must NOT cross threshold (assoc false fire).
        # For focal (A1/DC) we want a clear margin; for non-focal (A2/A3) the DESIGN is
        # that fast path never fires (slow path handles it), so we only need sim < threshold
        # at the engine's actual decision boundary — no margin.
        leak_margin = 0.03 if category in ("A1", "DC") else 0.0
        for i, sim in enumerate(max_sims[: trigger_turn - 1], 1):
            if sim >= threshold - leak_margin:
                errors.append(
                    f"early-turn {i} sim={sim:.3f} >= threshold-margin {threshold - leak_margin:.3f}"
                )
        # For non-focal, also guard against trigger_turn itself tripping fast path —
        # the whole point is that slow LLM monitor is what fires, not embedding match.
        if category in ("A2", "A3"):
            tt_sim = max_sims[trigger_turn - 1]
            if tt_sim >= threshold:
                errors.append(
                    f"trigger-turn sim={tt_sim:.3f} >= threshold {threshold:.3f} "
                    f"(A2/A3 should rely on slow path, not fast-path match)"
                )

        # For A1/DC (focal) trigger_turn MUST reach threshold (otherwise missing match)
        # For A2/A3 the monitor may carry — but we still want trigger_turn to be a strong peak
        if category in ("A1", "DC"):
            tt_sim = max_sims[trigger_turn - 1]
            if tt_sim < threshold:
                errors.append(f"trigger-turn sim={tt_sim:.3f} < threshold {threshold:.3f}")

        # For DC: later cue mentions should ALSO score high (commission test requires repetition)
        if category == "DC":
            later_high = [i for i, s in enumerate(max_sims[trigger_turn:], trigger_turn + 1)
                          if s >= threshold - leak_margin]
            if len(later_high) < 1:
                errors.append("DC: no later turn reaches threshold — commission test will be trivial")

    elif spec["trigger_type"] == "activity_based":
        # C2 uses activity_completions (list); C3 uses single activity_completion;
        # C1 (existing seed) uses single activity_completion. We only run the
        # peak check for C1 — C2/C3 are LLM-judged at evaluation time and have
        # construct-level rules in validate_sample.
        if category == "C1":
            completion = intention.get("activity_completion", "")
            if not completion:
                return False, ["no activity_completion"]
            comp_emb = embed_text(completion)
            turn_embs = embed_texts(turns)
            sims = [cosine_similarity(te, comp_emb) for te in turn_embs]
            tt_sim = sims[trigger_turn - 1]
            if trigger_turn >= 2 and sims[trigger_turn - 2] >= tt_sim:
                errors.append(
                    f"C1: turn {trigger_turn - 1} sim={sims[trigger_turn - 2]:.3f} >= "
                    f"trigger-turn sim {tt_sim:.3f} (trigger must be a peak)"
                )
            if tt_sim < 0.55:
                errors.append(f"C1: trigger-turn sim={tt_sim:.3f} too low (<0.55)")
        # C2/C3 fall through (no semantic gate); construct-level checks in
        # validate_sample suffice.

    # (time_based / time_based_dynamic already returned at the top of this function.)

    return (len(errors) == 0), errors


# =============================================================================
# Generation loop
# =============================================================================

JSON_EXTRACT_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from an LLM response."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = JSON_EXTRACT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def model_for_category(category: str, failure_mode: str | None = None) -> str:
    # A2/A3 (non-focal) and D3 omission probes require strict topical disjointedness
    # between early turns and the cue concept. D1/D2 require multi-intention or
    # interleaved-ongoing-task schema adherence. C2/C3 are composite activity classes
    # that DeepSeek-V3 generates with internal inconsistencies (ambiguous completion
    # points for C2; outcome label not matching content for C3) — Grok follows the
    # "single unambiguous completion / outcome-consistent" instructions better.
    if category in ("A2", "A3", "D1", "D2", "C2", "C3"):
        return "x-ai/grok-4.3"
    if category == "D3" and failure_mode == "omission":
        return "x-ai/grok-4.3"
    return "deepseek/deepseek-chat-v3-0324"


def generate_one(
    llm: LLMClient,
    category: str,
    domain: str,
    idx: int,
    max_attempts: int = 6,
    verbose: bool = True,
    failure_mode: str | None = None,
) -> dict | None:
    """Generate a single sample with validate-retry loop.

    `category` is one of the 12 public category IDs. For D3, `failure_mode`
    must be one of {'commission','omission','content'} and the sample's
    on-disk `id` follows the unified D3-XX sequence (commission D3-01..15,
    omission D3-16..23, content D3-24..30).
    """
    # Compute expected id. For D3, the per-failure_mode id_start offset comes
    # from CATEGORY_SPECS["D3"]["by_failure_mode"][failure_mode]["id_start"].
    if category == "D3":
        sub = CATEGORY_SPECS["D3"]["by_failure_mode"][failure_mode]
        expected_id = f"D3-{sub['id_start'] + (idx - 1):02d}"
    else:
        expected_id = f"{category}-{idx:02d}"
    model = model_for_category(category, failure_mode=failure_mode)
    for attempt in range(1, max_attempts + 1):
        prompt = build_generation_prompt(category, domain, idx, failure_mode=failure_mode)
        try:
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                model=model,
                temperature=0.7,
                max_tokens=2000,
            )
        except Exception as e:
            if verbose:
                print(f"    [{expected_id} attempt {attempt}] LLM error: {e}")
            continue

        sample = extract_json(resp)
        if sample is None:
            if verbose:
                print(f"    [{expected_id} attempt {attempt}] JSON parse failed")
            continue

        ok, errors = validate_sample(sample, category, expected_id, failure_mode=failure_mode)
        if not ok:
            if verbose:
                print(f"    [{expected_id} attempt {attempt}] struct: {'; '.join(errors[:3])}")
            continue

        sem_ok, sem_errors = semantic_validation(sample, category)
        if not sem_ok:
            if verbose:
                print(f"    [{expected_id} attempt {attempt}] semantic: {'; '.join(sem_errors[:2])}")
            continue

        # C3: confirm the conversation's ACTUAL ending outcome matches the labeled
        # ground_truth_outcome (DeepSeek/Grok sometimes generate failure-content under a
        # 'success' label or vice versa). Reject mismatches so the no-fire/fire split is honest.
        if category == "C3":
            ok_c3, why = _c3_outcome_consistent(llm, sample)
            if not ok_c3:
                if verbose:
                    print(f"    [{expected_id} attempt {attempt}] c3-consistency: {why}")
                continue

        if verbose:
            print(f"    [{expected_id} attempt {attempt}] OK ({domain})")
        return sample
    return None


def _c3_outcome_consistent(llm: LLMClient, sample: dict) -> tuple[bool, str]:
    """Ask the LLM whether the conversation's actual ending matches the labeled
    ground_truth_outcome. Returns (ok, reason)."""
    gt = sample.get("ground_truth_outcome")
    activity = sample.get("intention", {}).get("activity_completion", "the activity")
    convo = "\n".join(f"{i}. {t}" for i, t in enumerate(sample.get("turns", []), 1))
    prompt = (
        "Read the conversation and decide the FINAL real-world outcome of the watched "
        "activity.\n\n"
        f"Watched activity: {activity}\n\n"
        f"Conversation:\n{convo}\n\n"
        "Does the activity ultimately SUCCEED or FAIL by the end? Reply strictly with one "
        "word: SUCCESS or FAIL."
    )
    resp = llm.chat(
        [{"role": "user", "content": prompt}],
        model="x-ai/grok-4.3", temperature=0.0, max_tokens=5,
    ).strip().upper()
    observed = "FAIL" if "FAIL" in resp else ("SUCCESS" if "SUCCESS" in resp else "?")
    if observed == "?":
        return False, f"judge returned {resp!r}"
    if observed != str(gt).upper():
        return False, f"content outcome={observed} but ground_truth_outcome={gt}"
    return True, "ok"


def generate_category(
    llm: LLMClient,
    category: str,
    count: int,
    start_idx: int,
    verbose: bool = True,
    failure_mode: str | None = None,
) -> list[dict]:
    """Generate N samples for a category, cycling through domains for diversity.

    For D3 generation, `start_idx` is the SUB-MODE-LOCAL index (1-based within the
    failure_mode). The actual on-disk id reflects the unified D3-XX sequence (see
    by_failure_mode[failure_mode]["id_start"] offset applied in generate_one)."""
    samples = []
    idx = start_idx
    attempted = 0
    while len(samples) < count and attempted < count * 3:
        domain = DOMAINS[(idx - 1) % len(DOMAINS)]
        s = generate_one(
            llm, category, domain, idx, verbose=verbose, failure_mode=failure_mode
        )
        attempted += 1
        if s is not None:
            samples.append(s)
            idx += 1
        time.sleep(0.3)  # gentle rate-limit
    return samples


def existing_max_idx(category: str, extra: list[dict] | None = None) -> int:
    """Find the highest existing index for this category (seed + optionally extra)."""
    with open(SEED_PATH, encoding="utf-8") as f:
        all_seed = json.load(f)
    pool = [s for s in all_seed if s["category"] == category]
    if extra:
        pool += [s for s in extra if s.get("category") == category]
    max_idx = 0
    for s in pool:
        m = re.match(rf"{category}-(\d+)$", s["id"])
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/generated_samples.json")
    ap.add_argument("--category", help="Generate only for this category (one of the 12 public IDs)")
    ap.add_argument(
        "--failure-mode",
        choices=("commission", "omission", "content"),
        help="Required when --category=D3. Picks the parameter sub-table.",
    )
    ap.add_argument("--count", type=int, help="Override target count for single-category mode")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate 1 sample per category (prompt/schema validation only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    config = ProsMemConfig.from_env()
    if not config.llm_api_key:
        print("ERROR: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    config.http_proxy = ""
    llm = LLMClient(config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output if present (for resume/append)
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing samples from {out_path}")

    # Default batch run skips D3 (it requires explicit --failure-mode) and skips the
    # legacy DC entry (existing 15 samples will be renamed to D3+failure_mode=commission
    # at the data-rename step, not regenerated).
    if args.category:
        if args.category not in CATEGORY_SPECS:
            print(f"ERROR: unknown category {args.category}", file=sys.stderr)
            return 2
        if args.category == "D3" and args.failure_mode is None:
            print("ERROR: --category D3 requires --failure-mode commission|omission|content",
                  file=sys.stderr)
            return 2
        categories_to_run = [args.category]
    else:
        categories_to_run = [c for c in CATEGORY_SPECS if c not in ("D3", "DC")]

    all_new: list[dict] = []
    for cat in categories_to_run:
        spec = CATEGORY_SPECS[cat]
        fm = args.failure_mode if cat == "D3" else None
        if cat == "D3":
            sub = spec["by_failure_mode"][fm]
            target_count = sub["target_count"]
            display_name = f"{spec['name']} ({fm})"
            start_idx = 1   # sub-mode-local; id_start offset applied in generate_one
            id_label = f"D3-{sub['id_start']:02d}"
        else:
            target_count = spec["target_count"]
            display_name = spec["name"]
            start_idx = existing_max_idx(cat, existing) + 1
            id_label = f"{cat}-{start_idx:02d}"
        count = 1 if args.dry_run else (args.count if args.count else target_count)
        print(f"\n=== {cat} ({display_name}): generating {count} starting at {id_label} ===")
        samples = generate_category(llm, cat, count, start_idx, failure_mode=fm)
        all_new.extend(samples)
        print(f"  → produced {len(samples)}/{count}")

    # Write
    combined = existing + all_new
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_new)} new samples ({len(combined)} total) to {out_path}")
    print(f"Tokens used: {llm.total_tokens_used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
