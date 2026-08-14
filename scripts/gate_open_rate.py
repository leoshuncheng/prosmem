"""Gate-open rate of the strategic monitor's DMF gate on ProsMem-Bench.

For every intention the monitor watches (non-focal event cues, activity conditions, dynamic-time
anchors; focal event cues and plain timers are left to the associative path and the timer), each
turn up to the labeled trigger turn is one monitored turn evaluation. The gate is open when the
turn's similarity to the condition reference reaches GATING_THRESHOLD (0.4):
  - non-focal event cue: cues joined with the action description;
  - single-activity condition (incl. outcome-qualified): the action description;
  - multi-activity condition: the closest pending activity;
  - dynamic-time intention: its anchor event.
Embeddings only (local BGE-small); no LLM call.

Run from the repository root: python scripts/gate_open_rate.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosmem.retrieval.embedder import cosine_similarity, embed_text, embed_texts  # noqa: E402
from prosmem.retrieval.strategic import StrategicMonitor  # noqa: E402

GATE = StrategicMonitor.GATING_THRESHOLD
NON_FOCAL = StrategicMonitor.NON_FOCAL_MAX
BENCH = ROOT / "prosmem" / "bench" / "golden_samples.json"


def intentions(sample: dict) -> list[dict]:
    if sample.get("intentions"):
        return sample["intentions"]
    if sample.get("intention"):
        return [sample["intention"]]
    return []


def turns_of(sample: dict) -> list:
    return sample.get("turns") or sample.get("conversation") or []


def trigger_turn(sample: dict, intention: dict) -> int:
    return sample.get("trigger_turn") or intention.get("trigger_turn") or len(turns_of(sample))


def references(sample: dict, intention: dict):
    """Condition reference embeddings for a monitored intention, or None if not monitored."""
    tt = (intention.get("trigger_type") or "").lower()
    if tt == "time_based_dynamic":
        anchor = intention.get("anchor_event")
        return [embed_text(anchor)] if anchor else None
    if tt == "activity_based":
        activities = intention.get("activity_completions") or []
        if activities and intention.get("outcome") is None:
            return embed_texts(activities)
        return [embed_text(intention.get("action_description", ""))]
    if tt == "event_based":
        if intention.get("focality", sample.get("focality", 0.5)) >= NON_FOCAL:
            return None  # focal cue: associative path, not monitored
        parts = list(intention.get("event_cues") or []) + [intention.get("action_description", "")]
        return [embed_text(" ".join(p for p in parts if p))]
    return None  # absolute / interval time: handled by the timer


def main() -> int:
    samples = json.loads(BENCH.read_text(encoding="utf-8"))
    print("samples loaded:", len(samples))
    opened_total = evaluated_total = monitored_samples = 0
    by_cat = collections.defaultdict(lambda: [0, 0])  # category -> [open, evaluated]
    for s in samples:
        turns = turns_of(s)
        if not turns:
            continue
        turn_embs = embed_texts([str(t) for t in turns])
        monitored = False
        for it in intentions(s):
            refs = references(s, it)
            if refs is None:
                continue
            monitored = True
            t_trig = trigger_turn(s, it)
            window = turn_embs[:t_trig] if t_trig and t_trig <= len(turn_embs) else turn_embs
            for te in window:
                is_open = max(cosine_similarity(te, r) for r in refs) >= GATE
                evaluated_total += 1
                by_cat[s.get("category")][1] += 1
                if is_open:
                    opened_total += 1
                    by_cat[s.get("category")][0] += 1
        monitored_samples += monitored
    print(f"samples with monitored intentions: {monitored_samples}")
    print(f"OVERALL gate-open rate: {opened_total}/{evaluated_total} = "
          f"{opened_total / evaluated_total * 100:.1f}%\n")
    print(f'{"cat":>4} {"open":>6} {"eval":>6} {"rate":>7}')
    for cat in sorted(by_cat):
        o, e = by_cat[cat]
        print(f"{cat:>4} {o:>6} {e:>6} {o / e * 100:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
