"""LoCoMo dataset loader.

LoCoMo (Maharana et al., ACL 2024) is a long-term conversation QA benchmark:
- 10 conversations, up to 35 sessions each, spanning months of simulated time.
- 1986 QA items total, each labeled with one of 5 categories.

Category mapping inferred from (a) the official task_eval/evaluation.py eval
branches and (b) example QA content per cat:
    1 = multi_hop       (eval uses sub-answer split F1 — from LoCoMo code)
    2 = temporal        ("when did X" → date answers; evaluator special-cases)
    3 = commonsense     (counterfactual "would X if..." — reasoning)
    4 = open_domain     (inference over conversation knowledge)
    5 = adversarial     (unanswerable — empty answer field in data)

Authoritative cross-check should use the LoCoMo paper §3 / Table 2 category
definitions. Per-category counts observed: 1=282, 2=321, 3=96, 4=841, 5=446.

Schema emitted by `load_locomo()` — one entry per conversation:
    {
      "id":          "conv-26",
      "personas":    {"a": "...", "b": "..."},
      "turns":       [                       # flat chronological list
          {"session": 1, "date": "...", "speaker": "Caroline",
           "dia_id": "D1:1", "text": "..."},
          ...
      ],
      "qa":          [
          {"qa_id": "conv-26::qa-0", "question": "...", "answer": "...",
           "category": 2, "category_name": "single_hop",
           "evidence": ["D1:3"]},
          ...
      ],
    }

Rationale for the flat `turns` list: ProsMem (and the retrospective baselines)
ingest the conversation turn-by-turn the same way they do on ProsMem-Bench.
Sessions are preserved as a `session` field for temporal reasoning.

Data source: https://github.com/snap-research/locomo
Expected path: third_party/locomo/data/locomo10.json
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("third_party/locomo/data/locomo10.json")

# Mapping derived from LoCoMo's task_eval/evaluation.py branches + sample inspection.
# cat 1 is unambiguous (multi-hop split-F1 code path); 2/3/4 are inferred from QA
# content patterns; 5 is adversarial (empty answers).
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "commonsense",
    4: "open_domain",
    5: "adversarial",
}

# Session keys look like "session_1", "session_12". Date-time keys have _date_time.
_SESSION_RE = re.compile(r"^session_(\d+)$")
_DATETIME_RE = re.compile(r"^session_(\d+)_date_time$")


def _flatten_conversation(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested session structure into a chronologically ordered
    list of turns, each annotated with session number and date."""
    # Collect date-times keyed by session number
    dates: dict[int, str] = {}
    for k, v in conv.items():
        m = _DATETIME_RE.match(k)
        if m:
            dates[int(m.group(1))] = v

    turns: list[dict[str, Any]] = []
    for k, v in conv.items():
        m = _SESSION_RE.match(k)
        if not m:
            continue
        sess_num = int(m.group(1))
        if not isinstance(v, list):
            # Some sessions have date_time only, no content — skip.
            continue
        date = dates.get(sess_num, "")
        for t in v:
            turns.append({
                "session": sess_num,
                "date": date,
                "speaker": t.get("speaker", ""),
                "dia_id": t.get("dia_id", ""),
                "text": t.get("text", ""),
            })
    # Sort by session number, preserving in-session order (stable sort).
    turns.sort(key=lambda x: x["session"])
    return turns


def _parse_evidence(raw: Any) -> list[str]:
    """Evidence field is stored as a string repr of a list (e.g. "['D1:3']").
    Return a plain list of dialog-id strings; empty list if unparseable."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not raw or not isinstance(raw, str):
        return []
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v]
    except (ValueError, SyntaxError):
        pass
    return []


def load_locomo(path: str | Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    """Load LoCoMo as a list of conversation entries with flattened turns + QA.

    Args:
        path: path to locomo10.json (defaults to third_party/locomo/data/...)

    Returns:
        List of dicts, one per conversation. See module docstring for schema.

    Raises:
        FileNotFoundError: if path doesn't exist. Run `git clone
          https://github.com/snap-research/locomo third_party/locomo` first.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"LoCoMo data not found at {p}. "
            f"Clone with: git clone --depth 1 "
            f"https://github.com/snap-research/locomo third_party/locomo")

    raw = json.load(open(p, encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected top-level list in {p}, got {type(raw).__name__}")

    out: list[dict[str, Any]] = []
    for entry in raw:
        conv_id = entry["sample_id"]
        turns = _flatten_conversation(entry["conversation"])
        qa_items = []
        for i, qa in enumerate(entry.get("qa", [])):
            cat = qa.get("category")
            qa_items.append({
                "qa_id": f"{conv_id}::qa-{i}",
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "category": cat,
                "category_name": CATEGORY_NAMES.get(cat, f"cat_{cat}"),
                "evidence": _parse_evidence(qa.get("evidence")),
            })
        out.append({
            "id": conv_id,
            "personas": {
                "a": entry["conversation"].get("speaker_a", ""),
                "b": entry["conversation"].get("speaker_b", ""),
            },
            "turns": turns,
            "qa": qa_items,
        })
    return out


def category_distribution(data: list[dict[str, Any]]) -> dict[str, int]:
    """Return count of QA items per category across all conversations."""
    c = Counter()
    for entry in data:
        for qa in entry["qa"]:
            c[qa["category_name"]] += 1
    return dict(c)


def summary(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Quick dataset summary for sanity-check prints."""
    total_turns = sum(len(e["turns"]) for e in data)
    total_qa = sum(len(e["qa"]) for e in data)
    turns_per_conv = [len(e["turns"]) for e in data]
    qa_per_conv = [len(e["qa"]) for e in data]
    return {
        "n_conversations": len(data),
        "total_turns": total_turns,
        "total_qa": total_qa,
        "turns_per_conversation": {
            "min": min(turns_per_conv), "max": max(turns_per_conv),
            "mean": round(sum(turns_per_conv) / len(turns_per_conv), 1),
        },
        "qa_per_conversation": {
            "min": min(qa_per_conv), "max": max(qa_per_conv),
            "mean": round(sum(qa_per_conv) / len(qa_per_conv), 1),
        },
        "categories": category_distribution(data),
    }


if __name__ == "__main__":
    data = load_locomo()
    s = summary(data)
    print(f"Loaded {s['n_conversations']} conversations")
    print(f"  Total turns: {s['total_turns']}")
    print(f"  Total QA:    {s['total_qa']}")
    print(f"  Turns/conv:  {s['turns_per_conversation']}")
    print(f"  QA/conv:     {s['qa_per_conversation']}")
    print(f"  Categories:  {s['categories']}")
    print()
    print("Sample entry[0]:")
    e = data[0]
    print(f"  id: {e['id']}")
    print(f"  personas: {e['personas']}")
    print(f"  first turn: {e['turns'][0]}")
    print(f"  first qa:   {e['qa'][0]}")
