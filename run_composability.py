"""Composability runner: A-MEM+PM on ProsMem-Bench.

Runs the A-MEM+PM composition agent on ProsMem-Bench via the evaluator's
ProsMem scoring paths (agent_factory hook). Produces the A-MEM+PM row of the
composability experiment; the A-MEM-alone (~34%) and ProsMem-alone (~95%)
rows come from the main table. Bolting the ProsMem PM layer onto an
unmodified A-MEM host lifts prospective-memory F1 from ~34% to ~95%.

    python run_composability.py                       # full 185, DSV3
    python run_composability.py --samples A1-01,B1-01  # subset smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
_env = ROOT / ".env"
if _env.exists():
    for _ln in _env.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from prosmem.agent.amem_pm_agent import AMemPMAgent  # noqa: E402
from prosmem.bench.evaluator import (  # noqa: E402
    EvalReport, SampleResult, eval_prosmem, eval_d1_prosmem, eval_d2_prosmem, load_samples,
)
from prosmem.core.config import ProsMemConfig  # noqa: E402

BENCH_FILES = [
    "prosmem/bench/golden_samples.json",
]


def _amem_pm_factory(config, sample):
    return AMemPMAgent(config, collection_id=sample["id"])


def run_one(sample: dict, config: ProsMemConfig) -> SampleResult:
    cat = sample.get("category")
    if cat == "D1":
        return eval_d1_prosmem(sample, config, agent_factory=_amem_pm_factory)
    if cat == "D2":
        return eval_d2_prosmem(sample, config, agent_factory=_amem_pm_factory)
    return eval_prosmem(sample, config, agent_factory=_amem_pm_factory)


def load_combined() -> list[dict]:
    seen, out = set(), []
    for rel in BENCH_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        for s in load_samples(str(p)):
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="deepseek/deepseek-chat-v3-0324")
    ap.add_argument("--monitor-model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--samples", help="Comma-separated subset of sample IDs")
    ap.add_argument("--out", default="results/composability_amem_pm_deepseek.json")
    args = ap.parse_args()

    samples = load_combined()
    if args.samples:
        ids = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in ids]
    print(f"Loaded {len(samples)} samples")

    config = ProsMemConfig.from_env()
    if not config.llm_api_key:
        print("ERROR: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    config.main_model = args.backbone
    config.monitor_model = args.monitor_model
    print(f"Agent: A-MEM+PM   Backbone: {args.backbone}   Monitor: {args.monitor_model}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out_path.with_suffix(".partial.json")

    def serialize(r: SampleResult) -> dict:
        return {
            "sample_id": r.sample_id, "category": r.category, "agent": r.agent_name,
            "expected_turns": r.expected_trigger_turns,
            "actual_triggers": r.actual_trigger_turns,
            "response_mentions": r.response_mentions_action,
            "hit": r.hit, "precise": r.precise, "latency": r.latency,
            "tokens": r.tokens_used, "wall_time_s": round(r.wall_time, 2),
            "ongoing_task_accuracy": r.ongoing_task_accuracy,
            "completeness": r.completeness, "content_correct": r.content_correct,
            "intention_results": [
                {"intention_id": ir.intention_id, "trigger_turn": ir.trigger_turn,
                 "actual_trigger_turns": ir.actual_trigger_turns, "hit": ir.hit,
                 "precise": ir.precise, "latency": ir.latency}
                for ir in r.intention_results
            ],
        }

    report = EvalReport()
    done: set[str] = set()
    if checkpoint.exists():
        for r in json.loads(checkpoint.read_text(encoding="utf-8")).get("details", []):
            done.add(r["sample_id"])
            report.add(SampleResult(
                sample_id=r["sample_id"], category=r["category"], agent_name=r["agent"],
                expected_trigger_turns=r["expected_turns"],
                actual_trigger_turns=r["actual_triggers"],
                response_mentions_action=r["response_mentions"],
                hit=r["hit"], precise=r["precise"], latency=r["latency"],
                tokens_used=r["tokens"], wall_time=r["wall_time_s"],
                ongoing_task_accuracy=r.get("ongoing_task_accuracy"),
                completeness=r.get("completeness"), content_correct=r.get("content_correct"),
            ))
        print(f"Resumed {len(done)} rows from {checkpoint}")

    t0 = time.time()
    for i, s in enumerate(samples, 1):
        if s["id"] in done:
            print(f"[{i}/{len(samples)}] {s['id']} SKIP", flush=True)
            continue
        try:
            r = run_one(s, config)
            r.agent_name = "A-MEM+PM"  # eval_* hardcode "ProsMem"; relabel for this run
            report.add(r)
            done.add(s["id"])
            print(f"[{i}/{len(samples)}] {s['id']:<8} {'HIT ' if r.hit else 'MISS'} "
                  f"trig={r.actual_trigger_turns} exp={r.expected_trigger_turns} "
                  f"tok={r.tokens_used} {r.wall_time:.1f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(samples)}] {s['id']:<8} FINAL_ERROR {type(e).__name__}: {str(e)[:140]}",
                  flush=True)
        checkpoint.write_text(
            json.dumps({"meta": {"in_progress": True, "n": len(report.results)},
                        "details": [serialize(r) for r in report.results]},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    summary = report.summary_by_agent()
    payload = {
        "meta": {"agent": "A-MEM+PM", "backbone": args.backbone,
                 "monitor_model": args.monitor_model, "n_samples": len(samples),
                 "wall_time_min": round(elapsed / 60, 1),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "note": "Composability gain: A-MEM+PM vs the A-MEM(~34%)/ProsMem(~95%) main-table rows"},
        "summary": summary,
        "by_category": report.summary_by_category(next(iter(summary), "ProsMem")),
        "details": [serialize(r) for r in report.results],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults → {out_path}   ({elapsed/60:.1f} min)")
    for agent, s in summary.items():
        print(f"  {agent:<12} hit={s['hit_rate']:.1%}  prec={s['precision']:.1%}  "
              f"F1={s['f1']:.1%}  tok={s['total_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
