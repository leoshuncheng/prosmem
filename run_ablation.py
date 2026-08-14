"""Ablation runner: ProsMem-Bench samples × 4 ProsMem variants.

Variants (all-else-equal relative to Full):
- Assoc-Only   (enable_monitor=False)
- Monitor-Only (enable_associative=False)
- No-II        (enable_ii=False)
- Static       (enable_gating=False — monitor always on, no DMF gating)

Full (all enabled) is ALREADY available in main_deepseek_run1.json (ProsMem rows) —
do not re-run to save ~2.5h and preserve the same random stream.

Resume semantics match run_main_table.py: row-level checkpoint,
(sample_id, variant_label) as dedup key.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

_env = Path(__file__).parent / ".env"
if _env.exists():
    with open(_env, encoding="utf-8") as _f:
        for _ln in _f:
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from prosmem.bench.evaluator import (  # noqa: E402
    eval_prosmem, get_eval_fn, EvalReport, load_samples, SampleResult, IntentionResult,
)
from prosmem.core.config import ProsMemConfig  # noqa: E402


VARIANTS: dict[str, dict] = {
    "Assoc-Only":   {"enable_associative": True,  "enable_monitor": False},
    "Monitor-Only": {"enable_associative": False, "enable_monitor": True},
    "No-II":        {"enable_ii": False},
    "Static":       {"enable_gating": False},
}


def load_combined(bench_files: list[str]) -> list[dict]:
    seen, out = set(), []
    for p in bench_files:
        for s in load_samples(p):
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            out.append(s)
    return out


def build_config(base: ProsMemConfig, flags: dict) -> ProsMemConfig:
    cfg = copy.deepcopy(base)
    for k, v in flags.items():
        setattr(cfg, k, v)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=[
        "prosmem/bench/golden_samples.json",
    ])
    ap.add_argument("--backbone", default="deepseek/deepseek-chat-v3-0324")
    ap.add_argument("--monitor-model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--samples", help="Comma-separated subset of sample IDs")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help=f"Subset of: {','.join(VARIANTS)}")
    ap.add_argument("--out", default="results/ablation/ablation_185_deepseek.json")
    args = ap.parse_args()

    samples = load_combined(args.bench)
    if args.samples:
        ids = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in ids]
    print(f"Loaded {len(samples)} samples")

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip() in VARIANTS]
    print(f"Variants: {variant_names}")

    base_config = ProsMemConfig.from_env()
    if not base_config.llm_api_key:
        print("ERROR: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    base_config.main_model = args.backbone
    base_config.monitor_model = args.monitor_model
    print(f"Backbone: {args.backbone}   Monitor: {args.monitor_model}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out.replace(".json", ".partial.json")
    report = EvalReport()

    def serialize_row(r) -> dict:
        """Single source of truth (partial + final). Persists per-category
        fields: intention_results (D1), ongoing/interference (D2), completeness (B2)."""
        return {
            "sample_id": r.sample_id, "category": r.category, "agent": r.agent_name,
            "expected_turns": r.expected_trigger_turns,
            "actual_triggers": r.actual_trigger_turns,
            "response_mentions": r.response_mentions_action,
            "hit": r.hit, "precise": r.precise, "latency": r.latency,
            "tokens": r.tokens_used, "wall_time_s": round(r.wall_time, 2),
            "ongoing_task_accuracy": r.ongoing_task_accuracy,
            "interference_score": r.interference_score,
            "completeness": r.completeness,
            "content_correct": r.content_correct,
            "intention_results": [
                {
                    "intention_id": ir.intention_id,
                    "trigger_turn": ir.trigger_turn,
                    "actual_trigger_turns": ir.actual_trigger_turns,
                    "response_mentions_turns": ir.response_mentions_turns,
                    "hit": ir.hit, "precise": ir.precise, "latency": ir.latency,
                }
                for ir in r.intention_results
            ],
        }

    def dump_partial() -> None:
        raw = [serialize_row(r) for r in report.results]
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"in_progress": True, "completed_results": len(raw)},
                       "details": raw}, f, indent=2, ensure_ascii=False)

    done_keys: set[tuple[str, str]] = set()
    if Path(checkpoint_path).exists():
        try:
            prev = json.load(open(checkpoint_path, encoding="utf-8"))
            for r in prev.get("details", []):
                done_keys.add((r["sample_id"], r["agent"]))
                report.add(SampleResult(
                    sample_id=r["sample_id"], category=r["category"],
                    agent_name=r["agent"],
                    expected_trigger_turns=r["expected_turns"],
                    actual_trigger_turns=r["actual_triggers"],
                    response_mentions_action=r["response_mentions"],
                    hit=r["hit"], precise=r["precise"], latency=r["latency"],
                    tokens_used=r["tokens"], wall_time=r["wall_time_s"],
                    ongoing_task_accuracy=r.get("ongoing_task_accuracy"),
                    interference_score=r.get("interference_score"),
                    completeness=r.get("completeness"),
                    content_correct=r.get("content_correct"),
                    intention_results=[
                        IntentionResult(
                            intention_id=ir["intention_id"],
                            trigger_turn=ir["trigger_turn"],
                            actual_trigger_turns=ir["actual_trigger_turns"],
                            response_mentions_turns=ir["response_mentions_turns"],
                            hit=ir["hit"], precise=ir["precise"], latency=ir["latency"],
                        )
                        for ir in r.get("intention_results", [])
                    ],
                ))
            print(f"Resumed: {len(done_keys)} rows from {checkpoint_path}", flush=True)
        except Exception as e:
            print(f"Checkpoint load failed ({e}); starting fresh", flush=True)

    def run_with_retry(fn, sample, cfg, max_tries=3):
        last = None
        for attempt in range(1, max_tries + 1):
            try:
                return fn(sample, cfg)
            except Exception as e:
                last = e
                backoff = min(30.0, 2.0 ** attempt)
                print(f"    retry {attempt}/{max_tries} after {type(e).__name__}: "
                      f"{str(e)[:100]} — sleep {backoff}s", flush=True)
                time.sleep(backoff)
        raise last

    t0 = time.time()
    for i, s in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {s['id']} ({s['category']}): {s['description'][:60]}",
              flush=True)
        for variant in variant_names:
            label = f"ProsMem-{variant}"
            if (s["id"], label) in done_keys:
                print(f"  {label:<22} SKIP (done)", flush=True)
                continue
            cfg = build_config(base_config, VARIANTS[variant])
            # Route by category: D1 → multi-intention eval, D2 → interference eval,
            # else standard. All ablation variants are the ProsMem agent, so the
            # "ProsMem" dispatch key picks the right per-category function.
            fn = get_eval_fn(s["category"], "ProsMem") or eval_prosmem
            try:
                r = run_with_retry(fn, s, cfg, max_tries=3)
                r.agent_name = label
                report.add(r)
                done_keys.add((s["id"], label))
                status = "HIT " if r.hit else "MISS"
                trig = r.actual_trigger_turns or r.response_mentions_action
                print(f"  {label:<22} {status}  triggers={trig}  "
                      f"tok={r.tokens_used}  {r.wall_time:.1f}s", flush=True)
            except Exception as e:
                print(f"  {label:<22} FINAL_ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        dump_partial()

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed/60:.1f} min")

    raw = [serialize_row(r) for r in report.results]
    payload = {
        "meta": {
            "backbone": args.backbone,
            "monitor_model": args.monitor_model,
            "n_samples": len(samples),
            "variants": variant_names,
            "wall_time_min": round(elapsed / 60, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": report.summary_by_agent(),
        "details": raw,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {args.out}")
    print("\n=== Summary ===")
    for ag, s in payload["summary"].items():
        print(f"  {ag:<22}  hit={s['hit_rate']:.1%}  prec={s['precision']:.1%}  "
              f"F1={s['f1']:.1%}  tok={s['total_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
