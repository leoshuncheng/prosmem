"""Main-table runner: ProsMem-Bench samples × N agents × 1 backbone per invocation.

Usage:
    python run_main_table.py                                # DeepSeek V3, 3 agents, all 92
    python run_main_table.py --backbone openai/gpt-4o
    python run_main_table.py --samples A1-01,A2-04           # subset
    python run_main_table.py --out results/main_185_deepseek.json
"""
from __future__ import annotations

import argparse
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
    eval_amem, eval_genagents, eval_mem0, eval_memorybank,
    eval_naive, eval_prosmem, eval_vanilla, eval_lightmem, eval_everos, eval_graphiti,
    get_eval_fn, EvalReport, load_samples,
)
from prosmem.core.config import ProsMemConfig  # noqa: E402


def load_combined(bench_files: list[str]) -> list[dict]:
    seen = set()
    out = []
    for p in bench_files:
        for s in load_samples(p):
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=[
        "prosmem/bench/golden_samples.json",
    ])
    ap.add_argument("--backbone", default="deepseek/deepseek-chat-v3-0324",
                    help="Main LLM model id (provider catalog naming)")
    ap.add_argument("--monitor-model", default="qwen/qwen-2.5-7b-instruct",
                    help="StrategicMonitor slow-path model (small, YES/NO-calibrated)")
    ap.add_argument("--samples", help="Comma-separated subset of sample IDs")
    ap.add_argument("--agents", default="prosmem,vanilla,naive",
                    help="Subset of: prosmem,vanilla,naive,mem0,amem,genagents,memorybank")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (auto-named if omitted)")
    ap.add_argument("--proxy", default=os.environ.get("HTTP_PROXY", ""),
                    help="Optional HTTP(S) proxy URL for the LLM client")
    args = ap.parse_args()

    samples = load_combined(args.bench)
    if args.samples:
        ids = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in ids]
    print(f"Loaded {len(samples)} samples from {args.bench}")

    agent_names = [a.strip().lower() for a in args.agents.split(",")]

    config = ProsMemConfig.from_env()
    if not config.llm_api_key:
        print("ERROR: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    config.main_model = args.backbone
    config.monitor_model = args.monitor_model
    config.http_proxy = args.proxy

    # Backbone-sanity check so we fail fast before the long run
    print(f"Backbone: {args.backbone}   Monitor: {args.monitor_model}   "
          f"Proxy: {config.http_proxy or '(direct)'}")

    report = EvalReport()
    fns = {"prosmem": (eval_prosmem, "ProsMem"),
           "vanilla": (eval_vanilla, "Vanilla"),
           "naive": (eval_naive, "NaiveReminder"),
           "mem0": (eval_mem0, "Mem0"),
           "amem": (eval_amem, "A-MEM"),
           "genagents": (eval_genagents, "GenAgents"),
           "memorybank": (eval_memorybank, "MemoryBank"),
           "lightmem": (eval_lightmem, "LightMem"),
           "everos": (eval_everos, "EverOS"),
           "graphiti": (eval_graphiti, "Graphiti")}

    # Derive output path early so we can checkpoint
    if args.out is None:
        safe = args.backbone.replace("/", "_").replace(":", "_")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out = f"results/main_{safe}_{stamp}.json"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out.replace(".json", ".partial.json")

    def serialize_row(r) -> dict:
        """Single source of truth for row serialization (partial + final).
        Persists per-category fields: intention_results (D1),
        ongoing_task_accuracy / interference_score (D2), completeness (B2)."""
        row = {
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
        return row

    def dump_partial() -> None:
        raw = [serialize_row(r) for r in report.results]
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"in_progress": True, "completed_results": len(raw)},
                       "details": raw}, f, indent=2, ensure_ascii=False)

    # Resume: load previously-completed (sample_id, agent) rows from checkpoint if present
    done_keys: set[tuple[str, str]] = set()
    if Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                prev = json.load(f)
            for r in prev.get("details", []):
                done_keys.add((r["sample_id"], r["agent"]))
                # Rebuild in-memory report so final summary includes them
                from prosmem.bench.evaluator import SampleResult, IntentionResult
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

    def run_with_retry(fn, sample, cfg, max_tries: int = 3):
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                return fn(sample, cfg)
            except Exception as e:
                last_err = e
                backoff = min(30.0, 2.0 ** attempt)
                print(f"    retry {attempt}/{max_tries} after {type(e).__name__}: "
                      f"{str(e)[:100]} — sleep {backoff}s", flush=True)
                time.sleep(backoff)
        raise last_err

    t0 = time.time()
    for i, s in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {s['id']} ({s['category']}): {s['description'][:60]}",
              flush=True)
        for a in agent_names:
            if a not in fns:
                continue
            _default_fn, label = fns[a]
            # Route by category: D1 → multi-intention eval, D2 → interference eval,
            # everything else → standard single-intention eval. Static `fns` is used
            # only for the CLI-name → label mapping and as the standard default.
            fn = get_eval_fn(s["category"], label) or _default_fn
            if (s["id"], label) in done_keys:
                print(f"  {label:<14} SKIP (already done)", flush=True)
                continue
            try:
                r = run_with_retry(fn, s, config, max_tries=3)
                report.add(r)
                done_keys.add((s["id"], label))
                status = "HIT " if r.hit else "MISS"
                trig = r.actual_trigger_turns or r.response_mentions_action
                print(f"  {label:<14} {status}  triggers={trig}  "
                      f"tok={r.tokens_used}  {r.wall_time:.1f}s", flush=True)
            except Exception as e:
                print(f"  {label:<14} FINAL_ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        dump_partial()  # checkpoint after each sample

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed/60:.1f} min")

    # Final write (paths already computed above for checkpointing).
    # Reuse serialize_row so the per-category fields (intention_results / completeness /
    # ongoing_task_accuracy / interference_score) are preserved — partial already does.
    raw = [serialize_row(r) for r in report.results]
    out_payload = {
        "meta": {
            "backbone": args.backbone,
            "monitor_model": args.monitor_model,
            "proxy": bool(config.http_proxy),
            "n_samples": len(samples),
            "agents": agent_names,
            "wall_time_min": round(elapsed / 60, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": report.summary_by_agent(),
        "details": raw,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {args.out}")

    print("\n=== Summary ===")
    for agent, s in out_payload["summary"].items():
        print(f"  {agent:<14}  hit_rate={s['hit_rate']:.1%}  "
              f"prec={s['precision']:.1%}  F1={s['f1']:.1%}  "
              f"tok={s['total_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
