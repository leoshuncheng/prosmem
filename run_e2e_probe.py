"""End-to-end intention-encoding probe runner (ProsMem only, DSV3).

Reads the frozen derived probe (prosmem/bench/probes/e2e_intention_185_fullspec_v2.json) and
runs ProsMem where the intention is NOT supplied as a structured spec — instead
the IntentionEncoder reads the natural-language `encoder_utterance` (stage-1
automation) and auto-registers it, then the normal PM engine handles the
scenario. Scoring reuses the evaluator's ProsMem paths verbatim (register_fn hook),
so the number is directly comparable to the main-table ProsMem F1 (95.2%,
structured-registration upper bound). The gap = cost of stage-1 automation.

Usage:
    python run_e2e_probe.py                       # full probe, DSV3
    python run_e2e_probe.py --samples A1-01,B1-01  # subset
"""
from __future__ import annotations

import argparse
import hashlib
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

from prosmem.agent.encoder import IntentionEncoder  # noqa: E402
from prosmem.bench.evaluator import (  # noqa: E402
    EvalReport, SampleResult, eval_prosmem, eval_d1_prosmem, eval_d2_prosmem,
)
from prosmem.core.config import ProsMemConfig  # noqa: E402
from prosmem.retrieval.strategic import StrategicMonitor  # noqa: E402

PROBE_PATH = ROOT / "prosmem" / "bench" / "probes" / "e2e_intention_185_fullspec_v2.json"

# Stage-1 auto-registration (the IntentionEncoder) uses a STRONG, declared extraction
# model — independent of the agent backbone. Encoding is an e2e-only preprocessing step
# (the structured main table never calls the encoder), so a strong parser here does not
# touch any main-table result. Kept fixed across backbones for a clean, reproducible
# "auto-registration cost" measurement. Set via --encoder-model.
ENCODER_MODEL = "openai/gpt-4o"
ENCODER_EXPAND = False   # moderate non-focal cue expansion (set via --expand)


def _encode_with_retry(enc, agent, utterance, label: str) -> str | None:
    """encode_and_register with runner-level backoff. The encoder's own 3 retries do not
    outlast a provider degradation window (observed: minutes-long 400/429 bursts that
    silently zero whole categories); three more spaced attempts here ride it out."""
    for attempt in range(4):
        rid = enc.encode_and_register(agent, utterance)
        if rid:
            return rid
        if attempt < 3:
            time.sleep(15 * (attempt + 1))
    # Distinguish silent stage-1 failure (provider glitch / is_intention misfire)
    # from a genuine engine omission — a miss with this warning is a REGISTRATION
    # failure, not a retrieval failure.
    print(f"    WARN {label}: encoder registered NO intention", flush=True)
    return None


def _encoder_register_single(agent, sample) -> list[str]:
    """Register the single intention by ENCODING sample['encoder_utterance']."""
    enc = IntentionEncoder(agent.llm, model=ENCODER_MODEL, expand=ENCODER_EXPAND)
    rid = _encode_with_retry(enc, agent, sample["encoder_utterance"], sample.get("id", "?"))
    return [rid] if rid else []


def _encoder_register_d1(agent, sample) -> list[str]:
    """Register each D1 intention by encoding its parallel encoder_utterance.
    Returns rids IN ORDER (None placeholder if an encode fails) so the evaluator's
    position-based label mapping stays aligned."""
    enc = IntentionEncoder(agent.llm, model=ENCODER_MODEL, expand=ENCODER_EXPAND)
    return [
        _encode_with_retry(enc, agent, utt, f"{sample.get('id','?')}[{idx}]")
        for idx, utt in enumerate(sample["encoder_utterance"])
    ]


def run_one(sample: dict, config: ProsMemConfig) -> SampleResult:
    cat = sample.get("category")
    if cat == "D1":
        return eval_d1_prosmem(sample, config, register_fn=_encoder_register_d1)
    if cat == "D2":
        return eval_d2_prosmem(sample, config, register_fn=_encoder_register_single)
    return eval_prosmem(sample, config, register_fn=_encoder_register_single)


def main() -> int:
    global ENCODER_MODEL, ENCODER_EXPAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=str(PROBE_PATH))
    ap.add_argument("--backbone", default="deepseek/deepseek-chat-v3-0324")
    ap.add_argument("--monitor-model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--encoder-model", default=ENCODER_MODEL,
                    help="Strong stage-1 extraction model (e2e-only; does not affect main table)")
    ap.add_argument("--expand", action="store_true",
                    help="Moderate non-focal cue expansion (close paraphrases)")
    ap.add_argument("--fallback", choices=["on", "off"], default="on",
                    help="focal_fallback safety net: on = recover mis-inferred focality")
    ap.add_argument("--monitor-prompt", choices=["standard", "multihop"], default="standard",
                    help="EVENT non-focal monitor prompt: multihop = one-hop involve/lead-to reframe (e2e-only)")
    ap.add_argument("--proxy", default="",
                    help="Optional HTTP(S) proxy URL for the LLM client")
    ap.add_argument("--samples", help="Comma-separated subset of sample IDs")
    ap.add_argument("--out", default="results/e2e_probe_deepseek.json")
    args = ap.parse_args()
    ENCODER_MODEL = args.encoder_model
    ENCODER_EXPAND = args.expand

    samples = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    if args.samples:
        ids = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in ids]
    # Every sample must carry a frozen encoder_utterance.
    missing = [s["id"] for s in samples if "encoder_utterance" not in s]
    if missing:
        print(f"ERROR: {len(missing)} samples lack encoder_utterance: {missing[:5]}", file=sys.stderr)
        return 2
    print(f"Loaded {len(samples)} probe samples from {args.probe}")

    config = ProsMemConfig.from_env()
    if not config.llm_api_key:
        print("ERROR: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    config.main_model = args.backbone
    config.monitor_model = args.monitor_model
    # Engine is UNCHANGED from the main table. The ONLY e2e-specific machinery is stage-1
    # encoding (a strong extraction model parses the NL utterance -> structured fields).
    # Two disclosed knobs, both engine-config choices for the imperfect-inferred-focality
    # setting (default OFF main-table behaviour is preserved when not set):
    #   --fallback on/off : focal_fallback keeps focal-rated intentions monitor-eligible so a
    #                       mis-inferred focality is recovered rather than silently missed.
    #   --expand          : moderate close-paraphrase expansion of NON-FOCAL event cues.
    config.enable_focal_fallback = (args.fallback == "on")
    config.enable_monitor_multihop = (args.monitor_prompt == "multihop")
    config.http_proxy = args.proxy
    print(f"Backbone: {args.backbone}   Monitor: {args.monitor_model}   "
          f"Encoder: {ENCODER_MODEL}   focal_fallback={args.fallback.upper()}   "
          f"expand={'rich-cue-non-focal' if ENCODER_EXPAND else 'none'}   "
          f"monitor_prompt={args.monitor_prompt}")

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
            report.add(r)
            done.add(s["id"])
            print(f"[{i}/{len(samples)}] {s['id']:<8} {'HIT ' if r.hit else 'MISS'} "
                  f"trig={r.actual_trigger_turns} exp={r.expected_trigger_turns} "
                  f"tok={r.tokens_used} {r.wall_time:.1f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(samples)}] {s['id']:<8} FINAL_ERROR {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
        checkpoint.write_text(
            json.dumps({"meta": {"in_progress": True, "n": len(report.results)},
                        "details": [serialize(r) for r in report.results]},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    summary = report.summary_by_agent()
    # Reproducibility record: every knob that can change the number, plus a digest of the
    # frozen probe, so a result can be re-derived (or shown to be stale) without guesswork.
    try:
        probe_sha = hashlib.sha256(Path(args.probe).read_bytes()).hexdigest()[:16]
    except Exception:
        probe_sha = "unavailable"
    payload = {
        "meta": {"backbone": args.backbone, "monitor_model": args.monitor_model,
                 "encoder_model": ENCODER_MODEL,
                 "probe": args.probe, "probe_sha256_16": probe_sha,
                 "focal_fallback": args.fallback, "cue_expand": bool(ENCODER_EXPAND),
                 "monitor_prompt": args.monitor_prompt, "proxy": args.proxy,
                 "embedding_model": config.embedding_model,
                 "engine_defaults": {
                     "semantic_threshold": config.default_semantic_threshold,
                     "gating_threshold": StrategicMonitor.GATING_THRESHOLD,
                     "non_focal_max": StrategicMonitor.NON_FOCAL_MAX,
                     "max_checks_per_step": StrategicMonitor.MAX_CHECKS_PER_STEP},
                 "n_samples": len(samples), "wall_time_min": round(elapsed / 60, 1),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "note": "end-to-end encoder registration; compare vs structured ProsMem 95.2%"},
        "summary": summary,
        "by_category": report.summary_by_category("ProsMem"),
        "details": [serialize(r) for r in report.results],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults → {out_path}   ({elapsed/60:.1f} min)")
    for agent, s in summary.items():
        print(f"  {agent:<10} hit={s['hit_rate']:.1%}  prec={s['precision']:.1%}  "
              f"F1={s['f1']:.1%}  tok={s['total_tokens']}")
    print("\n(structured-registration upper bound: ProsMem F1 95.2% / hit 96.2%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
