"""End-to-end BASELINE comparison (EverOS, A-MEM) under the STRICT alignment protocol.

Single-variable substitution vs the main table: for each sample, the baseline's memory is
seeded with the VERBATIM delegation utterance that ProsMem's encoder received (same probe
file, byte-identical text) — no implementation_intention, no action_description, no parsed
fields. Everything else (eval functions, agents, scoring, backbone) is inherited from the
main-table protocol unchanged.

Alignment is ENFORCED, not assumed — the runner aborts unless:
  * probe sha256 matches the ProsMem e2e run (78f3cc8992451abb...)
  * backbone == deepseek/deepseek-chat-v3-0324 (ProsMem e2e backbone)
  * every sample consumes exactly its own utterances (count-checked per sample)
  * EverOS only: a fresh memory root (never the main-table .everos_data) and a reachable
    server; the pipeline must start the server with EVEROS_LLM__MODEL=<same DSV3>.

Implementation note: the verbatim seeding is done by replacing the agents'
`register_intention` bound method INSIDE THIS PROCESS ONLY (class-level, catches every
call site: standard, D1, D2 dispatch). Shared code (evaluator.py / baselines.py /
everos_agent.py) is untouched, so the main table cannot be affected by construction.
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

sys.path.insert(0, str(ROOT))

from prosmem.bench.evaluator import get_eval_fn, EvalReport, SampleResult  # noqa: E402
from prosmem.core.config import ProsMemConfig  # noqa: E402

# --- Alignment constants (the ProsMem e2e run this comparison must match) ---
PROBE = "prosmem/bench/probes/e2e_intention_185_fullspec_v2.json"
PROBE_SHA16 = "78f3cc8992451abb"
BACKBONE = "deepseek/deepseek-chat-v3-0324"

# Per-sample utterance queue consumed by the patched registration methods.
_UTTS: list[str] = []


def _next_utt() -> str:
    if not _UTTS:
        raise RuntimeError("ALIGNMENT VIOLATION: more registrations than utterances")
    return _UTTS.pop(0)


def _patch_agents(agent_label: str) -> None:
    """Replace register_intention with verbatim-utterance seeding (process-local)."""
    if agent_label == "A-MEM":
        from prosmem.agent.baselines import AMemAgent

        def _verbatim_amem(self, description: str, trigger: str) -> None:
            try:
                self._mem.add_note(content=_next_utt())
            except RuntimeError:
                raise
            except Exception:
                pass  # same non-fatal semantics as the original
        AMemAgent.register_intention = _verbatim_amem
    elif agent_label == "EverOS":
        from prosmem.agent.everos_agent import EverOSAgent

        def _verbatim_everos(self, description: str, trigger: str) -> None:
            self._add([("user", _next_utt())])
        EverOSAgent.register_intention = _verbatim_everos
    else:
        raise SystemExit(f"unsupported agent: {agent_label}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["A-MEM", "EverOS"])
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--proxy", default="",
                    help="Optional HTTP(S) proxy URL for the LLM client")
    ap.add_argument("--samples", help="Comma-separated subset of sample IDs (smoke)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # --- HARD ALIGNMENT GATES ---
    sha16 = hashlib.sha256(Path(args.probe).read_bytes()).hexdigest()[:16]
    if sha16 != PROBE_SHA16:
        print(f"ABORT: probe sha {sha16} != ProsMem e2e probe {PROBE_SHA16}", file=sys.stderr)
        return 2
    config = ProsMemConfig.from_env()
    config.main_model = BACKBONE
    config.http_proxy = args.proxy
    if args.agent == "EverOS":
        root = os.environ.get("EVEROS_MEMORY__ROOT", "")
        if not root or root.rstrip("/\\").endswith(".everos_data"):
            print("ABORT: EverOS needs a FRESH EVEROS_MEMORY__ROOT (not the main-table "
                  ".everos_data) — start the server via the e2e pipeline", file=sys.stderr)
            return 2
        model = os.environ.get("EVEROS_LLM__MODEL", "")
        if "deepseek-chat-v3" not in model and "deepseek-v3" not in model:
            print(f"ABORT: EverOS server model '{model}' != DSV3 — backbone misaligned",
                  file=sys.stderr)
            return 2

    _patch_agents(args.agent)

    samples = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    if args.samples:
        keep = set(args.samples.split(","))
        samples = [s for s in samples if s["id"] in keep]
    missing = [s["id"] for s in samples if "encoder_utterance" not in s]
    if missing:
        print(f"ABORT: {len(missing)} samples lack encoder_utterance", file=sys.stderr)
        return 2
    print(f"Probe: {args.probe} (sha16={sha16})  Agent: {args.agent}  "
          f"Backbone: {BACKBONE}  proxy={args.proxy}  n={len(samples)}")

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
            ))
        print(f"Resumed {len(done)} rows")

    t0 = time.time()
    for i, s in enumerate(samples, 1):
        if s["id"] in done:
            print(f"[{i}/{len(samples)}] {s['id']} SKIP", flush=True)
            continue
        utts = s["encoder_utterance"]
        _UTTS[:] = utts if isinstance(utts, list) else [utts]
        n_utts = len(_UTTS)
        eval_fn = get_eval_fn(s.get("category", "A1"), args.agent)
        if eval_fn is None:
            print(f"[{i}/{len(samples)}] {s['id']} ABORT no eval fn", file=sys.stderr)
            return 2
        try:
            r = eval_fn(s, config)
            if _UTTS:
                # fewer registrations than utterances -> the seed was NOT fully aligned
                raise RuntimeError(
                    f"ALIGNMENT VIOLATION: {len(_UTTS)}/{n_utts} utterances unconsumed")
            report.add(r)
            done.add(s["id"])
            print(f"[{i}/{len(samples)}] {s['id']:<8} {'HIT ' if r.hit else 'MISS'} "
                  f"mention={r.response_mentions_action} exp={r.expected_trigger_turns} "
                  f"tok={r.tokens_used} {r.wall_time:.1f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(samples)}] {s['id']:<8} FINAL_ERROR {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
        checkpoint.write_text(
            json.dumps({"meta": {"in_progress": True},
                        "details": [serialize(r) for r in report.results]},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    summary = report.summary_by_agent()
    payload = {
        "meta": {"agent": args.agent, "backbone": BACKBONE,
                 "probe": args.probe, "probe_sha256_16": sha16,
                 "seed_protocol": "verbatim utterance (single-variable substitution vs "
                                  "main table; no parsed fields)",
                 "proxy": args.proxy,
                 "n_samples": len(samples), "wall_time_min": round(elapsed / 60, 1),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "summary": summary,
        "details": [serialize(r) for r in report.results],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults → {out_path}   ({elapsed/60:.1f} min)")
    for agent, s2 in summary.items():
        print(f"  {agent:<10} hit={s2['hit_rate']:.1%}  prec={s2['precision']:.1%}  "
              f"F1={s2['f1']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
