"""Per-turn replay of the PM engine on individual ProsMem-Bench samples (case-study traces).

Runs the engine in its default (Full) configuration with the real strategic monitor and no
main-LLM calls, and logs per turn: the associative cue similarity (max over cues), the monitor
gate similarity, the monitor's verbatim verdicts, which pathway proposed the intention, and
whether it fired.

Usage (from the repository root):
    python scripts/replay_case.py A2-04 A1-02 B1-06
Output: results/replay_<id>.json (one file per sample).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_env = ROOT / ".env"
if _env.exists():
    for _ln in _env.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(ROOT))

from prosmem.agent.loop import ProsMemAgent  # noqa: E402
from prosmem.bench.evaluator import _register_intention_from_cfg  # noqa: E402
from prosmem.core.config import ProsMemConfig  # noqa: E402
from prosmem.retrieval.embedder import cosine_similarity, embed_text, embed_texts  # noqa: E402

BENCH = ROOT / "prosmem" / "bench" / "golden_samples.json"
OUT = ROOT / "results"


def replay(sample: dict) -> dict:
    it = sample["intention"]
    cfg = ProsMemConfig.from_env()
    agent = ProsMemAgent(cfg)
    _register_intention_from_cfg(agent, it)
    eng = agent.engine
    mon = eng.monitor
    cur: dict = {"asks": []}

    # Instrument the monitor's LLM call and each pathway's proposals (behavior unchanged).
    orig_ask = mon._ask

    def ask(prompt, _orig=orig_ask):
        r = _orig(prompt)
        cur["asks"].append(r.strip())
        return r

    orig_assoc = eng.associative.retrieve

    def assoc(text, step, _o=orig_assoc):
        out = _o(text, step)
        cur["assoc"] = [i.intention_id for i in out]
        return out

    orig_mon = mon.check

    def moncheck(text, step, _o=orig_mon):
        out = _o(text, step)
        cur["monitor"] = [i.intention_id for i in out]
        return out

    orig_time = eng.time_checker.check

    def timecheck(step, _o=orig_time):
        out = _o(step)
        cur["timer"] = [i.intention_id for i in out]
        return out

    mon._ask = ask
    eng.associative.retrieve = assoc
    mon.check = moncheck
    eng.time_checker.check = timecheck

    cue_e = embed_texts(it["event_cues"]) if it.get("event_cues") else None
    reg = next(iter(agent.registry.get_armed()))
    rows = []
    for step, turn in enumerate(sample["turns"], 1):
        cur.clear()
        cur["asks"] = []
        te = embed_text(turn)
        cue_max = max(cosine_similarity(te, c) for c in cue_e) if cue_e else None
        gate = cosine_similarity(te, reg.trigger_embedding)
        fired = eng.process_step(turn, step)
        rows.append({
            "turn": step, "text": turn,
            "cue_max_sim": round(cue_max, 4) if cue_max is not None else None,
            "gate_sim": round(gate, 4),
            "monitor_verdicts": list(cur["asks"]),
            "assoc_proposed": bool(cur.get("assoc")), "monitor_proposed": bool(cur.get("monitor")),
            "timer_proposed": bool(cur.get("timer")),
            "fired": bool(fired),
        })
        if fired:
            fired.complete(step)
    return {"sample_id": sample["id"], "trigger_turn": sample["trigger_turn"],
            "threshold": it.get("semantic_threshold"), "gate_threshold": mon.GATING_THRESHOLD,
            "trigger_type": it["trigger_type"], "focality": it.get("focality"),
            "target_step": it.get("target_step"),
            "monitor_model": cfg.monitor_model, "rows": rows}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    bench = {s["id"]: s for s in json.loads(BENCH.read_text(encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in sys.argv[1:]:
        rec = replay(bench[sid])
        (OUT / f"replay_{sid}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
        print(f"== {sid} (T={rec['trigger_turn']}, theta={rec['threshold']}, "
              f"gate={rec['gate_threshold']})")
        for r in rec["rows"]:
            cm = f"{r['cue_max_sim']:.3f}" if r["cue_max_sim"] is not None else "n/a"
            print(f"  T{r['turn']}: cue_max={cm} gate={r['gate_sim']:.3f} "
                  f"verdicts={r['monitor_verdicts']} assoc={r['assoc_proposed']} "
                  f"monitor={r['monitor_proposed']} timer={r['timer_proposed']} fired={r['fired']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
