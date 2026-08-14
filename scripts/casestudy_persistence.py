"""Cross-session persistence case: an intention survives a process restart via SQLite.

An intention registered in one process survives process death and fires in a SEPARATE later
process, with the SQLite-backed registry as the only bridge between the two.

Run the two sessions as two independent processes:

    python scripts/casestudy_persistence.py --session A --db /path/pm.sqlite
    python scripts/casestudy_persistence.py --session B --db /path/pm.sqlite

Or let the script orchestrate both as subprocesses and print the joined trace:

    python scripts/casestudy_persistence.py --demo
"""
from __future__ import annotations

import argparse
import os
import subprocess
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
from prosmem.core.config import ProsMemConfig  # noqa: E402
from prosmem.core.intention import TriggerType  # noqa: E402

DEFAULT_DB = ROOT / ".prosmem_casestudy.sqlite"

# The delegated intention (session A) and the scenario turns.
_ACTION = "remind the user to buy cold medicine"
_SESSION_A_TURNS = [
    "Let me organize my afternoon. First I'll clear my inbox.",
    "Then I have a quick sync with the design team.",
]
_SESSION_B_TURNS = [
    "I need to stop by the pharmacy for a few things.",
]


def _config() -> ProsMemConfig:
    cfg = ProsMemConfig.from_env()
    cfg.main_model = "deepseek/deepseek-chat-v3-0324"
    cfg.monitor_model = "qwen/qwen-2.5-7b-instruct"
    return cfg


def session_a(db: str) -> int:
    """Register the intention, run some unrelated turns (no trigger), persist, exit."""
    print(f"[SESSION A pid {os.getpid()}] open registry @ {db}")
    agent = ProsMemAgent(_config(), db_path=db)
    iid = agent.register_intention(
        trigger_type=TriggerType.EVENT_BASED,
        action_description=_ACTION,
        event_cues=["pharmacy", "drugstore", "chemist"],
        focality=0.9,
        semantic_threshold=0.75,
        implementation_intention="IF the user mentions going to a pharmacy THEN remind them to buy cold medicine",
    )
    print(f"[SESSION A] registered intention {iid} (status={agent.registry.get(iid).status.value})")
    for t in _SESSION_A_TURNS:
        r = agent.step(t)
        fired = r.pm_triggered is not None
        print(f"[SESSION A] step {r.step}: {'FIRED' if fired else 'no-fire'}  [{t[:55]}]")
    print(f"[SESSION A] persisted summary: {agent.registry.summary()}")
    agent.registry.close()
    print("[SESSION A] process ending; the intention now lives only in SQLite.\n")
    return 0


def session_b(db: str) -> int:
    """Fresh process: reload from SQLite, run the trigger turn, expect a firing."""
    print(f"[SESSION B pid {os.getpid()}] open registry @ {db}")
    agent = ProsMemAgent(_config(), db_path=db)
    reloaded = agent.registry.all()
    print(f"[SESSION B] reloaded {len(reloaded)} intention(s) from previous session: "
          f"{[(i.intention_id, i.status.value) for i in reloaded]}")
    fired_any = False
    for t in _SESSION_B_TURNS:
        r = agent.step(t)
        if r.pm_triggered is not None:
            fired_any = True
            print(f"[SESSION B] step {r.step}: *** PM FIRED *** -> {r.pm_triggered.action_description!r}")
            print(f"[SESSION B]   action output (full): {r.pm_action_response!r}")
            print(f"[SESSION B]   agent reply to user (full): {r.agent_response!r}")
        else:
            print(f"[SESSION B] step {r.step}: no-fire  [{t[:55]}]")
    agent.registry.close()
    print(f"\n[SESSION B] RESULT: {'SUCCESS, cross-session PM fired' if fired_any else 'FAIL, did not fire'}")
    return 0 if fired_any else 1


def demo() -> int:
    """Fresh DB, then run A and B as separate subprocesses and print the joined trace."""
    db = str(DEFAULT_DB)
    if os.path.exists(db):
        os.remove(db)
    print("=" * 70)
    print("Cross-session persistence: one intention across two separate processes")
    print("=" * 70)
    py = sys.executable
    ra = subprocess.run([py, __file__, "--session", "A", "--db", db], cwd=str(ROOT))
    if ra.returncode != 0:
        print("session A failed")
        return ra.returncode
    rb = subprocess.run([py, __file__, "--session", "B", "--db", db], cwd=str(ROOT))
    print("=" * 70)
    print(f"Verdict: {'PASS' if rb.returncode == 0 else 'FAIL'} "
          "(intention registered in process A fired in process B via SQLite only)")
    print("=" * 70)
    if os.path.exists(db):
        os.remove(db)
    return rb.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=("A", "B"))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo or args.session is None:
        return demo()
    return session_a(args.db) if args.session == "A" else session_b(args.db)


if __name__ == "__main__":
    sys.exit(main())
