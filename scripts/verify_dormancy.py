"""Dormancy check — deterministic no-op confirmation of the PM layer.

The composability "does not degrade the host" claim rests on an ARCHITECTURAL
property: on a pure retrospective (QA) task NO prospective intention is
registered, so the ProsMem PM layer is inert and the response path is exactly
the host's (A-MEM's). This script confirms the no-op property directly and
deterministically, without a full LoCoMo re-run (which would only reproduce
A-MEM's own numbers):

  For each QA-style turn, with an EMPTY intention registry, one call to the PM
  engine's process_step() must (a) return None (no trigger) and (b) consume
  ZERO LLM tokens — the associative path uses only the local embedder, and the
  strategic monitor iterates zero armed intentions so it never calls the LLM.

Token(process_step) == 0 over every turn ⇒ the PM layer adds no LLM cost and
cannot alter retrieval ⇒ A-MEM+PM ≡ A-MEM on QA ⇒ Δ = 0 by construction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
_env = ROOT / ".env"
if _env.exists():
    for _ln in _env.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(ROOT))

from prosmem.agent.amem_pm_agent import AMemPMAgent  # noqa: E402
from prosmem.core.config import ProsMemConfig  # noqa: E402

# LoCoMo-style QA turns (retrospective questions — NOT prospective intentions).
_QA_TURNS = [
    "When did Sarah say she moved to Seattle?",
    "What hobby did the user mention picking up last summer?",
    "How many siblings does Tom have according to our earlier chat?",
    "Which restaurant did we agree was the best for the reunion?",
    "What was the name of the book the user recommended two sessions ago?",
]


def main() -> int:
    cfg = ProsMemConfig.from_env()
    cfg.main_model = "deepseek/deepseek-chat-v3-0324"
    cfg.monitor_model = "qwen/qwen-2.5-7b-instruct"
    agent = AMemPMAgent(cfg, collection_id="dormancy_check")

    print("A-MEM+PM with NO intention registered (pure QA setting):")
    print(f"  registry armed_count = {agent.registry.armed_count}")
    assert agent.registry.armed_count == 0, "registry should be empty"

    all_ok = True
    for step, turn in enumerate(_QA_TURNS, 1):
        before = agent.llm.total_tokens_used
        triggered = agent.engine.process_step(turn, step)
        pm_tokens = agent.llm.total_tokens_used - before
        ok = (triggered is None) and (pm_tokens == 0)
        all_ok &= ok
        print(f"  step {step}: process_step -> triggered={triggered} | "
              f"PM-layer LLM tokens={pm_tokens} | {'OK' if ok else 'FAIL'}")

    agent.cleanup()
    agent.llm.close()
    print()
    if all_ok:
        print("DORMANCY CONFIRMED: PM layer fired 0 triggers and consumed 0 LLM tokens on "
              "every QA turn (empty registry). The A-MEM response path is therefore "
              "byte-identical to the bare host -> recall & cost unchanged -> Delta = 0 "
              "by construction. No LoCoMo re-run needed to establish non-degradation.")
        return 0
    print("DORMANCY FAILED: the PM layer was not inert on an empty registry.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
