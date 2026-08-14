"""Statistical significance for the main + ablation results (n=185).

Per-sample paired design: every system answers the identical 185 samples, so all
tests are paired by sample_id.

Outcomes per sample:
  hit      — fired/mentioned at (an) expected turn (Tier-1 hit)
  strict   — hit AND no spurious fire/mention (fires subset of expected turns);
             this is the per-sample binarization of the hit-rate x precision
             trade-off that drives the main-table F1 gap

Battery per comparison (ProsMem vs each of 9 baselines x 4 backbones, plus
ProsMem-Full vs 4 ablation variants on DeepSeek-V3):
  1. Exact McNemar on paired binary outcomes (hit, and strict)
  2. Cohen's h on the two proportions (hit, strict) + Cohen's d on paired
     per-sample strict differences
  3. Bootstrap 95% CI on the aggregate F1 gap (F1 = harmonic mean of hit rate
     and precision, mirroring evaluator.summary_by_agent)
  4. Wilcoxon signed-rank on per-category F1 (12 categories)

Expected inputs under results/ (produce them with the runners; pass a matching
--out where noted):
  main_185_{deepseek,llama,gpt4o,dsv4pro}.json          run_main_table.py --out ...
  {lightmem,graphiti,everos}_185_{stem}.json            run_main_table.py --agents ... --out ...
  ablation/ablation_185_deepseek.json                   run_ablation.py (default)

Output: results/SIGNIFICANCE_SUMMARY.json
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

BK_FILES = {"DeepSeek-V3-0324": "deepseek", "Llama-3.3-70B": "llama",
            "GPT-4o": "gpt4o", "DeepSeek-V4-Pro": "dsv4pro"}
NEW = {"LightMem": "lightmem", "Graphiti": "graphiti", "EverOS": "everos"}
BASE = ["Vanilla", "NaiveReminder", "Mem0", "A-MEM", "GenAgents", "MemoryBank",
        "LightMem", "Graphiti", "EverOS"]
VARIANTS = ["ProsMem-Assoc-Only", "ProsMem-Monitor-Only", "ProsMem-No-II",
            "ProsMem-Static"]

BOOT_N = 2000
BOOT_SEED = 42


# ---------------- outcome extraction (mirrors evaluator definitions) ----------------
def fires(r: dict) -> list:
    return r.get("actual_triggers") or r.get("response_mentions") or []


def strict_success(r: dict) -> bool:
    """Hit with zero spurious fires: fired/mentioned at expected turn(s) only."""
    return bool(r.get("hit")) and set(fires(r)) <= set(r.get("expected_turns") or [])


def agg_f1(rows: list[dict]) -> float:
    n = len(rows)
    hr = sum(1 for r in rows if r.get("hit")) / n if n else 0.0
    tot = sum(len(fires(r)) for r in rows)
    cor = sum(len(set(fires(r)) & set(r.get("expected_turns") or [])) for r in rows)
    pr = cor / tot if tot else 0.0
    return 2 * hr * pr / (hr + pr) if (hr + pr) else 0.0


# ---------------- test machinery ----------------
def mcnemar_test(a: list[int], b: list[int]) -> dict:
    """Exact two-sided McNemar on paired binary outcomes."""
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if not i and j)
    n = x + y
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(x, y)
    p = min(1.0, 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1)))
    return {"a_only": x, "b_only": y, "p": p,
            "log10_p": round(math.log10(p), 2) if p > 0 else float("-inf")}


def cohens_h(p1: float, p2: float) -> float:
    return round(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)), 3)


def cohens_d_paired(diffs: list[float]) -> float:
    if len(diffs) < 2:
        return 0.0
    s = statistics.stdev(diffs)
    return round(statistics.mean(diffs) / s, 3) if s > 0 else 0.0


def bootstrap_f1_gap_ci(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Paired bootstrap 95% CI on F1_a - F1_b, resampling sample_ids."""
    rng = random.Random(BOOT_SEED)
    n = len(rows_a)
    gaps = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(n) for _ in range(n)]
        gaps.append(agg_f1([rows_a[i] for i in idx]) - agg_f1([rows_b[i] for i in idx]))
    gaps.sort()
    return {"mean": round(statistics.mean(gaps), 4),
            "lo95": round(gaps[int(BOOT_N * 0.025)], 4),
            "hi95": round(gaps[int(BOOT_N * 0.975)], 4)}


def per_category_f1(rows: list[dict]) -> dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    return {c: agg_f1(v) for c, v in by.items()}


def compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Full battery on two agents' aligned row sets."""
    a_by = {r["sample_id"]: r for r in rows_a}
    b_by = {r["sample_id"]: r for r in rows_b}
    common = sorted(set(a_by) & set(b_by))
    ra = [a_by[s] for s in common]
    rb = [b_by[s] for s in common]
    ah = [int(bool(r.get("hit"))) for r in ra]
    bh = [int(bool(r.get("hit"))) for r in rb]
    asx = [int(strict_success(r)) for r in ra]
    bsx = [int(strict_success(r)) for r in rb]
    n = len(common)

    cat_a, cat_b = per_category_f1(ra), per_category_f1(rb)
    cats = sorted(set(cat_a) & set(cat_b))
    diffs = [cat_a[c] - cat_b[c] for c in cats]
    if any(d != 0 for d in diffs):
        _, wp = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided",
                         mode="exact")
        wp = float(wp)
    else:
        wp = 1.0

    return {
        "n": n,
        "hit_rate": {"a": round(sum(ah) / n, 4), "b": round(sum(bh) / n, 4)},
        "strict_rate": {"a": round(sum(asx) / n, 4), "b": round(sum(bsx) / n, 4)},
        "mcnemar_hit": mcnemar_test(ah, bh),
        "mcnemar_strict": mcnemar_test(asx, bsx),
        "cohens_h_hit": cohens_h(sum(ah) / n, sum(bh) / n),
        "cohens_h_strict": cohens_h(sum(asx) / n, sum(bsx) / n),
        "cohens_d_strict": cohens_d_paired([x - y for x, y in zip(asx, bsx)]),
        "wilcoxon_percat_f1_p": round(wp, 6),
        "f1_gap_boot95": bootstrap_f1_gap_ci(ra, rb),
    }


# ---------------- data loading ----------------
def load_rows(stem: str) -> list[dict]:
    rows = json.loads((RESULTS / f"main_185_{stem}.json").read_text(encoding="utf-8"))["details"]
    for fk in NEW.values():
        rows += json.loads((RESULTS / f"{fk}_185_{stem}.json").read_text(encoding="utf-8"))["details"]
    return rows


def main():
    out = {
        "experiment": "significance tests (paired, n=185 per comparison)",
        "generated_by": "scripts/stats.py",
        "definitions": {
            "hit": "Tier-1 hit (fired/mentioned at an expected turn)",
            "strict": "hit AND fires subset of expected turns (no spurious fire/mention)",
            "mcnemar": "exact two-sided binomial on discordant pairs",
            "cohens_h": "effect size on proportions; |h|: 0.2 small / 0.5 medium / 0.8 large",
            "f1_gap_boot95": "paired bootstrap (2000 resamples, seed 42) on aggregate F1 gap; "
                             "F1 mirrors evaluator.summary_by_agent",
            "wilcoxon_percat_f1_p": "signed-rank over 12 per-category F1 pairs (exact)",
        },
        "main_vs_prosmem": {},
        "ablation_dsv3_vs_full": {},
    }

    worst = {"mcnemar_strict_p": 0.0, "min_abs_h_strict": 99.0, "min_f1_gap_lo95": 99.0}
    for bk, stem in BK_FILES.items():
        rows = load_rows(stem)
        pm = [r for r in rows if r["agent"] == "ProsMem"]
        out["main_vs_prosmem"][bk] = {}
        for base in BASE:
            br = [r for r in rows if r["agent"] == base]
            c = compare(pm, br)
            out["main_vs_prosmem"][bk][base] = c
            worst["mcnemar_strict_p"] = max(worst["mcnemar_strict_p"], c["mcnemar_strict"]["p"])
            worst["min_abs_h_strict"] = min(worst["min_abs_h_strict"], abs(c["cohens_h_strict"]))
            worst["min_f1_gap_lo95"] = min(worst["min_f1_gap_lo95"], c["f1_gap_boot95"]["lo95"])
            print(f"{bk:18s} vs {base:14s} strict {c['strict_rate']['a']:.3f}/{c['strict_rate']['b']:.3f} "
                  f"McNemar p={c['mcnemar_strict']['p']:.2e} h={c['cohens_h_strict']:+.2f} "
                  f"F1gap95=[{c['f1_gap_boot95']['lo95']:+.3f},{c['f1_gap_boot95']['hi95']:+.3f}]")

    abl = json.loads((RESULTS / "ablation" / "ablation_185_deepseek.json")
                     .read_text(encoding="utf-8"))["details"]
    pm_dsv3 = [r for r in load_rows("deepseek") if r["agent"] == "ProsMem"]
    for v in VARIANTS:
        vr = [r for r in abl if r["agent"] == v]
        c = compare(pm_dsv3, vr)
        out["ablation_dsv3_vs_full"][v] = c
        print(f"{'DSV3 Full':18s} vs {v:22s} strict {c['strict_rate']['a']:.3f}/{c['strict_rate']['b']:.3f} "
              f"McNemar p={c['mcnemar_strict']['p']:.2e} h={c['cohens_h_strict']:+.2f}")

    out["headline"] = {
        "n_main_comparisons": 36,
        "max_mcnemar_strict_p_main": worst["mcnemar_strict_p"],
        "min_abs_cohens_h_strict_main": worst["min_abs_h_strict"],
        "min_f1_gap_lo95_main": worst["min_f1_gap_lo95"],
    }
    (RESULTS / "SIGNIFICANCE_SUMMARY.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nheadline:", json.dumps(out["headline"], indent=2))
    print("wrote", RESULTS / "SIGNIFICANCE_SUMMARY.json")


if __name__ == "__main__":
    main()
