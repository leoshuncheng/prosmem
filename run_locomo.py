"""LoCoMo QA harness (the dormancy verification setting in the paper).

Runs the selected agents over LoCoMo conversations; scope is controlled by
--n-convs (size-stratified conversation pick) and --n-qa (QA items per
conversation, answerable categories only).

Evaluation metrics follow LoCoMo official task_eval/evaluation.py:
  - normalize_answer (lowercase, drop punct/articles, collapse whitespace)
  - stemmed token-F1 (PorterStemmer)
  - multi-answer split on comma → mean(max F1)

Usage:
  PYTHONIOENCODING=utf-8 python run_locomo.py --n-convs 1 --n-qa 20
  PYTHONIOENCODING=utf-8 python run_locomo.py --agents vanilla,prosmem --n-convs 10 --n-qa 500
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_env = Path(__file__).parent / ".env"
if _env.exists():
    with open(_env, encoding="utf-8") as _f:
        for _ln in _f:
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np  # noqa: E402
from nltk.stem import PorterStemmer  # noqa: E402

from prosmem.agent.baselines import _mem0_config  # noqa: E402
from prosmem.agent.llm import LLMClient  # noqa: E402
from prosmem.bench.locomo import load_locomo  # noqa: E402
from prosmem.core.config import ProsMemConfig  # noqa: E402
from prosmem.retrieval.embedder import embed_text, embed_texts  # noqa: E402

PS = PorterStemmer()

# =====================================================================
# LoCoMo-compatible eval metrics (reimplemented from task_eval/evaluation.py)
# =====================================================================
def _normalize(s: str) -> str:
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def f1_token(prediction, ground_truth) -> float:
    # Coerce to str — LoCoMo bench has some int gold answers (years like 2022,
    # counts like 2). Without this, .split() raises AttributeError.
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    pred_tok = [PS.stem(w) for w in _normalize(prediction).split()]
    gt_tok = [PS.stem(w) for w in _normalize(ground_truth).split()]
    common = Counter(pred_tok) & Counter(gt_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tok)
    rec = num_same / len(gt_tok)
    return 2 * prec * rec / (prec + rec)


def f1_multianswer(prediction, ground_truth) -> float:
    """LoCoMo's multi-answer F1: split on comma, take mean(max)."""
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    preds = [p.strip() for p in prediction.split(",") if p.strip()]
    gts = [g.strip() for g in ground_truth.split(",") if g.strip()]
    if not preds or not gts:
        return 0.0
    return float(np.mean([
        max(f1_token(p, gt) for p in preds) for gt in gts
    ]))


# =====================================================================
# Agent implementations (LoCoMo mode, NOT PM-trigger mode)
# =====================================================================
QA_PROMPT_TEMPLATE = (
    "You are answering a question about a conversation between {a} and {b}.\n"
    "Below are turns from their conversation (ordered chronologically):\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer as concisely as possible (a phrase, date, or short sentence). "
    "If the question is unanswerable from the conversation, reply 'no information available'.\n"
    "Answer:"
)


def format_turns(turns: list[dict]) -> str:
    """Format a list of turn dicts into a readable conversation excerpt."""
    return "\n".join(
        f"[{t['date']}] {t['speaker']}: {t['text']}" for t in turns
    )


def agent_vanilla_truncate(conv: dict, qa: dict, config: ProsMemConfig,
                           window: int = 50) -> dict:
    """Vanilla baseline: pass last `window` turns as context."""
    client = LLMClient(config)
    ctx_turns = conv["turns"][-window:]
    prompt = QA_PROMPT_TEMPLATE.format(
        a=conv["personas"]["a"], b=conv["personas"]["b"],
        context=format_turns(ctx_turns), question=qa["question"])
    t0 = time.time()
    ans = client.chat(
        [{"role": "user", "content": prompt}],
        model=config.main_model, temperature=0.3, max_tokens=128)
    return {
        "prediction": ans.strip(),
        "tokens": client.total_tokens_used,
        "wall_time_s": round(time.time() - t0, 2),
        "context_turns": len(ctx_turns),
    }


# Caching embedder + indices across QAs within one conversation
_EMBED_CACHE: dict[str, tuple] = {}


def _get_or_build_index(conv: dict):
    """Precompute turn embeddings for fast QA retrieval."""
    key = conv["id"]
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    texts = [t["text"] for t in conv["turns"]]
    embs = np.array(embed_texts(texts))  # shape (N, 384)
    _EMBED_CACHE[key] = (texts, embs)
    return texts, embs


def agent_prosmem_retrieve(conv: dict, qa: dict, config: ProsMemConfig,
                          top_k: int = 20) -> dict:
    """ProsMem-style retrieval: embed all turns, pick top-K most similar to Q.
    This exercises ProsMem's associative-path primitive (embedding cosine)."""
    client = LLMClient(config)
    texts, embs = _get_or_build_index(conv)
    q_emb = embed_text(qa["question"])
    # Cosine sim (embeddings are L2-normalized by bge-small default)
    sims = np.dot(embs, q_emb)
    top_idx = sorted(np.argsort(sims)[-top_k:].tolist())  # chronological order
    ctx_turns = [conv["turns"][i] for i in top_idx]
    prompt = QA_PROMPT_TEMPLATE.format(
        a=conv["personas"]["a"], b=conv["personas"]["b"],
        context=format_turns(ctx_turns), question=qa["question"])
    t0 = time.time()
    ans = client.chat(
        [{"role": "user", "content": prompt}],
        model=config.main_model, temperature=0.3, max_tokens=128)
    return {
        "prediction": ans.strip(),
        "tokens": client.total_tokens_used,
        "wall_time_s": round(time.time() - t0, 2),
        "context_turns": len(ctx_turns),
        "retrieved_idx": top_idx,
    }


def agent_vanilla_fullctx(conv: dict, qa: dict, config: ProsMemConfig) -> dict:
    """Upper-bound baseline: pass the ENTIRE conversation as context.
    Validates that ProsMem's retrieval strategy is justified vs brute-force long context."""
    client = LLMClient(config)
    prompt = QA_PROMPT_TEMPLATE.format(
        a=conv["personas"]["a"], b=conv["personas"]["b"],
        context=format_turns(conv["turns"]), question=qa["question"])
    t0 = time.time()
    ans = client.chat(
        [{"role": "user", "content": prompt}],
        model=config.main_model, temperature=0.3, max_tokens=128)
    return {
        "prediction": ans.strip(),
        "tokens": client.total_tokens_used,
        "wall_time_s": round(time.time() - t0, 2),
        "context_turns": len(conv["turns"]),
    }


# Per-conversation memory-store caches (built on first QA of each conv, reused across QAs)
_MEM0_STORES: dict[str, tuple] = {}
_AMEM_STORES: dict[str, Any] = {}


def _group_by_session(turns: list[dict]) -> list[tuple[int, str, str]]:
    """Collapse flat turn list into session-level chunks: (session_num, date, joined_text)."""
    from collections import defaultdict
    by_sess: dict[int, list] = defaultdict(list)
    for t in turns:
        by_sess[t["session"]].append(t)
    out = []
    for sess_num in sorted(by_sess):
        sess_turns = by_sess[sess_num]
        date = sess_turns[0]["date"]
        text = "\n".join(f"{t['speaker']}: {t['text']}" for t in sess_turns)
        out.append((sess_num, date, text))
    return out


def _build_mem0(conv: dict, config: ProsMemConfig):
    """Build + ingest Mem0 store for one conversation. Session-granular ingestion
    keeps ingest cost tractable: ~24 sessions vs ~600 turns per conv."""
    from mem0 import Memory
    collection_id = f"locomo_{conv['id']}"
    mem = Memory.from_config(_mem0_config(config, collection_id))
    user_id = f"locomo_{conv['id']}"
    sessions = _group_by_session(conv["turns"])
    print(f"    Mem0 ingesting {len(sessions)} sessions for {conv['id']}...",
          flush=True)
    n_ok = 0
    for sess_num, date, text in sessions:
        try:
            # infer=False skips Mem0's LLM fact-extraction and stores the raw
            # session chunk verbatim. On session-level 200-500tok chunks the
            # extractor often returned {} → empty store → F1=0 (observed).
            mem.add(f"[Session {sess_num}, {date}]\n{text}",
                    user_id=user_id, infer=False)
            n_ok += 1
        except Exception as e:
            print(f"      ingest session {sess_num} failed: {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
    # Post-ingest sanity count
    try:
        all_mem = mem.get_all(filters={"user_id": user_id})
        items = all_mem.get("results", []) if isinstance(all_mem, dict) else all_mem
        print(f"    Mem0 ingest done: {n_ok}/{len(sessions)} adds OK, "
              f"store size={len(items)}", flush=True)
    except Exception as e:
        print(f"    Mem0 post-ingest count failed: {type(e).__name__}: {e}",
              flush=True)
    return mem, user_id


def agent_mem0_qa(conv: dict, qa: dict, config: ProsMemConfig, top_k: int = 5) -> dict:
    """Mem0 baseline in QA mode: session-level ingest, search top-K at QA time."""
    if conv["id"] not in _MEM0_STORES:
        _MEM0_STORES[conv["id"]] = _build_mem0(conv, config)
    mem, user_id = _MEM0_STORES[conv["id"]]

    client = LLMClient(config)
    t0 = time.time()
    try:
        # Match Mem0Agent.search signature (top_k + filters dict)
        retrieved = mem.search(query=qa["question"], top_k=top_k,
                               filters={"user_id": user_id})
        memories = retrieved.get("results", []) if isinstance(retrieved, dict) else retrieved
    except Exception as e:
        print(f"    Mem0 search failed: {type(e).__name__}: {str(e)[:120]}",
              flush=True)
        memories = []
    ctx_blocks = []
    for m in memories:
        txt = m.get("memory") if isinstance(m, dict) else str(m)
        if txt:
            ctx_blocks.append(txt)
    context = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "(no memories retrieved)"
    prompt = QA_PROMPT_TEMPLATE.format(
        a=conv["personas"]["a"], b=conv["personas"]["b"],
        context=context, question=qa["question"])
    ans = client.chat(
        [{"role": "user", "content": prompt}],
        model=config.main_model, temperature=0.3, max_tokens=128)
    return {
        "prediction": ans.strip(),
        "tokens": client.total_tokens_used,
        "wall_time_s": round(time.time() - t0, 2),
        "context_turns": len(ctx_blocks),
    }


def _build_amem(conv: dict, config: ProsMemConfig):
    """Build + ingest A-MEM for one conversation at session granularity."""
    os.environ["OPENAI_API_KEY"] = config.llm_api_key
    os.environ["OPENAI_BASE_URL"] = config.llm_base_url
    from agentic_memory.memory_system import AgenticMemorySystem
    mem = AgenticMemorySystem(
        model_name="all-MiniLM-L6-v2",
        llm_backend="openai",
        llm_model=config.main_model,
        api_key=config.llm_api_key,
        evo_threshold=100,  # suppress consolidation within per-conv store
    )
    sessions = _group_by_session(conv["turns"])
    print(f"    A-MEM ingesting {len(sessions)} sessions for {conv['id']}...",
          flush=True)
    for sess_num, date, text in sessions:
        try:
            mem.add_note(content=f"[Session {sess_num}, {date}]\n{text}")
        except Exception as e:
            print(f"      add_note session {sess_num} failed: {type(e).__name__}",
                  flush=True)
    return mem


def agent_amem_qa(conv: dict, qa: dict, config: ProsMemConfig, top_k: int = 5) -> dict:
    """A-MEM baseline in QA mode: session-level note ingest + agentic search."""
    if conv["id"] not in _AMEM_STORES:
        _AMEM_STORES[conv["id"]] = _build_amem(conv, config)
    mem = _AMEM_STORES[conv["id"]]

    client = LLMClient(config)
    t0 = time.time()
    try:
        retrieved = mem.search_agentic(query=qa["question"], k=top_k)
    except Exception:
        retrieved = []
    ctx_blocks = []
    for m in retrieved:
        c = m.get("content", "") if isinstance(m, dict) else str(m)
        if c:
            ctx_blocks.append(c)
    context = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "(no memories retrieved)"
    prompt = QA_PROMPT_TEMPLATE.format(
        a=conv["personas"]["a"], b=conv["personas"]["b"],
        context=context, question=qa["question"])
    ans = client.chat(
        [{"role": "user", "content": prompt}],
        model=config.main_model, temperature=0.3, max_tokens=128)
    return {
        "prediction": ans.strip(),
        "tokens": client.total_tokens_used,
        "wall_time_s": round(time.time() - t0, 2),
        "context_turns": len(ctx_blocks),
    }


AGENTS = {
    "vanilla": ("Vanilla_Truncate", agent_vanilla_truncate),
    "vanillafull": ("Vanilla_FullContext", agent_vanilla_fullctx),
    "prosmem": ("ProsMem_Retrieve", agent_prosmem_retrieve),
    "mem0": ("Mem0_QA", agent_mem0_qa),
    "amem": ("AMEM_QA", agent_amem_qa),
}


# =====================================================================
# Runner
# =====================================================================
def sample_qa(conv: dict, n: int = 20) -> list[dict]:
    """Pick `n` QAs with balanced category representation, skipping
    adversarial (empty answers, needs separate eval logic)."""
    by_cat: dict[str, list] = defaultdict(list)
    for qa in conv["qa"]:
        if qa["category_name"] == "adversarial":
            continue
        by_cat[qa["category_name"]].append(qa)
    # Round-robin pick
    out = []
    iters = {c: iter(lst) for c, lst in by_cat.items()}
    while len(out) < n and iters:
        for c in list(iters):
            try:
                out.append(next(iters[c]))
                if len(out) >= n:
                    break
            except StopIteration:
                del iters[c]
    return out


def pick_smallest_conv(data: list[dict]) -> dict:
    """Pick the conversation with the fewest turns (cheapest validation)."""
    return min(data, key=lambda e: len(e["turns"]))


def pick_size_sample(data: list[dict], n: int) -> list[dict]:
    """Pick `n` convs spanning the size distribution (smallest, median..., largest)."""
    sorted_data = sorted(data, key=lambda e: len(e["turns"]))
    if n >= len(sorted_data):
        return sorted_data
    if n == 1:
        return [sorted_data[0]]
    # Evenly-spaced quantile picks
    idxs = [round(i * (len(sorted_data) - 1) / (n - 1)) for i in range(n)]
    return [sorted_data[i] for i in idxs]


def run(conv: dict, qas: list[dict], agents: list[str],
        config: ProsMemConfig,
        checkpoint_path: str | None = None,
        done_keys: set | None = None,
        prior_rows: list[dict] | None = None) -> list[dict]:
    """Run agents on a conversation's QAs. Writes incremental checkpoint after
    each row (cumulative: prior_rows + new rows) if `checkpoint_path` provided.
    `done_keys` skips already-completed (qa_id, agent_label) pairs for resume.
    `prior_rows` is the set of rows from earlier convs/sessions; passing this
    keeps each flush cumulative so a mid-conv crash doesn't erase prior convs."""
    done_keys = done_keys or set()
    prior_rows = prior_rows or []
    rows = []
    for a in agents:
        label, fn = AGENTS[a]
        print(f"\n=== Running {label} on {conv['id']} ({len(qas)} QAs) ===",
              flush=True)
        for i, qa in enumerate(qas, 1):
            if (qa["qa_id"], label) in done_keys:
                print(f"  [{i:>3}/{len(qas)}] SKIP (done) {qa['qa_id']}/{label}",
                      flush=True)
                continue
            try:
                res = fn(conv, qa, config)
                score = f1_multianswer(res["prediction"], qa["answer"])
                row = {
                    "conv_id": conv["id"], "qa_id": qa["qa_id"],
                    "category": qa["category_name"], "agent": label,
                    "question": qa["question"], "gold": qa["answer"],
                    "prediction": res["prediction"], "f1": round(score, 4),
                    "tokens": res["tokens"],
                    "wall_time_s": res["wall_time_s"],
                    "context_turns": res["context_turns"],
                }
                rows.append(row)
                gold_str = str(qa['answer'])[:40]
                pred_str = str(res['prediction'])[:40]
                print(f"  [{i:>3}/{len(qas)}] {qa['category_name']:<12} "
                      f"F1={score:.3f}  tok={res['tokens']:>5}  "
                      f"{res['wall_time_s']:.1f}s  | gold={gold_str}  "
                      f"| pred={pred_str}", flush=True)
            except Exception as e:
                print(f"  [{i:>3}/{len(qas)}] ERROR {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
                rows.append({
                    "conv_id": conv["id"], "qa_id": qa["qa_id"],
                    "category": qa["category_name"], "agent": label,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                })
            # Cumulative checkpoint flush: prior convs' rows + this conv's progress
            if checkpoint_path:
                _dump_checkpoint(checkpoint_path, prior_rows + rows)
    return rows


def _dump_checkpoint(path: str, rows: list[dict]) -> None:
    """Write per-row checkpoint. Caller is responsible for passing accumulated rows.
    Includes retry on Windows PermissionError (race with concurrent readers
    holding short-lived file locks)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": {"in_progress": True,
                            "n_rows": len(rows),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                   "details": rows}, f, indent=2, ensure_ascii=False)
    # Windows can fail os.replace with PermissionError (concurrent reader
    # holding lock) OR FileNotFoundError (.tmp inexplicably gone between
    # retries). Catch both, retry, and never crash the run on a checkpoint
    # failure — the next flush will catch up.
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError, OSError) as e:
            if attempt == 5:
                print(f"    [checkpoint] WARN: replace failed after 6 retries "
                      f"({type(e).__name__}); skipping this flush", flush=True)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return
            # If .tmp got lost mid-retry, recreate it before retrying
            if not os.path.exists(tmp):
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"meta": {"in_progress": True,
                                        "n_rows": len(rows),
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                               "details": rows}, f, indent=2, ensure_ascii=False)
            time.sleep(0.2 * (2 ** attempt))


def _load_checkpoint(path: str) -> tuple[list[dict], set]:
    """Return (previously-completed rows, set of (qa_id, agent) keys)."""
    if not Path(path).exists():
        return [], set()
    try:
        d = json.load(open(path, encoding="utf-8"))
        rows = d.get("details", [])
        keys = {(r["qa_id"], r["agent"]) for r in rows if "qa_id" in r and "agent" in r}
        return rows, keys
    except Exception:
        return [], set()


def summarize(rows: list[dict]) -> dict:
    """Aggregate F1 by (agent, category)."""
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if "f1" in r:
            by[(r["agent"], r["category"])].append(r["f1"])
    out = {}
    agents = sorted({r["agent"] for r in rows if "agent" in r})
    for ag in agents:
        cats = {c for (a, c) in by if a == ag}
        per_cat = {c: round(float(np.mean(by[(ag, c)])), 4) for c in cats}
        all_f1 = [r["f1"] for r in rows if r.get("agent") == ag and "f1" in r]
        overall = round(float(np.mean(all_f1)) if all_f1 else 0.0, 4)
        total_tok = sum(r.get("tokens", 0) for r in rows if r.get("agent") == ag)
        out[ag] = {"overall_f1": overall, "per_category": per_cat,
                   "total_tokens": total_tok, "n_qa": len(all_f1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="vanilla,prosmem",
                    help="Comma-separated: vanilla,prosmem")
    ap.add_argument("--n-qa", type=int, default=20,
                    help="QA items per conversation")
    ap.add_argument("--n-convs", type=int, default=1,
                    help="Number of conversations (size-stratified: small/median/large)")
    ap.add_argument("--backbone", default="deepseek/deepseek-chat-v3-0324")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load_locomo()
    print(f"Loaded {len(data)} LoCoMo conversations")

    convs = pick_size_sample(data, args.n_convs)
    print(f"Scope: n_convs={len(convs)}  sizes={[len(c['turns']) for c in convs]}")

    config = ProsMemConfig.from_env()
    if not config.llm_api_key:
        print("ERROR: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    config.main_model = args.backbone
    agent_keys = [a.strip() for a in args.agents.split(",")]

    # Pre-compute output paths so we can register checkpoint
    conv_tag = f"{len(convs)}convs_{args.n_qa}qa" if len(convs) > 1 else convs[0]["id"]
    out = args.out or f"results/locomo/run_{conv_tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out.replace(".json", ".partial.json")

    # Resume from checkpoint if present
    prior_rows, done_keys = _load_checkpoint(checkpoint_path)
    if prior_rows:
        print(f"Resuming from checkpoint: {len(prior_rows)} rows "
              f"({len(done_keys)} (qa,agent) keys) from {checkpoint_path}")

    all_rows = list(prior_rows)
    per_conv_meta = []
    for conv in convs:
        qas = sample_qa(conv, args.n_qa)
        print(f"\n### conv={conv['id']} turns={len(conv['turns'])} qa_subset={len(qas)} ###",
              flush=True)
        # Pass all_rows as prior_rows so per-QA flush is cumulative across convs
        new_rows = run(conv, qas, agent_keys, config,
                       checkpoint_path=checkpoint_path,
                       done_keys=done_keys,
                       prior_rows=all_rows)
        all_rows.extend(new_rows)
        per_conv_meta.append({"conv_id": conv["id"],
                              "n_turns": len(conv["turns"]),
                              "n_qa": len(qas)})
        # Refresh done_keys with newly completed rows for cross-conv resume
        done_keys = {(r["qa_id"], r["agent"]) for r in all_rows
                     if "qa_id" in r and "agent" in r}
        _dump_checkpoint(checkpoint_path, all_rows)

    result = {
        "meta": {"convs": per_conv_meta,
                 "agents": [AGENTS[a][0] for a in agent_keys],
                 "backbone": args.backbone,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "details": all_rows,
        "summary": summarize(all_rows),
    }

    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nResults → {out}")

    print("\n=== Aggregate Summary (all convs) ===")
    for ag, s in result["summary"].items():
        per_cat = " ".join(f"{c}={v:.3f}" for c, v in s["per_category"].items())
        print(f"  {ag:<20} overall={s['overall_f1']:.3f}  "
              f"n={s['n_qa']}  tok={s['total_tokens']}  |  {per_cat}")

    # Per-conv breakdown for diagnosis
    print("\n=== Per-conv F1 ===")
    from collections import defaultdict
    bycv = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        if "f1" in r:
            bycv[r["conv_id"]][r["agent"]].append(r["f1"])
    for cid in sorted(bycv):
        print(f"  {cid}:")
        for ag in sorted(bycv[cid]):
            arr = bycv[cid][ag]
            print(f"    {ag:<20} n={len(arr):>3}  F1={sum(arr)/len(arr):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
