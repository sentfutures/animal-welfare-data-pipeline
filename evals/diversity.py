#!/usr/bin/env python3
"""Corpus-level SEMANTIC diversity audit, using OpenAI text embeddings.

evals/audit_sdf.py measures redundancy lexically (word-shingle cosine): it
catches copied skeletons and phrasing, but not paraphrase — two documents can
share zero 3-grams and still be the same document semantically. This tool
embeds the corpus (OpenAI text-embedding-3-small by default) and reads
diversity in meaning-space:

  Redundancy: each document's nearest-neighbor cosine similarity, the fraction
  of semantic near-duplicates at 0.80/0.90/0.95, and the most-similar pairs
  (with snippets) so paraphrase clusters can be inspected directly.

  Diversity: mean pairwise cosine (how tightly the corpus clusters overall)
  and the Vendi score — the effective number of semantically distinct
  documents (exp of the entropy of the similarity-matrix spectrum; N for N
  orthogonal docs, 1 for N identical docs).

  Spread: per-type intra-group cosine and the similarity between type
  centroids, when records carry a `type_id` (SDF).

SCOPE — this measures MEANING/topic diversity, NOT lexical/structural. "Similar"
here is embedding cosine, so two prompts that share templated phrasing ("So
I...", "gut-check", "can you help me") but different topics embed far apart and
count as fully distinct. For a shared-scenario set (e.g. the same DAD scenarios
drafted two ways) the Vendi is ~fixed regardless of surface phrasing — it tells
you about topic coverage (a 1a/scenario property), not about writing
fingerprints. For lexical/structural diversity use the offline checks in
evals/audit_dad.py (openers/closers, structural skeletons, and the
over-represented-n-gram + style-Vendi section), which read surface form.

Absolute numbers depend on the embedding model and on the corpus sharing one
broad topic BY DESIGN, so the headline use is comparing runs: rerun after a
pipeline change and diff with --compare. Verdict thresholds are provisional
and only attached where defensible (the near-duplicate rate).

Works on both corpora: SDF records embed `content`; DAD records embed their
user+assistant `messages` joined.

Usage:
  python evals/diversity.py                                   # outputs/sdf/latest
  python evals/diversity.py --input outputs/dad/latest        # a DAD run dir
  python evals/diversity.py --input some/corpus.jsonl         # a bare corpus file
  python evals/diversity.py --compare <previous diversity_report.json>

Embeddings cost ~$0.02 per million tokens and are cached per report dir
(audit/embeddings_cache.npz, gitignored), so reruns and threshold tweaks are
free. Writes audit/diversity_report.json next to audit_sdf's report.
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import embeddings, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags
from dad_pipeline.id_registry import prompt_key, prompt_keys

# ---------------------------------------------------------------- verdicts
# (same conventions as evals/audit_sdf.py)


def _verdict(value: float, good: float, ok: float, higher_better: bool = False) -> str:
    if higher_better:
        return "GOOD" if value >= good else ("OK" if value >= ok else "BAD")
    return "GOOD" if value <= good else ("OK" if value <= ok else "BAD")


# Display rows are also collected into report["sections"] (same convention as
# evals/audit_dad.py) so the viewer renders this report without re-deriving
# any threshold. Unlike audit_dad the prints here predate the collectors, so
# rows are recorded alongside the existing prints rather than through them.


def _sec(report: dict, title: str) -> dict:
    sec: dict = {"title": title, "rows": []}
    report.setdefault("sections", []).append(sec)
    return sec


def _r(sec: dict, label: str, value: str, verdict: str | None = None, note: str = "") -> None:
    sec["rows"].append({"label": label, "value": value, "verdict": verdict, "note": note})


def _d(sec: dict, line: str) -> None:
    sec.setdefault("detail", []).append(line)


def _fmt(label: str, value: str, verdict: str | None = None, note: str = "") -> str:
    tail = f"  [{verdict}]" if verdict else ""
    tail += f"  {note}" if note else ""
    return f"   {label:<34} {value}{tail}"


# ---------------------------------------------------------------- input resolution


def resolve_input(input_arg: str) -> tuple[list[dict], Path, str]:
    """Return (records, report_dir, corpus_name).

    Accepts a run dir (SDF or DAD — resolved via final/*_corpus.jsonl) or a
    bare JSONL file. Per-type names for the group breakdown come from each
    record's own ``type_name`` (the matrix pipeline writes no layer-1 type map).
    """
    path = Path(input_arg)
    if path.is_dir():
        for name in ("sdf_corpus.jsonl", "dad_corpus.jsonl"):
            corpus = path / "final" / name
            if corpus.exists():
                return utils.load_jsonl(corpus), path / "audit", name
        raise SystemExit(f"No final/sdf_corpus.jsonl or final/dad_corpus.jsonl under {path}")
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    return utils.load_jsonl(path), path.parent / "audit", path.name


def record_text(rec: dict) -> str:
    """The embeddable text of a record: SDF `content`, or DAD `messages` joined."""
    if rec.get("content"):
        return rec["content"].strip()
    if rec.get("messages"):
        return "\n\n".join(m.get("content", "") for m in rec["messages"]).strip()
    return ""


def record_id(rec: dict, index: int) -> str:
    # DAD finals: the stable example gid (E-####) beats the uuid record_id
    # for anything a human reads (pair lists, cloud hover)
    return (rec.get("doc_id") or rec.get("example_gid")
            or rec.get("record_id") or f"row{index}")


def scope_id(rec: dict, scope: str, fallback: str, prompt_gids: dict) -> str:
    """The stable id to LABEL a record with in a given scope, so the cloud hover
    names what the dot actually is: the response gid (R-) for the responses
    scope, the prompt gid (P-) for the prompts scope, the example gid (E-, the
    `fallback`) for the combined/document scope. Falls back to `fallback` when
    the scope's own gid is unavailable (e.g. non-DAD records)."""
    if scope == "responses":
        return rec.get("response_gid") or fallback
    if scope == "prompts":
        return prompt_gids.get(rec.get("record_id")) or fallback
    return fallback


def dad_prompt_gids(run_dir: Path) -> dict:
    """record_id -> prompt gid (P-), joined final→step3→step1 via each record's
    prompt key (prompt_gid, legacy prompt_id); {} for non-DAD runs or missing stages. The
    final corpus carries example_gid/response_gid but not the prompt gid, so the
    prompts-scope cloud needs this two-hop lookup to label dots P- not E-."""
    try:
        pid_by_rec = {r["record_id"]: prompt_key(r)
                      for r in utils.load_jsonl(run_dir / "step3" / "rewrites.jsonl")
                      if r.get("record_id") and prompt_key(r)}
        pgid_by_pid = {k: d["prompt_gid"]
                       for d in utils.load_jsonl(run_dir / "step1" / "dilemmas.jsonl")
                       if d.get("prompt_gid") for k in prompt_keys(d)}
    except (OSError, KeyError):
        return {}
    return {rec: pgid_by_pid[pid] for rec, pid in pid_by_rec.items() if pid in pgid_by_pid}


def stride_sample(items: list, cap: int) -> list:
    """Deterministic stride sample (same scheme as audit_sdf's --dup-sample)."""
    if len(items) <= cap:
        return list(items)
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


# ---------------------------------------------------------------- embedding cache


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    utils.ensure_dir(path.parent)
    np.savez_compressed(path, **cache)


def embed_with_cache(
    texts: list[str], model: str, cache_path: Path | None, chunk: int = 256
) -> tuple[np.ndarray, dict]:
    """Embed texts, reusing/extending the npz cache. Returns (matrix, stats).

    The cache is keyed by sha256(model + text), so edited documents re-embed
    and unchanged ones don't. It is saved after every chunk — like the
    pipelines' checkpoints, a crash mid-run must not discard paid API calls.
    """
    cache = load_cache(cache_path) if cache_path else {}
    keys = [_cache_key(model, t) for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache]

    for start in range(0, len(missing), chunk):
        idxs = missing[start : start + chunk]
        vectors = embeddings.embed_texts([texts[i] for i in idxs], model=model)
        for i, v in zip(idxs, vectors):
            cache[keys[i]] = v.astype(np.float32)
        if cache_path:
            save_cache(cache_path, cache)
        print(f"   embedded {min(start + chunk, len(missing))}/{len(missing)} new documents")

    X = np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)
    # cached rows are already unit-length; re-normalize to be safe
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    stats = {"cached": len(texts) - len(missing), "embedded": len(missing)}
    return X / norms, stats


# ---------------------------------------------------------------- metrics
# All take an L2-normalized float32 matrix, one row per document.


def mean_pairwise_cosine(X: np.ndarray) -> float:
    """Mean cosine over all distinct pairs, via the centroid identity
    (sum_ij <xi,xj> = ||sum xi||^2) — O(n·d), no n×n matrix.

    Assumes every row is unit-norm: the diagonal term subtracted is n, which
    equals sum_i ||xi||^2 only for unit rows. Callers must L2-normalize (and
    drop any zero rows) before calling — see centroid_mean_cosine.
    """
    n = len(X)
    if n < 2:
        return 0.0
    s = X.sum(axis=0, dtype=np.float64)
    return float((s @ s - n) / (n * (n - 1)))


def nearest_neighbors(X: np.ndarray, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Per-document nearest-neighbor (cosine, index), blockwise like
    textstats.nearest_neighbor_sims."""
    n = len(X)
    sims = np.zeros(n, dtype=np.float32)
    idx = np.zeros(n, dtype=np.int64)
    if n < 2:
        return sims, idx
    for i in range(0, n, block):
        S = X[i : i + block] @ X.T
        rows = S.shape[0]
        S[np.arange(rows), np.arange(i, i + rows)] = -2.0  # mask self-similarity
        idx[i : i + rows] = S.argmax(axis=1)
        sims[i : i + rows] = S[np.arange(rows), idx[i : i + rows]]
    return sims, idx


def vendi_score(X: np.ndarray) -> float:
    """Effective number of distinct documents: exp of the von Neumann entropy
    of X·Xᵀ/n (Friedman & Dieng 2023). n orthogonal docs → n; n copies → 1."""
    n = len(X)
    if n < 2:
        return float(n)
    K = (X @ X.T).astype(np.float64) / n
    ev = np.clip(np.linalg.eigvalsh(K), 0.0, None)
    total = ev.sum()  # = trace = 1 for unit rows, up to float error
    if total <= 0:
        return 1.0
    ev = ev / total
    nz = ev[ev > 1e-12]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def kmeans_evenness(X: np.ndarray, k: int | None = None, seed: int = 0,
                    iters: int = 60, return_labels: bool = False):
    """Topic evenness: k-means over the (unit) doc
    embeddings, then the normalized entropy of cluster sizes (1.0 = topics
    perfectly even) and the largest cluster's share. Plain numpy k-means++
    (cosine via normalized centroids) — no new dependency; deterministic via
    seed. k defaults to n/5 capped at 50.
    return_labels=True returns (stats, labels) so callers can say WHAT is in
    each cluster; the default stays the bare stats dict."""
    n = len(X)
    if k is None:
        k = min(50, max(2, n // 5))
    k = max(2, min(k, n))
    rng = np.random.default_rng(seed)
    centers = [X[int(rng.integers(n))]]
    for _ in range(k - 1):
        d2 = np.min(np.stack([((X - c) ** 2).sum(axis=1) for c in centers]), axis=0)
        total = float(d2.sum())
        pick = int(rng.choice(n, p=d2 / total)) if total > 0 else int(rng.integers(n))
        centers.append(X[pick])
    C = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = np.argmax(X @ C.T, axis=1)
        newC = np.stack([X[labels == j].mean(axis=0) if (labels == j).any() else C[j]
                         for j in range(k)])
        norms = np.linalg.norm(newC, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        newC = newC / norms
        if np.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    sizes = np.bincount(labels, minlength=k).astype(float)
    sizes = sizes[sizes > 0]
    p = sizes / sizes.sum()
    evenness = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    stats = {"k": int(k), "clusters_nonempty": int(len(p)),
             "evenness": round(evenness, 3), "largest_share": round(float(p.max()), 4),
             "sizes": sorted((int(s) for s in sizes), reverse=True)}
    return (stats, labels) if return_labels else stats


def _cluster_detail(ids: list, texts: list, X: np.ndarray, labels: np.ndarray,
                    snippet_width: int = 140) -> list[dict]:
    """What each cluster IS: one entry per nonempty cluster, largest first
    (aligned with the sorted `sizes` list and the viewer's topic-spread bars).
    Each carries the cluster size, its member ids, and the most CENTRAL member
    (highest cosine to the cluster centroid) as a representative snippet — an
    honest, offline answer to "what is cluster 1?" without a paid labeling
    call. Clusters are unlabeled k-means groups; the representative is a
    typical member, not a title."""
    out = []
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        centroid = X[idx].mean(axis=0)
        norm = float(np.linalg.norm(centroid)) or 1.0
        rep = idx[int(np.argmax(X[idx] @ (centroid / norm)))]
        out.append({"size": int(len(idx)),
                    "rep_id": ids[rep],
                    "rep": _snippet(texts[rep], snippet_width),
                    "ids": [ids[i] for i in idx]})
    out.sort(key=lambda d: -d["size"])
    return out


def pca_coords(X: np.ndarray) -> np.ndarray:
    """2-D PCA projection for the document-cloud scatter. PCA rather than
    UMAP/t-SNE deliberately: deterministic and dependency-free; good enough to
    show clumps and outliers, which is all the cloud is for."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :2] * S[:2]).astype(float)


# Idea-level diversity: each record is reduced to a
# one-line scenario summary; summaries are embedded and near neighbours
# counted. Catches the same idea re-skinned across registers/settings, which
# document-level embeddings miss.
_IDEA_PROMPT = (
    "Reduce the text below to ONE line naming its core scenario: who faces what "
    "decision or question about what. Strip register, names, place, and phrasing — "
    "two texts with the same underlying scenario should produce near-identical "
    "lines. BEGIN the line with the most specific actor or setting available "
    "(e.g. 'A town-board member', 'A backyard cricket farmer') — NEVER with a "
    "generic stem like 'A person', 'A user', 'Someone', or 'An AI': these lines "
    "are compared by embedding similarity, and a uniform opening shared across "
    "summaries inflates every pairwise score. Return ONLY the line.\n\nTEXT:\n"
)


def idea_text(rec: dict) -> str:
    """The text whose IDEA we summarize: for chat records the user message
    (the scenario lives there); for documents the content."""
    msgs = rec.get("messages")
    if isinstance(msgs, list) and msgs:
        return str(msgs[0].get("content") or "")
    return str(rec.get("content") or rec.get("text") or "")


def scope_text(rec: dict, scope: str) -> str:
    """One side of a chat record: 'prompts' = the user message, 'responses' =
    the assistant message. Non-chat records have no sides (empty string)."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    if scope == "prompts":
        return str(msgs[0].get("content") or "")
    if len(msgs) > 1:
        return str(msgs[1].get("content") or "")
    return ""


def idea_summaries(ids: list[str], texts: list[str], config: dict) -> dict[str, str]:
    """One summary call per record via shared.api (Anthropic; cents). Failed
    summaries are dropped (and counted by the caller via missing ids)."""
    from shared import api, utils as _utils

    def one(item):
        rid, text = item
        try:
            line = api.call_claude(user_message=_IDEA_PROMPT + text[:6000],
                                   max_tokens=120, temperature=0.2,
                                   stage="eval_diversity_ideas", item_id=rid)
            return rid, re.sub(r"\s+", " ", line).strip()
        except Exception:
            return rid, None
    out = {}
    for rid, line in _utils.parallel_map(one, list(zip(ids, texts)),
                                         config.get("workers", 1)):
        if line:
            out[rid] = line
    return out


def top_pairs(
    sims: np.ndarray, idx: np.ndarray, ids: list[str], texts: list[str],
    limit: int, floor: float,
) -> list[dict]:
    """The most-similar document pairs (deduplicated i↔j), sim >= floor."""
    seen = set()
    order = np.argsort(-sims)
    pairs = []
    for i in order[: limit * 4]:
        if sims[i] < floor:
            break
        j = int(idx[i])
        key = frozenset((int(i), j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "a": ids[i], "b": ids[j], "similarity": round(float(sims[i]), 4),
            "a_snippet": _snippet(texts[i]), "b_snippet": _snippet(texts[j]),
        })
        if len(pairs) >= limit:
            break
    return pairs


def _snippet(text: str, width: int = 80) -> str:
    return re.sub(r"\s+", " ", text.strip())[:width]


def group_breakdown(records: list[dict], X: np.ndarray) -> list[dict] | None:
    """Per-type_id intra-group cosine, for SDF records that carry one."""
    groups: dict = collections.defaultdict(list)
    names: dict = {}
    for i, rec in enumerate(records):
        if rec.get("type_id") is not None:
            groups[rec["type_id"]].append(i)
            names.setdefault(rec["type_id"], rec.get("type_name") or str(rec["type_id"]))
    if len(groups) < 2:
        return None
    rows = []
    for gid, idxs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        intra = mean_pairwise_cosine(X[idxs]) if len(idxs) >= 2 else None
        rows.append({
            "type_id": gid, "type_name": str(names.get(gid, ""))[:60], "n": len(idxs),
            "intra_mean_cosine": round(intra, 4) if intra is not None else None,
        })
    return rows


def centroid_mean_cosine(records: list[dict], X: np.ndarray) -> float | None:
    """Mean cosine between per-type centroids — how separable the types are."""
    groups: dict = collections.defaultdict(list)
    for i, rec in enumerate(records):
        if rec.get("type_id") is not None:
            groups[rec["type_id"]].append(i)
    if len(groups) < 2:
        return None
    C = np.stack([X[idxs].mean(axis=0) for idxs in groups.values()])
    # Drop degenerate centroids (a group whose unit vectors cancel to ~zero);
    # a zero row breaks mean_pairwise_cosine's unit-norm assumption. Real
    # embeddings sit in a positive cone so this ~never fires, but keep it exact.
    norms = np.linalg.norm(C, axis=1)
    C = C[norms > 0]
    if len(C) < 2:
        return None
    return mean_pairwise_cosine(C / np.linalg.norm(C, axis=1, keepdims=True))


# ---------------------------------------------------------------- report


def compare_reports(current: dict, previous_path: str) -> None:
    with open(previous_path, encoding="utf-8") as f:
        prev = json.load(f)
    if prev.get("embed_model") != current.get("embed_model"):
        print(f"\nCOMPARE: skipped — previous report used embed model "
              f"{prev.get('embed_model')!r}, this run {current.get('embed_model')!r} "
              "(cosines are not comparable across models)")
        return
    print(f"\nCOMPARE (vs {previous_path} — positive delta = more redundant/clustered)")
    rows = [
        ("mean nearest-neighbor sim", ("nn", "mean"), "{:+.3f}"),
        ("near-dup fraction >0.90", ("nn", "over_0.90"), "{:+.1%}"),
        ("mean pairwise cosine", ("mean_pairwise_cosine",), "{:+.3f}"),
        ("Vendi ratio", ("vendi", "ratio"), "{:+.3f}"),
        ("topic evenness", ("clusters", "evenness"), "{:+.3f}"),
    ]
    for label, keys, fmt in rows:
        cur_v, prev_v = current, prev
        for k in keys:
            cur_v = cur_v.get(k, {}) if isinstance(cur_v, dict) else {}
            prev_v = prev_v.get(k, {}) if isinstance(prev_v, dict) else {}
        if isinstance(cur_v, (int, float)) and isinstance(prev_v, (int, float)):
            print(_fmt(label, f"{cur_v} (was {prev_v}, {fmt.format(cur_v - prev_v)})"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic diversity audit of a corpus.")
    parser.add_argument("--input", default="outputs/sdf/latest",
                        help="Run directory (SDF or DAD) or corpus JSONL path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=2000,
                        help="Deterministic stride-sample cap (Vendi is O(n^3) past a few thousand)")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model (default: resolve_default_model() — "
                             "EMBEDDINGS_MODEL env var, else whichever provider has a key)")
    parser.add_argument("--ideas", action="store_true",
                        help="Idea-level diversity: one-line scenario summary per record "
                             "(Anthropic call each), summaries embedded and near "
                             "neighbours counted — catches re-skinned scenarios")
    # A first, cheap char cap (bounds cost and keeps runs consistent); English
    # is ~4 chars/token, so 7000 chars is ~1.7k tokens. It does NOT bound tokens
    # for CJK (~1.25-1.5 tok/char) — that safety is enforced under embed_texts,
    # which truncates any input to the model's 8192-token window before sending.
    parser.add_argument("--max-chars", type=int, default=7000,
                        help="First-pass char truncation before embedding; the hard "
                             "8192-token input cap is enforced by embed_texts")
    parser.add_argument("--top-pairs", type=int, default=10,
                        help="Most-similar pairs to list in the report")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip the embeddings cache (re-embeds everything)")
    parser.add_argument("--compare", default=None,
                        help="A previous diversity_report.json to print deltas against")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    records, report_dir, corpus_name = resolve_input(args.input)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Corpus is empty — nothing to audit.")

    embeddings.init(args.config)  # evals log to the global cost log
    args.embed_model = args.embed_model or embeddings.resolve_default_model()

    indexed = [(record_id(r, i), record_text(r), r) for i, r in enumerate(records)]
    empty = [rid for rid, text, _ in indexed if not text]
    indexed = [(rid, text, r) for rid, text, r in indexed if text]
    sampled = stride_sample(indexed, args.max_docs)
    truncated = sum(1 for _, text, _ in sampled if len(text) > args.max_chars)

    ids = [rid for rid, _, _ in sampled]
    texts = [text[: args.max_chars] for _, text, _ in sampled]
    recs = [r for _, _, r in sampled]
    # record_id -> prompt gid (P-), so the prompts-scope cloud labels dots P-
    # not the example gid E- (the final corpus lacks the prompt gid).
    prompt_gids = dad_prompt_gids(report_dir.parent)
    if len(texts) < 2:
        raise SystemExit("Need at least 2 non-empty documents for diversity metrics.")

    n = len(texts)
    print(f"=== Semantic diversity audit: {args.input} "
          f"({len(records)} records, {n} embedded) ===\n")

    cache_path = None if args.no_cache else report_dir / "embeddings_cache.npz"
    X, cache_stats = embed_with_cache(texts, args.embed_model, cache_path)

    report: dict = {}
    sec = _sec(report, "Embedding")
    print("EMBEDDING")
    print(_fmt("model", args.embed_model))
    _r(sec, "model", args.embed_model)
    print(_fmt("cache", f"{cache_stats['cached']} reused, {cache_stats['embedded']} embedded"))
    _r(sec, "embedded", f"{n} of {len(records)} records "
                        f"({cache_stats['cached']} cached, {cache_stats['embedded']} new)")
    if empty:
        print(_fmt("empty documents skipped", f"{len(empty)}", "BAD",
                   "(empty training records are a pipeline defect)"))
        _r(sec, "empty documents skipped", str(len(empty)), "BAD",
           "(empty training records are a pipeline defect)")
    if truncated:
        print(_fmt("truncated for embedding", f"{truncated} of {n} (> {args.max_chars} chars)"))
        _r(sec, "truncated for embedding", f"{truncated} of {n} (> {args.max_chars} chars)")

    sims, nn_idx = nearest_neighbors(X)
    over = {t: float((sims > t).sum()) / n for t in (0.80, 0.90, 0.95)}
    v = _verdict(over[0.90], 0.02, 0.08)
    pairs = top_pairs(sims, nn_idx, ids, texts, args.top_pairs, floor=0.80)

    sec = _sec(report, "Redundancy (semantic)")
    print("\nREDUNDANCY (embedding cosine — semantic; catches paraphrase, not just copied text)")
    print(_fmt("mean nearest-neighbor sim", f"{float(sims.mean()):.3f}"))
    _r(sec, "mean nearest-neighbor sim", f"{float(sims.mean()):.3f}")
    print(_fmt("p90 / max nearest-neighbor sim",
               f"{float(np.quantile(sims, 0.9)):.3f} / {float(sims.max()):.3f}"))
    _r(sec, "p90 / max nearest-neighbor sim",
       f"{float(np.quantile(sims, 0.9)):.3f} / {float(sims.max()):.3f}")
    print(_fmt("near-dup >0.80 / >0.90 / >0.95",
               f"{over[0.80]:.1%} / {over[0.90]:.1%} / {over[0.95]:.1%}", v,
               "(verdict on >0.90: near-verbatim in meaning-space)"))
    _r(sec, "near-dup >0.80 / >0.90 / >0.95",
       f"{over[0.80]:.1%} / {over[0.90]:.1%} / {over[0.95]:.1%}", v,
       "(verdict on >0.90: near-verbatim in meaning-space)")
    for p in pairs:
        print(f"      {p['similarity']:.3f}  {p['a'][:12]} ~ {p['b'][:12]}")
        print(f"             a: {p['a_snippet']}")
        print(f"             b: {p['b_snippet']}")
        _d(sec, f"{p['similarity']:.3f}  {p['a']} ~ {p['b']}  ·  a: {p['a_snippet'][:60]}  ·  "
                f"b: {p['b_snippet'][:60]}")

    mpc = mean_pairwise_cosine(X)
    vendi = vendi_score(X)
    sec = _sec(report, "Diversity")
    print("\nDIVERSITY")
    print(_fmt("mean pairwise cosine", f"{mpc:.3f}", None,
               "(shared-topic corpora sit well above 0; track the trend across runs)"))
    _r(sec, "mean pairwise cosine", f"{mpc:.3f}", None,
       "(shared-topic corpora sit well above 0; track the trend across runs)")
    print(_fmt("Vendi score", f"{vendi:.1f} effective docs of {n} (ratio {vendi / n:.3f})", None,
               "(higher = more semantically distinct documents)"))
    _r(sec, "Vendi score", f"{vendi:.1f} effective docs of {n} (ratio {vendi / n:.3f})", None,
       "(higher = more semantically distinct documents)")

    clusters = kmeans_evenness(X)
    print(_fmt("topic evenness", f"{clusters['evenness']:.3f} across {clusters['k']} clusters "
                                 f"(largest {clusters['largest_share']:.1%})", None,
               "(k-means on embeddings; 1.0 = perfectly even; compare across runs)"))
    _r(sec, "topic evenness", f"{clusters['evenness']:.3f} across {clusters['k']} clusters "
                              f"(largest {clusters['largest_share']:.1%})", None,
       "(k-means on embeddings; 1.0 = perfectly even; compare across runs)")

    groups = group_breakdown(recs, X)
    centroid_sim = centroid_mean_cosine(recs, X)
    if groups:
        sec = _sec(report, "Groups (by type_id)")
        print("\nGROUPS (by type_id)")
        for g in groups:
            intra = f"{g['intra_mean_cosine']:.3f}" if g["intra_mean_cosine"] is not None else "n/a"
            print(f"   type {g['type_id']}: n={g['n']}, intra-cosine {intra}  {g['type_name']}")
            _d(sec, f"type {g['type_id']}: n={g['n']}, intra-cosine {intra}  {g['type_name']}")
        print(_fmt("mean type-centroid cosine", f"{centroid_sim:.3f}", None,
                   "(types blurring together pushes this toward 1)"))
        _r(sec, "mean type-centroid cosine", f"{centroid_sim:.3f}", None,
           "(types blurring together pushes this toward 1)")

    # Per-scope distributions (prompts / responses / combined), stored with the
    # raw nearest-neighbour sims and sorted cluster sizes so the viewer can
    # draw the diversity chart rows (NN histogram · topic-spread bars · cloud).
    # Chat corpora get all three scopes; document corpora just the combined one.
    def _scope_block(s_ids: list, s_texts: list, S: np.ndarray) -> dict:
        s_sims, _ = nearest_neighbors(S)
        s_cloud = pca_coords(S)
        s_clusters, s_labels = kmeans_evenness(S, return_labels=True)
        s_clusters["detail"] = _cluster_detail(s_ids, s_texts, S, s_labels)
        return {
            "n": len(s_ids),
            "nn_sims": [round(float(v), 3) for v in s_sims],
            "over": {f"{t:.2f}": round(float((s_sims > t).sum()) / len(s_ids), 4)
                     for t in (0.80, 0.90, 0.95)},
            "mean_pairwise_cosine": round(mean_pairwise_cosine(S), 4),
            "vendi_ratio": round(vendi_score(S) / len(s_ids), 4),
            "clusters": s_clusters,
            "cloud": [{"id": rid, "x": round(float(x), 3), "y": round(float(y), 3),
                       "snippet": _snippet(t, 90)}
                      for rid, t, (x, y) in zip(s_ids, s_texts, s_cloud)],
        }

    scopes = {"combined": _scope_block(ids, texts, X)}
    if any(isinstance(r.get("messages"), list) and r.get("messages") for r in recs):
        for scope, label in (("prompts", "user messages"), ("responses", "assistant messages")):
            # label each dot with the scope's own gid (P- / R-), not the E- used
            # for the combined scope
            kept = [(scope_id(r, scope, rid, prompt_gids), scope_text(r, scope))
                    for rid, r in zip(ids, recs)]
            kept = [(rid, t) for rid, t in kept if t.strip()]
            if len(kept) < 2:
                continue
            s_ids = [rid for rid, _ in kept]
            s_texts = [t[: args.max_chars] for _, t in kept]
            S, _ = embed_with_cache(s_texts, args.embed_model, cache_path)
            blk = _scope_block(s_ids, s_texts, S)
            scopes[scope] = blk
            sec = _sec(report, f"Scope: {scope} ({label})")
            print(f"\nSCOPE {scope} ({label})")
            nd = (f"{blk['over']['0.80']:.1%} / {blk['over']['0.90']:.1%} / "
                  f"{blk['over']['0.95']:.1%}")
            ndv = _verdict(blk["over"]["0.90"], 0.02, 0.08)
            print(_fmt("near-dup >0.80 / >0.90 / >0.95", nd, ndv))
            _r(sec, "near-dup >0.80 / >0.90 / >0.95", nd, ndv)
            dv = (f"cosine {blk['mean_pairwise_cosine']:.3f} · "
                  f"Vendi ratio {blk['vendi_ratio']:.3f}")
            print(_fmt("mean pairwise / Vendi", dv))
            _r(sec, "mean pairwise / Vendi", dv)
            c = blk["clusters"]
            ev = f"{c['evenness']:.3f} across {c['k']} clusters (largest {c['largest_share']:.1%})"
            print(_fmt("topic evenness", ev))
            _r(sec, "topic evenness", ev)

    ideas_report = None
    if args.ideas:
        from shared import api
        api.init(args.config)  # summary calls log to the global cost log too
        idea_texts_all = [idea_text(r) for r in recs]
        keep = [i for i, t in enumerate(idea_texts_all) if t.strip()]
        summaries = idea_summaries([ids[i] for i in keep],
                                   [idea_texts_all[i] for i in keep],
                                   utils.load_config(args.config))
        sec = _sec(report, "Idea-level diversity (LLM summaries)")
        print("\nIDEA-LEVEL (one-line scenario summaries, embedded — catches re-skinned ideas)")
        s_ids = sorted(summaries)
        failures = len(keep) - len(s_ids)
        line = f"{len(s_ids)} of {len(keep)}" + (f" ({failures} summary failures)" if failures else "")
        print(_fmt("records summarized", line))
        _r(sec, "records summarized", line)
        if len(s_ids) >= 2:
            s_texts = [summaries[i] for i in s_ids]
            S, s_cache = embed_with_cache(s_texts, args.embed_model, cache_path)
            s_sims, s_nn = nearest_neighbors(S)
            s_over = {t: float((s_sims > t).sum()) / len(s_ids) for t in (0.80, 0.90, 0.95)}
            sv = _verdict(s_over[0.95], 0.05, 0.15)
            print(_fmt("idea near-dup >0.80 / >0.90 / >0.95",
                       f"{s_over[0.80]:.1%} / {s_over[0.90]:.1%} / {s_over[0.95]:.1%}", sv,
                       "(verdict on >0.95: the same idea re-skinned)"))
            _r(sec, "idea near-dup >0.80 / >0.90 / >0.95",
               f"{s_over[0.80]:.1%} / {s_over[0.90]:.1%} / {s_over[0.95]:.1%}", sv,
               "(verdict on >0.95: the same idea re-skinned)")
            s_pairs = top_pairs(s_sims, s_nn, s_ids, s_texts, args.top_pairs, floor=0.80)
            for p in s_pairs:
                print(f"      {p['similarity']:.3f}  {p['a'][:12]} ~ {p['b'][:12]}")
                print(f"             a: {p['a_snippet']}")
                print(f"             b: {p['b_snippet']}")
                _d(sec, f"{p['similarity']:.3f}  a: {p['a_snippet'][:70]}  ·  b: {p['b_snippet'][:70]}")
            ideas_report = {
                "n": len(s_ids), "failures": failures,
                "nn_mean": round(float(s_sims.mean()), 4),
                "nn_sims": [round(float(v), 3) for v in s_sims],
                **{f"over_{t:.2f}": round(frac, 4) for t, frac in s_over.items()},
                "top_pairs": s_pairs,
                "summaries": {rid: summaries[rid] for rid in s_ids},
            }

    report.update({
        "input": utils.repo_relative(args.input),
        "corpus": corpus_name,
        "embed_model": args.embed_model,
        "n_records": len(records),
        "n_embedded": n,
        "n_empty": len(empty),
        "empty_ids": empty[:20],
        "n_truncated": truncated,
        "max_chars": args.max_chars,
        "cache": cache_stats,
        "nn": {
            "mean": round(float(sims.mean()), 4),
            "p90": round(float(np.quantile(sims, 0.9)), 4),
            "max": round(float(sims.max()), 4),
            **{f"over_{t:.2f}": round(frac, 4) for t, frac in over.items()},
        },
        "top_pairs": pairs,
        "mean_pairwise_cosine": round(mpc, 4),
        "vendi": {"score": round(vendi, 2), "ratio": round(vendi / n, 4)},
        "clusters": clusters,
        "scopes": scopes,
        "ideas": ideas_report,
        "groups": groups,
        "type_centroid_mean_cosine": round(centroid_sim, 4) if centroid_sim is not None else None,
    })

    if args.compare:
        compare_reports(report, args.compare)

    utils.ensure_dir(report_dir)
    out = report_dir / "diversity_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to {out}")
    print(f"Embedding cost is appended to the global cost log "
          f"({utils.load_config(args.config)['outputs']['cost_log']}); "
          f"~$0.02 per 1M tokens, so a full run is cents.")


if __name__ == "__main__":
    main()
