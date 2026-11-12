"""measures the SDF corpus's semantic diversity against real web text.

evals/diversity.py answers "how much ground does this corpus cover" only against
itself, or against a structureless corpus of the same size — neither of which
says whether the answer is low. This script supplies the missing reference: a
sample of FineWeb-Edu matched to our own documents on size, length and topic, so
the two Vendi scores are comparable and their ratio means something.

Five cells, each matching on more:

  full_length      all documents, length-matched only. Our side is multilingual
                   and FineWeb-Edu is English, which INFLATES our score (an
                   embedding space separates languages by construction), so this
                   cell flatters us and is reported last.
  english_density  English only, matched on length AND welfare-phrase density
                   — the like-for-like cell, and the headline.
  phrases_ge_{1,2,3}
                   both sides carry at least k distinct welfare phrases, then
                   length-matched. Tightening k squeezes FineWeb onto our topic,
                   so the ratio across these cells separates "they cover more
                   topics" from "they write more differently about one topic".

Two readings are reported alongside each raw Vendi, because the raw number alone
cannot carry the argument:

  structureless reference / clumping cost
      Vendi the same n documents would score if every pair sat at their measured
      mean cosine. All text on one topic occupies a narrow cone, which costs
      diversity without being a defect; measured-minus-this is the part that is.

  n-matched null
      Vendi is NOT size-invariant (it rises with n), so cells cannot be compared
      to each other. Each side's own unfiltered pool is resampled at the cell's n
      to show what the filter did beyond what the sample size did.

The phrase list is derived from our corpus, not hand-written — see
evals/welfare_lexicon.py. That biases the comparison AGAINST the headline:
selecting FineWeb documents by our own vocabulary pulls them toward our region of
the embedding space and lowers their measured diversity, so a gap that survives
is a floor, not a ceiling.

    python evals/welfare_lexicon.py --input outputs/sdf/latest
    python evals/external_baseline.py --input outputs/sdf/latest --sanity
    python evals/external_baseline.py --input outputs/sdf/latest
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import diversity, welfare_lexicon
from shared import utils

DEFAULT_POOL = "outputs/external/fineweb_pool.jsonl"
DEFAULT_CACHE = "outputs/external/embeddings_cache.npz"
DEFAULT_LEXICON = "evals/welfare_phrases.yaml"

# a reservoir of ordinary documents, for the one cell with no phrase floor. It
# only has to be large enough that length-matching the whole corpus still has
# choices, so it is capped well below the phrase pool.
RESERVOIR = 15_000


def mean_field_vendi(n: int, rho: float) -> float:
    """Vendi n documents would score if every pair sat at cosine rho.

    The Gram matrix of that corpus has eigenvalues (1+(n-1)rho)/n once and
    (1-rho)/n with multiplicity n-1, so this is closed form.
    """
    if n < 2:
        return float(n)
    rho = max(0.0, min(1.0, rho))
    ev = np.array([(1 + (n - 1) * rho) / n] + [(1 - rho) / n] * (n - 1), dtype=np.float64)
    ev = ev / ev.sum()
    nz = ev[ev > 1e-12]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def side_metrics(X: np.ndarray, n_boot: int = 200) -> dict:
    """the reported shape of one side of one cell."""
    n = len(X)
    rho = diversity.mean_pairwise_cosine(X)
    measured = diversity.vendi_score(X)
    structureless = mean_field_vendi(n, rho)
    block = {
        "n": n,
        "vendi": round(measured, 2),
        "vendi_ratio": round(measured / n, 4) if n else None,
        "mean_pairwise_cosine": round(rho, 4),
        "vendi_if_structureless": round(structureless, 2),
        "clumping_cost": round(structureless - measured, 2),
    }
    ci = diversity.vendi_ci(X, n_boot=n_boot)
    if ci:
        block["ci_lo"], block["ci_hi"] = round(ci[0], 2), round(ci[1], 2)
    return block


def subsample_vendi(X: np.ndarray, n: int, draws: int = 30, seed: int = 0) -> float | None:
    """mean Vendi of random n-subsets of X — the n-matched null.

    Returns None when the pool is not strictly larger than n, where the
    "null" would be the measurement itself.
    """
    if n < 2 or len(X) <= n:
        return None
    rng = np.random.default_rng(seed)
    scores = [diversity.vendi_score(X[rng.choice(len(X), n, replace=False)])
              for _ in range(draws)]
    return round(float(np.mean(scores)), 2)


def match(targets: list[dict], pool: list[dict], features: tuple[str, ...],
          seed: int = 0) -> tuple[list[dict], float]:
    """greedily pair each target with the nearest unused pool item.

    Features are standardized over the pool first, so a char length and a phrase
    density contribute comparably. Targets are visited in a seeded random order
    rather than corpus order: greedy matching is order-dependent, and corpus
    order would systematically hand the best matches to whatever the corpus
    happens to list first.

    Returns:
        (matched pool items, mean standardized distance) — the distance is the
        matching quality, and belongs in the report.
    """
    if not pool or not targets:
        return [], float("nan")
    P = np.array([[float(p[f]) for f in features] for p in pool])
    mu, sigma = P.mean(axis=0), P.std(axis=0)
    sigma[sigma == 0] = 1.0
    P = (P - mu) / sigma
    T = (np.array([[float(t[f]) for f in features] for t in targets]) - mu) / sigma

    used = np.zeros(len(pool), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(targets))
    out: list[tuple[int, dict]] = []
    dists: list[float] = []
    for i in order:
        d = np.linalg.norm(P - T[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            break
        used[j] = True
        out.append((int(i), pool[j]))
        dists.append(float(d[j]))
    out.sort(key=lambda kv: kv[0])
    return [p for _, p in out], float(np.mean(dists)) if dists else float("nan")


def build_pool(dataset: str, config: str, stream_docs: int, patterns: list,
               reservoir: int, seed: int, out_path: Path) -> list[dict]:
    """stream the external corpus once, keeping what the cells need.

    Kept: every document carrying at least one welfare phrase (the topic cells),
    plus a reservoir sample of ordinary documents (the length-only cell). Texts
    are kept — the pool is re-read on every later run, and re-streaming hundreds
    of thousands of documents to rebuild it is the expensive mistake.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    rng = np.random.default_rng(seed)
    kept: list[dict] = []
    res: list[dict] = []
    n = 0
    for rec in ds:
        if n >= stream_docs:
            break
        text = rec.get("text") or ""
        n += 1
        if not text.strip():
            continue
        count = welfare_lexicon.count_phrases(text, patterns)
        item = {"id": rec.get("id"), "text": text, "chars": len(text),
                "phrases": count, "density": count / max(len(text), 1) * 1000}
        if count >= 1:
            kept.append(item)
        elif len(res) < reservoir:
            res.append(item)
        elif rng.random() < reservoir / n:  # reservoir sampling, keeps it unbiased
            res[rng.integers(0, reservoir)] = item
        if n % 25_000 == 0:
            print(f"   streamed {n:,}  phrase-docs {len(kept):,}  reservoir {len(res):,}")

    pool = kept + res
    utils.ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in pool:
            f.write(json.dumps(item) + "\n")
    print(f"   pool: {len(kept):,} phrase-docs + {len(res):,} reservoir "
          f"from {n:,} streamed -> {out_path}")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="SDF run dir or corpus jsonl")
    parser.add_argument("--lexicon", default=DEFAULT_LEXICON)
    parser.add_argument("--pool", default=DEFAULT_POOL,
                        help="cached external sample; built on first run, reused after")
    parser.add_argument("--rebuild-pool", action="store_true")
    parser.add_argument("--stream-docs", type=int, default=300_000)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE)
    parser.add_argument("--language", default="English")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vendi-boot", type=int, default=200)
    parser.add_argument("--sanity", action="store_true",
                        help="stream 20k docs instead of 300k — fast, cheap, and "
                             "enough to prove the path end to end")
    args = parser.parse_args()

    if args.sanity:
        args.stream_docs = 20_000

    terms = welfare_lexicon.load_lexicon(args.lexicon)
    patterns = welfare_lexicon.compile_matcher(terms)
    print(f"welfare lexicon: {len(terms)} terms from {args.lexicon}")

    records, report_dir, corpus_name = diversity.resolve_input(args.input)
    sdf = []
    for i, r in enumerate(records):
        text = diversity.record_text(r)
        if not text:
            continue
        count = welfare_lexicon.count_phrases(text, patterns)
        sdf.append({"id": diversity.record_id(r, i), "text": text,
                    "chars": len(text), "phrases": count,
                    "density": count / max(len(text), 1) * 1000,
                    "language": r.get("language")})
    english = [d for d in sdf if d["language"] == args.language]
    print(f"corpus: {len(sdf)} documents, {len(english)} {args.language}")
    if not english:
        raise SystemExit(f"No {args.language} documents in {corpus_name}.")

    pool_path = Path(args.pool)
    if args.rebuild_pool or not pool_path.exists():
        print(f"building external pool from {args.dataset} [{args.dataset_config}], "
              f"streaming {args.stream_docs:,} documents")
        pool = build_pool(args.dataset, args.dataset_config, args.stream_docs,
                          patterns, RESERVOIR, args.seed, pool_path)
    else:
        pool = utils.load_jsonl(pool_path)
        print(f"external pool: {len(pool):,} documents (cached at {pool_path}; "
              "--rebuild-pool to refresh)")

    phrase_pool = [p for p in pool if p["phrases"] >= 1]
    reservoir = [p for p in pool if p["phrases"] == 0]

    # ---- cells -----------------------------------------------------------
    # each carries the two pools it was drawn FROM, so its n-matched null is
    # resampled from the same populations the cell itself was selected out of
    cells = []
    cells.append(("english_density", english, phrase_pool, ("chars", "density"),
                  f"{args.language} only, length + welfare-phrase density matched",
                  english, phrase_pool))
    for k in (1, 2, 3):
        ours = [d for d in english if d["phrases"] >= k]
        theirs = [p for p in phrase_pool if p["phrases"] >= k]
        cells.append((f"phrases_ge_{k}", ours, theirs, ("chars",),
                      f">={k} distinct welfare phrase(s) both sides, length matched",
                      english, phrase_pool))
    cells.append(("full_length", sdf, reservoir or pool, ("chars",),
                  "whole corpus, length matched (ours multilingual, theirs English)",
                  sdf, reservoir or pool))

    # ---- embed and measure ----------------------------------------------
    cache = Path(args.cache_path)
    results = []
    for name, ours, theirs, features, note, pool_ours, pool_theirs in cells:
        n = min(len(ours), len(theirs))
        if n < 10:
            print(f"\n[{name}] SKIPPED — only {n} documents available "
                  f"(ours {len(ours)}, theirs {len(theirs)})")
            results.append({"cell": name, "note": note, "skipped": True,
                            "available_ours": len(ours), "available_theirs": len(theirs)})
            continue
        # whichever side is larger is cut down to the other's size by the same
        # matching, so a cell is never a random truncation of the bigger side
        ours_cell = ours if len(ours) <= n else match(theirs, ours, features, args.seed)[0]
        matched, dist = match(ours_cell, theirs, features, args.seed)
        n = min(len(ours_cell), len(matched))
        ours_cell, matched = ours_cell[:n], matched[:n]

        print(f"\n[{name}] n={n} per side — {note}")
        Xs, _ = diversity.embed_with_cache([d["text"] for d in ours_cell],
                                           args.embed_model, cache)
        Xt, _ = diversity.embed_with_cache([d["text"] for d in matched],
                                           args.embed_model, cache)
        ours_m = side_metrics(Xs, args.vendi_boot)
        theirs_m = side_metrics(Xt, args.vendi_boot)
        ratio = theirs_m["vendi"] / ours_m["vendi"] if ours_m["vendi"] else None
        results.append({
            "cell": name, "note": note, "n": n,
            "match_mean_distance": round(dist, 3),
            "sdf": ours_m, "external": theirs_m,
            "vendi_ratio": round(ratio, 2) if ratio else None,
            "median_chars": {"sdf": int(np.median([d["chars"] for d in ours_cell])),
                             "external": int(np.median([d["chars"] for d in matched]))},
            "mean_phrases": {"sdf": round(float(np.mean([d["phrases"] for d in ours_cell])), 2),
                             "external": round(float(np.mean([d["phrases"] for d in matched])), 2)},
            "_pools": (pool_ours, pool_theirs),
        })
        print(f"   ours     Vendi {ours_m['vendi']:7.2f}  cos {ours_m['mean_pairwise_cosine']:.3f}"
              f"  structureless {ours_m['vendi_if_structureless']:7.2f}")
        print(f"   external Vendi {theirs_m['vendi']:7.2f}  cos {theirs_m['mean_pairwise_cosine']:.3f}"
              f"  structureless {theirs_m['vendi_if_structureless']:7.2f}")
        print(f"   ratio    {ratio:.2f}x" if ratio else "   ratio    n/a")

    # ---- n-matched nulls -------------------------------------------------
    # Vendi rises with n, so a cell cannot be read against another cell. Each
    # side is resampled at the cell's own n from the pool it was selected out
    # of: measured-above-null is what the cell's FILTER did, as distinct from
    # what its sample size did.
    print("\nn-matched nulls (each side resampled from its own pool at the cell's n)")
    null_cap = 800  # embedding a whole pool to draw subsamples is not worth cents
    embedded: dict[int, np.ndarray] = {}

    def pool_matrix(pool: list[dict]) -> np.ndarray:
        if id(pool) not in embedded:
            sample = diversity.stride_sample(pool, null_cap)
            embedded[id(pool)], _ = diversity.embed_with_cache(
                [d["text"] for d in sample], args.embed_model, cache)
        return embedded[id(pool)]

    for r in results:
        if r.get("skipped"):
            continue
        pool_ours, pool_theirs = r.pop("_pools")
        r["null_at_n"] = {
            "sdf": subsample_vendi(pool_matrix(pool_ours), r["n"], seed=args.seed),
            "external": subsample_vendi(pool_matrix(pool_theirs), r["n"], seed=args.seed),
        }
        print(f"   {r['cell']:18s} n={r['n']:4d}  ours {r['null_at_n']['sdf']}"
              f"   external {r['null_at_n']['external']}")

    report = {
        "corpus": corpus_name,
        "external_dataset": f"{args.dataset}:{args.dataset_config}",
        "embed_model": args.embed_model,
        "lexicon": {"path": args.lexicon, "n_terms": len(terms), "terms": terms},
        "pool": {"path": str(pool_path), "size": len(pool),
                 "phrase_docs": len(phrase_pool), "reservoir": len(reservoir)},
        "cells": results,
    }
    utils.ensure_dir(report_dir)
    out = Path(report_dir) / "external_baseline.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'cell':20s} {'n':>5s} {'ours':>8s} {'external':>9s} {'ratio':>7s}")
    for r in results:
        if r.get("skipped"):
            print(f"{r['cell']:20s} {'--':>5s} {'skipped':>8s}")
            continue
        print(f"{r['cell']:20s} {r['n']:5d} {r['sdf']['vendi']:8.1f} "
              f"{r['external']['vendi']:9.1f} {str(r['vendi_ratio']) + 'x':>7s}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
