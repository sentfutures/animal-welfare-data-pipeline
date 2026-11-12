#!/usr/bin/env python3
"""Build a blind rating run from two arms of an SDF revision experiment.

The arms of an experiment rewrite ONE shared set of drafts, so two arms hold
two versions of the same document differing only in the intervention. That is
already a preference pair; this stages them as one so pref_pipeline/rate.py can
rate them blind. Nothing is generated and nothing is billed.

Pairs are shuffled within each comparison and then interleaved proportionally
between them, so a rater who stops early leaves a random sample of each
comparison rather than a prefix of the corpus or a run of whichever one the
shuffle happened to front-load.

    python pref_pipeline/from_sdf_arms.py --exp outputs/sdf/experiments/split \\
        --pair control:split --pair control:control-b
    streamlit run pref_pipeline/rate.py
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline import compose_prompts as cp
from shared import utils
# Imported rather than reimplemented: the blinded side assignment has to match
# what the rating app and its tests already assume, and a second copy of a
# hashing rule is a copy that drifts.
from pref_pipeline.run import _left_arm

PROMPT_HEADER = (
    "Two versions of the same document, rewritten from the same draft. "
    "Which is the better document?\n\n"
    "Judge on NATURALNESS above all — would this exact document exist on the "
    "real internet, written by this author for this audience? Then on whether "
    "it embodies what it set out to show rather than merely stating it. The "
    "spec both versions were written to is below for reference; neither is "
    "trying to be shorter or longer than the other.\n\n"
    "---\n\n"
)


def arm_documents(run_dir: Path) -> dict[str, dict]:
    """{doc_id: rewrite record} for one arm."""
    return {r["doc_id"]: r
            for r in utils.load_jsonl(utils.sdf_stage_file(run_dir, "rewrite"))}


def build_pairs(exp_dir: Path, comparisons: list[tuple[str, str]]) -> list[dict]:
    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    arms = manifest["arms"]
    cache: dict[str, dict[str, dict]] = {}

    pairs = []
    for name_a, name_b in comparisons:
        for name in (name_a, name_b):
            if name not in arms:
                raise SystemExit(
                    f"arm {name!r} is not in {exp_dir / 'manifest.json'} "
                    f"(have: {', '.join(sorted(arms))})"
                )
            cache.setdefault(name, arm_documents(Path(arms[name])))
        docs_a, docs_b = cache[name_a], cache[name_b]
        shared = sorted(set(docs_a) & set(docs_b))
        if not shared:
            raise SystemExit(f"{name_a} and {name_b} share no documents")
        print(f"  {name_a} vs {name_b}: {len(shared)} shared document(s)")
        for doc_id in shared:
            a, b = docs_a[doc_id], docs_b[doc_id]
            pairs.append({
                # Opaque on purpose: the rating app prints the pair id above the
                # documents, so an id naming its arms would tell the rater which
                # comparison they are in — and a calibration comparison the
                # rater can recognise has stopped calibrating anything.
                "pair_id": "pair_" + hashlib.md5(
                    f"{name_a}:{name_b}:{doc_id}".encode()).hexdigest()[:12],
                "prompt_id": doc_id,
                "user_message": PROMPT_HEADER + a["description"],
                "arm_names": {"a": name_a, "b": name_b},
                "response_a": a["content"],
                "response_b": b["content"],
                # analysis-only; the rating app ignores unknown keys
                "comparison": f"{name_a}-vs-{name_b}",
                "doc_id": doc_id,
                # SDF deals {culture}; the language is derived from it. None
                # rather than "English" so a record that carries neither is a
                # visible gap in the analysis instead of a wrong label.
                "language": cp.record_language(a, default=None),
            })
    return pairs


def interleave(pairs: list[dict], rng: random.Random) -> list[dict]:
    """Shuffle within each comparison, then merge them proportionally.

    A flat shuffle over the combined list can front-load one comparison — seed 0
    opened on six of the same kind — which costs a rater who stops early the
    other comparison entirely. Emitting whichever group is furthest behind its
    share keeps every prefix balanced without an alternating pattern a rater
    could read off.
    """
    groups: dict[str, list[dict]] = {}
    for p in pairs:
        groups.setdefault(p["comparison"], []).append(p)
    for g in groups.values():
        rng.shuffle(g)

    totals = {k: len(v) for k, v in groups.items()}
    taken = dict.fromkeys(groups, 0)
    out = []
    while len(out) < len(pairs):
        live = [k for k in groups if taken[k] < totals[k]]
        behind = min(taken[k] / totals[k] for k in live)
        choices = [k for k in live if taken[k] / totals[k] == behind]
        key = rng.choice(choices)
        out.append(groups[key][taken[key]])
        taken[key] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True, help="Experiment dir holding manifest.json.")
    parser.add_argument("--pair", action="append", metavar="ARM_A:ARM_B",
                        help="Comparison to stage; repeatable. Default control:split.")
    parser.add_argument("--label", default=None, help="Run label (default: the experiment id).")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the pair order.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep only the first N pairs AFTER shuffling.")
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    if not (exp_dir / "manifest.json").exists():
        raise SystemExit(f"no manifest at {exp_dir / 'manifest.json'}")

    comparisons = []
    for spec in (args.pair or ["control:split"]):
        if spec.count(":") != 1:
            raise SystemExit(f"--pair must be ARM_A:ARM_B, got {spec!r}")
        comparisons.append(tuple(spec.split(":")))

    pairs = interleave(build_pairs(exp_dir, comparisons), random.Random(args.seed))
    if args.limit:
        pairs = pairs[:args.limit]

    repo_root = Path(__file__).parent.parent
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", repo_root / "outputs"))
    exp_id = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))["exp_id"]
    run_dir = utils.create_run_dir(
        outputs_root / "pref" / "runs",
        label=args.label or f"sdf-{exp_id}",
        config={"pref": {"source_experiment": str(exp_dir),
                         "comparisons": [f"{a}:{b}" for a, b in comparisons],
                         "seed": args.seed}},
    )

    for pair in pairs:
        pair["left_arm"] = _left_arm(pair["pair_id"], salt=run_dir.name)
    utils.save_jsonl(pairs, run_dir / "pairs" / "pairs.jsonl")

    print(f"\n{len(pairs)} pair(s) -> {run_dir / 'pairs' / 'pairs.jsonl'}")
    for name_a, name_b in comparisons:
        n = sum(1 for p in pairs if p["comparison"] == f"{name_a}-vs-{name_b}")
        print(f"  {name_a}-vs-{name_b}: {n}")
    print("\nRate them:  streamlit run pref_pipeline/rate.py")


if __name__ == "__main__":
    main()
