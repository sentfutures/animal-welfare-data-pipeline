#!/usr/bin/env python3
"""Run several layer-3 variants ("arms") over one shared set of drafts.

The diversity experiments all intervene at the revision stage, so the only
honest comparison holds layers 1-2 fixed: drafts are sampled once from a donor
run, the revision cards are dealt once, and every arm re-runs layers 3-4 over
that same material. Any difference between arms is then attributable to the
revision prompt and nothing else.

Each arm is an ORDINARY run dir under outputs/sdf/runs/, so every eval,
--resume and the cost log work on it unchanged; outputs/sdf/experiments/<id>/
is an index tying the arms together with the sample and the deal.

    python experiments/experiment.py --donor-run <run_id> \\
        --arms experiments/arms.example.yaml --exp-id exp1 --n 150 --dry-run
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, costs, matrix, sdf_run_layouts, utils
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.layer4_draft_critique_and_revise import run_layer as draft_critique_and_revise_layer
from sdf_pipeline.layer5_score_and_final_selection import run_layer as score_and_final_selection_layer

DEFAULT_ARMS_FILE = "experiments/arms.example.yaml"


# ------------------------------------------------------------------ sampling

def _cap_quotas(quotas: list[int], capacity: list[int], n: int) -> list[int]:
    """Clamp each quota to its bucket's capacity, redistributing the shortfall.

    A language holding fewer documents than its share demands cannot fill it,
    and silently returning a short sample would understate every arm's n.
    """
    q = [min(a, b) for a, b in zip(quotas, capacity)]
    while sum(q) < n:
        slack = [i for i in range(len(q)) if q[i] < capacity[i]]
        if not slack:
            break
        # widest remaining slack first, so no single bucket absorbs everything
        i = max(slack, key=lambda j: capacity[j] - q[j])
        q[i] += 1
    return q


def stratified_draw(buckets: dict[str, list], n: int, rng: random.Random) -> list:
    """Draw n records matching the buckets' current mix, REMOVING them.

    Mutating is the point: calling this twice gives a sample and a holdout that
    are disjoint by construction rather than by a later set difference.
    """
    keys = sorted(buckets)
    total = sum(len(buckets[k]) for k in keys)
    if n > total:
        raise SystemExit(f"cannot draw {n} records — only {total} available")
    capacity = [len(buckets[k]) for k in keys]
    quotas = _cap_quotas(
        matrix.largest_remainder([c / total for c in capacity], n), capacity, n
    )
    drawn = []
    for key, quota in zip(keys, quotas):
        pool = buckets[key]
        for i in sorted(rng.sample(range(len(pool)), quota), reverse=True):
            drawn.append(pool.pop(i))
    rng.shuffle(drawn)
    return drawn


# shared with experiments/compare_arms.py, which reads lengths by the same derivation
language_of = cp.record_language


def language_mix(records: list[dict]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for r in records:
        mix[language_of(r)] = mix.get(language_of(r), 0) + 1
    return dict(sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))


# ----------------------------------------------------------------- the deal

def deal_revision(
    records: list[dict],
    variables_path: Path | None,
    seed: int,
) -> dict[str, dict]:
    """{doc_id: {"cards": {...}}} dealt once and shared by every arm.

    Cards come from the revision half of the matrix by deck sample, so their
    marginals match the declared weights exactly. Dealt here rather than read
    off each record because the donor's drafts were composed before these axes
    existed, and because arms are only comparable if they saw the same cards.
    """
    cards: list[dict[str, str]] = [{} for _ in records]
    if variables_path and variables_path.exists():
        values, weights = matrix.split_weights(matrix.parse_variables(variables_path))
        # Only the revision half: the file is variables.txt, whose composition
        # axes were already dealt when the donor's drafts were made.
        axes = sorted(a for a in values if a in cp.REVISION_AXES)
        if axes:
            cards = matrix.deck_sample(
                values, weights, axes, len(records), random.Random(seed)
            )

    return {r["doc_id"]: {"cards": c} for r, c in zip(records, cards)}


# ------------------------------------------------------------------- arms

def merge_arm_config(base: dict, arm: dict) -> dict:
    """Base config with this arm's revision knobs applied."""
    merged = copy.deepcopy(base)
    sdf = merged.setdefault("sdf", {})
    rev = sdf.setdefault("revision", {})
    for key in ("style_mode", "use_length", "split"):
        if key in arm:
            rev[key] = arm[key]
    for key in ("rewrite_model", "revise_model", "critique_model", "score_model"):
        if key in arm:
            sdf[key] = arm[key]
    if "rewrite_model" in arm and "revise_model" in sdf and "revise_model" not in arm:
        sdf["revise_model"] = arm["rewrite_model"]   # the old name is this arm's setting; the new key would win over it
    return merged


def import_rewrites(donor_dir: Path, doc_ids: set[str], revisions_dir: Path) -> int:
    """Copy the donor's own rewrites for the sampled ids — a free reference arm.

    Nothing is generated and nothing is billed: this arm is the corpus as it
    actually shipped, which is the only arm that needs no model call at all.
    """
    src = utils.sdf_stage_file(donor_dir, "rewrite")
    if not src.exists():
        raise SystemExit(f"import_rewrites: donor has no rewrites at {src}")
    kept = [r for r in utils.load_jsonl(src) if r["doc_id"] in doc_ids]
    utils.save_jsonl(kept, revisions_dir / "rewrites.jsonl")
    checkpoint = utils.Checkpoint(revisions_dir / "_checkpoint.json")
    for r in kept:
        checkpoint.mark_done(r["doc_id"])
    return len(kept)


def run_arm(
    arm: dict,
    *,
    sample: list[dict],
    deal: dict[str, dict],
    base_config: dict,
    config_path: str,
    runs_root: Path,
    repo_root: Path,
    exp_id: str,
    donor_dir: Path,
    max_docs: int | None,
    existing_run: Path | None,
    project_to: int | None = None,
) -> Path:
    name = arm["name"]
    merged = merge_arm_config(base_config, arm)

    if existing_run is not None:
        run_dir = existing_run
        print(f"  resuming arm '{name}' in {run_dir.name}")
    else:
        run_dir = utils.create_run_dir(
            runs_root,
            label=f"{exp_id}-{name}",
            config=merged,
            snapshot_dirs={
                "prompts": repo_root / "prompts" / "sdf",
                "constitution": repo_root / "constitution",
            },
            extra_manifest_fields={sdf_run_layouts.LAYOUT_RECORDED_IN_MANIFEST_KEY: sdf_run_layouts.FIVE_NAMED_LAYERS.name},
        )
        print(f"  arm '{name}' -> {run_dir.name}")

    prompts_dir = run_dir / "inputs" / "prompts"
    drafts_dir = utils.ensure_dir(utils.sdf_stage_folder(run_dir, sdf_run_layouts.DRAFT))
    revisions_dir = utils.ensure_dir(utils.sdf_stage_folder(run_dir, sdf_run_layouts.CRITIQUE_AND_REVISE))
    scores_dir = utils.ensure_dir(utils.sdf_stage_folder(run_dir, sdf_run_layouts.SCORE))
    final_dir = utils.ensure_dir(run_dir / "final")

    drafts = sample if max_docs is None else sample[:max_docs]
    # The deal rides on the draft record, so layer 3 needs no experiment
    # vocabulary of its own and an arm's drafts.jsonl is self-describing.
    drafts = [{**d, "revision": deal[d["doc_id"]]} for d in drafts]
    utils.save_jsonl(drafts, drafts_dir / "drafts.jsonl")

    api.init(config_path, cost_log_path=run_dir / "cost_log.jsonl")
    cost_start = costs.mark(run_dir / "cost_log.jsonl")

    if arm.get("import_rewrites"):
        n = import_rewrites(donor_dir, {d["doc_id"] for d in drafts}, revisions_dir)
        print(f"    imported {n} donor rewrite(s) — no API calls")
        rewrites = utils.load_jsonl(revisions_dir / "rewrites.jsonl")
    else:
        rewrites = draft_critique_and_revise_layer.run_critique_and_revise(merged, prompts_dir, revisions_dir, drafts)

    final = score_and_final_selection_layer.run(merged, prompts_dir, scores_dir, final_dir, rewrites)
    print(f"    {len(final)} document(s) in {final_dir / 'sdf_corpus.jsonl'}")
    costs.report(run_dir / "cost_log.jsonl", title=f"Arm '{name}' cost",
                 n_done=len(drafts), project_to=project_to, since=cost_start)
    return run_dir


# -------------------------------------------------------------------- main

def load_arms_file(path: Path) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    arms = spec.get("arms") or []
    if not arms:
        raise SystemExit(f"{path}: no arms defined — see experiments/arms.example.yaml")
    names = [a.get("name") for a in arms]
    if len(set(names)) != len(names) or not all(names):
        raise SystemExit(f"{path}: every arm needs a unique 'name'")
    return spec.get("experiment") or {}, arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--arms", default=DEFAULT_ARMS_FILE, help="YAML defining the arms.")
    parser.add_argument("--exp-id", default=None, help="Experiment name (dir under outputs/sdf/experiments).")
    parser.add_argument("--donor-run", default=None, help="Run supplying the shared drafts.")
    parser.add_argument("--n", type=int, default=None, help="Documents per arm.")
    parser.add_argument("--holdout", type=int, default=None, help="Disjoint holdout size.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--arm", default=None, help="Run only this arm.")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap each arm to the first N documents (human read before scaling).")
    parser.add_argument("--resume", action="store_true", help="Re-enter the arms named in the manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no run dirs, no API calls.")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", repo_root / "outputs"))
    runs_root = outputs_root / "sdf" / "runs"
    config = utils.load_config(args.config)

    spec, arms = load_arms_file(Path(args.arms))
    if args.arm:
        arms = [a for a in arms if a["name"] == args.arm]
        if not arms:
            raise SystemExit(f"--arm {args.arm!r} is not defined in {args.arms}")

    exp_id = args.exp_id or spec.get("exp_id")
    if not exp_id:
        raise SystemExit("--exp-id is required (or set experiment.exp_id in the arms file)")
    exp_dir = outputs_root / "sdf" / "experiments" / exp_id
    manifest_path = exp_dir / "manifest.json"

    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if args.resume and prior is None:
        raise SystemExit(f"--resume: no manifest at {manifest_path}")

    # A resumed experiment replays its OWN sample and deal, never a re-derived
    # one: the arms are only comparable if they all saw the same material, and
    # a changed donor, seed or variables file would silently break that.
    if args.resume:
        donor_dir = Path(prior["donor_run"])
        sample = utils.load_jsonl(exp_dir / "sample.jsonl")
        deal = {r["doc_id"]: r["revision"] for r in utils.load_jsonl(exp_dir / "revision_deals.jsonl")}
        seed = prior["seed"]
    else:
        donor_arg = args.donor_run or spec.get("donor_run")
        if not donor_arg:
            raise SystemExit("--donor-run is required (or set experiment.donor_run in the arms file)")
        donor_dir = Path(donor_arg) if Path(donor_arg).is_dir() else utils.resolve_run_dir(runs_root, donor_arg)

        drafts_path = utils.sdf_stage_file(donor_dir, "draft")
        if not drafts_path.exists():
            raise SystemExit(f"donor run has no drafts at {drafts_path}")
        donor_drafts = utils.load_jsonl(drafts_path)

        # Precedence: CLI flag > config > arms file. An `sdf.experiment` block
        # in the config outranks the arms file so one config can shrink ANY
        # arms file to a cheap run without editing it; nothing sets it by
        # default, so normally the arms file decides.
        sizing = (config.get("sdf") or {}).get("experiment") or {}
        n = args.n if args.n is not None else sizing.get(
            "n", spec.get("n", len(donor_drafts)))
        holdout = args.holdout if args.holdout is not None else sizing.get(
            "holdout", spec.get("holdout", 0))
        seed = args.seed if args.seed is not None else sizing.get(
            "seed", spec.get("seed", 0))

        rng = random.Random(seed)
        buckets: dict[str, list[dict]] = {}
        for r in donor_drafts:
            buckets.setdefault(language_of(r), []).append(r)
        sample = stratified_draw(buckets, n, rng)
        holdout_records = stratified_draw(buckets, holdout, rng) if holdout else []

        variables_path = Path(spec.get("revision_variables") or repo_root / "prompts/sdf/variables.txt")
        deal = deal_revision(sample, variables_path, seed=seed)

        print(f"=== Experiment {exp_id} ===")
        print(f"donor: {donor_dir.name} ({len(donor_drafts)} drafts)")
        print(f"donor language mix: {language_mix(donor_drafts)}")
        print(f"sample n={len(sample)}: {language_mix(sample)}")
        if holdout_records:
            print(f"holdout n={len(holdout_records)}: {language_mix(holdout_records)}")
        dealt_axes = sorted({k for d in deal.values() for k in d["cards"]})
        print(f"dealt axes: {dealt_axes or '(none — variables.txt defines no revision axes)'}")
        print(f"arms: {[a['name'] for a in arms]}")

        if args.dry_run:
            print("\n--dry-run: nothing written, no API calls.")
            return

        utils.ensure_dir(exp_dir)  # written only past the dry-run guard
        utils.save_jsonl(sample, exp_dir / "sample.jsonl")
        utils.save_jsonl(holdout_records, exp_dir / "holdout.jsonl")
        utils.save_jsonl(
            [{"doc_id": k, "revision": v} for k, v in deal.items()],
            exp_dir / "revision_deals.jsonl",
        )

    if args.dry_run:  # reached only via --resume --dry-run
        print("--dry-run: nothing written, no API calls.")
        return

    arm_runs = dict((prior or {}).get("arms", {}))
    for arm in arms:
        existing = arm_runs.get(arm["name"])
        run_dir = run_arm(
            arm,
            sample=sample,
            deal=deal,
            base_config=config,
            config_path=args.config,
            runs_root=runs_root,
            repo_root=repo_root,
            exp_id=exp_id,
            donor_dir=donor_dir,
            max_docs=args.max_docs,
            existing_run=Path(existing) if (args.resume and existing) else None,
            project_to=(
                len(sample) if args.max_docs
                else ((config.get("sdf") or {}).get("experiment") or {}).get("project_to")
            ),
        )
        arm_runs[arm["name"]] = str(run_dir)
        manifest_path.write_text(json.dumps({
            "exp_id": exp_id,
            "donor_run": str(donor_dir),
            "seed": seed,
            "n": len(sample),
            "max_docs": args.max_docs,
            "arms": arm_runs,
            "arm_specs": {a["name"]: a for a in arms},
            "success": spec.get("success", {}),
            "git_commit": utils._git_status()[0],
        }, indent=2), encoding="utf-8")

    print(f"\n=== Done. Manifest: {manifest_path} ===")
    print(f"Compare the arms: python experiments/compare_arms.py --exp {exp_dir}")

    # Whole-experiment total: every arm's log, summed, with the full-scale
    # projection a capped (--max-docs) run is actually run to get.
    per_arm = len(sample) if args.max_docs is None else args.max_docs
    grand = 0.0
    for name, run in arm_runs.items():
        grand += costs.summarize(Path(run) / "cost_log.jsonl")["totals"]["usd"]
    print(f"\n{'=' * 62}")
    print(f"EXPERIMENT TOTAL: ${grand:.4f} across {len(arm_runs)} arm(s), "
          f"{per_arm} document(s) each")
    if args.max_docs:
        print(f"  This was a capped run (--max-docs {args.max_docs}). "
              f"Each arm's own projection to the full sample of {len(sample)} is above.")
    print("=" * 62)


if __name__ == "__main__":
    main()
