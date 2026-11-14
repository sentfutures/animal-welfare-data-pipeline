#!/usr/bin/env python3
"""SDF matrix pipeline orchestrator: plan, draft, rewrite, score.

Layer 1 deals and repairs the hands (offline), screens them for extreme
implausibility with redeal rounds, plans one document per accepted hand, rewrites
each description for plausibility and conformance, and selects considerations for
each surviving specification. Layers 2-4 draft, rewrite, and score/gate.

Steps a document goes through (full table with module detail in CLAUDE.md):

   1. Deal (compose_prompts, no model)                      cannot drop
   2. Implausibility screen (layer1_screen), batches of 10,
      up to 3 redeal rounds                                 can drop: extremely implausible
   3. Compose planning prompt (compose_prompts, no model)   cannot drop
   4. Plan (layer1_plan)                                    can drop: declared incoherent
   5. Description rewrite (layer1_description_rewrite)      can drop: dealt-value conflict
   6. Consideration scoring (layer1_select_considerations)  cannot drop
   7. Consideration assignment (considerations_assignment,
      no model)                                             cannot drop
   8. Draft (layer2_draft)                                  cannot drop
   9. Critique (layer3_rewrite, first call)                 cannot drop
  10. Revise (layer3_rewrite, second call)                  cannot drop
  11. Score (layer4_score)                                  cannot drop
  12. Gate and near-duplicate cull (layer4_score, no model):
      alignment and realism must each reach 7, near-duplicates
      above 0.90 are dropped                                can drop: below threshold,
                                                            near-duplicate

Runs made in the five-layer layout or before the layer renumbering can be read
(report, audits, viewer) but not resumed: --resume refuses them.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, costs, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags
from sdf_pipeline.dealt_value_identifiers import RECORDS_VALUE_IDS_MANIFEST_KEY
from sdf_pipeline.document_state import RunLayout
from sdf_pipeline.document_state_loader import UnknownRunLayoutError, detect_run_layout
from sdf_pipeline import (
    layer1_description_rewrite,
    layer1_plan,
    layer1_screen,
    layer1_select_considerations,
    layer2_draft,
    layer3_rewrite,
    layer4_score,
    seeding,
)


def refuse_to_resume_run_in_older_layout(run_dir: Path) -> None:
    """Exit before any folder is created or call made when ``run_dir`` is not in the current layout.

    Older layouts can be read but not resumed: a resume would create a second set of ``layer1`` to
    ``layer4`` folders beside the run's own, and the layer numbers name different stages.
    """
    try:
        layout = detect_run_layout(run_dir)
    except UnknownRunLayoutError as error:
        raise SystemExit(f"Cannot resume {run_dir.name}: {error}")
    if layout is not RunLayout.CURRENT:
        raise SystemExit(
            f"Cannot resume {run_dir.name}: it is in the {layout.name} layout. Runs in that layout can be read "
            "(report, audits, viewer) but not resumed; only runs in the current layout (layer1 to layer4) can be.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SDF matrix pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Start from this layer (1-4: plan, draft, rewrite, score). Default 1.")
    parser.add_argument("--label", default="dev", help="Run label, e.g. dev or full-scale.")
    parser.add_argument("--run-id", default=None, help="Run to resume (with --resume; defaults to latest).")
    parser.add_argument("--seed-from", default=None, metavar="RUN",
                        help="Start a NEW run from a chosen step of this earlier run (a run id, or a run folder). "
                             "Needs --from-step.")
    parser.add_argument("--from-step", default=None,
                        help="With --seed-from: the step to start at (" + ", ".join(seeding.SEEDABLE_STEPS) + ").")
    parser.add_argument("--document-ids", default=None,
                        help="With --seed-from: comma-separated document ids to seed (default: every document the step can read).")
    parser.add_argument("--use-current-prompts", action="store_true",
                        help="With --seed-from: run on this checkout's current prompts and constitution instead of the "
                             "source run's frozen inputs.")
    parser.add_argument("--revision-override-file", default=None, type=Path,
                        help="With --seed-from at critique or revise: a JSON file mapping document ids (or \"*\") to "
                             "revision records ({\"cards\": {...}}) written into the seeded draft records.")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    seeding_requested = args.seed_from is not None
    if seeding_requested and (args.resume or args.layer is not None or args.run_id):
        parser.error("--seed-from starts a new run at --from-step; it cannot be combined with --resume, --layer or --run-id.")
    if seeding_requested and not args.from_step:
        parser.error("--seed-from needs --from-step.")
    if not seeding_requested and (args.from_step or args.document_ids or args.use_current_prompts
                                  or args.revision_override_file):
        parser.error("--from-step, --document-ids, --use-current-prompts and --revision-override-file need --seed-from.")
    if args.layer is None:
        args.layer = 1

    if not 1 <= args.layer <= 4:
        parser.error(f"--layer must be 1-4, got {args.layer}.")

    config = utils.load_config(args.config)

    root = Path(__file__).parent.parent
    # PIPELINE_OUTPUT_ROOT redirects all run output (used by the test suite)
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", root / "outputs"))
    runs_root = outputs_root / "sdf" / "runs"

    start_layer = args.layer
    if seeding_requested:
        seeded_run = seeding.seed_run_from_chosen_step(
            config=config,
            source_run=seeding.resolve_source_run(args.seed_from, runs_root),
            from_step=args.from_step,
            document_ids=[d.strip() for d in args.document_ids.split(",") if d.strip()] if args.document_ids else None,
            runs_root=runs_root,
            label=args.label,
            repository_root=root,
            use_current_prompts=args.use_current_prompts,
            revision_override_path=args.revision_override_file,
        )
        run_dir = seeded_run.run_directory
        start_layer = seeded_run.start_layer
        print(f"Seeded {run_dir.name} from {args.seed_from} at step {args.from_step} "
              f"with {len(seeded_run.document_ids)} document(s); starting at layer {start_layer}")
    elif args.resume:
        run_dir = utils.resolve_run_dir(runs_root, args.run_id)
        refuse_to_resume_run_in_older_layout(run_dir)
        utils.warn_if_backend_changed(run_dir, config)
    else:
        run_dir = utils.create_run_dir(
            runs_root,
            label=args.label,
            config=config,
            snapshot_dirs={
                "prompts": root / "prompts" / "sdf",
                "constitution": root / "constitution",
            },
            # Decided once, here: this run writes dealt value ids into its stage files. A run created
            # before this key existed lacks it and is resumed without ever gaining the field.
            extra_manifest_fields={RECORDS_VALUE_IDS_MANIFEST_KEY: True},
        )

    # Read templates from the run's frozen snapshot so prompts stay reproducible
    # (and --resume replays the run's own templates, not the repo's current ones).
    prompts_dir = run_dir / "inputs" / "prompts"
    if not prompts_dir.is_dir():
        raise SystemExit(f"Run {run_dir.name} has no inputs/prompts snapshot, so it cannot be resumed on its own templates.")

    api.init(args.config, cost_log_path=run_dir / "cost_log.jsonl")
    # On --resume the run's log already holds the earlier attempt's calls;
    # mark here so the closing report is what THIS invocation spent.
    cost_start = costs.mark(run_dir / "cost_log.jsonl")

    plan_dir = utils.sdf_stage_dir(run_dir, 1)
    layer_dirs = {n: utils.sdf_stage_dir(run_dir, n) for n in (2, 3, 4)}
    final_dir = run_dir / "final"
    for d in [plan_dir, *layer_dirs.values(), final_dir]:
        utils.ensure_dir(d)

    print(f"=== SDF Matrix Pipeline — run {run_dir.name} ===")
    print(f"Outputs: {run_dir}")

    plans = drafts = rewrites = None

    if start_layer <= 1:
        layer1_description_rewrite.require_openai_key_if_configured(config)
        print("[Layer 1] Deal hands and screen them for extreme implausibility (with redeals)")
        hands = layer1_screen.run(config, prompts_dir, plan_dir)
        print("[Layer 1] Compose prompts + plan documents")
        plans = layer1_plan.run(config, prompts_dir, plan_dir, hands)
        print("[Layer 1] Rewrite descriptions for plausibility and conformance")
        plans = layer1_description_rewrite.run(config, prompts_dir, plan_dir, plans)
        print("[Layer 1] Select considerations for each surviving specification")
        plans = layer1_select_considerations.run(config, prompts_dir, plan_dir, plans)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    if start_layer <= 2:
        if plans is None:
            plans = layer1_description_rewrite.load_rewritten_plans(
                config, plan_dir, utils.load_jsonl(plan_dir / "plans.jsonl"))
            plans = layer1_select_considerations.attach_selections(
                config, prompts_dir, plan_dir, plans)
        print("[Layer 2] Draft documents")
        drafts = layer2_draft.run(config, prompts_dir, layer_dirs[2], plans)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    if start_layer <= 3:
        if drafts is None:
            drafts = utils.load_jsonl(layer_dirs[2] / "drafts.jsonl")
        print("[Layer 3] Review and rewrite")
        rewrites = layer3_rewrite.run(config, prompts_dir, layer_dirs[3], drafts)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    if start_layer <= 4:
        if rewrites is None:
            rewrites = utils.load_jsonl(layer_dirs[3] / "rewrites.jsonl")
        print("[Layer 4] Score and gate")
        final = layer4_score.run(config, prompts_dir, layer_dirs[4], final_dir, rewrites)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")
        print(f"=== Done. {len(final)} documents in {final_dir / 'sdf_corpus.jsonl'} ===")

    costs.report(run_dir / "cost_log.jsonl", title="Run cost (this invocation)",
                 n_done=config["sdf"]["n_prompts"], since=cost_start,
                 project_to=(config["sdf"].get("experiment") or {}).get("project_to"))
    if cost_start:
        costs.report(run_dir / "cost_log.jsonl", title="Run cost (total, incl. earlier attempts)",
                     n_done=config["sdf"]["n_prompts"])


if __name__ == "__main__":
    main()
