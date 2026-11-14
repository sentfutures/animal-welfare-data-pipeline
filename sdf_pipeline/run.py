#!/usr/bin/env python3
"""SDF matrix pipeline orchestrator: deal and screen, plan and rewrite descriptions, score considerations, draft, critique and
revise, score.

Five layers, each named for what it does (decision 17); each layer has a folder in ``sdf_pipeline/`` holding its
driver (``run_layer.py``) and one file per step:

  1. ``layer1_deal_and_screen``: deal and screen the hands for extreme implausibility, with redeals (offline deal)
  2. ``layer2_plan_and_description_rewrite``: compose the planning prompts, plan, rewrite descriptions
  3. ``layer3_consideration_scoring_and_selection``: score the considerations, select them (a linear programme)
  4. ``layer4_draft_critique_and_revise``: draft, critique, revise
  5. ``layer5_score_and_final_selection``: score, score gate, near-duplicate removal, write ``final/``

Steps a document goes through (full table with module detail in sdf_pipeline/CLAUDE.md; the folder of each step's
file is its layer's, above):

   1. Deal (compose_prompts, no model)                      cannot drop
   2. Implausibility screen (screen.py), batches of 10,
      up to 3 redeal rounds                                 can drop: extremely implausible
   3. Compose planning prompt (compose_prompts, no model)   cannot drop
   4. Plan (plan.py)                                        can drop: declared incoherent
   5. Description rewrite (description_rewrite.py)          can drop: dealt-value conflict
   6. Consideration scoring (consideration_scoring.py)      cannot drop
   7. Consideration selection (consideration_selection.py,
      considerations_assignment.py, no model)               cannot drop
   8. Draft (draft.py)                                      cannot drop
   9. Critique (critique.py)                                cannot drop
  10. Revise (revise.py)                                    cannot drop
  11. Score (score.py)                                      cannot drop
  12. Score gate and near-duplicate removal (score_gate.py,
      near_duplicate_removal.py, no model):
      alignment and realism must each reach 7, near-duplicates
      above 0.90 are dropped                                can drop: below threshold,
                                                            near-duplicate

A run keeps the layout it was created in (``shared/sdf_run_layouts.py``). Runs made before decision 17 have four
numbered layers (``layer1`` to ``layer4``: layer 1 holds steps 1 to 7, layer 2 the draft, layer 3 the critique and
revise, layer 4 the score), and keep their folders, ``--layer`` numbers and cost-log tags when resumed. Runs made in
the five-layer layout before the variables matrix, or before the layer renumbering, can be read (report, audits,
viewer) but not resumed: --resume refuses them.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, costs, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import RECORDS_VALUE_IDS_MANIFEST_KEY
from sdf_pipeline.shared_step_machinery.document_state import RunLayout
from sdf_pipeline.shared_step_machinery.document_state_loader import UnknownRunLayoutError, detect_run_layout
from shared import sdf_run_layouts
from sdf_pipeline.layer1_deal_and_screen import run_layer as deal_and_screen_layer
from sdf_pipeline.layer3_consideration_scoring_and_selection import run_layer as consideration_scoring_and_selection_layer
from sdf_pipeline.layer4_draft_critique_and_revise import run_layer as draft_critique_and_revise_layer
from sdf_pipeline.layer5_score_and_final_selection import run_layer as score_and_final_selection_layer
from sdf_pipeline import pipeline_steps
from sdf_pipeline import start_a_run_from_an_earlier_runs_step as seeding
from sdf_pipeline.layer2_plan_and_description_rewrite import description_rewrite, plan
from sdf_pipeline.layer4_draft_critique_and_revise import draft


def refuse_to_resume_run_in_older_layout(run_dir: Path) -> sdf_run_layouts.SdfRunLayout:
    """Exit before any folder is created or call made when ``run_dir`` is not in a layout that is still written;
    otherwise the run's layout.

    The five named layers and the four numbered layers are resumed in their own layout. Older layouts can be read
    but not resumed: a resume would create a second set of folders beside the run's own, and the layer numbers
    name different stages.
    """
    try:
        layout = detect_run_layout(run_dir)
    except UnknownRunLayoutError as error:
        raise SystemExit(f"Cannot resume {run_dir.name}: {error}")
    if layout is RunLayout.FIVE_NAMED_LAYERS:
        return sdf_run_layouts.FIVE_NAMED_LAYERS
    if layout is RunLayout.FOUR_NUMBERED_LAYERS:
        return sdf_run_layouts.FOUR_NUMBERED_LAYERS
    raise SystemExit(
        f"Cannot resume {run_dir.name}: it is in the {layout.name} layout. Runs in that layout can be read "
        "(report, audits, viewer) but not resumed; only runs in the five-layer layout (layer1_deal_and_screen to "
        "layer5_score_and_final_selection) or the four-layer layout (layer1 to layer4) can be.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SDF matrix pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Start from this layer. A run in the five-layer layout (every new run) has layers 1-5: deal and "
                             "screen, plan and description rewrite, consideration scoring and selection, draft/critique/"
                             "revise, score and final selection. A run made in the four-layer layout keeps its 1-4: "
                             "everything up to the considerations, draft, critique and revise, score. Default 1.")
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

    config = utils.load_config(args.config)

    root = Path(__file__).parent.parent
    # PIPELINE_OUTPUT_ROOT redirects all run output (used by the test suite)
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", root / "outputs"))
    runs_root = outputs_root / "sdf" / "runs"

    # The layout is decided once, when the run is created, and kept: a new or seeded run is in the five-layer
    # layout; a resumed run keeps the one it was made in.
    layout = sdf_run_layouts.FIVE_NAMED_LAYERS
    run_dir = None
    if args.resume and not seeding_requested:
        run_dir = utils.resolve_run_dir(runs_root, args.run_id)
        layout = refuse_to_resume_run_in_older_layout(run_dir)
    if not seeding_requested and args.layer not in layout.layer_numbers:
        parser.error(f"--layer must be {layout.layer_numbers[0]}-{layout.layer_numbers[-1]}, got {args.layer}.")

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
            # The layout is decided here too, and recorded, so a run keeps its folders and tags.
            extra_manifest_fields={RECORDS_VALUE_IDS_MANIFEST_KEY: True,
                                   sdf_run_layouts.LAYOUT_RECORDED_IN_MANIFEST_KEY: layout.name},
        )

    with sdf_run_layouts.using_layout(layout):
        run_in_layout(args, config, layout, run_dir, start_layer)


def documents_read_when_restarting_at(folder_key: str, layout, reading, config: dict, prompts_dir: Path, folders: dict):
    """The documents the runner of ``folder_key`` is given when the run starts there: what the stage files hold from the
    step before. The plan step is given the hands the screen accepted (None when the screen is off)."""
    if folder_key == sdf_run_layouts.PLAN_AND_DESCRIPTION_REWRITE:
        if deal_and_screen_layer.reason_screen_is_inactive(config, prompts_dir):
            return None
        accepted_path = folders[sdf_run_layouts.DEAL_AND_SCREEN] / deal_and_screen_layer.SCREENED_HANDS_FILENAME
        if not accepted_path.is_file():
            raise SystemExit(f"Cannot start at layer {layout.layer_number_of_key[folder_key]}: the screen has not finished, so "
                             f"{accepted_path} does not exist. Run layer {layout.layer_number_of_key[sdf_run_layouts.DEAL_AND_SCREEN]} first.")
        return utils.load_jsonl(accepted_path)
    entry = next(entry for entry in pipeline_steps.PIPELINE_STEPS if entry.folder_key == folder_key and entry.load_documents)
    return entry.load_documents(reading)


def run_in_layout(args, config: dict, layout: sdf_run_layouts.SdfRunLayout, run_dir: Path, start_layer: int) -> None:
    """Run the layers of ``layout`` from ``start_layer`` on ``run_dir`` (the tags and folders of ``layout`` are active)."""
    # Read templates from the run's frozen snapshot so prompts stay reproducible
    # (and --resume replays the run's own templates, not the repo's current ones).
    prompts_dir = run_dir / "inputs" / "prompts"
    if not prompts_dir.is_dir():
        raise SystemExit(f"Run {run_dir.name} has no inputs/prompts snapshot, so it cannot be resumed on its own templates.")

    api.init(args.config, cost_log_path=run_dir / "cost_log.jsonl")
    # On --resume the run's log already holds the earlier attempt's calls;
    # mark here so the closing report is what THIS invocation spent.
    cost_start = costs.mark(run_dir / "cost_log.jsonl")

    folders = {key: layout.folder_of(run_dir, key) for key in sdf_run_layouts.FOLDER_KEYS}
    final_dir = run_dir / "final"
    for folder in [*folders.values(), final_dir]:
        utils.ensure_dir(folder)

    print(f"=== SDF Matrix Pipeline — run {run_dir.name} ===")
    print(f"Outputs: {run_dir}")

    reading = pipeline_steps.RunReading(config, prompts_dir, folders)

    def refuse_when_snapshot_lacks_templates(folder_key: str) -> None:
        for step_name, missing in pipeline_steps.prompt_files_missing_from_snapshot(folder_key, config, prompts_dir):
            raise SystemExit(f"Run {run_dir.name}'s frozen prompt snapshot lacks {', '.join(missing)}, which step "
                             f"{step_name} needs. Stopping before any call.")

    def layer_label(folder_key: str) -> str:
        return f"[Layer {layout.layer_number_of_key[folder_key]}]"

    def deal_and_screen(unused_documents):
        print(f"{layer_label(sdf_run_layouts.DEAL_AND_SCREEN)} Deal hands and screen them for extreme implausibility (with redeals)")
        return deal_and_screen_layer.run(config, prompts_dir, folders[sdf_run_layouts.DEAL_AND_SCREEN])

    def plan_and_rewrite_descriptions(hands):
        label = layer_label(sdf_run_layouts.PLAN_AND_DESCRIPTION_REWRITE)
        directory = folders[sdf_run_layouts.PLAN_AND_DESCRIPTION_REWRITE]
        print(f"{label} Compose prompts + plan documents")
        plans = plan.run(config, prompts_dir, directory, hands)
        print(f"{label} Rewrite descriptions for plausibility and conformance")
        return description_rewrite.run(config, prompts_dir, directory, plans)

    def select_considerations(plans):
        print(f"{layer_label(sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION)} Select considerations for each surviving specification")
        return consideration_scoring_and_selection_layer.run(
            config, prompts_dir, folders[sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION], plans)

    def draft_documents(plans):
        print(f"{layer_label(sdf_run_layouts.DRAFT)} Draft documents")
        return draft.run(config, prompts_dir, folders[sdf_run_layouts.DRAFT], plans, layout.draft_checkpoint_file_name)

    def critique_and_revise(drafts):
        print(f"{layer_label(sdf_run_layouts.CRITIQUE_AND_REVISE)} Review and rewrite")
        return draft_critique_and_revise_layer.run_critique_and_revise(
            config, prompts_dir, folders[sdf_run_layouts.CRITIQUE_AND_REVISE], drafts)

    def score_and_select(rewrites):
        print(f"{layer_label(sdf_run_layouts.SCORE)} Score and gate")
        final = score_and_final_selection_layer.run(config, prompts_dir, folders[sdf_run_layouts.SCORE], final_dir, rewrites)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")
        print(f"=== Done. {len(final)} documents in {final_dir / 'sdf_corpus.jsonl'} ===")
        return final

    # In pipeline order. Each runner takes what the one before returned and returns what the next takes.
    runners = {
        sdf_run_layouts.DEAL_AND_SCREEN: deal_and_screen,
        sdf_run_layouts.PLAN_AND_DESCRIPTION_REWRITE: plan_and_rewrite_descriptions,
        sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION: select_considerations,
        sdf_run_layouts.DRAFT: draft_documents,
        sdf_run_layouts.CRITIQUE_AND_REVISE: critique_and_revise,
        sdf_run_layouts.SCORE: score_and_select,
    }
    four_layer_number_of_key = sdf_run_layouts.FOUR_NUMBERED_LAYERS.layer_number_of_key
    keys_to_run = [key for key in runners if layout.layer_number_of_key[key] >= start_layer]
    openai_check_layer = layout.layer_number_of_key[sdf_run_layouts.PLAN_AND_DESCRIPTION_REWRITE]
    if start_layer <= openai_check_layer:   # before the first paid call of the run, as the four-layer run did at its layer 1
        description_rewrite.require_openai_key_if_configured(config)

    layers_with_a_run = sorted({layout.layer_number_of_key[key] for key in keys_to_run})
    last_key_of_layer = {layout.layer_number_of_key[key]: key for key in keys_to_run}
    documents_from_the_step_before = None
    templates_checked_for_group = set()
    for key in keys_to_run:
        layer = layout.layer_number_of_key[key]
        group = four_layer_number_of_key[key]   # templates are checked a group at a time, before its first call
        if group not in templates_checked_for_group:
            templates_checked_for_group.add(group)
            for key_in_group in (k for k in keys_to_run if four_layer_number_of_key[k] == group):
                refuse_when_snapshot_lacks_templates(key_in_group)
        if documents_from_the_step_before is None and key != sdf_run_layouts.DEAL_AND_SCREEN:   # restarting here: read the stage files
            documents_from_the_step_before = documents_read_when_restarting_at(key, layout, reading, config, prompts_dir, folders)
        documents_from_the_step_before = runners[key](documents_from_the_step_before)
        if layer != layers_with_a_run[-1] and last_key_of_layer[layer] == key:
            print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    costs.report(run_dir / "cost_log.jsonl", title="Run cost (this invocation)",
                 n_done=config["sdf"]["n_prompts"], since=cost_start,
                 project_to=(config["sdf"].get("experiment") or {}).get("project_to"))
    if cost_start:
        costs.report(run_dir / "cost_log.jsonl", title="Run cost (total, incl. earlier attempts)",
                     n_done=config["sdf"]["n_prompts"])


if __name__ == "__main__":
    main()
