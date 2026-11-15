#!/usr/bin/env python3
"""SDF matrix pipeline orchestrator: five layers (deal variables and screen; description write and description rewrite; consideration scoring and selection;
draft, critique and revise; score and final selection), each a folder in ``sdf_pipeline/`` with a driver ``run_layer.py`` and one file per step.
The layouts are in ``shared/sdf_run_layouts.py``; a run keeps the layout it was created in."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, costs, sdf_run_layouts, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags
from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.layouts_that_can_be_resumed import layout_of_run_that_can_be_resumed
from sdf_pipeline.layer1_deal_variables_and_screen import run_layer as deal_variables_and_screen_layer
from sdf_pipeline.layer2_description_write_and_description_rewrite import description_rewrite, description_write
from sdf_pipeline.layer3_consideration_scoring_and_selection import run_layer as consideration_scoring_and_selection_layer
from sdf_pipeline.layer4_draft_critique_and_revise import draft, run_layer as draft_critique_and_revise_layer
from sdf_pipeline.layer5_score_and_final_selection import run_layer as score_and_final_selection_layer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SDF matrix pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    parser.add_argument("--layer", type=int, default=1,
                        help="Start from this layer. A run in the five-layer layout (every new run) has layers 1-5: deal variables and "
                             "screen, description write and description rewrite, consideration scoring and selection, draft/critique/"
                             "revise, score and final selection. A run made in the four-layer layout keeps its 1-4: "
                             "everything up to the considerations, draft, critique and revise, score. Default 1.")
    parser.add_argument("--label", default="dev", help="Run label, e.g. dev or full-scale.")
    parser.add_argument("--run-id", default=None, help="Run to resume (with --resume; defaults to latest).")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    config = utils.load_config(args.config)

    root = Path(__file__).parent.parent
    # PIPELINE_OUTPUT_ROOT redirects all run output (used by the test suite)
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", root / "outputs"))
    runs_root = outputs_root / "sdf" / "runs"

    # The layout is decided once, when the run is created, and kept: a new run is in the five-layer layout; a
    # resumed run keeps the one it was made in.
    layout = sdf_run_layouts.FIVE_NAMED_LAYERS
    if args.resume:
        run_dir = utils.resolve_run_dir(runs_root, args.run_id)
        layout = layout_of_run_that_can_be_resumed(run_dir)
    if args.layer not in layout.layer_numbers:
        parser.error(f"--layer must be {layout.layer_numbers[0]}-{layout.layer_numbers[-1]}, got {args.layer}.")

    if args.resume:
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
            extra_manifest_fields={sdf_run_layouts.LAYOUT_RECORDED_IN_MANIFEST_KEY: layout.name},
        )

    with sdf_run_layouts.using_layout(layout):
        run_in_layout(args, config, layout, run_dir, root)


def documents_read_when_starting_at(folder_key: str, config: dict, prompts_dir: Path, folders: dict, layout):
    """What the layer's first runner is given when the run starts there: the stage files of the layers before it."""
    if folder_key == sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE:
        if not config["sdf"].get("screen_enabled", True) or deal_variables_and_screen_layer.load_screen_prompts(prompts_dir) is None:
            return None   # the screen is off: the description write deals its own hands
        accepted_path = folders[sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN] / deal_variables_and_screen_layer.SCREENED_HANDS_FILENAME
        if not accepted_path.is_file():
            raise SystemExit(
                f"Cannot start at layer {layout.layer_number_of_key[folder_key]}: the screen has not finished, so {accepted_path} "
                f"does not exist. Run layer {layout.layer_number_of_key[sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN]} first.")
        return utils.load_jsonl(accepted_path)
    plan_dir = folders[sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE]
    if folder_key in (sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION, sdf_run_layouts.DRAFT):
        plans = description_rewrite.load_rewritten_plans(config, plan_dir, utils.load_jsonl(plan_dir / "plans.jsonl"))
        if folder_key == sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION:
            return plans
        return consideration_scoring_and_selection_layer.attach_selections(
            config, prompts_dir, folders[sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION], plans)
    if folder_key == sdf_run_layouts.CRITIQUE_AND_REVISE:
        return utils.load_jsonl(folders[sdf_run_layouts.DRAFT] / "drafts.jsonl")
    return utils.load_jsonl(folders[sdf_run_layouts.CRITIQUE_AND_REVISE] / "rewrites.jsonl")


def run_in_layout(args, config: dict, layout: sdf_run_layouts.SdfRunLayout, run_dir: Path, root: Path) -> None:
    """Run the layers of ``layout`` from ``args.layer`` on ``run_dir`` (the tags and folders of ``layout`` are active)."""
    # Read templates from the run's frozen snapshot so prompts stay reproducible
    # (and --resume replays the run's own templates, not the repo's current ones).
    prompts_dir = run_dir / "inputs" / "prompts"
    if not prompts_dir.is_dir():
        prompts_dir = root / "prompts" / "sdf"
        print("WARNING: run has no inputs/ snapshot (pre-snapshot run); using live prompts/.")

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

    def layer_label(folder_key: str) -> str:
        return f"[Layer {layout.layer_number_of_key[folder_key]}]"

    def deal_variables_and_screen(unused_documents):
        print(f"{layer_label(sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN)} Deal hands and screen them for extreme implausibility (with redeals)")
        return deal_variables_and_screen_layer.run(config, prompts_dir, folders[sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN])

    def write_and_rewrite_descriptions(hands):
        label = layer_label(sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE)
        directory = folders[sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE]
        print(f"{label} Compose prompts + write descriptions")
        plans = description_write.run(config, prompts_dir, directory, hands)
        print(f"{label} Rewrite descriptions for plausibility and conformance")
        return description_rewrite.run(config, prompts_dir, directory, plans)

    def select_considerations(plans):
        print(f"{layer_label(sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION)} Select considerations for each surviving specification")
        return consideration_scoring_and_selection_layer.run(
            config, prompts_dir, folders[sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION], plans)

    def draft_documents(plans):
        print(f"{layer_label(sdf_run_layouts.DRAFT)} Draft documents")
        return draft.run(config, prompts_dir, folders[sdf_run_layouts.DRAFT], plans)

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
        sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN: deal_variables_and_screen,
        sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE: write_and_rewrite_descriptions,
        sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION: select_considerations,
        sdf_run_layouts.DRAFT: draft_documents,
        sdf_run_layouts.CRITIQUE_AND_REVISE: critique_and_revise,
        sdf_run_layouts.SCORE: score_and_select,
    }
    keys_to_run = [key for key in runners if layout.layer_number_of_key[key] >= args.layer]
    if args.layer <= layout.layer_number_of_key[sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE]:   # before the first paid call of the run
        description_rewrite.require_openai_key_if_configured(config)

    last_layer = layout.layer_number_of_key[keys_to_run[-1]]
    last_key_of_layer = {layout.layer_number_of_key[key]: key for key in keys_to_run}
    documents_from_the_step_before = None
    for key in keys_to_run:
        layer = layout.layer_number_of_key[key]
        if documents_from_the_step_before is None and key != sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN:   # restarting here: read the stage files
            documents_from_the_step_before = documents_read_when_starting_at(key, config, prompts_dir, folders, layout)
        documents_from_the_step_before = runners[key](documents_from_the_step_before)
        if layer != last_layer and last_key_of_layer[layer] == key:
            print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    costs.report(run_dir / "cost_log.jsonl", title="Run cost (this invocation)",
                 n_done=config["sdf"]["n_prompts"], since=cost_start,
                 project_to=(config["sdf"].get("experiment") or {}).get("project_to"))
    if cost_start:
        costs.report(run_dir / "cost_log.jsonl", title="Run cost (total, incl. earlier attempts)",
                     n_done=config["sdf"]["n_prompts"])


if __name__ == "__main__":
    main()
