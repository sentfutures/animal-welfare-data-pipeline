#!/usr/bin/env python3
"""SDF matrix pipeline orchestrator: plan, draft, rewrite, score.

Layer 1 is a single stage: deterministic composition of the prompt matrix
(offline) followed by one plan call per document. Layers 2-4 draft, rewrite,
and score/gate.

The layers were renumbered from the old 1-2/3/4/5 scheme (in which composition
and planning were counted as two layers) to today's 1/2/3/4. --layer takes the
new numbers; 5 is rejected rather than remapped, because every old number from
3 up now names a DIFFERENT stage and a silent remap would resume a paid run at
the wrong one.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from sdf_pipeline import (
    layer1_plan,
    layer2_draft,
    layer3_rewrite,
    layer4_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SDF matrix pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    parser.add_argument("--layer", type=int, default=1,
                        help="Start from this layer (1-4: plan, draft, rewrite, score).")
    parser.add_argument("--label", default="dev", help="Run label, e.g. dev or full-scale.")
    parser.add_argument("--run-id", default=None, help="Run to resume (with --resume; defaults to latest).")
    args = parser.parse_args()

    if not 1 <= args.layer <= 4:
        parser.error(
            f"--layer must be 1-4, got {args.layer}. The layers were renumbered: the old "
            "1-2/3/4/5 (compose+plan / draft / rewrite / score) are now 1/2/3/4. Old 3 "
            "(draft) is now 2, old 4 (rewrite) is now 3, old 5 (score) is now 4."
        )

    config = utils.load_config(args.config)

    root = Path(__file__).parent.parent
    # PIPELINE_OUTPUT_ROOT redirects all run output (used by the test suite)
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", root / "outputs"))
    runs_root = outputs_root / "sdf" / "runs"

    if args.resume:
        run_dir = utils.resolve_run_dir(runs_root, args.run_id)
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
        )

    # Read templates from the run's frozen snapshot so prompts stay reproducible
    # (and --resume replays the run's own templates, not the repo's current ones).
    prompts_dir = run_dir / "inputs" / "prompts"
    if not prompts_dir.is_dir():
        prompts_dir = root / "prompts" / "sdf"
        print("WARNING: run has no inputs/ snapshot (pre-snapshot run); using live prompts/.")

    api.init(args.config, cost_log_path=run_dir / "cost_log.jsonl")

    # Resuming a run made before the renumber has to write beside that run's own
    # stage dirs, not create a second set under the new names.
    plan_dir = utils.sdf_stage_dir(run_dir, 1)
    layer_dirs = {n: utils.sdf_stage_dir(run_dir, n) for n in (2, 3, 4)}
    final_dir = run_dir / "final"
    for d in [plan_dir, *layer_dirs.values(), final_dir]:
        utils.ensure_dir(d)

    start_layer = args.layer

    print(f"=== SDF Matrix Pipeline — run {run_dir.name} ===")
    print(f"Outputs: {run_dir}")

    plans = drafts = rewrites = None

    if start_layer <= 1:
        print("[Layer 1] Compose matrix + plan documents")
        plans = layer1_plan.run(config, prompts_dir, plan_dir)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    if start_layer <= 2:
        if plans is None:
            plans = utils.load_jsonl(plan_dir / "plans.jsonl")
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

    print(f"Total API cost this session: ${api.get_total_cost():.4f}")


if __name__ == "__main__":
    main()
