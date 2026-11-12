#!/usr/bin/env python3
"""DAD pipeline orchestrator. Runs steps 1-3 with checkpointing.

Steps: 1 dilemma prompts (1a scenario deal + plan: a stratified variable
combination dealt per example from prompts/dad/variables.txt, then one plan
call writes its scenario description; 1b first attempt drafted to fit each
scenario; 1c latent-welfare rewrite) → 2 responses (2a scope the case from
the user's message; 2a.5 flag
which reasoning-library trigger conditions fire, in a dedicated selection
call; 2b respond over the scope plus the triggered library rows) → 3 rewrite
against the distilled constitution principles (the alignment-critical pass).

A baseline control arm (dad_pipeline/baseline.py) rides along with step 2:
one plain-model call per dilemma, no system prompt — viewer comparison data,
never a training input. Toggled by dad.baseline.enabled.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from shared.api_keys import (add_api_key_source_flags, api_key_source_flags_for_subprocess,
                             apply_api_key_source_flags)
from dad_pipeline import (
    baseline,
    profiles,
    step1_dilemmas,
    step2_responses,
    step3_rewrite,
)


def auto_evals_enabled(config: dict) -> bool:
    """The post-run evals fire unless dad.evals.auto is explicitly false — a
    config without the block (older configs, pared-down dev configs) gets them
    by default (same convention as the baseline arm)."""
    return bool((config["dad"].get("evals") or {}).get("auto", True))


def run_auto_evals(run_dir: Path, config_path: str, root: Path) -> None:
    """Standard post-run evals: the corpus audit with the paid --judges pass,
    then the embedding diversity audit. Each runs as a subprocess — the eval
    scripts call api.init pointed at the global cost log, which would clobber
    this process's run-scoped cost-log state if imported. The corpus is
    already complete before these start, so a failing eval (missing
    GEMINI_API_KEY, network blip) warns and never fails the run."""
    config_abs = str(Path(config_path).resolve())
    jobs = [
        ("corpus audit + judges pass",
         [sys.executable, str(root / "evals" / "audit_dad.py"),
          "--input", str(run_dir), "--judges", "--config", config_abs, *api_key_source_flags_for_subprocess()]),
        ("semantic diversity",
         [sys.executable, str(root / "evals" / "diversity.py"),
          "--input", str(run_dir), "--config", config_abs, *api_key_source_flags_for_subprocess()]),
    ]
    for name, cmd in jobs:
        print(f"[Evals] Running {name}...")
        try:
            result = subprocess.run(cmd, cwd=root)
        except OSError as exc:
            print(f"  WARNING: {name} failed to launch ({exc}).")
            continue
        if result.returncode != 0:
            print(f"  WARNING: {name} exited {result.returncode}; the run itself is "
                  f"complete — re-run manually: {' '.join(cmd[1:])}")
        print()


def _refuse_profile_change(run_dir: Path, name: str) -> None:
    """A run is one profile for life: resuming it under another would mix two
    domains' prompts and tables in one corpus. Runs that predate profiles are
    animal runs."""
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    was = profiles.name_from_config(manifest.get("config") or {})
    if was != name:
        raise SystemExit(f"{run_dir.name} was made with DAD profile {was!r}; resuming it "
                         f"with {name!r} would mix profiles. Use a config with "
                         f"dad.profile: {was}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DAD pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    parser.add_argument("--step", type=int, default=1, choices=(1, 2, 3),
                        help="Start from this step (1-3).")
    parser.add_argument("--stop-after", type=int, default=3, dest="stop_after", choices=(1, 2, 3),
                        help="Stop after this step (1-3); e.g. --stop-after 1 runs only prompt generation.")
    parser.add_argument("--label", default="dev", help="Run label, e.g. dev or full-scale.")
    parser.add_argument("--run-id", default=None, help="Run to resume (with --resume; defaults to latest).")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)
    if args.stop_after < args.step:
        parser.error(f"--stop-after {args.stop_after} is before --step {args.step} — nothing would run.")

    config = utils.load_config(args.config)

    root = Path(__file__).parent.parent
    # The content profile (dad.profile, default "animal") picks the prompt
    # folder, the principles file and where runs live; the code is shared.
    profile = profiles.use(profiles.name_from_config(config))
    # PIPELINE_OUTPUT_ROOT redirects all run output (used by the test suite)
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", root / "outputs"))
    runs_root = outputs_root / profile.OUTPUT_SUBDIR / "runs"

    if args.resume:
        run_dir = utils.resolve_run_dir(runs_root, args.run_id)
        utils.warn_if_backend_changed(run_dir, config)
        _refuse_profile_change(run_dir, profile.NAME)
    else:
        run_dir = utils.create_run_dir(
            runs_root,
            label=args.label,
            config=config,
            snapshot_dirs={
                "prompts": root / "prompts" / profile.PROMPTS_SUBDIR,
                "constitution": root / "constitution",
            },
        )

    # Read templates from the run's frozen snapshot so prompts stay reproducible
    # (and --resume replays the run's own templates, not the repo's current ones).
    prompts_dir = run_dir / "inputs" / "prompts"
    if not prompts_dir.is_dir():
        prompts_dir = root / "prompts" / profile.PROMPTS_SUBDIR
        print("WARNING: run has no inputs/ snapshot (pre-snapshot run); using live prompts/.")

    api.init(args.config, cost_log_path=run_dir / "cost_log.jsonl")

    step_dirs = {i: run_dir / f"step{i}" for i in range(1, 4)}
    final_dir = run_dir / "final"
    for d in step_dirs.values():
        utils.ensure_dir(d)
    utils.ensure_dir(final_dir)

    start_step = args.step
    stop_after = args.stop_after

    print(f"=== DAD Pipeline — run {run_dir.name} ===")
    print(f"Outputs: {run_dir}")

    dilemmas = responses = None

    if start_step <= 1 <= stop_after:
        print("[Step 1] Scenario deal + plan (1a) and first-attempt drafts (1b)")
        dilemmas = step1_dilemmas.run(config, prompts_dir, step_dirs[1])
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    # Baseline: one plain-model call per dilemma, no system prompt. Doubles as
    # the viewer's control arm and as the advisory "first take" 2b reads
    # (reference notes, never trained on). Optional: with the stage disabled,
    # 2b's first-take slot renders empty and drafting proceeds unaided.
    baselines = None
    if baseline.enabled(config) and start_step <= 2 <= stop_after:
        if dilemmas is None:
            dilemmas = utils.load_jsonl(step_dirs[1] / "dilemmas.jsonl")
        print("[Baseline] Plain-model first takes / control responses (no system prompt)")
        baselines = baseline.run(config, run_dir / "baseline", dilemmas)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    if start_step <= 2 <= stop_after:
        if dilemmas is None:
            dilemmas = utils.load_jsonl(step_dirs[1] / "dilemmas.jsonl")
        print("[Step 2] Generate responses from the reasoning library")
        responses = step2_responses.run(config, prompts_dir, step_dirs[2], dilemmas, baselines)
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    if start_step <= 3 <= stop_after:
        if responses is None:
            # Resume: take all step-2 responses. `kept` is legacy (the ruthless
            # judge that set it false was removed); default to kept for old runs.
            all_responses = utils.load_jsonl(step_dirs[2] / "responses.jsonl")
            responses = [r for r in all_responses if r.get("kept", True)]
        print("[Step 3] Rewrite against the distilled principles")
        final = step3_rewrite.run(
            config, prompts_dir, step_dirs[3], final_dir, responses
        )
        print(f"  Running cost: ${api.get_total_cost():.4f}\n")
        print(f"=== Done. {len(final)} records in {final_dir / 'dad_corpus.jsonl'} ===")

        # Standard evals ride at the end of every full run (dad.evals.auto).
        # Partial runs (--stop-after 1/2) skip them: no final corpus to audit.
        if final and auto_evals_enabled(config):
            run_auto_evals(run_dir, args.config, root)

    total = api.get_total_cost()
    print(f"Total API cost this session: ${total:.4f}")


if __name__ == "__main__":
    main()
