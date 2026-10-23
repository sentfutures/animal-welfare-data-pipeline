#!/usr/bin/env python3
"""Rerun only the layer 1 considerations selector on plans stored by an earlier SDF pipeline run.

Uses the CURRENT selector code and CURRENT prompt files of this worktree
(prompts/sdf/layer1_select_considerations.txt and prompts/sdf/considerations.txt), not the
source run's prompt snapshot, so it tests a changed selector on the exact specifications
(post description rewrite) that the original selector saw. The source run is only read.

Everything is written into one new run folder:
    <output_root>/considerations_selector_rerun_on_<label>_<YYYY-MM-DD_HH-MM>/
        launch_record_<timestamp>.json          source run, git commit, model, workers
        prompt_snapshot/                        copies of the two prompt files used
        selector_outputs/                       consideration_selections.jsonl (checkpoint source of truth),
                                                _select_considerations_checkpoint.json,
                                                consideration_coverage.json
        cost_log.jsonl, model_calls.jsonl       cost and full call records
        comparison_table_<timestamp>.md         per entry: target, achieved, rejected over cap, original achieved

Example:
    python scripts/rerun_considerations_selector_on_stored_plans.py \\
      --source-run-dir <an earlier SDF pipeline run folder> \\
      --config <the config file that run used> \\
      --output-root <folder for the rerun outputs> \\
      --run-label opus_5_5_smoke_descriptions [--dry-run]
    Resume an interrupted rerun:  ... --resume <existing rerun folder>

The API key comes only from the .env one folder above this worktree (loaded with override, so an OS-environment key
never wins); the script refuses to run without it. --dry-run makes no calls and needs no key.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE_ROOT))

DOTENV_PATH = WORKTREE_ROOT.parent / ".env"   # the .env one folder above this worktree, which holds this project's key
PROMPT_FILENAMES = ("layer1_select_considerations.txt", "considerations.txt", "variables.txt")   # the catalog resolves [id]s against variables.txt
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
LAUNCH_RECORD_PREFIX = "launch_record_"


def load_project_api_key_or_exit() -> None:
    """Load ANTHROPIC_API_KEY from the .env one folder above this worktree, overriding the OS environment."""
    from dotenv import dotenv_values, load_dotenv
    if not DOTENV_PATH.is_file():
        sys.exit(f"ERROR: project .env not found at {DOTENV_PATH}. Not falling back to the OS environment.")
    if not dotenv_values(DOTENV_PATH).get("ANTHROPIC_API_KEY"):
        sys.exit(f"ERROR: {DOTENV_PATH} has no ANTHROPIC_API_KEY. Not falling back to the OS environment.")
    load_dotenv(DOTENV_PATH, override=True)
    print(f"API key source: ANTHROPIC_API_KEY from {DOTENV_PATH} (value not printed)")


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(WORKTREE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_comparison_table(report: dict, original_report: dict | None, total_cost: float, path: Path) -> str:
    original_achieved = {e["label"]: e["achieved"] for e in (original_report or {}).get("entries", [])}
    lines = [
        "| List | Entry | Target | Achieved | Rejected over cap | Original run achieved |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report["entries"]:
        original = original_achieved.get(entry["label"], "n/a")
        lines.append(f"| {entry['list']} | {entry['label']} | {entry['target_count']} | {entry['achieved']} "
                     f"| {entry.get('rejected_over_cap', 0)} | {original} |")
    table = "\n".join(lines)
    text = (f"# Considerations selector rerun: comparison with the original run\n\n"
            f"Eligible documents: {report['eligible_documents']}\n\n{table}\n\n"
            f"Total cost of this rerun (from the cost log): ${total_cost:.4f}\n")
    path.write_text(text, encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run-dir", type=Path, help="Earlier SDF run directory holding layer1/ (read only).")
    parser.add_argument("--config", type=Path, help="Config yaml the source run used.")
    parser.add_argument("--output-root", type=Path, help="Folder in which the new rerun folder is created.")
    parser.add_argument("--run-label", default="stored_plans", help="Descriptive label used in the folder name.")
    parser.add_argument("--resume", type=Path, default=None, help="Existing rerun folder to continue.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan counts, targets and the first prompt; no calls.")
    args = parser.parse_args()

    resume_dir = args.resume.resolve() if args.resume else None
    if resume_dir:
        record_files = sorted(resume_dir.glob(f"{LAUNCH_RECORD_PREFIX}*.json"))
        if not record_files:
            sys.exit(f"ERROR: no launch record in {resume_dir}; cannot resume.")
        launch_record = json.loads(record_files[0].read_text(encoding="utf-8"))
        source_run_dir = Path(launch_record["source_run_dir"])
        config_path = Path(launch_record["config_path"])
        run_timestamp = launch_record["run_timestamp"]
        run_dir = resume_dir
    else:
        if not (args.source_run_dir and args.config and args.output_root):
            parser.error("--source-run-dir, --config and --output-root are required unless --resume is given.")
        source_run_dir, config_path = args.source_run_dir.resolve(), args.config.resolve()
        run_timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        run_dir = args.output_root.resolve() / f"considerations_selector_rerun_on_{args.run_label}_{run_timestamp}"

    if not args.dry_run:
        load_project_api_key_or_exit()   # must precede importing shared.api, which calls load_dotenv() without override

    from shared import api, utils
    from sdf_pipeline import layer1_description_rewrite, layer1_select_considerations as selector
    from sdf_pipeline.considerations_catalog import load_considerations

    source_layer1_dir = utils.sdf_stage_dir(source_run_dir, 1)
    config = utils.load_config(str(config_path))
    plans = layer1_description_rewrite.load_rewritten_plans(
        config, source_layer1_dir, utils.load_jsonl(source_layer1_dir / "plans.jsonl"))

    prompt_snapshot_dir = run_dir / "prompt_snapshot"
    selector_output_dir = run_dir / "selector_outputs"

    if args.dry_run:
        prompts_dir = prompt_snapshot_dir if prompt_snapshot_dir.is_dir() else WORKTREE_ROOT / "prompts" / "sdf"
        catalog = load_considerations(prompts_dir)
        eligible = selector.eligible_plans(config, catalog, plans)
        tracker = selector._build_tracker(config, catalog, plans)
        print(f"DRY RUN (no API calls). Source run: {source_layer1_dir}")
        print(f"Plans loaded: {len(plans)}; eligible documents: {len(eligible)}; "
              f"digital-mind documents: {tracker.digital_count}")
        print(f"Model: {config['sdf'].get('select_model') or config['sdf'].get('plan_model') or config['model']}; "
              f"workers: {config.get('workers')}; max passes: {config['sdf'].get('max_selection_passes', 3)}")
        print("Targets per entry:")
        for entry in tracker.coverage_report()["entries"]:
            print(f"  [{entry['list']}] {entry['label']}: target {entry['target_count']} "
                  f"(share {entry['target_share']})")
        first_id = selector.seeded_document_order(
            [p["prompt_id"] for p in eligible], config["sdf"].get("seed", 0))[0]
        first_plan = next(p for p in eligible if p["prompt_id"] == first_id)
        facts = selector.DocumentFacts(
            document_description=first_plan["description"],
            document_type=first_plan["variables"].get("document_type", ""),
            centrality=first_plan["variables"].get("centrality", ""),
            sentient_category=first_plan["variables"].get("sentient_category", ""))
        offered = {name: tracker.offered_entries(first_id, name) for name in selector.LIST_NAMES}
        system, user = selector.build_selection_prompt(
            selector.load_selection_template(prompts_dir), facts, offered)
        print(f"\n--- First document ({first_id}) rendered selector prompt: system ---\n{system}")
        print(f"\n--- user ---\n{user}")
        print("\n--- Answer schema ---")
        print(json.dumps(selector.build_selection_schema(offered), indent=2))
        return

    # Real run: snapshot the current prompts once (a resume keeps the original snapshot).
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume_dir:
        prompt_snapshot_dir.mkdir(exist_ok=True)
        for filename in PROMPT_FILENAMES:
            shutil.copy2(WORKTREE_ROOT / "prompts" / "sdf" / filename, prompt_snapshot_dir / filename)
        launch_record = {
            "source_run_dir": str(source_run_dir),
            "config_path": str(config_path),
            "git_commit": current_git_commit(),
            "model": config["sdf"].get("select_model") or config["sdf"].get("plan_model") or config["model"],
            "workers": config.get("workers"),
            "seed": config["sdf"].get("seed"),
            "run_timestamp": run_timestamp,
            "launched_at": datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / f"{LAUNCH_RECORD_PREFIX}{run_timestamp}.json").write_text(
            json.dumps(launch_record, indent=2), encoding="utf-8")
    utils.ensure_dir(selector_output_dir)

    api.init(str(config_path), cost_log_path=run_dir / "cost_log.jsonl")
    print(f"=== Considerations selector rerun -> {run_dir} ===")
    selector.run(config, prompt_snapshot_dir, selector_output_dir, plans)

    report = json.loads((selector_output_dir / selector.COVERAGE_FILENAME).read_text(encoding="utf-8"))
    original_path = source_layer1_dir / selector.COVERAGE_FILENAME
    original_report = json.loads(original_path.read_text(encoding="utf-8")) if original_path.is_file() else None
    total_cost = api.get_total_cost()
    table = write_comparison_table(report, original_report, total_cost,
                                   run_dir / f"comparison_table_{datetime.now().strftime(TIMESTAMP_FORMAT)}.md")
    print("\n" + table)
    print(f"\nTotal cost of this rerun: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
