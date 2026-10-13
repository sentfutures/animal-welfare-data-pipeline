#!/usr/bin/env python3
"""Fabricate a mini DAD run for the model-rerun experiment.

Takes a subset of an existing run's step-1 dilemmas (the selection picked from
its welfare-impact scores), copies the matching plain-baseline responses so the
control arm is byte-identical to the original run's (zero baseline calls, and
the judges re-score the same control text), snapshots the ORIGINAL run's own
prompt/constitution inputs so templates match exactly, and freezes the arm's
model overrides into inputs/config_override.yaml for drive.py and api.init.

The run label starts with `local_` so the run dir stays untracked (.gitignore).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from shared import utils  # noqa: E402
from dad_pipeline.id_registry import prompt_key, prompt_keys  # noqa: E402

SOURCE_RUN_DEFAULT = "outputs/dad/runs/2026-07-29_12-26_archetype200"


def selection_ids(selection: dict) -> list[str]:
    return [row["prompt_id"]
            for bucket in ("low", "mid", "high")
            for row in selection["buckets"][bucket]]


def arm_config(base_config: dict, model: str) -> dict:
    """The arm's config: every response-side stage on the target model, no
    cross-model refusal fallback (a refusal is a measurement here, and a
    fallback to a different model would contaminate the arm), auto-evals off
    (the judge pass is run separately, on the base config's judge model)."""
    config = json.loads(json.dumps(base_config))  # deep copy
    config["model"] = model
    dad = config["dad"]
    for knob in ("response_scope_model", "response_select_model",
                 "response_draft_model", "constitution_rewrite_model"):
        dad[knob] = model
    dad["scope_refusal_fallback_model"] = None
    dad.setdefault("baseline", {})["model"] = model  # unused: baseline is copied in
    dad.setdefault("evals", {})["auto"] = False
    return config


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(ROOT / SOURCE_RUN_DEFAULT))
    parser.add_argument("--selection",
                        default=str(Path(__file__).parent / "selection.json"))
    parser.add_argument("--label", required=True,
                        help="Run label; must contain 'local_' to stay untracked, "
                             "e.g. local_rerun-opus55")
    parser.add_argument("--model", required=True,
                        help="Model id (Anthropic) or slug (OpenRouter) for every "
                             "response-side stage")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args(argv)
    if "local_" not in args.label:
        raise SystemExit("--label must contain 'local_' so the run dir stays untracked")

    source = Path(args.source_run)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    wanted = set(selection_ids(selection))

    dilemmas = [d for d in utils.load_jsonl(source / "step1" / "dilemmas.jsonl")
                if d.get("prompt_id") in wanted or d.get("prompt_gid") in wanted]
    missing = wanted - {d.get("prompt_id") for d in dilemmas} \
                     - {d.get("prompt_gid") for d in dilemmas}
    if missing:
        raise SystemExit(f"selection ids not found in source dilemmas: {sorted(missing)}")

    # The source run's baseline records predate prompt gids (they carry only the
    # legacy prompt_id), while the dilemmas carry prompt_gid — which is what the
    # current baseline/step-2 code keys on. Join through every id each side
    # carries, and stamp the copied record with the dilemma's current key so the
    # baseline stage sees it as done and 2b's first-take lookup hits.
    source_baselines = utils.load_jsonl(source / "baseline" / "baseline_responses.jsonl")
    by_any_key = {}
    for b in source_baselines:
        for k in prompt_keys(b):
            by_any_key[k] = b
    baselines = []
    for d in dilemmas:
        b = next((by_any_key[k] for k in prompt_keys(d) if k in by_any_key), None)
        if b is not None:
            baselines.append({**b, "prompt_gid": prompt_key(d)})

    base_config = utils.load_config(args.config)
    config = arm_config(base_config, args.model)

    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", ROOT / "outputs"))
    run_dir = utils.create_run_dir(
        outputs_root / "dad" / "runs",
        label=args.label,
        config=config,
        snapshot_dirs={
            "prompts": source / "inputs" / "prompts",
            "constitution": source / "inputs" / "constitution",
        },
    )

    utils.ensure_dir(run_dir / "step1")
    with open(run_dir / "step1" / "dilemmas.jsonl", "w", encoding="utf-8") as f:
        for d in dilemmas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    utils.ensure_dir(run_dir / "baseline")
    with open(run_dir / "baseline" / "baseline_responses.jsonl", "w", encoding="utf-8") as f:
        for b in baselines:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    with open(run_dir / "inputs" / "config_override.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"run_dir: {run_dir}")
    print(f"dilemmas: {len(dilemmas)} (selection {len(wanted)})")
    print(f"baselines copied: {len(baselines)}"
          + ("" if len(baselines) == len(dilemmas) else
             f"  WARNING: {len(dilemmas) - len(baselines)} selected examples have no "
             "baseline in the source run — their 2b first-take slot renders empty "
             "and the judges score fewer plain-arm records"))
    return run_dir


if __name__ == "__main__":
    main()
