#!/usr/bin/env python3
"""Preference pipeline: generate two candidate responses per prompt for human A/B rating.

Each input prompt yields one pair record with a response from each of two
configured arms (config: pref.arms.a / pref.arms.b) — e.g. the plain model vs.
a candidate response spec, or spec v1 vs. spec v2. Which arm renders on which
side in the rating UI is fixed per pair by hashing the pair id, so raters stay
blind and reloads don't flip sides. Arms are resolved once at run creation and
frozen into inputs/arm_prompts.yaml so --resume replays the run's own arms.

Rate the pairs afterwards with:  streamlit run pref_pipeline/rate.py
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags

# Two independent samples from the bare model when config has no pref.arms.
_DEFAULT_ARMS = {
    "a": {"name": "baseline", "system_prompt": ""},
    "b": {"name": "candidate", "system_prompt": ""},
}


def _load_prompts(path: Path) -> list[dict]:
    """Load prompts from JSONL, accepting user_message (handwritten sets, DAD
    step-1 dilemmas) or prompt as the text field."""
    records = utils.load_jsonl(path)
    prompts = []
    for i, r in enumerate(records):
        text = (r.get("user_message") or r.get("prompt") or "").strip()
        if not text:
            print(f"  WARNING: record {i} has no user_message/prompt text, skipping.")
            continue
        pid = str(r.get("prompt_id") or r.get("scenario_id") or f"row{i:04d}")
        prompts.append({"prompt_id": pid, "user_message": text})
    return prompts


def _resolve_arms(config: dict, root: Path) -> dict:
    """Resolve the two arm definitions: inline system_prompt, or system_prompt_file
    (path relative to the repo root, e.g. a candidate response-spec doc)."""
    arms_cfg = (config.get("pref") or {}).get("arms") or _DEFAULT_ARMS
    arms = {}
    for key in ("a", "b"):
        if key not in arms_cfg:
            raise SystemExit(f"config pref.arms must define arm '{key}'")
        cfg = arms_cfg[key]
        system_prompt = cfg.get("system_prompt", "") or ""
        if cfg.get("system_prompt_file"):
            system_prompt = (root / cfg["system_prompt_file"]).read_text(encoding="utf-8")
        arms[key] = {
            "name": cfg.get("name", key),
            "system_prompt": system_prompt,
            "model": cfg.get("model"),
            "max_tokens": cfg.get("max_tokens"),
        }
    return arms


def _left_arm(pair_id: str, salt: str = "") -> str:
    """Deterministic blinded side assignment, stable across resumes. Salting with
    the run id varies the ordering across runs, so a rater doing two studies on
    the same prompt set doesn't see the same left/right layout each time."""
    return "a" if int(hashlib.md5(f"{salt}:{pair_id}".encode()).hexdigest(), 16) % 2 == 0 else "b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate response pairs for human preference rating.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompts", default=None,
                        help="JSONL of prompts (user_message/refined/prompt per record); "
                             "falls back to config pref.prompts_path.")
    parser.add_argument("--label", default="dev", help="Run label, e.g. dev or spec-v1-vs-plain.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    parser.add_argument("--run-id", default=None, help="Run to resume (with --resume; defaults to latest).")
    parser.add_argument("--limit", type=int, default=None, help="Only pair the first N prompts.")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    config = utils.load_config(args.config)
    root = Path(__file__).parent.parent
    # PIPELINE_OUTPUT_ROOT redirects all run output (used by the test suite)
    outputs_root = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", root / "outputs"))
    runs_root = outputs_root / "pref" / "runs"

    prompts_path = args.prompts or (config.get("pref") or {}).get("prompts_path")
    if not prompts_path:
        raise SystemExit("No prompts source: pass --prompts <file.jsonl> or set pref.prompts_path in config.")
    prompts = _load_prompts(Path(prompts_path))
    if not prompts:
        raise SystemExit(f"No usable prompts in {prompts_path}")
    if args.limit:
        prompts = prompts[: args.limit]

    if args.resume:
        run_dir = utils.resolve_run_dir(runs_root, args.run_id)
        utils.warn_if_backend_changed(run_dir, config)
    else:
        run_dir = utils.create_run_dir(runs_root, label=args.label, config=config)

    # Freeze the resolved arms into the run so --resume replays them even if
    # config.yaml or a referenced spec file changes afterwards.
    arms_path = run_dir / "inputs" / "arm_prompts.yaml"
    if arms_path.exists():
        with open(arms_path, encoding="utf-8") as f:
            arms = yaml.safe_load(f)
    else:
        arms = _resolve_arms(config, root)
        utils.ensure_dir(arms_path.parent)
        with open(arms_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(arms, f)

    api.init(args.config, cost_log_path=run_dir / "cost_log.jsonl")

    pairs_dir = utils.ensure_dir(run_dir / "pairs")
    pairs_path = pairs_dir / "pairs.jsonl"
    # Per-arm responses persist here as they succeed, so a failure on one arm
    # never discards (or re-bills on resume) its sibling's paid response.
    arm_cache_path = pairs_dir / "arm_responses.jsonl"
    checkpoint = utils.Checkpoint(pairs_dir / "_checkpoint.json")
    done_ids = {r["pair_id"] for r in utils.load_jsonl(pairs_path)}
    arm_cache = {(r["pair_id"], r["arm"]): r["response"]
                 for r in utils.load_jsonl(arm_cache_path)}

    print(f"=== Preference pipeline — run {run_dir.name} ===")
    print(f"Prompts: {len(prompts)} from {prompts_path}")
    print(f"Arms: a={arms['a']['name']}, b={arms['b']['name']}")

    generated = failed = 0
    for p in prompts:
        pair_id = f"pair_{p['prompt_id']}"
        if pair_id in done_ids or checkpoint.is_done(pair_id):
            continue

        print(f"  Generating pair for {p['prompt_id'][:40]}...")
        responses = {}
        for key in ("a", "b"):
            if (pair_id, key) in arm_cache:
                responses[key] = arm_cache[(pair_id, key)]
                continue
            arm = arms[key]
            try:
                text, stop_reason = api.call_claude(
                    user_message=p["user_message"],
                    system_prompt=arm.get("system_prompt") or "",
                    model=arm.get("model"),
                    max_tokens=arm.get("max_tokens"),
                    return_stop_reason=True,
                    stage=f"pref_arm_{key}",
                )
            except Exception as e:  # keep the sibling arm's paid work; move on
                print(f"    Arm {key} ({arm['name']}) failed for {pair_id}: {e} — "
                      "pair deferred; --resume will retry only this arm.")
                failed += 1
                break
            text = text.strip()
            # A truncated or empty response must not become an A/B candidate.
            if not text or stop_reason == "max_tokens":
                why = "truncated at max_tokens" if stop_reason == "max_tokens" else "empty"
                print(f"    Arm {key} ({arm['name']}) {why} for {pair_id} — "
                      "pair deferred; --resume will retry only this arm.")
                failed += 1
                break
            responses[key] = text
            arm_cache[(pair_id, key)] = text
            utils.append_jsonl({"pair_id": pair_id, "arm": key, "response": text},
                               arm_cache_path)
        if len(responses) < 2:
            continue

        record = {
            "pair_id": pair_id,
            "prompt_id": p["prompt_id"],
            "user_message": p["user_message"],
            "arm_names": {k: arms[k]["name"] for k in ("a", "b")},
            "response_a": responses["a"],
            "response_b": responses["b"],
            "left_arm": _left_arm(pair_id, salt=run_dir.name),
        }
        utils.append_jsonl(record, pairs_path)
        checkpoint.mark_done(pair_id)
        done_ids.add(pair_id)
        generated += 1

    print(f"  Generated {generated} new pairs ({len(done_ids)} total) in {pairs_path}")
    if failed:
        print(f"  {failed} pair(s) deferred on a failed/unusable arm — rerun with --resume "
              "to retry just the missing arms.")
    print(f"Total API cost this session: ${api.get_total_cost():.4f}")
    print("\nNext: streamlit run pref_pipeline/rate.py")


if __name__ == "__main__":
    main()
