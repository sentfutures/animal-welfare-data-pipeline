#!/usr/bin/env python3
"""Drive DAD steps 2-3 on a fabricated mini run (see build_run.py).

Mirrors dad_pipeline/run.py's step calls on the run's frozen
inputs/config_override.yaml, with two experiment additions:

- --backend openrouter swaps shared.api.call_claude for the OpenRouter shim
  (or_backend.py), so a non-Anthropic model serves every response-side call.
- Every call's outcome is recorded to <run>/call_log.jsonl (stage, item_id,
  model, stop_reason, chars, attempts are in cost_log.jsonl) so refusal and
  truncation rates can be reported per arm afterwards.

Judging happens separately, on the repo's own audit with the base config
(the same claude-opus-5 judges that scored the original run):

    .venv/bin/python evals/audit_dad.py --input <run_dir> --judges --config config.yaml
"""

import argparse
import json
import sys
import threading
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import api, utils  # noqa: E402
from dad_pipeline import baseline, step2_responses, step3_rewrite  # noqa: E402


def install_recorder(run_dir: Path) -> None:
    """Wrap the active shared.api.call_claude so every call's stop_reason is
    persisted — the refusal-rate data this experiment exists to collect. The
    wrapper always requests the stop_reason and honors the caller's own
    return_stop_reason flag on the way out."""
    base = api.call_claude
    lock = threading.Lock()
    log_path = run_dir / "call_log.jsonl"

    def recorded(user_message, **kw):
        want_sr = kw.pop("return_stop_reason", False)
        text, stop_reason = base(user_message, return_stop_reason=True, **kw)
        rec = {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": kw.get("stage"),
            "item_id": kw.get("item_id"),
            "model": kw.get("model"),
            "stop_reason": stop_reason,
            "chars": len(text or ""),
        }
        with lock, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return (text, stop_reason) if want_sr else text

    api.call_claude = recorded


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--backend", choices=("anthropic", "openrouter"),
                        default="anthropic")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config_path = run_dir / "inputs" / "config_override.yaml"
    config = utils.load_config(str(config_path))
    prompts_dir = run_dir / "inputs" / "prompts"

    api.init(str(config_path), cost_log_path=run_dir / "cost_log.jsonl")
    if args.backend == "openrouter":
        import or_backend
        or_backend.install()
    install_recorder(run_dir)

    step2_dir = run_dir / "step2"
    step3_dir = run_dir / "step3"
    final_dir = run_dir / "final"
    for d in (step2_dir, step3_dir, final_dir):
        utils.ensure_dir(d)

    dilemmas = utils.load_jsonl(run_dir / "step1" / "dilemmas.jsonl")
    print(f"=== model rerun — {run_dir.name} ({len(dilemmas)} dilemmas, "
          f"backend {args.backend}, model {config['dad']['response_draft_model']}) ===")

    # All baseline records were copied in by build_run.py, so this returns them
    # with zero API calls — the control arm stays byte-identical to the source run.
    baselines = baseline.run(config, run_dir / "baseline", dilemmas)

    print("[Step 2] Generate responses (2a scope / 2a.5 select / 2b respond)")
    responses = step2_responses.run(config, prompts_dir, step2_dir, dilemmas, baselines)
    print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    print("[Step 3] Rewrite against the distilled principles")
    final = step3_rewrite.run(config, prompts_dir, step3_dir, final_dir, responses)
    print(f"  Running cost: ${api.get_total_cost():.4f}\n")

    print(f"=== Done. {len(final)} of {len(dilemmas)} examples produced a final "
          f"record ({run_dir / 'final' / 'dad_corpus.jsonl'}) ===")
    print("Judge next (base config = the original run's judges):\n"
          f"  .venv/bin/python evals/audit_dad.py --input {run_dir} --judges --config config.yaml")


if __name__ == "__main__":
    main()
