#!/usr/bin/env python3
"""Run the repo's own judge pass (evals/audit_dad.py::audit_judges — delivery
quality + welfare impact + composite, pipeline arm vs plain baseline) on a mini
run under a CHOSEN judge model and backend.

Results go to audit/judge_<name>.json rather than audit_report.json, so several
judges can score the same run side by side — the cross-judge bias design: each
generation arm judged once by a Claude-family judge and once by a GPT-family
judge, through the identical rubric prompts.

The pass's calls are logged to cost_log_judge_<name>.jsonl in the run dir, and
the audit's {delivery,impact}_failures.jsonl files are renamed per judge so a
second pass doesn't overwrite the first's evidence.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "evals"))

from shared import api, utils  # noqa: E402
import audit_dad  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True,
                        help="Judge label, e.g. opus55 or gpt61sol — names the "
                             "output file audit/judge_<name>.json")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--backend", choices=("anthropic", "openrouter"),
                        default="anthropic")
    parser.add_argument("--reasoning-effort", default=None,
                        help="openrouter only: reasoning effort for the judge "
                             "model (e.g. medium)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = utils.load_config(str(ROOT / "config.yaml"))
    config.setdefault("evals", {})["judge_model"] = args.judge_model

    api.init(str(ROOT / "config.yaml"),
             cost_log_path=run_dir / f"cost_log_judge_{args.name}.jsonl")
    if args.backend == "openrouter":
        import or_backend
        or_backend.install(reasoning_effort=args.reasoning_effort)

    print(f"=== judge pass '{args.name}' ({args.judge_model}, {args.backend}) "
          f"on {run_dir.name} ===")
    report: dict = {}
    audit_dad.resolve_gids(run_dir, report)
    audit_dad.audit_judges(run_dir, config, report)

    # Keep each judge's failure evidence apart — audit_judges writes these
    # fresh per pass.
    for stem in ("delivery_failures", "impact_failures"):
        src = run_dir / "audit" / f"{stem}.jsonl"
        if src.exists():
            src.rename(run_dir / "audit" / f"{stem}_{args.name}.jsonl")

    out = {
        "judge_name": args.name,
        "judge_model": args.judge_model,
        "backend": args.backend,
        "reasoning_effort": args.reasoning_effort,
        "delivery": report.get("delivery"),
        "welfare_impact": report.get("welfare_impact"),
        "composite": report.get("composite"),
    }
    out_path = run_dir / "audit" / f"judge_{args.name}.json"
    utils.ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"judge-pass cost: ${api.get_total_cost():.4f}")


if __name__ == "__main__":
    main()
