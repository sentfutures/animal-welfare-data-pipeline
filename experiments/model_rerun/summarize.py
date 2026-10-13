#!/usr/bin/env python3
"""Build the results report for the model-rerun experiment.

Reads the original run's audit (the scores the 30 examples were selected on),
each arm's mini-run audit (written by evals/audit_dad.py --judges), and each
arm's call_log.jsonl (written by drive.py), and writes a markdown report:
per-example welfare-impact and delivery scores next to the originals, bucket
and overall means, judge-stability check on the shared plain arm, refusal and
truncation rates, and cost.

Usage:
  python summarize.py --arm opus55=outputs/dad/runs/<...> --arm gpt61=<...>
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared import utils  # noqa: E402
from dad_pipeline.id_registry import prompt_key  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE_RUN_DEFAULT = ROOT / "outputs/dad/runs/2026-07-29_12-26_archetype200"

# Conservative text-level refusal markers — flags for manual review, not verdicts.
REFUSAL_MARKERS = (
    "i can't help with", "i cannot help with", "i can't assist",
    "i cannot assist", "i must decline", "i won't be able to help",
    "i'm not able to help", "i can't provide", "i cannot provide",
)


def load_report(run_dir: Path) -> dict:
    p = Path(run_dir) / "audit" / "audit_report.json"
    if not p.exists():
        raise SystemExit(f"{p} missing — run evals/audit_dad.py --judges on {run_dir} first")
    return json.loads(p.read_text(encoding="utf-8"))


def per_case_lookup(report: dict, section: str) -> dict:
    return (report.get(section) or {}).get("per_case") or {}


def case_for(per_case: dict, keys: tuple) -> dict | None:
    for k in keys:
        if k in per_case:
            return per_case[k]
    return None


def arm_blended(case: dict | None, arm: str):
    if case and arm in case:
        return case[arm].get("blended_score", case[arm].get("score"))
    return None


def fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.mean(vals) if vals else None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(SOURCE_RUN_DEFAULT))
    parser.add_argument("--selection", default=str(HERE / "selection.json"))
    parser.add_argument("--arm", action="append", required=True,
                        metavar="NAME=RUN_DIR",
                        help="Arm name and its mini-run dir; repeatable")
    parser.add_argument("--out", default=str(HERE / "results.md"))
    args = parser.parse_args(argv)

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    source_report = load_report(Path(args.source_run))
    orig_welfare = per_case_lookup(source_report, "welfare_impact")
    orig_delivery = per_case_lookup(source_report, "delivery")

    arms: dict[str, dict] = {}
    for spec in args.arm:
        name, _, run_dir = spec.partition("=")
        run_dir = Path(run_dir)
        report = load_report(run_dir)
        dilemmas = utils.load_jsonl(run_dir / "step1" / "dilemmas.jsonl")
        keys_by_pid = {d["prompt_id"]: tuple(k for k in (d.get("prompt_id"), d.get("prompt_gid"), prompt_key(d)) if k)
                       for d in dilemmas}
        produced = {r.get("prompt_gid") for r in utils.load_jsonl(run_dir / "step3" / "rewrites.jsonl")}
        calls = utils.load_jsonl(run_dir / "call_log.jsonl")
        cost = sum(r.get("cost_usd", 0.0) for r in utils.load_jsonl(run_dir / "cost_log.jsonl"))
        finals = utils.load_jsonl(run_dir / "final" / "dad_corpus.jsonl")
        flagged = []
        for rec in finals:
            text = ""
            for m in rec.get("messages", []):
                if m.get("role") == "assistant":
                    text = m.get("content", "")
            low = text.lower()
            if any(marker in low for marker in REFUSAL_MARKERS):
                flagged.append(rec.get("example_gid") or rec.get("record_id"))
        arms[name] = {
            "run_dir": run_dir, "report": report,
            "welfare": per_case_lookup(report, "welfare_impact"),
            "delivery": per_case_lookup(report, "delivery"),
            "keys_by_pid": keys_by_pid, "produced": produced,
            "calls": calls, "cost": cost, "n_final": len(finals),
            "text_flags": flagged,
        }

    lines = ["# Model rerun — results", ""]
    lines.append(f"Original run: `{Path(args.source_run).name}`. "
                 "Scores are the judges' blended welfare-impact / delivery scores "
                 "(0–100) for the pipeline arm; the same claude-opus-5 judges "
                 "scored every arm.")
    lines.append("")

    bucket_rows = {b: selection["buckets"][b] for b in ("low", "mid", "high")}

    # Per-example tables, one per bucket, welfare impact
    for metric, orig_pc, key in (("Welfare impact", orig_welfare, "welfare"),
                                 ("Delivery", orig_delivery, "delivery")):
        lines.append(f"## {metric} (pipeline arm, blended)")
        lines.append("")
        header = "| bucket | example | original |" + "".join(f" {a} |" for a in arms)
        lines.append(header)
        lines.append("|---" * (3 + len(arms)) + "|")
        all_by_arm = {a: [] for a in arms}
        orig_all = []
        for bucket, rows in bucket_rows.items():
            bucket_by_arm = {a: [] for a in arms}
            bucket_orig = []
            for row in rows:
                pid = row["prompt_id"]
                o = arm_blended(orig_pc.get(pid), "pipeline")
                bucket_orig.append(o)
                cells = [bucket, pid, fmt(o)]
                for a, arm in arms.items():
                    case = case_for(arm[key], arm["keys_by_pid"].get(pid, (pid,)))
                    v = arm_blended(case, "pipeline")
                    bucket_by_arm[a].append(v)
                    cells.append(fmt(v))
                lines.append("| " + " | ".join(cells) + " |")
            cells = [f"**{bucket} mean**", "", f"**{fmt(mean(bucket_orig))}**"]
            for a in arms:
                cells.append(f"**{fmt(mean(bucket_by_arm[a]))}**")
                all_by_arm[a].extend(bucket_by_arm[a])
            orig_all.extend(bucket_orig)
            lines.append("| " + " | ".join(cells) + " |")
        cells = ["**overall mean**", "", f"**{fmt(mean(orig_all))}**"]
        for a in arms:
            cells.append(f"**{fmt(mean(all_by_arm[a]))}**")
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Judge stability: the plain arm is the same text in every run
    lines.append("## Judge stability (plain arm — identical text, re-judged)")
    lines.append("")
    lines.append("| metric | original | " + " | ".join(arms) + " |")
    lines.append("|---" * (2 + len(arms)) + "|")
    for metric, orig_pc, key in (("welfare impact", orig_welfare, "welfare"),
                                 ("delivery", orig_delivery, "delivery")):
        pids = [r["prompt_id"] for rows in bucket_rows.values() for r in rows]
        o = mean([arm_blended(orig_pc.get(p), "plain") for p in pids])
        cells = [metric, fmt(o)]
        for a, arm in arms.items():
            vals = [arm_blended(case_for(arm[key], arm["keys_by_pid"].get(p, (p,))), "plain")
                    for p in pids]
            cells.append(fmt(mean(vals)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Refusals and failures
    lines.append("## Refusals, truncations, and drop-outs")
    lines.append("")
    for a, arm in arms.items():
        calls = arm["calls"]
        refusals = [c for c in calls if c.get("stop_reason") == "refusal"]
        truncated = [c for c in calls if c.get("stop_reason") == "max_tokens"]
        n_sel = sum(len(v) for v in bucket_rows.values())
        missing = n_sel - arm["n_final"]
        lines.append(f"### {a} (`{arm['run_dir'].name}`)")
        lines.append("")
        lines.append(f"- generation calls: {len(calls)}; hard refusals "
                     f"(stop_reason=refusal): **{len(refusals)}** "
                     f"({len(refusals)/len(calls):.1%} of calls)" if calls else
                     f"- generation calls: 0")
        if refusals:
            by_stage: dict = {}
            for c in refusals:
                by_stage.setdefault(c.get("stage"), []).append(c.get("item_id"))
            for stage, items in by_stage.items():
                lines.append(f"  - {stage}: {len(items)} ({', '.join(str(i) for i in items[:12])})")
        lines.append(f"- truncations at max_tokens (retried per pipeline rules): {len(truncated)}")
        lines.append(f"- examples with no final record: **{missing}** of {n_sel}")
        lines.append(f"- final responses matching refusal phrasing (manual-review flags): "
                     f"{len(arm['text_flags'])}"
                     + (f" ({', '.join(arm['text_flags'])})" if arm["text_flags"] else ""))
        lines.append(f"- generation + judging cost on this run: ${arm['cost']:.2f}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
