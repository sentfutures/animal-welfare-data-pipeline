#!/usr/bin/env python3
"""Build the results report for the model-rerun experiment (cross-judge design).

Inputs per generation arm: the mini-run dir, its call_log.jsonl (absent on the
judge-only originals arm), and every audit/judge_<name>.json written by
judge.py. The report:

- judge × arm mean matrices (welfare impact and delivery, pipeline arm)
- the same matrix for the plain baseline — identical text in every run, so it
  anchors each judge's calibration
- a same-family bias readout: how much more each judge likes the Claude-made
  responses than the GPT-made ones, and the difference between the judges
- per-example tables per judge, with the original run's claude-opus-5 score
  as a reference column
- refusal / truncation / drop-out rates per generation arm, and costs

Usage:
  python summarize.py --arm original=<run> --arm opus55=<run> --arm gpt61sol=<run>
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared import utils  # noqa: E402
from dad_pipeline.id_registry import prompt_key, prompt_keys  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE_RUN_DEFAULT = ROOT / "outputs/dad/runs/2026-07-29_12-26_archetype200"

REFUSAL_MARKERS = (
    "i can't help with", "i cannot help with", "i can't assist",
    "i cannot assist", "i must decline", "i won't be able to help",
    "i'm not able to help", "i can't provide", "i cannot provide",
)

METRICS = (("welfare impact", "welfare_impact"), ("delivery", "delivery"))


def per_case(block: dict | None) -> dict:
    return (block or {}).get("per_case") or {}


def case_for(pc: dict, keys: tuple) -> dict | None:
    """Merge the per-case entries found under any of the example's ids — a
    mixed-era run can key the pipeline arm by legacy prompt_id and the plain
    arm by prompt_gid."""
    merged: dict = {}
    for k in keys:
        if k in pc:
            for arm_name, v in pc[k].items():
                merged.setdefault(arm_name, v)
    return merged or None


def blended(case: dict | None, arm: str):
    if case and arm in case:
        return case[arm].get("blended_score", case[arm].get("score"))
    return None


def fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.mean(vals) if vals else None


def load_arm(run_dir: Path) -> dict:
    dilemmas = utils.load_jsonl(run_dir / "step1" / "dilemmas.jsonl")
    keys_by_pid = {d["prompt_id"]: tuple(dict.fromkeys(
        k for k in (*prompt_keys(d), prompt_key(d)) if k)) for d in dilemmas}
    judges = {}
    for jf in sorted((run_dir / "audit").glob("judge_*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        judges[data.get("judge_name") or jf.stem.removeprefix("judge_")] = data
    finals = utils.load_jsonl(run_dir / "final" / "dad_corpus.jsonl")
    flagged = []
    for rec in finals:
        text = ""
        for m in rec.get("messages", []):
            if m.get("role") == "assistant":
                text = m.get("content", "")
        if any(marker in text.lower() for marker in REFUSAL_MARKERS):
            flagged.append(rec.get("example_gid") or rec.get("record_id"))
    return {
        "run_dir": run_dir,
        "keys_by_pid": keys_by_pid,
        "judges": judges,
        "n_final": len(finals),
        "text_flags": flagged,
        "calls": utils.load_jsonl(run_dir / "call_log.jsonl"),
        "gen_cost": sum(r.get("cost_usd", 0.0)
                        for r in utils.load_jsonl(run_dir / "cost_log.jsonl")),
        "judge_costs": {p.stem.removeprefix("cost_log_judge_"):
                        sum(r.get("cost_usd", 0.0) for r in utils.load_jsonl(p))
                        for p in sorted(run_dir.glob("cost_log_judge_*.jsonl"))},
    }


def arm_judge_scores(arm: dict, judge: str, metric_key: str, side: str,
                     pids: list[str]) -> list:
    jdata = arm["judges"].get(judge)
    if not jdata:
        return [None] * len(pids)
    pc = per_case(jdata.get(metric_key))
    return [blended(case_for(pc, arm["keys_by_pid"].get(p, (p,))), side)
            for p in pids]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(SOURCE_RUN_DEFAULT))
    parser.add_argument("--selection", default=str(HERE / "selection.json"))
    parser.add_argument("--arm", action="append", required=True,
                        metavar="NAME=RUN_DIR")
    parser.add_argument("--claude-arms", default="original,opus55",
                        help="Comma-separated arm names whose responses are "
                             "Claude-made (for the bias readout)")
    parser.add_argument("--gpt-arms", default=None,
                        help="Comma-separated arm names whose responses are "
                             "GPT-made; default: every non-Claude arm. Set it "
                             "when a third-family arm (e.g. Gemini) is present")
    parser.add_argument("--claude-judges", default="opus55",
                        help="Comma-separated judge names that are Claude-family")
    parser.add_argument("--gpt-judges", default="gpt61sol",
                        help="Comma-separated judge names that are GPT-family; any "
                             "other judge (e.g. a Gemini arbiter) stays out of the "
                             "two-family bias estimate but gets its own delta line")
    parser.add_argument("--out", default=str(HERE / "results.md"))
    args = parser.parse_args(argv)

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    buckets = {b: [r["prompt_id"] for r in selection["buckets"][b]]
               for b in ("low", "mid", "high")}
    pids = [p for rows in buckets.values() for p in rows]

    source_report = json.loads(
        (Path(args.source_run) / "audit" / "audit_report.json").read_text(encoding="utf-8"))

    arms = {}
    for spec in args.arm:
        name, _, run_dir = spec.partition("=")
        arms[name] = load_arm(Path(run_dir))
    judge_names = sorted({j for a in arms.values() for j in a["judges"]})

    lines = ["# Model rerun — cross-judge results", ""]
    lines.append(f"30 examples from `{Path(args.source_run).name}` (10 lowest / 10 middle / "
                 "10 highest by that run's welfare-impact score). Every arm's responses "
                 "judged by every judge through the repo's own rubric prompts; scores are "
                 "blended 0–100 (70% holistic, 30% dimension mean). The 'original ref' "
                 "column is the source run's own claude-opus-5 judge.")
    lines.append("")

    # --- judge × arm matrices ---
    for side, title in (("pipeline", "Pipeline arm (the responses under test)"),
                        ("plain", "Plain baseline (identical text in every run — judge calibration)")):
        lines.append(f"## {title}")
        lines.append("")
        for mname, mkey in METRICS:
            lines.append(f"**{mname}** (mean blended, n per cell in parens)")
            lines.append("")
            lines.append("| generation arm | " + " | ".join(f"judge: {j}" for j in judge_names) + " |")
            lines.append("|---" * (1 + len(judge_names)) + "|")
            for aname, arm in arms.items():
                cells = [aname]
                for j in judge_names:
                    vals = arm_judge_scores(arm, j, mkey, side, pids)
                    ok = [v for v in vals if v is not None]
                    cells.append(f"{fmt(mean(ok))} ({len(ok)})")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    # --- bias readout ---
    claude_arms = [a.strip() for a in args.claude_arms.split(",") if a.strip() in arms]
    gpt_arms = ([a.strip() for a in args.gpt_arms.split(",") if a.strip() in arms]
                if args.gpt_arms else [a for a in arms if a not in claude_arms])
    claude_judges = [j.strip() for j in args.claude_judges.split(",") if j.strip() in judge_names]
    gpt_judges = [j.strip() for j in args.gpt_judges.split(",") if j.strip() in judge_names]
    if claude_arms and gpt_arms and claude_judges and gpt_judges:
        lines.append("## Same-family bias readout (welfare impact, pipeline arm)")
        lines.append("")
        lines.append("Each judge's preference for the Claude-made over the GPT-made "
                     "responses, on the same 30 prompts. If the two judges disagree "
                     "mostly in their own family's favor, that gap is same-family bias.")
        lines.append("")
        deltas = {}
        for j in judge_names:
            c = mean([v for a in claude_arms
                      for v in arm_judge_scores(arms[a], j, "welfare_impact", "pipeline", pids)
                      if v is not None])
            g = mean([v for a in gpt_arms
                      for v in arm_judge_scores(arms[a], j, "welfare_impact", "pipeline", pids)
                      if v is not None])
            if c is not None and g is not None:
                deltas[j] = c - g
                fam = ("Claude-family" if j in claude_judges
                       else "GPT-family" if j in gpt_judges else "neutral arbiter")
                lines.append(f"- judge **{j}** ({fam}): Claude-made − GPT-made = "
                             f"**{c - g:+.1f}**")
        if len(deltas) >= 2 and claude_judges and gpt_judges:
            cj = mean([deltas[j] for j in claude_judges if j in deltas])
            gj = mean([deltas[j] for j in gpt_judges if j in deltas])
            if cj is not None and gj is not None:
                lines.append(f"- **difference between the judges (the bias estimate): "
                             f"{cj - gj:+.1f} points** — positive means each family "
                             "favors its own side relative to the other judge")
        lines.append("")

    # --- per-example tables, per judge ---
    orig_ref = {mkey: per_case(source_report.get(mkey)) for _, mkey in METRICS}
    for j in judge_names:
        for mname, mkey in METRICS:
            lines.append(f"## Per-example — judge {j}, {mname} (pipeline arm)")
            lines.append("")
            lines.append("| bucket | example | original ref (opus-5 judge) | "
                         + " | ".join(arms) + " |")
            lines.append("|---" * (3 + len(arms)) + "|")
            for bucket, rows in buckets.items():
                per_arm = {a: [] for a in arms}
                refs = []
                for pid in rows:
                    ref = blended(orig_ref[mkey].get(pid), "pipeline")
                    refs.append(ref)
                    cells = [bucket, pid, fmt(ref)]
                    for aname, arm in arms.items():
                        v = arm_judge_scores(arm, j, mkey, "pipeline", [pid])[0]
                        per_arm[aname].append(v)
                        cells.append(fmt(v))
                    lines.append("| " + " | ".join(cells) + " |")
                cells = [f"**{bucket} mean**", "", f"**{fmt(mean(refs))}**"]
                cells += [f"**{fmt(mean(per_arm[a]))}**" for a in arms]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    # --- refusals and failures ---
    lines.append("## Refusals, truncations, and drop-outs (generation arms)")
    lines.append("")
    for aname, arm in arms.items():
        lines.append(f"### {aname} (`{arm['run_dir'].name}`)")
        lines.append("")
        calls = arm["calls"]
        if not calls:
            lines.append("- no generation in this run (judge-only arm: the source "
                         "run's own responses)")
        else:
            refusals = [c for c in calls if c.get("stop_reason") == "refusal"]
            truncated = [c for c in calls if c.get("stop_reason") == "max_tokens"]
            lines.append(f"- generation calls: {len(calls)}; hard refusals "
                         f"(stop_reason=refusal): **{len(refusals)}** "
                         f"({len(refusals) / len(calls):.1%} of calls)")
            if refusals:
                by_stage: dict = {}
                for c in refusals:
                    by_stage.setdefault(c.get("stage"), []).append(c.get("item_id"))
                for stage, items in by_stage.items():
                    lines.append(f"  - {stage}: {len(items)} "
                                 f"({', '.join(str(i) for i in items[:12])})")
            lines.append(f"- truncations at max_tokens (each retried once bigger): "
                         f"{len(truncated)}")
        lines.append(f"- examples with a final record: **{arm['n_final']}** of {len(pids)}")
        lines.append(f"- final responses matching refusal phrasing (manual-review "
                     f"flags): {len(arm['text_flags'])}"
                     + (f" ({', '.join(str(x) for x in arm['text_flags'])})"
                        if arm["text_flags"] else ""))
        costs = " + ".join(f"${v:.2f} ({k} judge)" for k, v in arm["judge_costs"].items())
        lines.append(f"- cost: ${arm['gen_cost']:.2f} (generation)"
                     + (f" + {costs}" if costs else ""))
        lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
