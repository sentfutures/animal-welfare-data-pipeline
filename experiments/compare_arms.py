#!/usr/bin/env python3
"""Compare the arms of a revision-stage experiment: diversity against quality.

Reads the manifest experiments/experiment.py wrote, then for each arm the
diversity report, the layer-4 judge scores, the final corpus and the cost log.
The question it answers is the one the experiment was set up to ask: did this
revision variant buy diversity, and what did it cost in judge pass rate?

Every threshold is read from the arms file's `success:` block — this script
hardcodes none, because a success criterion decided after seeing the numbers
is not a success criterion.

    python experiments/compare_arms.py --exp outputs/sdf/experiments/exp1
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import revision_style
from shared import textstats, utils

DIVERSITY_REPORT = "audit/diversity_report.json"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _num(x: float | None, places: int = 1) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def judge_stats(run_dir: Path, threshold: int) -> dict:
    """Layer-4 pass rate and per-dimension means for one arm."""
    scores = utils.load_jsonl(utils.sdf_stage_file(run_dir, "score"))
    if not scores:
        return {"n": 0, "pass_rate": None, "means": {}}
    dims = ("alignment", "realism", "spec_conformance")
    means = {
        d: round(sum(s["scores"].get(d, 0) for s in scores) / len(scores), 2)
        for d in dims
    }
    passed = sum(
        1 for s in scores
        if s["scores"].get("alignment", 0) >= threshold
        and s["scores"].get("realism", 0) >= threshold
    )
    return {"n": len(scores), "pass_rate": passed / len(scores), "means": means}


# A p10/p90 off fewer documents than this is a sample artifact, not a spread.
_MIN_SPREAD_DOCS = 10


def _percentile(values: list[int], p: float) -> int:
    """Nearest-rank percentile, matching how audit_sdf.py reports lengths."""
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def _length_score(actual: int, target: int, sigma: float) -> float:
    """Graded proximity to a length target: 1.0 on target, falling off smoothly.

    A Gaussian on the LOG ratio, because length is a ratio scale: half the
    target and twice it are equally wrong. At sigma = log(1 + tolerance) a
    document on the tolerance edge scores exp(-0.5) ~ 0.61.
    """
    if actual <= 0 or target <= 0:
        return 0.0
    return math.exp(-0.5 * (math.log(actual / target) / sigma) ** 2)


def draft_lengths(run_dir: Path) -> dict[str, int]:
    """{doc_id: length of the draft layer 3 was handed}, in its own unit.

    The denominator a relative target is checked against. Each arm carries its
    own, since the driver writes the sampled drafts into the arm's layer-2 dir.
    """
    drafts = utils.load_jsonl(utils.sdf_stage_file(run_dir, "draft"))
    out = {}
    for rec in drafts:
        unit = textstats.unit_for_language(cp.record_language(rec))
        out[rec["doc_id"]] = textstats.count_units(rec.get("content", ""), unit)
    return out


def length_compliance(run_dir: Path, tolerance: float, threshold: int) -> dict | None:
    """How closely documents landed on the length target they were dealt.

    Targets are a multiple of the document's own draft length, so the expected
    length is per-document and the rungs compare across languages. A corpus
    with no length cards returns None rather than a vacuous 100%.

    Three figures, because one in-band percentage cannot separate "ignored the
    instruction" from "moved the right way and overshot": `rate` is the share
    inside +/- tolerance, `score` the graded mean, and `levels` the response
    curve, whose `unchanged` row is the undirected baseline from the same arm.
    """
    rewrites = utils.load_jsonl(utils.sdf_stage_file(run_dir, "rewrite"))
    if not rewrites:
        return None
    drafts = draft_lengths(run_dir)
    judged = {
        s["doc_id"]: s.get("scores") or {}
        for s in utils.load_jsonl(utils.sdf_stage_file(run_dir, "score"))
    }
    sigma = math.log(1 + tolerance)

    buckets: dict[float | None, list[dict]] = {}
    for rec in rewrites:
        did = rec["doc_id"]
        unit = textstats.unit_for_language(cp.record_language(rec))
        card = ((rec.get("revision") or {}).get("cards") or {}).get("length")
        ratio = revision_style.length_ratio(card)
        started = drafts.get(did)
        buckets.setdefault(ratio if started else None, []).append({
            "actual": textstats.count_units(rec.get("content", ""), unit),
            "target": round(started * ratio) if (ratio and started) else None,
            "scores": judged.get(did) or {},
        })

    targeted = [(k, v) for k, v in buckets.items() if k is not None]
    if not targeted:
        return None

    def level(ratio: float | None, items: list[dict]) -> dict:
        actuals = [i["actual"] for i in items]
        row = {"ratio": ratio, "n": len(items),
               "median_actual": _percentile(actuals, 0.50)}
        scored = [i for i in items if i["scores"]]
        if scored:
            row["pass_rate"] = round(sum(
                1 for i in scored
                if i["scores"].get("alignment", 0) >= threshold
                and i["scores"].get("realism", 0) >= threshold
            ) / len(scored), 4)
        if ratio is None:
            return row
        hit = [i for i in items if i["target"]]
        if not hit:
            return row
        row["median_target"] = _percentile([i["target"] for i in hit], 0.50)
        row["median_ratio"] = round(
            _percentile([i["actual"] for i in hit], 0.50) / row["median_target"], 3)
        row["rate"] = round(sum(
            1 for i in hit if abs(i["actual"] - i["target"]) <= tolerance * i["target"]
        ) / len(hit), 4)
        row["score"] = round(
            sum(_length_score(i["actual"], i["target"], sigma) for i in hit) / len(hit), 4)
        return row

    levels = [level(r, items) for r, items in
              sorted(buckets.items(), key=lambda kv: (kv[0] is not None, kv[0] or 0))]

    flat = [i for _, items in targeted for i in items if i["target"]]
    if not flat:
        return None
    n = len(flat)

    # Spread over word-counted documents only: a p90/p10 mixing word counts
    # with character counts is not a number.
    words = [textstats.count_units(r.get("content", ""), "words") for r in rewrites
             if textstats.unit_for_language(cp.record_language(r)) == "words"]
    spread = None
    if len(words) >= _MIN_SPREAD_DOCS:
        p10, p90 = _percentile(words, 0.10), _percentile(words, 0.90)
        spread = {"unit": "words", "n": len(words), "p10": p10,
                  "median": _percentile(words, 0.50), "p90": p90,
                  "ratio": round(p90 / p10, 3) if p10 else None}

    return {
        "n_targeted": n,
        "tolerance": tolerance,
        "sigma": round(sigma, 4),
        "relative": True,
        "rate": round(sum(
            1 for i in flat if abs(i["actual"] - i["target"]) <= tolerance * i["target"]
        ) / n, 4),
        "score": round(
            sum(_length_score(i["actual"], i["target"], sigma) for i in flat) / n, 4),
        "levels": levels,
        "spread": spread,
    }


def arm_cost(run_dir: Path) -> float:
    total = 0.0
    for rec in utils.load_jsonl(run_dir / "cost_log.jsonl"):
        total += float(rec.get("cost_usd") or 0.0)
    return total


def read_diversity(run_dir: Path, *, run_evals: bool, config: str,
                   group_by: str, ideas: bool) -> dict | None:
    path = run_dir / DIVERSITY_REPORT
    if not path.exists() and run_evals:
        cmd = [sys.executable, "evals/diversity.py", "--input", str(run_dir),
               "--config", config, "--group-by", group_by]
        if ideas:
            cmd.append("--ideas")
        print(f"  running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect(name: str, run_dir: Path, *, threshold: int, tolerance: float,
            run_evals: bool, config: str, group_by: str, ideas: bool) -> dict:
    div = read_diversity(run_dir, run_evals=run_evals, config=config,
                         group_by=group_by, ideas=ideas) or {}
    corpus = utils.load_jsonl(run_dir / "final" / "sdf_corpus.jsonl")
    judge = judge_stats(run_dir, threshold)
    return {
        "arm": name,
        "run_dir": str(run_dir),
        "n_scored": judge["n"],
        "n_final": len(corpus),
        "vendi": div.get("vendi"),
        "idea_vendi": (div.get("ideas") or {}).get("vendi"),
        "group_vendi": div.get("group_vendi"),
        "grouped_by": div.get("grouped_by"),
        "near_dup_over_0.90": (div.get("nn") or {}).get("over_0.90"),
        "form": div.get("form"),
        "pass_rate": judge["pass_rate"],
        "score_means": judge["means"],
        "length_compliance": length_compliance(run_dir, tolerance, threshold),
        "cost_usd": round(arm_cost(run_dir), 4),
    }


def verdicts(rows: list[dict], success: dict, control: str) -> None:
    """Mark each arm against the criteria written down before the run."""
    base = next((r for r in rows if r["arm"] == control), None)
    min_gain = success.get("min_vendi_gain")
    max_drop = success.get("max_pass_rate_drop")
    min_in_band = success.get("min_length_compliance")
    min_length_score = success.get("min_length_score")
    for row in rows:
        if base is None or row is base:
            row["verdict"] = "control" if row is base else "no control arm"
            continue
        notes, ok, unmeasured = [], True, []
        b, a = (base.get("vendi") or {}).get("ratio"), (row.get("vendi") or {}).get("ratio")
        if min_gain is not None:
            if b and a is not None:
                gain = (a - b) / b
                notes.append(f"Vendi ratio {gain:+.1%}")
                ok &= gain >= min_gain
            else:
                unmeasured.append("min_vendi_gain")
        if max_drop is not None:
            if base["pass_rate"] is not None and row["pass_rate"] is not None:
                drop = base["pass_rate"] - row["pass_rate"]
                notes.append(f"pass rate {-drop:+.1%}")
                ok &= drop <= max_drop
            else:
                unmeasured.append("max_pass_rate_drop")
        # Length is judged on the arm's own documents, not against the control:
        # the control renders no length directive, so its compliance is the
        # chance rate, which is a baseline to read rather than a bar to clear.
        lc = row.get("length_compliance") or {}
        if min_in_band is not None:
            if lc.get("rate") is not None:
                notes.append(f"length in-band {lc['rate']:.1%}")
                ok &= lc["rate"] >= min_in_band
            else:
                unmeasured.append("min_length_compliance")
        if min_length_score is not None:
            if lc.get("score") is not None:
                notes.append(f"length score {lc['score']:.2f}")
                ok &= lc["score"] >= min_length_score
            else:
                unmeasured.append("min_length_score")
        # A criterion that was declared but could not be evaluated is not a
        # pass. Reporting PASS off the surviving half would let an arm clear a
        # bar nobody measured — the exact mistake the success block exists to
        # prevent.
        if unmeasured:
            notes.append("not measured: " + ", ".join(unmeasured))
            verdict = "INCOMPLETE"
        else:
            verdict = "PASS" if ok else "FAIL"
        row["verdict"] = verdict + (f" ({'; '.join(notes)})" if notes else "")


def length_curves(rows: list[dict]) -> list[str]:
    """Per-arm response curve: did the dealt target actually move the document?

    The question the in-band percentage cannot answer. The `unchanged` row is
    the undirected baseline from the same arm over the same drafts, so a ladder
    that fails to separate from it is a knob that did nothing.
    """
    out: list[str] = []
    for r in rows:
        lc = r.get("length_compliance")
        if not lc:
            continue
        out.append("")
        out.append(f"### Length response — `{r['arm']}`")
        out.append("")
        out.append("| target | n | median | median/target | in-band | score | judge pass |")
        out.append("|" + "---|" * 7)
        for lv in lc["levels"]:
            target = f"{lv['ratio']}x" if lv.get("ratio") else "unchanged (baseline)"
            out.append(
                f"| {target} | {lv['n']} | {lv['median_actual']} "
                f"| {_num(lv.get('median_ratio'), 2)} | {_pct(lv.get('rate'))} "
                f"| {_num(lv.get('score'), 2)} | {_pct(lv.get('pass_rate'))} |"
            )
        sp = lc.get("spread")
        if sp:
            out.append("")
            out.append(
                f"Corpus spread over the {sp['n']} word-counted documents: "
                f"p10 {sp['p10']} / median {sp['median']} / p90 {sp['p90']} "
                f"words, p90/p10 = {_num(sp['ratio'], 2)}."
            )
    return out


def markdown(rows: list[dict], success: dict) -> str:
    head = ("| arm | final/scored | Vendi (ratio) | idea Vendi | form struct / lex | "
            "near-dup >0.90 | judge pass | align / real / spec | length in-band / score "
            "| cost | verdict |")
    out = [head, "|" + "---|" * 11]
    for r in rows:
        v = r.get("vendi") or {}
        iv = r.get("idea_vendi") or {}
        means = r["score_means"]
        ci = f" [{v['ci_lo']:.0f}-{v['ci_hi']:.0f}]" if "ci_lo" in v else ""
        vendi_cell = f"{_num(v.get('score'))} ({_num(v.get('ratio'), 3)}){ci}"
        idea_cell = f"{_num(iv.get('score'))} ({_num(iv.get('ratio'), 3)})"
        dims = " / ".join(
            str(means.get(d, "—")) for d in ("alignment", "realism", "spec_conformance")
        )
        lc = r.get("length_compliance")
        length_cell = (f"{_pct(lc['rate'])} / {lc['score']:.2f} (n={lc['n_targeted']})"
                       if lc else "—")
        fm = r.get("form") or {}
        form_cell = (f"{_num(fm.get('structural_spread'), 3)} / "
                     f"{_num(fm.get('lexical_spread'), 3)}" if fm else "—")
        out.append(
            f"| {r['arm']} | {r['n_final']}/{r['n_scored']} | {vendi_cell} | {idea_cell} "
            f"| {form_cell} | {_pct(r.get('near_dup_over_0.90'))} | {_pct(r.get('pass_rate'))} "
            f"| {dims} | {length_cell} | ${r['cost_usd']:.2f} | {r.get('verdict', '')} |"
        )
    if success:
        out.append("")
        out.append("Success criteria, fixed before the run: "
                   + ", ".join(f"`{k}` = {v}" for k, v in success.items()))
    out.extend(length_curves(rows))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True, help="Experiment dir holding manifest.json.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--control", default="control", help="Arm every other arm is measured against.")
    parser.add_argument("--group-by", default="language", choices=("language", "type_id"))
    parser.add_argument("--run-evals", action="store_true",
                        help="Run diversity.py for an arm that has no report yet (PAID: embeddings).")
    parser.add_argument("--ideas", action="store_true",
                        help="With --run-evals, also run the summary pass (PAID: one call per document).")
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    manifest_path = exp_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    success = manifest.get("success") or {}
    config = utils.load_config(args.config)
    threshold = config["sdf"]["min_score_threshold"]
    tolerance = success.get("length_tolerance", 0.20)

    rows = []
    for name, run in manifest["arms"].items():
        run_dir = Path(run)
        if not run_dir.exists():
            print(f"  arm '{name}': run dir missing ({run_dir}) — skipped")
            continue
        print(f"  arm '{name}' <- {run_dir.name}")
        rows.append(collect(name, run_dir, threshold=threshold, tolerance=tolerance,
                            run_evals=args.run_evals, config=args.config,
                            group_by=args.group_by, ideas=args.ideas))

    if not rows:
        raise SystemExit("no arms to compare")
    verdicts(rows, success, args.control)

    table = markdown(rows, success)
    print("\n" + table)
    (exp_dir / "comparison.md").write_text(
        f"# {manifest['exp_id']}\n\nDonor run `{manifest['donor_run']}`, "
        f"n={manifest['n']}, seed {manifest['seed']}.\n\n{table}\n",
        encoding="utf-8",
    )
    (exp_dir / "comparison.json").write_text(
        json.dumps({"exp_id": manifest["exp_id"], "success": success, "arms": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    missing = [r["arm"] for r in rows if not r.get("vendi")]
    if missing:
        print(f"\nNo diversity report for: {', '.join(missing)}. "
              f"Re-run with --run-evals (paid) or run evals/diversity.py on each arm.")
    print(f"\nWritten: {exp_dir / 'comparison.md'} and {exp_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
