#!/usr/bin/env python3
"""Constitutional-compliance audit of an SDF corpus (paid LLM pass).

The failure-mode appendix (``principle_id`` 14) is the rubric: present/absent/not_applicable and a quote per mode. The prompt encodes the corpus's design slices as scope rules,
or the judge reports intended variety as violations. Writes ``<run>/audit/compliance_report.json``."""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.audit_sdf import resolve_input
from shared import api, constitution_loader, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags

TEMPLATE = Path(__file__).parent.parent / "prompts" / "tools" / "compliance_scan.txt"

# The appendix section of the sentient-beings reading that enumerates observed
# failure modes. Its id is stable in META_PRINCIPLE_IDS; the heading regex below
# is the contract with its formatting.
_TYPOLOGY_PRINCIPLE_ID = 14
_MODE_HEADING_RE = re.compile(r"^###\s+(\d+)\.\s+(.+?)\s*$", re.M)

# A mode present in more than this share of judged documents is a corpus-level
# pattern rather than a scattered miss — worth a prompt fix, not a per-doc one.
PREVALENCE_FLAG = 0.10


def load_typology(constitution_dir: Path | None = None) -> tuple[str, dict[int, str]]:
    """``(appendix text, {mode number: title})`` from the sentient-beings reading; raises if the appendix is missing or its headings do not parse."""
    segments = constitution_loader.load_segments(constitution_dir)
    body = next(
        (s["content"] for s in segments if s.get("principle_id") == _TYPOLOGY_PRINCIPLE_ID),
        None,
    )
    if not body:
        raise SystemExit(
            f"no section with principle_id {_TYPOLOGY_PRINCIPLE_ID} in the "
            "sentient-beings reading — the violation typology moved or was removed"
        )
    modes = {int(n): title.strip() for n, title in _MODE_HEADING_RE.findall(body)}
    if not modes:
        raise SystemExit(
            "the violation-typology appendix has no '### N. Title' headings — "
            "its formatting changed; update _MODE_HEADING_RE"
        )
    return body, modes


def judge_documents(records: list[dict], config: dict, typology: str,
                    modes: dict[int, str], sample: int) -> list[dict]:
    """One judge call per sampled document. Returns the parsed verdict objects
    (documents whose call failed or parsed badly are dropped, not defaulted)."""
    stride = max(len(records) / max(sample, 1), 1.0)
    picked = [records[int(i * stride)] for i in range(min(sample, len(records)))]

    def judge_one(record: dict) -> dict | None:
        prompt = utils.load_prompt(
            TEMPLATE, typology=typology, document=(record.get("content") or "")[:6000]
        )
        try:
            parsed = utils.extract_json(api.call_claude(user_message=prompt, stage="eval_compliance") or "")
            verdicts = {}
            for entry in parsed.get("modes", []):
                n, verdict = entry.get("mode"), entry.get("verdict")
                if isinstance(n, int) and n in modes and verdict in (
                    "present", "absent", "not_applicable"
                ):
                    verdicts[n] = entry
            if not verdicts:
                return None
            return {
                "doc_id": record.get("doc_id"),
                "language": record.get("language"),
                "verdicts": verdicts,
                "overall": str(parsed.get("overall", ""))[:400],
            }
        except Exception:
            return None  # malformed judge output: unjudged, not compliant-by-default

    workers = config.get("workers", 1)
    return [r for r in utils.parallel_map(judge_one, picked, workers) if r]


def summarize(results: list[dict], modes: dict[int, str], report: dict) -> None:
    judged = len(results)
    present = collections.Counter()
    applicable = collections.Counter()
    findings = []
    for r in results:
        for n, entry in r["verdicts"].items():
            if entry["verdict"] != "not_applicable":
                applicable[n] += 1
            if entry["verdict"] == "present":
                present[n] += 1
                findings.append({
                    "doc_id": r["doc_id"], "mode": n, "mode_title": modes[n],
                    "evidence": str(entry.get("evidence", ""))[:400],
                    "note": str(entry.get("note", ""))[:300],
                })

    clean = sum(1 for r in results if not any(
        e["verdict"] == "present" for e in r["verdicts"].values()))
    print(f"\nCONSTITUTIONAL COMPLIANCE ({judged} documents judged against "
          f"{len(modes)} failure modes)")
    print(f"   {'documents with zero violations':<52} {clean}/{judged} ({clean / judged:.0%})")
    print(f"   {'total findings':<52} {len(findings)}")
    print(f"\n   {'failure mode':<52}{'present':>9}{'of judged':>11}{'of applicable':>15}")
    by_mode = {}
    for n in sorted(modes):
        share = present[n] / judged
        app = f"{present[n] / applicable[n]:.0%}" if applicable[n] else "n/a"
        flag = "  <-- CORPUS PATTERN" if share > PREVALENCE_FLAG else ""
        print(f"   {f'{n}. {modes[n]}'[:52]:<52}{present[n]:>9}{share:>11.0%}{app:>15}{flag}")
        by_mode[n] = {
            "title": modes[n], "present": present[n],
            "share_of_judged": round(share, 3),
            "applicable": applicable[n],
            "share_of_applicable": round(present[n] / applicable[n], 3) if applicable[n] else None,
        }

    report.update({
        "judged": judged,
        "clean_documents": clean,
        "clean_frac": round(clean / judged, 3),
        "prevalence_flag": PREVALENCE_FLAG,
        "by_mode": by_mode,
        "findings": findings,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="outputs/sdf/latest",
                        help="Run directory or sdf_corpus.jsonl path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=int, default=80,
                        help="Documents to judge (evenly strided across the corpus)")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    records, report_dir = resolve_input(args.input)
    if not records:
        raise SystemExit("Corpus is empty — nothing to audit.")
    config = utils.load_config(args.config)
    typology, modes = load_typology()

    print(f"=== SDF constitutional-compliance audit: {args.input} "
          f"({len(records)} documents, judging {min(args.sample, len(records))}) ===")
    api.init(args.config)  # evals log to the global cost log and record no full model calls

    results = judge_documents(records, config, typology, modes, args.sample)
    if not results:
        raise SystemExit("no documents judged (every judge call failed) — nothing written")

    report: dict = {"input": str(args.input), "n_docs": len(records),
                    "modes": {n: t for n, t in sorted(modes.items())}}
    summarize(results, modes, report)

    utils.ensure_dir(report_dir)
    out = report_dir / "compliance_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
