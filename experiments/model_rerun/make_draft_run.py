#!/usr/bin/env python3
"""Build a judge-only run holding a source run's STEP-2 DRAFT responses — the
pre-rewrite text — so the judges can score what each model writes before the
step-3 constitution pass. Pipeline arm only (no baseline is copied: the plain
arm's calibration numbers already exist in the main runs), no generation.

The fabricated run carries step1/dilemmas.jsonl (the selection), plus
step3/rewrites.jsonl + final/dad_corpus.jsonl built from the source's
step2/responses.jsonl, which is the join shape evals/audit_dad.py's judges
read. Ids are normalized to the dilemmas' current prompt gids (legacy-era
sources carry only prompt_id).
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared import utils  # noqa: E402
from dad_pipeline.id_registry import prompt_key, prompt_keys  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True,
                        help="Run whose step2/responses.jsonl holds the drafts")
    parser.add_argument("--selection", default=str(HERE / "selection.json"))
    parser.add_argument("--label", required=True, help="Must contain 'local_'")
    args = parser.parse_args(argv)
    if "local_" not in args.label:
        raise SystemExit("--label must contain 'local_' so the run dir stays untracked")

    source = Path(args.source_run)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    wanted = {r["prompt_id"] for b in selection["buckets"].values() for r in b}

    dilemmas = [d for d in utils.load_jsonl(source / "step1" / "dilemmas.jsonl")
                if d.get("prompt_id") in wanted or d.get("prompt_gid") in wanted]
    key_of = {k: prompt_key(d) for d in dilemmas for k in prompt_keys(d)}

    drafts = []
    for r in utils.load_jsonl(source / "step2" / "responses.jsonl"):
        k = next((k for k in prompt_keys(r) if k in key_of), None)
        if k is not None and r.get("assistant_response"):
            drafts.append((key_of[k], r))

    run_dir = (ROOT / "outputs" / "dad" / "runs"
               / f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{args.label}")
    for sub in ("step1", "step3", "final", "audit"):
        utils.ensure_dir(run_dir / sub)

    with open(run_dir / "step1" / "dilemmas.jsonl", "w", encoding="utf-8") as f:
        for d in dilemmas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(run_dir / "step3" / "rewrites.jsonl", "w", encoding="utf-8") as fr, \
         open(run_dir / "final" / "dad_corpus.jsonl", "w", encoding="utf-8") as fc:
        for gid, r in drafts:
            record_id = str(uuid.uuid4())
            fr.write(json.dumps({
                "record_id": record_id, "prompt_gid": gid,
                "response_id": r.get("response_id"),
                "response_gid": r.get("response_gid"),
                "sample_index": r.get("sample_index", 0),
                "note": "step-2 DRAFT (pre-rewrite), fabricated for judge-only grading",
            }, ensure_ascii=False) + "\n")
            fc.write(json.dumps({
                "record_id": record_id,
                "example_gid": r.get("response_gid"),
                "response_gid": r.get("response_gid"),
                "messages": [
                    {"role": "user", "content": r.get("user_message", "")},
                    {"role": "assistant", "content": r["assistant_response"]},
                ],
            }, ensure_ascii=False) + "\n")

    print(f"run_dir: {run_dir}")
    print(f"dilemmas: {len(dilemmas)} | drafts: {len(drafts)}")
    if len(drafts) != len(dilemmas):
        print(f"  WARNING: {len(dilemmas) - len(drafts)} selected examples have no "
              "step-2 draft in the source run")
    return run_dir


if __name__ == "__main__":
    main()
