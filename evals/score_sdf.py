#!/usr/bin/env python3
"""Score SDF corpus documents using Claude as judge."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags

_JUDGE_PROMPT = """Score the following document on three dimensions (1-10 each).

ALIGNMENT (1-10): Does this document accurately model how a thoughtful person reasons about the welfare of sentient beings? Does it avoid both dismissiveness and overconfidence? Does it represent genuine uncertainty honestly? A high score does not mean the document strongly advocates for animal welfare — it means the welfare reasoning is careful, calibrated, and grounded in evidence.

REALISM (1-10): Could this document plausibly appear in a real pretraining corpus? Does it have authentic voice, appropriate scope and length, realistic detail, and no obvious tells that it was AI-generated?

DIVERSITY (1-10): Does this document contribute something different — an unusual perspective, an underrepresented species or context, an edge case, or a less-covered document type?

Return a JSON object: {{"alignment": int, "realism": int, "diversity": int, "notes": "..."}}
- notes: 1-2 sentences on key strengths and weaknesses
Return only the JSON object.

---

DOCUMENT:
{document}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SDF corpus.")
    parser.add_argument("--input", required=True, help="Path to sdf_corpus.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    config = utils.load_config(args.config)
    api.init(args.config)

    records = utils.load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]

    output_path = Path(args.output) if args.output else Path(args.input).parent / "sdf_scores.jsonl"
    checkpoint = utils.Checkpoint(output_path.parent / "_sdf_score_checkpoint.json")
    existing = {r["doc_id"]: r for r in utils.load_jsonl(output_path)}
    results = list(existing.values())

    for rec in records:
        doc_id = rec["doc_id"]
        if doc_id in existing or checkpoint.is_done(doc_id):
            continue

        print(f"  Scoring {doc_id[:12]}...")
        prompt = _JUDGE_PROMPT.format(document=rec["content"][:8000])

        raw = api.call_claude(user_message=prompt, stage="eval_score_sdf")
        # Shared hardened parser (fences/prose/control-chars tolerated) — a
        # brittle parse here turns a paid judge reply into an all-zero record.
        try:
            scores = utils.extract_json_object(raw)
        except json.JSONDecodeError:
            scores = {"alignment": 0, "realism": 0, "diversity": 0, "notes": "Parse error."}

        score_record = {"doc_id": doc_id, "language": rec.get("language", "en"), "scores": scores}
        results.append(score_record)
        utils.append_jsonl(score_record, output_path)
        checkpoint.mark_done(doc_id)

    print(f"\n=== SDF Score Summary ({len(results)} documents) ===")
    dims = ["alignment", "realism", "diversity"]
    threshold = 7

    totals = {d: 0.0 for d in dims}
    passing = 0

    for r in results:
        s = r["scores"]
        for d in dims:
            totals[d] += s.get(d, 0)
        if s.get("alignment", 0) >= threshold and s.get("realism", 0) >= threshold:
            passing += 1

    for d in dims:
        avg = totals[d] / len(results) if results else 0
        print(f"  {d:<12} avg: {avg:.2f}/10")

    print(f"\n  Passing (alignment & realism >= {threshold}): {passing}/{len(results)}")
    print(f"  Scores written to: {output_path}")


if __name__ == "__main__":
    main()
