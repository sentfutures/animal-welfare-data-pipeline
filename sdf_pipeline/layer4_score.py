"""Layer 4: score each rewrite and gate the final corpus.

The judge scores alignment, realism, and spec_conformance (1-10 each) against
the constitution and the document's generating spec. The gate keeps documents
with alignment AND realism >= sdf.min_score_threshold; spec_conformance is
recorded and reported but does not gate (drift diagnostics belong to humans
while the dimension is young). Survivors pass a deterministic near-duplicate
cull before the final corpus is written.

A scoring response that fails to parse is recorded as 5/5/5 "Parse error."
and checkpointed — a re-run would re-bill without new information; the score
report flags these for manual review instead.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, constitution_loader, textstats, utils
from sdf_pipeline import compose_prompts as cp


def run(
    config: dict,
    prompts_dir: Path,
    output_dir: Path,
    final_dir: Path,
    rewrites: list[dict],
) -> list[dict]:
    output_path = output_dir / "scores.jsonl"
    final_path = final_dir / "sdf_corpus.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    sdf = config["sdf"]
    threshold = sdf["min_score_threshold"]

    constitution_dir = utils.resolve_constitution_dir(prompts_dir)
    constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)

    existing = {r["doc_id"]: r for r in utils.load_jsonl(output_path)}
    results = [existing[rw["doc_id"]] for rw in rewrites if rw["doc_id"] in existing]
    pending = [
        rw for rw in rewrites
        if rw["doc_id"] not in existing and not checkpoint.is_done(rw["doc_id"])
    ]

    def score_one(rw: dict) -> dict | None:
        system, user = cp.split_sections(utils.load_prompt(
            utils.sdf_template_path(prompts_dir, 4),
            constitution_claude=constitution_claude,
            document_description=rw["description"],
            improved_document=rw["content"],
        ))
        try:
            raw = api.call_claude(
                user,
                system_prompt=system or "",
                model=sdf.get("score_model"),
                stage="layer4_score",
                item_id=rw["doc_id"],
                cache_system=True,  # constitution + rubric are identical across scoring calls
            )
        except Exception as e:
            # Per-item failures skip the doc instead of killing the layer;
            # unmarked work is retried by --resume.
            print(f"  {rw['doc_id']}: API call failed ({type(e).__name__}: {e}) — will retry on resume")
            return None
        try:
            scores = utils.extract_json_object(raw)
        except json.JSONDecodeError:
            scores = {
                "alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error.",
            }
        return {
            "doc_id": rw["doc_id"],
            "variables": rw["variables"],
            "scores": {
                "alignment": scores.get("alignment", 0),
                "realism": scores.get("realism", 0),
                "spec_conformance": scores.get("spec_conformance", 0),
                "notes": scores.get("notes", ""),
            },
        }

    workers = config.get("workers", 1)
    rewrites_by_id = {rw["doc_id"]: rw for rw in rewrites}
    failed_calls = 0
    for record in utils.parallel_map(score_one, pending, workers):
        if record is None:
            failed_calls += 1
            continue
        results.append(record)
        utils.append_jsonl(record, output_path)
        checkpoint.mark_done(record["doc_id"])
        s = record["scores"]
        print(f"  Scored {record['doc_id']} A{s['alignment']} R{s['realism']} S{s['spec_conformance']}")
    if pending and failed_calls == len(pending):
        raise SystemExit(
            "layer4_score: every pending API call failed — this is systemic "
            "(auth, backend, or network), not per-document; fix and --resume."
        )

    # Gate on alignment + realism; spec_conformance is advisory (reported below).
    passed = [
        r for r in results
        if r["scores"]["alignment"] >= threshold and r["scores"]["realism"] >= threshold
    ]
    failed = len(results) - len(passed)
    if failed:
        print(f"  {failed} document(s) below threshold {threshold} — dropped")

    # Deterministic near-duplicate cull over survivors (greedy keep-first).
    dup_threshold = sdf.get("near_dup_threshold")
    contents = [rewrites_by_id[r["doc_id"]]["content"] for r in passed]
    if dup_threshold and len(passed) > 1:
        keep_idx, dropped = textstats.near_dup_filter(contents, dup_threshold)
        if dropped:
            utils.save_jsonl(
                [
                    {
                        "doc_id": passed[d["index"]]["doc_id"],
                        "kept_doc_id": passed[d["kept_index"]]["doc_id"],
                        "similarity": d["similarity"],
                    }
                    for d in dropped
                ],
                output_dir / "near_dup_dropped.jsonl",
            )
            print(f"  {len(dropped)} near-duplicate(s) culled")
        passed = [passed[i] for i in keep_idx]

    # Final corpus records carry the eval-facing fields (audit_sdf/diversity
    # read content/type/language/register) alongside the full matrix lineage.
    corpus = []
    for r in passed:
        rw = rewrites_by_id[r["doc_id"]]
        v = rw["variables"]
        corpus.append({
            "doc_id": r["doc_id"],
            "type_id": v.get("document_type", ""),
            "type_name": v.get("document_type", ""),
            "language": cp.derive_language(v.get("culture", "")),
            "register": v.get("tone", ""),
            "variables": v,
            "description": rw["description"],
            "content": rw["content"],
            "scores": r["scores"],
        })
    utils.save_jsonl(corpus, final_path)

    low_spec = [r for r in results if r["scores"]["spec_conformance"] < threshold]
    if low_spec:
        print(f"  Note: {len(low_spec)} document(s) scored <{threshold} on spec_conformance (advisory, not gated)")
    return corpus
