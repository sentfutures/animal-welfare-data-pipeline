"""Step 3: Rewrite responses into training-ready form — the alignment-critical pass.

The rewrite is anchored on the distilled constitution principles (each with
its verbatim constitution quote), per prompts/dad/step3_rewrite.txt. That
template splits (via utils.load_split_prompt) into a system half — the
rewrite instructions plus the principles block — and a user half carrying
the draft and user message. The step-1 dealt cards are not passed to the call
(they ride along in the audit record, and are projected onto the corpus
record's `variables` for inspection): the full constitution was context for
distilling the principles, not a per-call dependency.
"""

import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils, constitution_loader
from dad_pipeline.id_registry import IdRegistry, example_fingerprint, prompt_key, registry_path
from dad_pipeline.compose_scenarios import dealt_variables
from dad_pipeline.step1_dilemmas import dealt_cards


def run(
    config: dict,
    prompts_dir: Path,
    output_dir: Path,
    final_dir: Path,
    kept_responses: list[dict],
) -> list[dict]:
    audit_path = output_dir / "rewrites.jsonl"
    final_path = final_dir / "dad_corpus.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    # Stable content-keyed example ids (E-####: the finished user/assistant
    # pair), shared across runs; response_gid (R-####) rides in from step 2.
    registry = IdRegistry(registry_path(output_dir))

    constitution_dir = utils.resolve_constitution_dir(prompts_dir)
    try:
        principles = constitution_loader.load_principles(constitution_dir)
    except FileNotFoundError:
        # Snapshot predates the principles CSV — fall back to the live repo copy
        print("  WARNING: run snapshot has no constitution_principles.csv; using the repo's live copy.")
        principles = constitution_loader.load_principles()
    principles_block = constitution_loader.format_principles(principles)

    existing_audit = utils.load_jsonl(audit_path)
    results = list(existing_audit)
    done_response_ids = {r["response_id"] for r in results}

    pending = [r for r in kept_responses
               if r["response_id"] not in done_response_ids
               and not checkpoint.is_done(r["response_id"])]

    def rewrite_response(resp: dict) -> dict:
        """API call + parsing only — all writes and checkpoint marks stay on
        the main thread, in input order (the parallel_map contract)."""
        print(f"  Rewriting response for {prompt_key(resp)}...")
        system_prompt, user_prompt = utils.load_split_prompt(
            prompts_dir / "step3_rewrite.txt",
            principles_block=principles_block,
            user_message=resp["user_message"],
            draft_response=resp["assistant_response"],
        )

        rewritten, stop_reason = api.call_claude(
            user_message=user_prompt, system_prompt=system_prompt,
            max_tokens=4000, return_stop_reason=True,
            model=config["dad"].get("constitution_rewrite_model"),
            stage="constitution_rewrite", item_id=resp["response_id"])
        # A legitimately long rewrite can exceed the cap; retry once with a
        # doubled budget before deferring, so long-form cases aren't silently
        # re-skipped on every resume.
        if stop_reason == "max_tokens":
            print(f"    {prompt_key(resp)}: rewrite hit the 4000-token cap — retrying at 8000.")
            rewritten, stop_reason = api.call_claude(
                user_message=user_prompt, system_prompt=system_prompt,
                max_tokens=8000, return_stop_reason=True,
                model=config["dad"].get("constitution_rewrite_model"),
                stage="constitution_rewrite", item_id=resp["response_id"])
        return {"resp": resp, "rewritten": rewritten.strip(), "stop_reason": stop_reason}

    workers = int(config.get("workers", 1))
    for out in utils.parallel_map(rewrite_response, pending, workers):
        resp, rewritten, stop_reason = out["resp"], out["rewritten"], out["stop_reason"]
        rid = resp["response_id"]
        scenario_cards = resp.get("scenario_cards") or resp.get("annotation") or {}

        # A truncated (max_tokens), refusal-cut, empty, or transcript-echoed
        # rewrite must never become a training record. Skip it without
        # checkpointing so a later --resume retries it, and log it so
        # repeated failures are visible rather than silent.
        if (not rewritten or stop_reason in ("max_tokens", "refusal")
                or utils.looks_like_transcript_echo(rewritten)):
            why = ("truncated at max_tokens (even at 8000)" if stop_reason == "max_tokens"
                   else "cut by the refusal classifier (stop_reason=refusal)" if stop_reason == "refusal"
                   else "transcript echo (reply wrapped in USER:/ASSISTANT: replay)"
                   if rewritten else "empty")
            print(f"    Skipping {prompt_key(resp)}: rewrite {why} — not written, will retry on resume.")
            utils.append_jsonl(
                {"response_id": rid, "prompt_gid": prompt_key(resp), "reason": why},
                output_dir / "rewrite_failures.jsonl",
            )
            continue

        record_id = str(uuid.uuid4())

        # Full audit record (includes the annotation + retrieval trail for inspection)
        audit_record = {
            "record_id": record_id,
            "example_gid": registry.gid(
                "example", example_fingerprint(resp["user_message"], rewritten)),
            "response_id": rid,
            "response_gid": resp.get("response_gid"),
            "prompt_gid": prompt_key(resp),
            "sample_index": resp.get("sample_index", 0),
            "tensions": resp.get("tensions", []),
            "entry_ids": resp.get("entry_ids", []),
            "user_message": resp["user_message"],
            "draft_response": resp["assistant_response"],
            "rewritten_response": rewritten,
            "scenario_cards": scenario_cards,
        }
        results.append(audit_record)
        utils.append_jsonl(audit_record, audit_path)
        registry.save()
        checkpoint.mark_done(rid)
        done_response_ids.add(rid)

    # Write final training-ready corpus — user + assistant messages plus the
    # stable ids (lineage keys, stripped with record_id at export; never text
    # the model trains on) and the dealt cards. The cards are metadata about
    # the record in the same way the ids are, not scaffolding the model reads:
    # they say which slice of the matrix this example came from, and the SDF
    # corpus carries its own dealt combination for the same reason
    # (sdf_pipeline/layer4_score.py). The scope, the library rows and the
    # constitution stay stripped.
    utils.ensure_dir(final_dir)
    final_records = []
    for r in results:
        final_records.append({
            "record_id": r["record_id"],
            "example_gid": r.get("example_gid"),
            "response_gid": r.get("response_gid"),
            "variables": dealt_variables(dealt_cards(r)),
            "messages": [
                {"role": "user", "content": r["user_message"]},
                {"role": "assistant", "content": r["rewritten_response"]},
            ],
        })

    utils.save_jsonl(final_records, final_path)
    print(f"  Rewrote {len(results)} responses. Final corpus: {len(final_records)} training records.")
    return final_records
