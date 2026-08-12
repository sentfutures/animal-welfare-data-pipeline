"""Baseline responses: the unguided control arm for each dilemma.

For every step-1 dilemma, one extra call sends the finished (1c) user prompt
verbatim to a plain model — no system prompt, no scope, no reasoning library,
no constitution. The result shows what an off-the-shelf model says to the same
question the pipeline answers; the viewer renders it side by side with the
final response.

Config: dad.baseline.enabled toggles the stage (absent means on);
dad.baseline.model names the plain model (falls back to the global `model`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from dad_pipeline.id_registry import IdRegistry, prompt_key, registry_path, response_fingerprint


# Token budget for the plain arm, and the doubled budget it escalates to on a
# truncation. Deliberately not the global `max_tokens`: the pipeline stages get
# their length discipline from their system prompts, and this call has none.
BASE_MAX_TOKENS = 4000
BASE_MAX_TOKENS_RETRY = 8000


def enabled(config: dict) -> bool:
    """The stage runs unless dad.baseline.enabled is explicitly false — a
    config without the block (older configs, pared-down dev configs) gets the
    control arm by default."""
    return bool((config["dad"].get("baseline") or {}).get("enabled", True))


def run(config: dict, output_dir: Path, dilemmas: list[dict]) -> list[dict]:
    output_path = output_dir / "baseline_responses.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    # Stable content-keyed control-arm ids (C-####), shared across runs.
    registry = IdRegistry(registry_path(output_dir))

    existing = utils.load_jsonl(output_path)
    results = list(existing)
    done = {prompt_key(r) for r in existing}

    pending = [d for d in dilemmas
               if prompt_key(d) not in done
               and not checkpoint.is_done(prompt_key(d))]

    model = (config["dad"].get("baseline") or {}).get("model")

    def baseline_call(d: dict) -> dict:
        """API call only — all writes and checkpoint marks stay on the main
        thread, in input order (the parallel_map contract)."""
        pid = prompt_key(d)
        print(f"  Baseline response for {pid}...")
        response, stop_reason = api.call_claude(
            user_message=d["user_message"], system_prompt="",
            max_tokens=BASE_MAX_TOKENS, return_stop_reason=True,
            model=model,
            stage="baseline_response", item_id=pid)
        # The plain arm is the ONE call with no system prompt and so no length
        # guidance at all, which makes it the most truncation-prone stage in the
        # pipeline: a full run lost 18 of 190 control arms this way, and every
        # skip costs a paid call whose replacement is skipped again on the next
        # resume. Retry once with a doubled budget before giving up (same shape
        # as the step-3 rewrite escalation).
        if stop_reason == "max_tokens":
            print(f"    {pid}: baseline hit the {BASE_MAX_TOKENS}-token cap — "
                  f"retrying at {BASE_MAX_TOKENS_RETRY}.")
            response, stop_reason = api.call_claude(
                user_message=d["user_message"], system_prompt="",
                max_tokens=BASE_MAX_TOKENS_RETRY, return_stop_reason=True,
                model=model,
                stage="baseline_response", item_id=pid)
        return {"dilemma": d, "response": response.strip(), "stop_reason": stop_reason}

    workers = int(config.get("workers", 1))
    for out in utils.parallel_map(baseline_call, pending, workers):
        d, response, stop_reason = out["dilemma"], out["response"], out["stop_reason"]
        pid = prompt_key(d)
        # A truncated, refusal-cut, or empty reply is not a usable comparison
        # arm. Skip without checkpointing so --resume retries it (fail-soft: a
        # baseline failure never stops the run — it only costs the comparison).
        if not response or stop_reason in ("max_tokens", "refusal"):
            why = (f"truncated at max_tokens (even at {BASE_MAX_TOKENS_RETRY})"
                   if stop_reason == "max_tokens"
                   else "cut by the refusal classifier (stop_reason=refusal)" if stop_reason == "refusal"
                   else "empty")
            print(f"    Skipping {pid}: baseline {why} — not written, will retry on resume.")
            continue
        record = {
            "prompt_gid": pid,
            "plain_gid": registry.gid("plain", response_fingerprint(response)),
            "user_message": d["user_message"],
            "baseline_response": response,
            "model": model or config.get("model"),
        }
        results.append(record)
        utils.append_jsonl(record, output_path)
        checkpoint.mark_done(pid)
        registry.save()

    print(f"  Total baseline responses: {len(results)}.")
    return results
