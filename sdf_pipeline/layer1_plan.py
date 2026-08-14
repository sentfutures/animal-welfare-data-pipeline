"""Layer 1: compose the prompt matrix (offline), then plan each document.

The composition half replaces the old LLM layers entirely: a deck sample
of the variables matrix (prompts/sdf/variables.txt x layer1.txt) yields one
fully-specified prompt per planned document, at zero API cost. The plan half
sends each composed prompt to the model, which works through the template's
questions and emits a self-contained DOCUMENT DESCRIPTION spec (or declares
the combination INCOHERENT). Only the extracted description travels
downstream; the working notes are scaffolding.

Checkpoint semantics (protects paid work):
- prompts.jsonl is composed once per run and reloaded on --resume (the deck
  sample is seeded, but reusing the file guarantees identity even if the
  variables file later changes in the repo — the run's snapshot is authority).
- A plan is checkpointed when it yields a description OR a deliberate
  INCOHERENT; malformed output (no tags, truncation) is NOT checkpointed, so
  --resume retries exactly the failed calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from sdf_pipeline import compose_prompts as cp


def run(config: dict, prompts_dir: Path, output_dir: Path) -> list[dict]:
    prompts_path = output_dir / "prompts.jsonl"
    plans_path = output_dir / "plans.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    sdf = config["sdf"]

    if prompts_path.exists():
        prompts = utils.load_jsonl(prompts_path)
        print(f"  Reusing {len(prompts)} composed prompts from this run")
    else:
        template = utils.sdf_template_path(prompts_dir, 1).read_text(encoding="utf-8")
        values, weights = cp.split_weights(cp.parse_variables(prompts_dir / "variables.txt"))
        preamble = (prompts_dir / "preamble.txt").read_text(encoding="utf-8")
        prompts = list(cp.compose_records(
            template, values, weights, preamble,
            n_prompts=sdf["n_prompts"],
            seed=sdf.get("seed", 0),
            entity_seed=sdf.get("entity_pool_seed", 137),
        ))
        for rec in prompts:
            utils.append_jsonl(rec, prompts_path)
        print(f"  Composed {len(prompts)} prompts (offline, $0.00)")

    existing = {r["prompt_id"]: r for r in utils.load_jsonl(plans_path)}
    results = [existing[p["prompt_id"]] for p in prompts if p["prompt_id"] in existing]
    pending = [
        p for p in prompts
        if p["prompt_id"] not in existing and not checkpoint.is_done(p["prompt_id"])
    ]

    def plan_one(rec: dict):
        try:
            return api.call_claude(
                rec["prompt"],
                system_prompt=rec.get("system") or "",
                model=sdf.get("plan_model"),
                stage="layer1_plan",
                item_id=rec["prompt_id"],
                return_stop_reason=True,
            )
        except Exception as e:
            # One poison prompt (e.g. a usage-policy false positive on the
            # claude_code backend) must not kill the layer and discard its
            # siblings' in-flight work. Failed work is not checkpointed, so
            # --resume retries exactly this call.
            return None, f"error: {type(e).__name__}: {e}"

    workers = config.get("workers", 1)
    failed_calls = 0
    for rec, (raw, stop) in zip(pending, utils.parallel_map(plan_one, pending, workers)):
        pid = rec["prompt_id"]
        if raw is None:
            failed_calls += 1
            print(f"  {pid}: API call failed ({stop}) — will retry on resume")
            continue
        if stop != "end_turn":
            print(f"  {pid}: truncated plan (stop_reason={stop}) — will retry on resume")
            continue
        description = cp.extract_description(raw)
        incoherent = cp.is_incoherent(raw)
        if description is None and not incoherent:
            print(f"  {pid}: no DOCUMENT DESCRIPTION found — will retry on resume")
            continue
        record = {
            "prompt_id": pid,
            "variables": rec["variables"],
            "plan": raw,
            "description": description,
            "incoherent": incoherent,
        }
        results.append(record)
        utils.append_jsonl(record, plans_path)
        checkpoint.mark_done(pid)
        print(f"  Planned {pid}" + (" (INCOHERENT)" if incoherent else ""))

    if pending and failed_calls == len(pending):
        raise SystemExit(
            "layer1_plan: every pending API call failed — this is systemic "
            "(auth, backend, or network), not per-document; fix and --resume."
        )
    incoherent_n = sum(1 for r in results if r.get("incoherent"))
    if incoherent_n:
        print(f"  {incoherent_n} combination(s) declared INCOHERENT — dropped")
    return results
