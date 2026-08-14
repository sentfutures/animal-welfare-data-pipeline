"""Layer 2: draft one document per plan, from its DOCUMENT DESCRIPTION spec.

The layer2.txt template carries the constitution and principles in its SYSTEM
section (rendered from the run's constitution snapshot) and the per-document
spec in its USER section. The draft must come back inside <document> tags;
anything else — missing tags, truncation — is not checkpointed, so --resume
retries exactly the failed calls (temperature 1.0 makes retries productive).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, constitution_loader, utils
from sdf_pipeline import compose_prompts as cp

DOC_TAG_RE = re.compile(r"<document>(.*?)</document>", re.DOTALL)

_MAX_TOKENS = 6000  # real documents are often long; the cap bounds api-backend cost


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    output_path = output_dir / "drafts.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    sdf = config["sdf"]

    constitution_dir = utils.resolve_constitution_dir(prompts_dir)
    constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
    principles = constitution_loader.format_principles(
        constitution_loader.load_principles(constitution_dir)
    )
    preamble = (prompts_dir / "preamble.txt").read_text(encoding="utf-8").strip()

    planned = [p for p in plans if p.get("description")]
    existing = {r["doc_id"]: r for r in utils.load_jsonl(output_path)}
    results = [existing[p["prompt_id"]] for p in planned if p["prompt_id"] in existing]
    pending = [
        p for p in planned
        if p["prompt_id"] not in existing and not checkpoint.is_done(p["prompt_id"])
    ]

    def draft_one(plan: dict):
        system, user = cp.split_sections(utils.load_prompt(
            utils.sdf_template_path(prompts_dir, 2),
            preamble=preamble,
            constitution_claude=constitution_claude,
            constitution_principles=principles,
            document_description=plan["description"],
            # Downstream-only matrix axis (sampled in layer 1, recorded in
            # variables). Default to the neutral value for plans composed
            # before this axis existed, so --resume on an old run never crashes.
            reasoning_featured=plan.get("variables", {}).get(
                "reasoning_featured", "any relevant principles from the constitution"
            ),
        ))
        try:
            return api.call_claude(
                user,
                system_prompt=system or "",
                model=sdf.get("draft_model"),
                max_tokens=_MAX_TOKENS,
                stage="layer2_draft",
                item_id=plan["prompt_id"],
                return_stop_reason=True,
                cache_system=True,  # constitution-laden system prompt is identical across drafts
            )
        except Exception as e:
            # One poison document (e.g. a usage-policy false positive on the
            # claude_code backend) must not kill the layer and discard its
            # siblings' in-flight work. Failed work is not checkpointed, so
            # --resume retries exactly this call.
            return None, f"error: {type(e).__name__}: {e}"

    workers = config.get("workers", 1)
    failed_calls = 0
    for plan, (raw, stop) in zip(pending, utils.parallel_map(draft_one, pending, workers)):
        pid = plan["prompt_id"]
        if raw is None:
            failed_calls += 1
            print(f"  {pid}: API call failed ({stop}) — will retry on resume")
            continue
        if stop != "end_turn":
            print(f"  {pid}: truncated draft (stop_reason={stop}) — will retry on resume")
            continue
        m = DOC_TAG_RE.search(raw)
        content = m.group(1).strip() if m else ""
        if not content:
            print(f"  {pid}: no <document> tags — will retry on resume")
            continue
        record = {
            "doc_id": pid,
            "variables": plan["variables"],
            "description": plan["description"],
            "content": content,
        }
        results.append(record)
        utils.append_jsonl(record, output_path)
        checkpoint.mark_done(pid)
        print(f"  Drafted {pid} ({len(content)} chars)")

    if pending and failed_calls == len(pending):
        raise SystemExit(
            "layer2_draft: every pending API call failed — this is systemic "
            "(auth, backend, or network), not per-document; fix and --resume."
        )
    return results
