"""Layer 3: review and rewrite each draft against the constitution and its spec.

The alignment-critical pass — do not skip or abbreviate it. The layer3.txt
template holds the constitution, principles, and the nine review checks in its
SYSTEM section; the USER section delivers the generating spec and the draft.
The rewrite must come back inside <improved_document> tags; the review text
preceding the tags is kept as the review record. Missing tags or truncation
are not checkpointed, so --resume retries exactly the failed calls.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, constitution_loader, utils
from sdf_pipeline import compose_prompts as cp

IMPROVED_TAG_RE = re.compile(r"<improved_document>(.*?)</improved_document>", re.DOTALL)

_MAX_TOKENS = 8000  # review record + a full rewritten document


def run(config: dict, prompts_dir: Path, output_dir: Path, drafts: list[dict]) -> list[dict]:
    output_path = output_dir / "rewrites.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    sdf = config["sdf"]

    constitution_dir = utils.resolve_constitution_dir(prompts_dir)
    constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
    principles = constitution_loader.format_principles(
        constitution_loader.load_principles(constitution_dir)
    )

    existing = {r["doc_id"]: r for r in utils.load_jsonl(output_path)}
    results = [existing[d["doc_id"]] for d in drafts if d["doc_id"] in existing]
    pending = [
        d for d in drafts
        if d["doc_id"] not in existing and not checkpoint.is_done(d["doc_id"])
    ]

    def rewrite_one(draft: dict):
        system, user = cp.split_sections(utils.load_prompt(
            utils.sdf_template_path(prompts_dir, 3),
            constitution_claude=constitution_claude,
            constitution_principles=principles,
            document_description=draft["description"],
            document=draft["content"],
        ))
        try:
            return api.call_claude(
                user,
                system_prompt=system or "",
                model=sdf.get("rewrite_model"),
                max_tokens=_MAX_TOKENS,
                stage="layer3_rewrite",
                item_id=draft["doc_id"],
                return_stop_reason=True,
                cache_system=True,  # constitution + nine checks are identical across rewrites
            )
        except Exception as e:
            # Per-item failures (e.g. a usage-policy false positive on the
            # claude_code backend) skip the doc instead of killing the layer;
            # unmarked work is retried by --resume.
            return None, f"error: {type(e).__name__}: {e}"

    workers = config.get("workers", 1)
    failed_calls = 0
    for draft, (raw, stop) in zip(pending, utils.parallel_map(rewrite_one, pending, workers)):
        did = draft["doc_id"]
        if raw is None:
            failed_calls += 1
            print(f"  {did}: API call failed ({stop}) — will retry on resume")
            continue
        if stop != "end_turn":
            print(f"  {did}: truncated rewrite (stop_reason={stop}) — will retry on resume")
            continue
        m = IMPROVED_TAG_RE.search(raw)
        content = m.group(1).strip() if m else ""
        if not content:
            print(f"  {did}: no <improved_document> tags — will retry on resume")
            continue
        record = {
            "doc_id": did,
            "variables": draft["variables"],
            "description": draft["description"],
            "review": raw[:m.start()].strip(),
            "content": content,
        }
        results.append(record)
        utils.append_jsonl(record, output_path)
        checkpoint.mark_done(did)
        print(f"  Rewrote {did} ({len(content)} chars)")

    if pending and failed_calls == len(pending):
        raise SystemExit(
            "layer3_rewrite: every pending API call failed — this is systemic "
            "(auth, backend, or network), not per-document; fix and --resume."
        )
    return results
