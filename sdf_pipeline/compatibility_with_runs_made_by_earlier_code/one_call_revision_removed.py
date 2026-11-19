"""Refuses ``sdf.revision.split: false``, the removed one-call revision, in a configuration and in the saved configuration of a run being resumed."""

import json
from pathlib import Path

from shared import utils


def _sets_one_call_revision(config: dict | None) -> bool:
    return ((((config or {}).get("sdf") or {}).get("revision") or {}).get("split", True)) is False


def refuse_configuration_with_one_call_revision(config: dict, source: str = "the configuration") -> None:
    """Exit before any folder is created or call made when ``config`` sets ``sdf.revision.split: false``; ``true`` is ignored."""
    if _sets_one_call_revision(config):
        raise SystemExit(f"{source} sets sdf.revision.split: false, but the one-call revision was removed: layer 4 always "
                         "critiques and then revises in two calls. Remove the sdf.revision.split key.")


def refuse_resuming_run_with_one_call_revision(run_dir: Path) -> None:
    """Exit before any call when the run's saved configuration sets ``split: false`` and some draft has no rewrite yet."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return
    if not _sets_one_call_revision(json.loads(manifest_path.read_text(encoding="utf-8")).get("config")):
        return
    draft_ids = {record["doc_id"] for record in utils.load_jsonl(utils.sdf_stage_file(run_dir, "draft"))}
    rewritten_ids = {record["doc_id"] for record in utils.load_jsonl(utils.sdf_stage_file(run_dir, "rewrite"))}
    if draft_ids and draft_ids <= rewritten_ids:
        return
    raise SystemExit(f"Cannot resume {Path(run_dir).name}: its saved configuration sets sdf.revision.split: false, but the "
                     "one-call revision was removed, and its layer 4 is not finished. Only a run whose layer 4 is finished "
                     "can be resumed (from layer 5), or start a new run.")
