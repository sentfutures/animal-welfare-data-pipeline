"""Layer 5 step: the near-duplicate removal (no model call), a deterministic greedy keep-first filter."""

from shared import textstats
from sdf_pipeline.update_steps import Accepted, CorpusStep, Removed


class NearDuplicateRemovalStep(CorpusStep):
    """Greedy keep-first near-duplicate removal over the documents given, in the order given. The
    documents are score records carrying ``content``. A falsy threshold, or fewer than two
    documents, removes nothing. A removal's ``detail`` is {"kept_doc_id", "similarity"}."""

    step_name = "near_duplicate_removal"

    def __init__(self, near_duplicate_threshold):
        self.near_duplicate_threshold = near_duplicate_threshold

    def apply(self, records: list) -> dict:
        outcomes = {record["doc_id"]: Accepted(None) for record in records}
        if self.near_duplicate_threshold and len(records) > 1:
            _, dropped = textstats.near_dup_filter(
                [record["content"] for record in records], self.near_duplicate_threshold)
            for entry in dropped:
                outcomes[records[entry["index"]]["doc_id"]] = Removed(
                    "near_duplicate",
                    {"kept_doc_id": records[entry["kept_index"]]["doc_id"], "similarity": entry["similarity"]})
        return outcomes
