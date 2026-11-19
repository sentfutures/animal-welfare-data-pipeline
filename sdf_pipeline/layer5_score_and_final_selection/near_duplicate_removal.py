"""Layer 5: the near-duplicate removal (no model call), a deterministic greedy keep-first filter."""

from shared import textstats


def remove_near_duplicates(records: list[dict], near_duplicate_threshold) -> tuple[list[dict], list[dict]]:
    """``(records kept, records dropped as {"doc_id", "kept_doc_id", "similarity"})``, keeping the first of each pair in the
    order given; a falsy threshold removes nothing."""
    if not near_duplicate_threshold or len(records) < 2:
        return records, []
    _, dropped_entries = textstats.near_dup_filter([record["content"] for record in records], near_duplicate_threshold)
    dropped_by_index = {entry["index"]: entry for entry in dropped_entries}
    kept = [record for index, record in enumerate(records) if index not in dropped_by_index]
    dropped = [{"doc_id": records[index]["doc_id"], "kept_doc_id": records[entry["kept_index"]]["doc_id"],
                "similarity": entry["similarity"]}
               for index, entry in sorted(dropped_by_index.items())]
    return kept, dropped
