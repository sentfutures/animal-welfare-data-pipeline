"""Considerations targets and coverage measurement: pure code, no API calls, no files.

``entry_targets`` turns each catalog entry's target share into a document count
(the assignment in considerations_assignment.py enforces it as a cap).
``document_coverage_shares`` and ``corpus_coverage`` measure how documents ended
up covered, the latter on the final corpus records.
"""

from sdf_pipeline.considerations_catalog import (
    LIST_NAMES, MORAL_LIST, ConsiderationCatalog, target_count,
)


def entry_targets(catalog: ConsiderationCatalog, eligible_documents: int, digital_mind_documents: int) -> dict[str, int]:
    """{label: target_count}: digital-minds-only entries take a share of the digital-mind
    documents, every other entry a share of all eligible documents."""
    return {
        entry.label: target_count(
            entry.target_share, digital_mind_documents if entry.digital_minds_only else eligible_documents)
        for entry in catalog.all_entries
    }


def document_coverage_shares(per_document_counts: list[dict[str, int]]) -> dict:
    """How documents are covered, from each document's pick count per list:
    shares with no consideration from either list, with picks from only one list,
    and from both, and the number carrying two or more picks from the same list."""
    total = len(per_document_counts)
    none = only_one = both = multiple_in_a_list = 0
    for counts in per_document_counts:
        lists_used = sum(counts[name] > 0 for name in LIST_NAMES)
        none += lists_used == 0
        only_one += lists_used == 1
        both += lists_used == 2
        multiple_in_a_list += any(counts[name] >= 2 for name in LIST_NAMES)
    share = (lambda n: round(n / total, 4)) if total else (lambda n: None)
    return {
        "documents_with_no_consideration": none,
        "documents_with_only_one_list": only_one,
        "documents_with_both_lists": both,
        "share_with_no_consideration": share(none),
        "share_with_only_one_list": share(only_one),
        "share_with_both_lists": share(both),
        "documents_with_two_or_more_in_one_list": multiple_in_a_list,
    }


def corpus_coverage(catalog: ConsiderationCatalog, variables_list: list[dict]) -> dict:
    """The same report measured on finished documents' ``variables`` (the final
    corpus), where drafting failures and the score gate have shifted the shares
    since selection. Documents with a skipped resolution are left out; there is
    no pass or rejection data here, so those columns are absent."""
    documents = [v for v in variables_list if not catalog.is_skipped_resolution(v)]
    digital = [v for v in documents if catalog.is_digital_mind_document(v)]
    entries = []
    for entry in catalog.all_entries:
        pool = digital if entry.digital_minds_only else documents
        field = "moral_considerations" if entry.list_name == MORAL_LIST else "epistemic_considerations"
        achieved = sum(entry.label in (v.get(field) or []) for v in pool)
        entries.append({
            "label": entry.label, "list": entry.list_name,
            "digital_minds_only": entry.digital_minds_only,
            "target_share": entry.target_share, "eligible_documents": len(pool),
            "achieved": achieved,
            "achieved_share": round(achieved / len(pool), 4) if pool else None,
        })
    per_document = [
        {MORAL_LIST: len(v.get("moral_considerations") or []),
         "epistemic": len(v.get("epistemic_considerations") or [])}
        for v in documents
    ]
    return {
        "eligible_documents": len(documents),
        "digital_mind_documents": len(digital),
        "entries": entries,
        **document_coverage_shares(per_document),
    }
