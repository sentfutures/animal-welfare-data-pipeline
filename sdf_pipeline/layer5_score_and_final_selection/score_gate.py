"""Layer 5: the score gate (no model call): keeps documents with alignment AND realism >= ``sdf.min_score_threshold``.
spec_conformance is reported but does not gate; a parse-error 5/5/5 falls below the default threshold of 7."""


def keep_documents_at_or_above_threshold(score_records: list[dict], threshold) -> list[dict]:
    return [record for record in score_records
            if record["scores"]["alignment"] >= threshold and record["scores"]["realism"] >= threshold]
