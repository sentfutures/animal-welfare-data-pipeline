"""Derive today's ``final/sdf_corpus.jsonl`` records from ``DocumentState`` records.

``final_corpus_record`` is the one definition of a final corpus record: ``layer4_score`` calls it too.
Fields, field order and values are the ones the final corpus has today, and the document order is the
order of the states handed in (deal order from the loader). Today's file has that order for a run that
was not resumed; for a resumed run today's order depends on what was finished before the resume.

Each record is built from the document's latest text version and what that version was written from,
never from the latest plan, exactly as ``layer4_score`` reads the rewrite record.

Layouts:
* current and matrix layouts: ``doc_id, type_id, type_name, language, register, variables, [variable_value_ids,] description,
  considerations_spec, content, scores``; ``considerations_spec`` is written as "" when the record
  lacked it in the current layout, and left out in the matrix layout before the renumbering, whose
  corpus never had the field.
* five-layer layout: the score record as stored (``doc_id``, the legacy fields, ``content``, ``scores``,
  then ``latent_beat_ok`` where the record had it).
"""

from sdf_pipeline import compose_prompts
from sdf_pipeline.document_state import DocumentOutcome, DocumentState, RunLayout

_FIVE_LAYER_FIELDS_WRITTEN_AFTER_SCORES = ("latent_beat_ok",)


def final_corpus_record(
        document_id: str, variables: dict, description: str, considerations_spec: str | None, content: str,
        scores: dict, variable_value_ids: dict | None = None) -> dict:
    """The one definition of a record of ``final/sdf_corpus.jsonl`` (current and matrix layouts), with
    its field order. Both ``layer4_score`` (from the rewrite and score records) and
    ``final_corpus_record_for_state`` (from a ``DocumentState``) build the record here.
    ``considerations_spec`` None leaves the field out (the matrix layout before the renumbering).
    ``variable_value_ids`` (the dealt value ids, ``variable_value_ids`` of the stage files) is written
    right after ``variables`` when given, and left out when None (a run that does not record them)."""
    record = {
        "doc_id": document_id,
        "type_id": variables.get("document_type", ""),
        "type_name": variables.get("document_type", ""),
        "language": compose_prompts.derive_language(variables.get("culture", "")),
        "register": variables.get("tone", ""),
        "variables": variables,
    }
    if variable_value_ids is not None:
        record["variable_value_ids"] = variable_value_ids
    record["description"] = description
    if considerations_spec is not None:
        record["considerations_spec"] = considerations_spec
    record["content"] = content
    record["scores"] = scores
    return record


def final_corpus_record_for_state(state: DocumentState) -> dict:
    latest = state.latest_text_version
    scores = state.scores.recorded_values
    if state.layout is RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX:
        before = {k: v for k, v in state.legacy_fields.items() if k not in _FIVE_LAYER_FIELDS_WRITTEN_AFTER_SCORES}
        after = {k: v for k, v in state.legacy_fields.items() if k in _FIVE_LAYER_FIELDS_WRITTEN_AFTER_SCORES}
        return {"doc_id": state.document_id, **before, "content": latest.text, "scores": scores, **after}
    considerations_spec = latest.written_from.considerations_spec
    if state.layout is RunLayout.CURRENT:
        considerations_spec = considerations_spec or ""
    return final_corpus_record(
        state.document_id, latest.written_from.variables, latest.written_from.description,
        considerations_spec, latest.text, scores, latest.written_from.variable_value_ids)


def derive_final_corpus_records(states) -> list:
    """The final corpus records of every state whose outcome is final, in the order of ``states``."""
    return [final_corpus_record_for_state(state) for state in states if state.outcome is DocumentOutcome.FINAL]
