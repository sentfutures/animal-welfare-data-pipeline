"""Build ``DocumentState`` records from the stage files an SDF run folder holds today.

``load_document_states(run_directory)`` reads a run folder in any of the three layouts
(``RunLayout`` in ``document_state.py``), whatever stage the run stopped at, and returns one
``DocumentState`` per document. It only reads: it never creates, modifies or deletes a file in the
run folder, and it does not use ``utils.Checkpoint`` or any other class that could write.

Order
-----
States come back in deal order (round, then hand index) for every hand of the deal. Where a layout
has a deal, the deal is ``screen_hands.jsonl`` (every hand ever dealt, restricted to the rounds that
have a record in ``screen_rounds.jsonl``, as the screen does), or ``prompts.jsonl`` when there is no
screen (screen disabled, or a run from before the screen). Ids that appear in later files but not in
the deal (hand-seeded runs) follow, in the order the stage files first mention them, with a note.
The five-layer layout has no deal: its documents come in the order of ``layer3/drafts.jsonl`` (the
order the stages wrote them), then any id only later files mention. Chosen because that is the order
today's final corpus has for a run that was not resumed.

Damaged files
-------------
An unparseable last line of a jsonl file (what an interrupted write leaves) is skipped and reported in
the run notes. An unparseable line anywhere else raises ``StageFileCorruptError`` naming the file and
line, because skipping it would silently lose a record. A missing file is an empty file.

How the outcome is decided (from stage files only)
--------------------------------------------------
* removed at the screen: a record in ``screen_rejections.jsonl``;
* removed at the plan: ``plans.jsonl`` ``incoherent`` with no description;
* removed at the description rewrite: ``dealt_value_conflict`` in ``description_rewrites.jsonl``;
* gate and cull: only for a document that has a draft, a revision and scores, each reachable the way
  today's stages reach them (a stage works from the previous stage's list, so a score whose draft
  record is missing is ignored by a full run), and only once ``final/sdf_corpus.jsonl`` exists, which is
  the evidence that the gate ran. The gate applies ``sdf.min_score_threshold`` from ``run_manifest.json``
  (or the ``gate_score_threshold`` argument) to alignment and realism; ``near_dup_dropped.jsonl`` names the
  cull's removals. The final corpus is read only as a witness: a document that passes both but is not in it
  is marked removed with ``ABSENT_FROM_FINAL_CORPUS_WITHOUT_RECORDED_REASON``.
* everything else is in progress, with the furthest step that has a record.

Limits worth knowing: ``--layer N`` runs start from a stage file without the earlier stages' files, so
whether an upstream gap made today's code skip a document depends on the invocation, which the files do
not record; the loader assumes a full run (and says so in the document's notes when it finds a gap).
The gate threshold comes from the manifest's config as the run was created, so a resume under another
threshold is not seen.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from sdf_pipeline.dealt_value_identifiers import load_frozen_value_ids, value_ids_of_dealt_hand_from_snapshot
from sdf_pipeline.document_event_log import DOCUMENT_EVENT_LOG_FILE_NAME, fold_dealt_value_ids, read_document_events
from sdf_pipeline.document_state import (
    AssignedConsideration, ConsiderationScoringRecord, DescriptionRewriteDecision,
    DescriptionRewriteVerdict, DescriptionVersion, DescriptionVersionKind, DocumentOutcome,
    DocumentRemoval, DocumentScores, DocumentState, PipelineStep, RecordedFailure,
    RemovalReasonCode, RunLayout, ScreenJudgment, TextVersion, TextVersionKind, TextWrittenFrom,
    position_of_step,
)

FINAL_CORPUS_RELATIVE_PATH = "final/sdf_corpus.jsonl"
MANIFEST_FILE_NAME = "run_manifest.json"
NEAR_DUPLICATE_FILE_NAME = "near_dup_dropped.jsonl"

# stage folder names per layout: (plan, draft, rewrite, score); the five-layer layout is read separately
_MATRIX_STAGE_FOLDERS = {
    RunLayout.CURRENT: ("layer1", "layer2", "layer3", "layer4"),
    RunLayout.MATRIX_BEFORE_LAYER_RENUMBERING: ("layer12", "layer3", "layer4", "layer5"),
}


class UnknownRunLayoutError(ValueError):
    """The folder is not an SDF run folder in any layout the loader knows."""


class StageFileCorruptError(ValueError):
    """A jsonl line that is not the last line of its file could not be parsed."""


@dataclass(frozen=True)
class LoadedDocumentStates:
    run_directory: Path
    layout: RunLayout
    states: list
    run_notes: list


def detect_run_layout(run_directory) -> RunLayout:
    run = Path(run_directory)
    if not run.is_dir():
        raise UnknownRunLayoutError(f"{run} is not a folder, so it is not an SDF run folder")
    if (run / "layer12").is_dir():
        return RunLayout.MATRIX_BEFORE_LAYER_RENUMBERING
    if ((run / "layer5").is_dir() or (run / "layer1" / "document_types.jsonl").is_file()
            or (run / "layer2" / "subtypes.jsonl").is_file()):
        return RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX
    if any((run / f"layer{number}").is_dir() for number in (1, 2, 3, 4)):
        return RunLayout.CURRENT
    raise UnknownRunLayoutError(
        f"{run} has none of the stage folders of an SDF run (layer1 to layer4, layer12, or layer1 to layer5), "
        "so its layout is unknown and it was not read")


def load_document_states(run_directory, *, gate_score_threshold=None) -> list:
    """The states of every document of the run, in the order described in the module docstring."""
    return load_document_states_with_run_notes(run_directory, gate_score_threshold=gate_score_threshold).states


def load_document_states_with_run_notes(run_directory, *, gate_score_threshold=None) -> LoadedDocumentStates:
    run = Path(run_directory)
    layout = detect_run_layout(run)
    files = _RunFiles(run)
    threshold = gate_score_threshold
    if threshold is None:
        manifest = files.json_object(run / MANIFEST_FILE_NAME)
        threshold = ((manifest.get("config") or {}).get("sdf") or {}).get("min_score_threshold")
    if layout is RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX:
        states = _build_five_layer_states(files, run, threshold)
    else:
        states = _build_matrix_states(files, run, layout, threshold)
    return LoadedDocumentStates(run, layout, states, files.run_notes)


def load_document_states_or_empty_list_when_not_a_run_folder(run_directory, *, gate_score_threshold=None) -> list:
    """``load_document_states``, except that a folder with none of the stage folders of an SDF run (or no
    folder at all) gives an empty list instead of ``UnknownRunLayoutError``.

    For readers that used to open stage files one by one and treated a missing file as empty (the report and
    the audits, which accept a bare corpus file that has no run folder around it). A damaged stage file still
    raises ``StageFileCorruptError``."""
    try:
        return load_document_states(run_directory, gate_score_threshold=gate_score_threshold)
    except UnknownRunLayoutError:
        return []


def plan_records_of_states(states) -> list:
    """The plan stage's records in the shape the description rewrite's share rows and the report read them
    (``prompt_id``, ``variables``, ``description``), one per state that has a plan record, in state order.
    ``description`` is the planner's own text, never the rewritten one, and is None for a plan declared
    incoherent (the plan file stores such a plan with no description)."""
    return [{"prompt_id": state.document_id, "variables": state.plan_record_variables,
             "description": state.planned_description}
            for state in states if state.has_plan_record]


# ------------------------------------------------------------------------------ reading files

class _RunFiles:
    """Read-only access to a run folder's files, collecting notes about damaged lines."""

    def __init__(self, run: Path):
        self.run = run
        self.run_notes: list = []

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.run))

    def records(self, path: Path) -> list:
        if not path.is_file():
            return []
        lines = [(number, line.strip()) for number, line in
                 enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip()]
        records = []
        for position, (number, line) in enumerate(lines):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("not a JSON object")
            except ValueError as error:
                if position == len(lines) - 1:
                    self.run_notes.append(
                        f"{self.relative(path)} line {number} is cut off or unparseable ({error}); it was skipped")
                    continue
                raise StageFileCorruptError(
                    f"{path} line {number} cannot be parsed ({error}) and is not the last line, "
                    "so reading it would silently lose a record") from error
            records.append(record)
        return records

    def json_object(self, path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            self.run_notes.append(f"{self.relative(path)} cannot be parsed ({error}); it was ignored")
            return {}
        return value if isinstance(value, dict) else {}

    def checkpoint_ids(self, path: Path) -> set:
        return {str(identifier) for identifier in self.json_object(path).get("completed", [])}


def _last_record_per_id(records: list, key: str) -> dict:
    """{id: record}, a later record replacing an earlier one in place of its position of first
    appearance, as ``{r[key]: r for r in records}`` does in the pipeline's stages."""
    return {record[key]: record for record in records if record.get(key) is not None}


def _hand_index(document_id: str):
    try:
        return int(document_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _written_from(record: dict) -> TextWrittenFrom:
    return TextWrittenFrom(
        description=record.get("description"), variables=record.get("variables"),
        considerations_spec=record.get("considerations_spec"))


def _label_pairs_of_assignment(assignment) -> frozenset:
    return frozenset((item.list_name, item.label) for item in assignment or ())


def _label_pairs_of_variables(variables) -> frozenset:
    pairs = set()
    for list_name in ("moral", "epistemic"):
        for label in (variables or {}).get(f"{list_name}_considerations") or []:
            pairs.add((list_name, label))
    return frozenset(pairs)


def _furthest_step(steps_with_evidence: list):
    return max(steps_with_evidence, key=position_of_step) if steps_with_evidence else None


def _gate_and_cull_outcome(document_id, scores, threshold, near_duplicate_record, final_ids, score_folder_name):
    """(removal or None, steps reached by the gate and cull, is_final) for a scored document."""
    values = scores.recorded_values
    alignment, realism = values.get("alignment") or 0, values.get("realism") or 0
    gate_source = f"derived from {score_folder_name}/scores.jsonl and the gate threshold {threshold}"
    if not (alignment >= threshold and realism >= threshold):
        return DocumentRemoval(
            RemovalReasonCode.BELOW_THRESHOLD_AT_GATE, PipelineStep.GATE,
            f"alignment {alignment} and realism {realism} against a threshold of {threshold} on both",
            gate_source,
            {"alignment": alignment, "realism": realism, "threshold": threshold,
             "score_was_parse_error": scores.was_parse_error},
        ), [PipelineStep.GATE], False
    steps = [PipelineStep.GATE, PipelineStep.NEAR_DUPLICATE_CULL]
    if near_duplicate_record is not None:
        return DocumentRemoval(
            RemovalReasonCode.NEAR_DUPLICATE_AT_CULL, PipelineStep.NEAR_DUPLICATE_CULL,
            f"similarity {near_duplicate_record.get('similarity')} to {near_duplicate_record.get('kept_doc_id')}, "
            "which the cull kept",
            f"{score_folder_name}/{NEAR_DUPLICATE_FILE_NAME}",
            {"kept_doc_id": near_duplicate_record.get("kept_doc_id"),
             "similarity": near_duplicate_record.get("similarity")},
        ), steps, False
    if document_id not in final_ids:
        return DocumentRemoval(
            RemovalReasonCode.ABSENT_FROM_FINAL_CORPUS_WITHOUT_RECORDED_REASON, PipelineStep.NEAR_DUPLICATE_CULL,
            "passed the gate and is not in near_dup_dropped.jsonl, but final/sdf_corpus.jsonl does not hold it",
            FINAL_CORPUS_RELATIVE_PATH, {"threshold": threshold},
        ), steps, False
    return None, steps, True


# ------------------------------------------------------------------------------ current and matrix layouts

def _build_matrix_states(files: _RunFiles, run: Path, layout: RunLayout, threshold) -> list:
    plan_name, draft_name, rewrite_name, score_name = _MATRIX_STAGE_FOLDERS[layout]
    plan_dir, draft_dir, rewrite_dir, score_dir = (run / name for name in _MATRIX_STAGE_FOLDERS[layout])
    rel = files.relative

    screen_hands = files.records(plan_dir / "screen_hands.jsonl")
    screen_rounds = files.records(plan_dir / "screen_rounds.jsonl")
    judgments = _last_record_per_id(files.records(plan_dir / "screen_judgments.jsonl"), "prompt_id")
    rejections = _last_record_per_id(files.records(plan_dir / "screen_rejections.jsonl"), "prompt_id")
    screened_hands_path = plan_dir / "screened_hands.jsonl"
    accepted_by_screen = {r.get("prompt_id") for r in files.records(screened_hands_path)}
    screen_summary = files.json_object(plan_dir / "screen_summary.json")
    leftover_ids = set(screen_summary.get("leftover_prompt_ids") or [])
    prompt_records = files.records(plan_dir / "prompts.jsonl")
    plans = _last_record_per_id(files.records(plan_dir / "plans.jsonl"), "prompt_id")
    rewrite_results = _last_record_per_id(files.records(plan_dir / "description_rewrites.jsonl"), "prompt_id")
    scoring_rows = _last_record_per_id(files.records(plan_dir / "consideration_scores.jsonl"), "prompt_id")
    selection_rows = _last_record_per_id(files.records(plan_dir / "consideration_selections.jsonl"), "prompt_id")
    scoring_failures = files.records(plan_dir / "consideration_scoring_failures.jsonl")
    drafts = _last_record_per_id(files.records(draft_dir / "drafts.jsonl"), "doc_id")
    critiques = _last_record_per_id(files.records(rewrite_dir / "critiques.jsonl"), "doc_id")
    rewrites = _last_record_per_id(files.records(rewrite_dir / "rewrites.jsonl"), "doc_id")
    score_records = _last_record_per_id(files.records(score_dir / "scores.jsonl"), "doc_id")
    near_duplicates = _last_record_per_id(files.records(score_dir / NEAR_DUPLICATE_FILE_NAME), "doc_id")
    final_path = run / FINAL_CORPUS_RELATIVE_PATH
    final_exists = final_path.is_file()
    final_ids = {r.get("doc_id") for r in files.records(final_path)}
    checkpoints = {
        "plan": (files.checkpoint_ids(plan_dir / "_checkpoint.json"), plans, f"{plan_name}/plans.jsonl"),
        "description rewrite": (files.checkpoint_ids(plan_dir / "_description_rewrite_checkpoint.json"),
                                rewrite_results, f"{plan_name}/description_rewrites.jsonl"),
        "draft": (files.checkpoint_ids(draft_dir / "_checkpoint.json"), drafts, f"{draft_name}/drafts.jsonl"),
        "rewrite": (files.checkpoint_ids(rewrite_dir / "_checkpoint.json"), rewrites, f"{rewrite_name}/rewrites.jsonl"),
        "score": (files.checkpoint_ids(score_dir / "_checkpoint.json"), score_records, f"{score_name}/scores.jsonl"),
    }

    # --- the deal
    recorded_round_ids = {pid for r in screen_rounds for pid in r.get("prompt_ids", [])}
    deal = {}
    if screen_hands:
        for hand in screen_hands:
            if hand.get("prompt_id") in recorded_round_ids:
                deal[hand["prompt_id"]] = (hand.get("round", 0), hand.get("variables"))
        deal_order = sorted(deal, key=lambda pid: (deal[pid][0], _hand_index(pid) if _hand_index(pid) is not None else -1))
        screen_ran = True
    else:
        for record in prompt_records:
            deal.setdefault(record["prompt_id"], (0, record.get("variables")))
        deal_order = list(deal)
        screen_ran = False
    prompt_by_id = _last_record_per_id(prompt_records, "prompt_id")
    rounds_with_hands = {r.get("round") for r in screen_rounds if r.get("prompt_ids")}

    other_ids = []
    for source in (prompt_by_id, plans, rewrite_results, scoring_rows, selection_rows, drafts, critiques, rewrites,
                   score_records, judgments, rejections):
        other_ids += [identifier for identifier in source if identifier not in deal and identifier not in other_ids]
    run_has_rewrite_results = bool(rewrite_results)
    run_has_selection_rows = bool(selection_rows)

    # Dealt value ids: the deal's own record where the run's event log has one, else a lookup of each dealt
    # text in the run's frozen variables.txt (both read-only; neither touches a stage file).
    try:
        recorded_value_ids = fold_dealt_value_ids(read_document_events(run / DOCUMENT_EVENT_LOG_FILE_NAME))
    except ValueError:
        recorded_value_ids = {}   # a damaged event log is reported by the event-log readers, not here
    frozen_value_ids, why_no_snapshot = load_frozen_value_ids(run)

    states = []
    for document_id in deal_order + other_ids:
        notes = []
        in_deal = document_id in deal
        if not in_deal:
            notes.append("has no record in the deal (screen_hands.jsonl or prompts.jsonl) but appears in other stage files")
        round_number, dealt_variables = deal.get(document_id, (None, None))
        plan = plans.get(document_id)
        if dealt_variables is None:
            dealt_variables = (prompt_by_id.get(document_id) or plan or {}).get("variables")
        if in_deal and plan is not None and plan.get("variables") is not None and dealt_variables != _strip_consideration_lists(plan["variables"]):
            notes.append("the plan's variables differ from the dealt variables of the deal record")

        # screen
        judgment_record = judgments.get(document_id)
        screen_judgment = None if judgment_record is None else ScreenJudgment(
            round_number=judgment_record.get("round"), batch_id=judgment_record.get("batch_id"),
            local_number=judgment_record.get("number"), reasoning=judgment_record.get("reasoning", ""),
            remove=bool(judgment_record.get("remove")), reason=judgment_record.get("reason", ""))
        rejection = rejections.get(document_id)

        # plan and description
        description_versions = []
        decision = None
        plan_has_description = bool(plan and plan.get("description"))
        plan_declared_incoherent = bool(plan and plan.get("incoherent"))
        if plan_has_description:
            description_versions.append(DescriptionVersion(DescriptionVersionKind.PLANNED, plan["description"]))
        result = rewrite_results.get(document_id)
        if result is not None:
            if result.get("dealt_value_conflict"):
                verdict = DescriptionRewriteVerdict.DEALT_VALUE_CONFLICT
            elif result.get("rewrite_needed"):
                verdict = DescriptionRewriteVerdict.REWRITTEN
            else:
                verdict = DescriptionRewriteVerdict.KEPT_ORIGINAL
            decision = DescriptionRewriteDecision(
                verdict, result.get("reasoning", ""), tuple(result.get("changes") or ()), result.get("conflict_reason", ""))
            if verdict is DescriptionRewriteVerdict.REWRITTEN:
                description_versions.append(DescriptionVersion(
                    DescriptionVersionKind.REWRITTEN, result.get("rewritten_description", ""), decision.changes))

        # considerations
        scoring_row = scoring_rows.get(document_id)
        consideration_scoring = None if scoring_row is None else ConsiderationScoringRecord(
            scoring_row.get("scores") or {}, scoring_row.get("attempts", 1))
        selection_row = selection_rows.get(document_id)
        assignment = None if selection_row is None else tuple(
            AssignedConsideration(o.get("list"), o.get("label"), o.get("reason", ""), o.get("score"))
            for o in selection_row.get("outcomes", []) if o.get("status") == "accepted")
        failures = tuple(
            RecordedFailure(PipelineStep.CONSIDERATION_SCORING, f.get("problem", ""),
                            f"{plan_name}/consideration_scoring_failures.jsonl", scoring_row is not None)
            for f in scoring_failures if f.get("prompt_id") == document_id)

        # text and scores
        draft, critique_record, rewrite, score_record = (
            drafts.get(document_id), critiques.get(document_id), rewrites.get(document_id), score_records.get(document_id))
        text_versions = []
        if draft is not None:
            text_versions.append(TextVersion(1, TextVersionKind.DRAFT, draft.get("content", ""), None,
                                             _written_from(draft), f"{draft_name}/drafts.jsonl"))
        if rewrite is not None:
            text_versions.append(TextVersion(2, TextVersionKind.REVISION, rewrite.get("content", ""), rewrite.get("review"),
                                             _written_from(rewrite), f"{rewrite_name}/rewrites.jsonl"))
        banked_critique = critique_record.get("review") if critique_record and rewrite is None else None
        override = (rewrite or {}).get("revision") or (draft or {}).get("revision")
        scores = None if score_record is None else DocumentScores(score_record.get("scores") or {})

        # what today's stages carry forward
        held_out = run_has_rewrite_results and plan_has_description and result is None
        conflict = decision is not None and decision.verdict is DescriptionRewriteVerdict.DEALT_VALUE_CONFLICT
        # A draft record is proof the document was carried past the description rewrite (hand-seeded
        # runs have drafts whose rewrite result was never stored), so it only holds a document out
        # while there is no draft.
        plan_carried = plan_has_description and not (held_out and draft is None) and not conflict
        draft_carried = draft is not None and plan_carried
        rewrite_carried = rewrite is not None and draft_carried
        score_carried = scores is not None and rewrite_carried
        for label, present, carried, previous in (
                ("draft", draft is not None, draft_carried, "plan"), ("rewrite", rewrite is not None, rewrite_carried, "draft"),
                ("score", scores is not None, score_carried, "rewrite")):
            if present and not carried:
                notes.append(f"has a {label} record that a full run ignores because the {previous} stage did not carry the document forward")
        if held_out and draft is None:
            notes.append("its description rewrite has no stored result, so a resumed run holds it out of drafting")
        elif held_out:
            notes.append("it was drafted although its description rewrite has no stored result")
        for stage_label, (completed, records, path_text) in checkpoints.items():
            if document_id in completed and document_id not in records:
                notes.append(f"{path_text}: marked done in the {stage_label} checkpoint but has no record there, so a resume does not retry it")

        # outcome
        removal = None
        steps = []
        if in_deal:
            steps.append(PipelineStep.DEAL)
        if judgment_record is not None or (document_id in accepted_by_screen):
            steps.append(PipelineStep.SCREEN)
        if document_id in prompt_by_id:
            steps.append(PipelineStep.COMPOSE_PLANNING_PROMPT)
        if plan is not None:
            steps.append(PipelineStep.PLAN)
        if result is not None:
            steps.append(PipelineStep.DESCRIPTION_REWRITE)
        if scoring_row is not None:
            steps.append(PipelineStep.CONSIDERATION_SCORING)
        if selection_row is not None:
            steps.append(PipelineStep.CONSIDERATION_ASSIGNMENT)
        if draft is not None:
            steps.append(PipelineStep.DRAFT)
        if banked_critique is not None or rewrite is not None or critique_record is not None:
            steps.append(PipelineStep.CRITIQUE)
        if rewrite is not None:
            steps.append(PipelineStep.REVISE)
        if scores is not None:
            steps.append(PipelineStep.SCORE)
        is_final = False
        if rejection is not None:
            next_round = (rejection.get("round") or 0) + 1
            removal = DocumentRemoval(
                RemovalReasonCode.EXTREMELY_IMPLAUSIBLE_AT_SCREEN, PipelineStep.SCREEN, rejection.get("reason", ""),
                f"{plan_name}/screen_rejections.jsonl",
                {"round": rejection.get("round"), "batch_id": rejection.get("batch_id"),
                 "local_number": rejection.get("number"),
                 "cards_were_redealt_into_round": next_round if next_round in rounds_with_hands else None,
                 "dropped_after_the_last_round": document_id in leftover_ids})
        elif plan is not None and not plan_has_description and plan_declared_incoherent:
            removal = DocumentRemoval(
                RemovalReasonCode.DECLARED_INCOHERENT_AT_PLAN, PipelineStep.PLAN,
                "the planner declared the variable combination incoherent", f"{plan_name}/plans.jsonl", {})
        elif conflict:
            removal = DocumentRemoval(
                RemovalReasonCode.DEALT_VALUE_CONFLICT_AT_DESCRIPTION_REWRITE, PipelineStep.DESCRIPTION_REWRITE,
                decision.conflict_reason, f"{plan_name}/description_rewrites.jsonl and description_rewrite_rejections.jsonl", {})
        elif score_carried and final_exists and threshold is not None:
            removal, gate_steps, is_final = _gate_and_cull_outcome(
                document_id, scores, threshold, near_duplicates.get(document_id), final_ids, score_name)
            steps += gate_steps
        elif plan is not None and not plan_has_description:
            notes.append("the plan record has no description and is not marked incoherent")
        if removal is not None:
            steps.append(removal.step)
        outcome = DocumentOutcome.REMOVED if removal else (DocumentOutcome.FINAL if is_final else DocumentOutcome.IN_PROGRESS)
        if document_id in final_ids and outcome is not DocumentOutcome.FINAL:
            notes.append("is in final/sdf_corpus.jsonl but the loader does not find it passing every stage")

        latest = text_versions[-1] if text_versions else None
        current_description = description_versions[-1].text if description_versions else None
        if document_id in recorded_value_ids:
            dealt_value_ids, dealt_value_id_notes = dict(recorded_value_ids[document_id]), []
            for axis in dealt_variables or {}:
                if axis not in dealt_value_ids:
                    dealt_value_id_notes.append(f"dealt value of {{{axis}}} has no id: the deal event recorded none")
        else:
            dealt_value_ids, dealt_value_id_notes = value_ids_of_dealt_hand_from_snapshot(
                dealt_variables, frozen_value_ids, why_no_snapshot)
        different_assignment = bool(
            run_has_selection_rows and latest is not None and latest.written_from.variables is not None
            and _label_pairs_of_variables(latest.written_from.variables) != _label_pairs_of_assignment(assignment))
        different_description = bool(
            latest is not None and latest.written_from.description is not None and current_description is not None
            and latest.written_from.description != current_description)
        states.append(DocumentState(
            document_id=document_id, layout=layout, deal_round_number=round_number,
            hand_index=_hand_index(document_id), dealt_variables=dealt_variables, screen_judgment=screen_judgment,
            plan_reply=plan.get("plan") if plan else None, plan_declared_incoherent=plan_declared_incoherent,
            description_versions=tuple(description_versions), description_rewrite_decision=decision,
            consideration_scoring=consideration_scoring, considerations_assignment=assignment,
            text_versions=tuple(text_versions), banked_critique=banked_critique, revision_override=override,
            scores=scores, furthest_step_reached=_furthest_step(steps), recorded_failures=failures,
            outcome=outcome, removal=removal,
            latest_text_was_written_from_a_different_assignment=different_assignment,
            latest_text_was_written_from_a_different_description=different_description,
            legacy_fields={}, loader_notes=tuple(notes),
            has_plan_record=plan is not None, plan_record_variables=plan.get("variables") if plan else None,
            dealt_value_ids=dealt_value_ids, dealt_value_id_notes=tuple(dealt_value_id_notes)))
    return states


def _strip_consideration_lists(variables: dict) -> dict:
    return {k: v for k, v in variables.items() if k not in ("moral_considerations", "epistemic_considerations")}


# ------------------------------------------------------------------------------ five-layer layout

def _build_five_layer_states(files: _RunFiles, run: Path, threshold) -> list:
    subtypes = _last_record_per_id(files.records(run / "layer2" / "subtypes.jsonl"), "subtype_id")
    draft_records = files.records(run / "layer3" / "drafts.jsonl")
    drafts = _last_record_per_id(draft_records, "doc_id")
    rewrites = _last_record_per_id(files.records(run / "layer4" / "rewrites.jsonl"), "doc_id")
    score_records = _last_record_per_id(files.records(run / "layer5" / "scores.jsonl"), "doc_id")
    near_duplicates = _last_record_per_id(files.records(run / "layer5" / NEAR_DUPLICATE_FILE_NAME), "doc_id")
    final_path = run / FINAL_CORPUS_RELATIVE_PATH
    final_exists = final_path.is_file()
    final_ids = {r.get("doc_id") for r in files.records(final_path)}
    checkpoints = {
        "draft": (files.checkpoint_ids(run / "layer3" / "_checkpoint.json"), drafts, "layer3/drafts.jsonl"),
        "rewrite": (files.checkpoint_ids(run / "layer4" / "_checkpoint.json"), rewrites, "layer4/rewrites.jsonl"),
        "score": (files.checkpoint_ids(run / "layer5" / "_checkpoint.json"), score_records, "layer5/scores.jsonl"),
    }
    ids = list(drafts)
    ids += [i for source in (rewrites, score_records) for i in source if i not in ids]
    states = []
    for document_id in ids:
        notes = []
        draft, rewrite, score_record = drafts.get(document_id), rewrites.get(document_id), score_records.get(document_id)
        subtype = subtypes.get((draft or rewrite or score_record or {}).get("subtype_id"))
        description = subtype.get("description") if subtype else None
        description_versions = (DescriptionVersion(DescriptionVersionKind.FIVE_LAYER_SUBTYPE, description),) if description else ()
        written_from = TextWrittenFrom(description, None, None)
        text_versions = []
        if draft is not None:
            text_versions.append(TextVersion(1, TextVersionKind.DRAFT, draft.get("content", ""), None, written_from, "layer3/drafts.jsonl"))
        if rewrite is not None:
            text_versions.append(TextVersion(2, TextVersionKind.REVISION, rewrite.get("rewritten", ""),
                                             rewrite.get("review_notes"), written_from, "layer4/rewrites.jsonl"))
        scores = None if score_record is None else DocumentScores(score_record.get("scores") or {})
        rewrite_carried = rewrite is not None and draft is not None
        score_carried = scores is not None and rewrite_carried
        if rewrite is not None and draft is None:
            notes.append("has a rewrite record that a full run ignores because it has no draft record")
        if scores is not None and not rewrite_carried:
            notes.append("has a score record that a full run ignores because it has no carried rewrite")
        for stage_label, (completed, records, path_text) in checkpoints.items():
            if document_id in completed and document_id not in records:
                notes.append(f"{path_text}: marked done in the {stage_label} checkpoint but has no record there")
        steps = []
        if draft is not None:
            steps.append(PipelineStep.DRAFT)
        if rewrite is not None:
            steps += [PipelineStep.CRITIQUE, PipelineStep.REVISE]
        if scores is not None:
            steps.append(PipelineStep.SCORE)
        removal, is_final = None, False
        if score_carried and final_exists and threshold is not None:
            removal, gate_steps, is_final = _gate_and_cull_outcome(
                document_id, scores, threshold, near_duplicates.get(document_id), final_ids, "layer5")
            steps += gate_steps
            if removal:
                steps.append(removal.step)
        outcome = DocumentOutcome.REMOVED if removal else (DocumentOutcome.FINAL if is_final else DocumentOutcome.IN_PROGRESS)
        if document_id in final_ids and outcome is not DocumentOutcome.FINAL:
            notes.append("is in final/sdf_corpus.jsonl but the loader does not find it passing every stage")
        legacy_source = score_record or draft or rewrite or {}
        legacy_fields = {k: v for k, v in legacy_source.items() if k not in ("doc_id", "content", "scores", "original", "rewritten", "review_notes")}
        latest = text_versions[-1] if text_versions else None
        states.append(DocumentState(
            document_id=document_id, layout=RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX, deal_round_number=None,
            hand_index=None, dealt_variables=None, screen_judgment=None, plan_reply=None, plan_declared_incoherent=False,
            description_versions=description_versions, description_rewrite_decision=None,
            consideration_scoring=None, considerations_assignment=None, text_versions=tuple(text_versions),
            banked_critique=None, revision_override=None, scores=scores, furthest_step_reached=_furthest_step(steps),
            recorded_failures=(), outcome=outcome, removal=removal,
            latest_text_was_written_from_a_different_assignment=False,
            latest_text_was_written_from_a_different_description=False,
            legacy_fields=legacy_fields, loader_notes=tuple(notes)))
    return states
