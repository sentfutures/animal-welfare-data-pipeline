"""The state of one SDF document: one record, keyed by one id, for the document's whole life.

``DocumentState`` holds everything the pipeline knows about one document: the hand of
variables it was dealt, the planner's notes, every version of its description, the
considerations assigned to it, every version of its text (each revision paired with the
critique that produced it), its scores, the furthest step it reached, the failures recorded
against it, and its outcome (in progress, final, or removed with a reason code).

The module only defines the data. ``sdf_pipeline/document_state_loader.py`` builds it from a
run folder's stage files, and ``sdf_pipeline/document_state_final_corpus.py`` derives today's
``final/sdf_corpus.jsonl`` records from it. Nothing here reads or writes a file, so the later
event log and step runner can build the same records by folding events.

The id
------
``document_id`` is the screen's hand id (``matrix_000007``): the same string today's files
call ``prompt_id`` (screen, plan, description rewrite and consideration files) and ``doc_id``
(draft, critique, rewrite, score and final files). Runs from before the variables matrix have
no hand id; their ``document_id`` is the ``doc_id`` (a UUID) minted when the document was
drafted, and they have no deal, plan or screen.

Where each field comes from, per layout
---------------------------------------
Layouts (``RunLayout``): ``CURRENT`` (stage folders layer1 to layer4), ``MATRIX_BEFORE_LAYER_RENUMBERING``
(layer12, layer3, layer4, layer5; the website's pinned run) and ``FIVE_LAYER_BEFORE_VARIABLES_MATRIX``
(layer1 document types, layer2 subtypes, layer3 drafts, layer4 rewrites, layer5 scores). In the
table "plan folder" is layer1 (current) or layer12 (matrix before renumbering), "draft folder" is
layer2 or layer3, "rewrite folder" is layer3 or layer4, "score folder" is layer4 or layer5.
A run made before the renumbering and resumed on newer code keeps its old folder names but may
hold the newer file names in them, so the loader reads every file name in either folder scheme.

====================================  =========================================================  ============================================
Field                                 Current and matrix layouts                                 Five-layer layout
====================================  =========================================================  ============================================
dealt_variables, deal_round_number    plan folder: screen_hands.jsonl (variables, round); when   none (no deal)
                                      there is no screen, prompts.jsonl (round 0)
screen_judgment                       plan folder: screen_judgments.jsonl                        none
planning_notes / plan_reply           plan folder: plans.jsonl ``plan``                          none
plan_declared_incoherent              plan folder: plans.jsonl ``incoherent``                    none
description_versions                  plans.jsonl ``description`` (planned); then                the subtype description (layer2 subtypes.jsonl,
                                      description_rewrites.jsonl ``rewritten_description``       found through the draft's ``subtype_id``)
                                      and ``changes`` (rewritten)
description_rewrite_decision          plan folder: description_rewrites.jsonl                    none
consideration_scoring                 plan folder: consideration_scores.jsonl                    none
considerations_assignment             plan folder: consideration_selections.jsonl (accepted      none
                                      outcomes)
text_versions[0] (the draft)          draft folder: drafts.jsonl ``content``, written from       layer3 drafts.jsonl ``content``
                                      its ``description``, ``variables``, ``considerations_spec``
text_versions[1] (the revision)       rewrite folder: rewrites.jsonl ``content``, critique       layer4 rewrites.jsonl ``rewritten``, critique
                                      ``review``, written from the record's own ``description``,  ``review_notes``
                                      ``variables``, ``considerations_spec``
banked_critique                       rewrite folder: critiques.jsonl ``review`` when no         none
                                      rewrite record exists yet
revision_override                     ``revision`` of the draft or rewrite record (set when a    none
                                      run is seeded by hand)
scores                                score folder: scores.jsonl ``scores``                      layer5 scores.jsonl ``scores``
recorded_failures                     plan folder: consideration_scoring_failures.jsonl (the     none
                                      only stage that records its failures)
outcome and removal                   see ``RemovalReasonCode``                                  gate only (threshold from run_manifest.json)
legacy_fields                         empty                                                      score record fields other than doc_id,
                                                                                                 content and scores (type_id, subtype_id, role,
                                                                                                 register, language, ...)
====================================  =========================================================  ============================================

What a text version was written from
------------------------------------
Each ``TextVersion`` records the description, variables and considerations text that its own
stage-file record carried. Today the draft record copies these from the latest plan when the
draft is made, and the critique, revise and score requests read them from the draft or rewrite
record, never from the latest plan. A later change to the plan or to the assignment therefore
leaves existing text versions as they were, and ``DocumentState`` marks the document
(``latest_text_was_written_from_a_different_assignment`` and
``latest_text_was_written_from_a_different_description``) instead of rewriting history.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from sdf_pipeline.compose_prompts import REVISION_AXES
from sdf_pipeline.considerations_catalog import SECTION_EPISTEMIC, SECTION_MORAL

# The two keys the consideration assignment adds to a plan's ``variables``
# (``layer1_select_considerations`` writes them: lists of the assigned labels).
ASSIGNED_CONSIDERATION_KEYS = (SECTION_MORAL, SECTION_EPISTEMIC)


class RunLayout(Enum):
    """The three on-disk layouts of an SDF run folder."""
    FIVE_LAYER_BEFORE_VARIABLES_MATRIX = "five_layer_before_variables_matrix"
    MATRIX_BEFORE_LAYER_RENUMBERING = "matrix_before_layer_renumbering"
    CURRENT = "current"


class PipelineStep(Enum):
    """The steps a document goes through, in order (the step table of the plan, with the gate
    and the near-duplicate cull as two steps)."""
    DEAL = "deal"
    SCREEN = "screen"
    COMPOSE_PLANNING_PROMPT = "compose_planning_prompt"
    PLAN = "plan"
    DESCRIPTION_REWRITE = "description_rewrite"
    CONSIDERATION_SCORING = "consideration_scoring"
    CONSIDERATION_ASSIGNMENT = "consideration_assignment"
    DRAFT = "draft"
    CRITIQUE = "critique"
    REVISE = "revise"
    SCORE = "score"
    GATE = "gate"
    NEAR_DUPLICATE_CULL = "near_duplicate_cull"


PIPELINE_STEPS_IN_ORDER = tuple(PipelineStep)


def position_of_step(step: PipelineStep) -> int:
    """0 for the deal, rising along the pipeline; for comparing which step is further."""
    return PIPELINE_STEPS_IN_ORDER.index(step)


class DocumentOutcome(Enum):
    IN_PROGRESS = "in_progress"
    FINAL = "final"
    REMOVED = "removed"


class RemovalReasonCode(Enum):
    """Why a document left the pipeline, with where today's files record it.

    EXTREMELY_IMPLAUSIBLE_AT_SCREEN
        Recorded in ``screen_rejections.jsonl`` (one record per removed hand: reason, round,
        batch) and in ``screen_judgments.jsonl`` (``remove: true``). A removed hand is redealt in
        the next round while rounds remain; a hand removed in the last round, or whose cards
        could not be redealt, is dropped (``screen_summary.json`` ``leftover_prompt_ids``).
    DECLARED_INCOHERENT_AT_PLAN
        Recorded in ``plans.jsonl`` as ``incoherent: true`` with no description. The plan stays
        in the file, and every later stage skips a plan with no description.
    DEALT_VALUE_CONFLICT_AT_DESCRIPTION_REWRITE
        Recorded twice: ``description_rewrites.jsonl`` (``dealt_value_conflict: true``, with the
        reason) and ``description_rewrite_rejections.jsonl``. Later stages skip the plan.
    BELOW_THRESHOLD_AT_GATE
        Recorded nowhere. The gate keeps documents whose alignment and realism both reach
        ``sdf.min_score_threshold`` (the run manifest's config) and the rest are simply left out
        of ``final/sdf_corpus.jsonl``. The loader finds them by applying the threshold to
        ``scores.jsonl``. A score parse error is stored as 5/5/5 and so counts here;
        ``DocumentScores.was_parse_error`` tells the two apart.
    NEAR_DUPLICATE_AT_CULL
        Recorded in ``near_dup_dropped.jsonl`` of the score folder (``doc_id``, ``kept_doc_id``,
        ``similarity``), which is written only when something was culled and never cleared.
    ABSENT_FROM_FINAL_CORPUS_WITHOUT_RECORDED_REASON
        Not a drop today's code records. The loader uses it only when a document passes the gate,
        is not in ``near_dup_dropped.jsonl``, and still is not in ``final/sdf_corpus.jsonl``,
        which means the final corpus was written some other way (by a script or by hand).
    """
    EXTREMELY_IMPLAUSIBLE_AT_SCREEN = "extremely_implausible_at_screen"
    DECLARED_INCOHERENT_AT_PLAN = "declared_incoherent_at_plan"
    DEALT_VALUE_CONFLICT_AT_DESCRIPTION_REWRITE = "dealt_value_conflict_at_description_rewrite"
    BELOW_THRESHOLD_AT_GATE = "below_threshold_at_gate"
    NEAR_DUPLICATE_AT_CULL = "near_duplicate_at_cull"
    ABSENT_FROM_FINAL_CORPUS_WITHOUT_RECORDED_REASON = "absent_from_final_corpus_without_recorded_reason"


class DescriptionRewriteVerdict(Enum):
    KEPT_ORIGINAL = "kept_original"
    REWRITTEN = "rewritten"
    DEALT_VALUE_CONFLICT = "dealt_value_conflict"


class DescriptionVersionKind(Enum):
    PLANNED = "planned"
    REWRITTEN = "rewritten"
    FIVE_LAYER_SUBTYPE = "five_layer_subtype"


class TextVersionKind(Enum):
    DRAFT = "draft"
    REVISION = "revision"


@dataclass(frozen=True)
class ScreenJudgment:
    """The screen's verdict on a hand (``screen_judgments.jsonl``), kept for keeps as well as removals."""
    round_number: int
    batch_id: str
    local_number: int
    reasoning: str
    remove: bool
    reason: str


@dataclass(frozen=True)
class DescriptionVersion:
    """One version of the document description. ``changes`` is the description rewrite's list of
    {"check", "reason"} for a rewritten version, and empty otherwise."""
    kind: DescriptionVersionKind
    text: str
    changes: tuple = ()


@dataclass(frozen=True)
class DescriptionRewriteDecision:
    """The description rewrite's stored answer. A document whose plan has a description but no
    decision was never checked (a failed call, an unusable answer, or the step did not run)."""
    verdict: DescriptionRewriteVerdict
    reasoning: str
    changes: tuple
    conflict_reason: str


@dataclass(frozen=True)
class ConsiderationScoringRecord:
    """The consideration scorer's answer: ``scores[list_name][label] = {"score", "where_it_arises"}``,
    and how many attempts the answer took."""
    scores: dict
    attempts: int


@dataclass(frozen=True)
class AssignedConsideration:
    list_name: str          # "moral" or "epistemic"
    label: str
    reason: str             # the scorer's sentence on where it arises
    score: int | None


@dataclass(frozen=True)
class TextWrittenFrom:
    """What a text version was written from, as its own stage-file record carried it. ``None`` in a
    field means the record had no such field (records from before it existed)."""
    description: str | None
    variables: dict | None
    considerations_spec: str | None


@dataclass(frozen=True)
class TextVersion:
    """One version of the document text. Version 1 is the draft. Each later version is a revision
    paired with the critique that produced it (``critique`` is None for the draft)."""
    version_number: int
    kind: TextVersionKind
    text: str
    critique: str | None
    written_from: TextWrittenFrom
    source_file: str


@dataclass(frozen=True)
class DocumentScores:
    """The score record. ``recorded_values`` is the ``scores`` dict exactly as stored (field order
    included); the scores carry no link to the text version they scored."""
    recorded_values: dict

    @property
    def alignment(self):
        return self.recorded_values.get("alignment")

    @property
    def realism(self):
        return self.recorded_values.get("realism")

    @property
    def spec_conformance(self):
        return self.recorded_values.get("spec_conformance")

    @property
    def notes(self) -> str:
        return self.recorded_values.get("notes") or ""

    @property
    def was_parse_error(self) -> bool:
        """The score answer did not parse and the 5/5/5 "Parse error." default was stored (the
        website's gate counts these by the same note)."""
        return "parse error" in self.notes.lower()


@dataclass(frozen=True)
class RecordedFailure:
    """A failure today's files record against a document. Only consideration scoring records any
    (``consideration_scoring_failures.jsonl``). ``resolved`` is true when a later result of the same
    step exists."""
    step: PipelineStep
    detail: str
    source_file: str
    resolved: bool


@dataclass(frozen=True)
class DocumentRemoval:
    """Why and where a document was removed. ``detail`` is the reason as the stage file words it,
    ``recorded_in`` names the file that records the drop (or says it is derived), and ``evidence``
    holds the structured facts (screen round and batch, gate scores and threshold, the kept
    near-duplicate and its similarity)."""
    reason_code: RemovalReasonCode
    step: PipelineStep
    detail: str
    recorded_in: str
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentState:
    """Everything known about one document. See the module docstring for the source of each field.

    Tuples and the frozen dataclass keep a state from being edited by accident; the ``dict`` fields
    hold JSON values exactly as the stage files do and must be treated as read-only.
    """
    document_id: str
    layout: RunLayout

    # The deal
    deal_round_number: int | None
    hand_index: int | None
    dealt_variables: dict | None
    screen_judgment: ScreenJudgment | None

    # Planning and description
    plan_reply: str | None
    plan_declared_incoherent: bool
    description_versions: tuple[DescriptionVersion, ...]
    description_rewrite_decision: DescriptionRewriteDecision | None

    # Considerations. ``considerations_assignment`` is None when no assignment row exists for the
    # document, and an empty tuple when the assignment ran and gave it nothing.
    consideration_scoring: ConsiderationScoringRecord | None
    considerations_assignment: tuple[AssignedConsideration, ...] | None

    # Text
    text_versions: tuple[TextVersion, ...]
    banked_critique: str | None
    revision_override: dict | None

    # Scores, progress, outcome
    scores: DocumentScores | None
    furthest_step_reached: PipelineStep | None
    recorded_failures: tuple[RecordedFailure, ...]
    outcome: DocumentOutcome
    removal: DocumentRemoval | None

    # Disagreements between the latest text and the document's current inputs
    latest_text_was_written_from_a_different_assignment: bool
    latest_text_was_written_from_a_different_description: bool

    # Fields only the five-layer layout carries (type_id, subtype_id, role, register, language, ...)
    legacy_fields: dict = field(default_factory=dict)

    # Things in the files the state cannot represent or that look wrong (a checkpoint mark with no
    # record, a record the later stages ignore, an id with no deal record, ...)
    loader_notes: tuple[str, ...] = ()

    # The plan stage file's record for the document, as readers that used to open plans.jsonl need it:
    # whether the file has a record for the id, and that record's own ``variables`` (None when it has
    # none). ``dealt_variables`` is the deal's copy; the two are equal on every committed run, and the
    # loader notes any run where they differ. Never set for the five-layer layout, which has no plans.
    has_plan_record: bool = False
    plan_record_variables: dict | None = None

    @property
    def current_description(self) -> str | None:
        """The description every later step uses: the rewritten one if the rewrite changed it."""
        return self.description_versions[-1].text if self.description_versions else None

    @property
    def variables(self) -> dict:
        """The ``variables`` dict exactly as stored: the one the latest text version was written from,
        or the dealt hand when no text exists yet (``{}`` when there is neither, as in the five-layer
        layout). Never copied or edited here. It holds three groups, which the three properties below
        return; ``dealt_cards`` takes every key the other two do not claim, so the three always
        add up to this dict."""
        latest = self.latest_text_version
        if latest is not None and latest.written_from.variables is not None:
            return latest.written_from.variables
        return self.dealt_variables or {}

    @property
    def revision_cards(self) -> dict:
        """The keys of ``variables`` that are revision axes (``compose_prompts.REVISION_AXES``), the cards
        only layer 3 uses."""
        return {key: value for key, value in self.variables.items() if key in REVISION_AXES}

    @property
    def assigned_considerations(self) -> dict:
        """The ``moral_considerations`` and ``epistemic_considerations`` lists the consideration assignment
        added to ``variables`` (absent before the assignment ran, or when it was disabled)."""
        return {key: value for key, value in self.variables.items() if key in ASSIGNED_CONSIDERATION_KEYS}

    @property
    def dealt_cards(self) -> dict:
        """The dealt composition cards: every key of ``variables`` that is neither a revision card nor an
        assigned-considerations list. Defined by exclusion so that a key from a run's older or newer variables
        file (an axis since renamed or removed) is still returned here and never dropped; the tests check each
        run's keys against the variables file the run froze."""
        return {key: value for key, value in self.variables.items()
                if key not in REVISION_AXES and key not in ASSIGNED_CONSIDERATION_KEYS}

    @property
    def planned_description(self) -> str | None:
        """The description exactly as the planner wrote it (before any rewrite), or None when the plan
        record has no description (a plan declared incoherent) or there is no plan record."""
        for version in self.description_versions:
            if version.kind is DescriptionVersionKind.PLANNED:
                return version.text
        return None

    @property
    def draft_text(self) -> str | None:
        """The text of the draft (text version 1), or None when the document has no draft record."""
        for version in self.text_versions:
            if version.kind is TextVersionKind.DRAFT:
                return version.text
        return None

    @property
    def latest_text_version(self) -> TextVersion | None:
        return self.text_versions[-1] if self.text_versions else None

    @property
    def planning_notes(self) -> str | None:
        """The planner's working notes: the text inside ``<document_planning>`` tags, or the whole
        reply when the planner left the tags out (as the website shows them)."""
        if self.plan_reply is None:
            return None
        match = _PLANNING_NOTES_PATTERN.search(self.plan_reply)
        return (match.group(1) if match else self.plan_reply).strip()

    @property
    def last_failure(self) -> RecordedFailure | None:
        return self.recorded_failures[-1] if self.recorded_failures else None

    @property
    def is_final(self) -> bool:
        return self.outcome is DocumentOutcome.FINAL


_PLANNING_NOTES_PATTERN = re.compile(r"<document_planning>(.*?)</document_planning>", re.DOTALL)
