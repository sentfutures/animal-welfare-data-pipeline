"""Layer 3 step: the consideration scoring, where the model scores and code assigns (``consideration_selection.py``).

Runs after the implausibility screen, planning and the description rewrite, and
before drafting (layer 3). The planner writes a scenario description with no consideration
attached; this stage then makes ONE scoring call per eligible document. The model
sees every eligible entry of both lists (moral and epistemic, from
prompts/sdf/considerations.txt; label plus gloss), in an order shuffled per
document with a seeded RNG, and no target or fullness information, and scores
each entry 1 to 5 for how naturally the scenario raises it (the scale is in
prompts/sdf/layer1_select_considerations.txt), with one short sentence on where
it arises for scores of 3 or more. Which entry each document actually gets is
then decided in code (considerations_assignment.py): a linear programme over all
scores that respects every entry's target count and the per-document caps.

Eligibility: documents with a skipped resolution (the no-welfare-stake arm) get
nothing; digital-minds-only entries are offered only on digital-mind documents.

Answer format: a JSON schema built per call (``build_scoring_schema``) with, per
list, an array of {label, score, where_it_arises} items, where ``label`` is an
enum of that list's offered labels (listed in the shuffled order), ``score`` is
an integer enum 1-5. Structured outputs cannot fix the array length or the label
at each position (no usable prefixItems / minItems above 1), so
``parse_scoring_answer`` checks in code that every label appears exactly once with
a valid score. Arbitrary labels are deliberately NOT used as JSON property names
(spaces and typographic quotes in property names are not guaranteed to be
supported). The model never reproduces ids.

Files in the layer's folder (``layer3_consideration_scoring_and_selection``): see
``sdf_pipeline/layer3_consideration_scoring_and_selection/run_layer.py``. This step writes consideration_scores.jsonl: one
row per scored document ({prompt_id, scores: {list: {label: {score, where_it_arises}}}, attempts}), appended as
each answer arrives; it is the checkpoint, and a rerun or ``--resume`` scores only the documents without a row. An
unusable answer is retried up to ``sdf.max_scoring_attempts`` times (default 3) and then recorded in
consideration_scoring_failures.jsonl; the layer then stops before assigning and that document is scored on the
next resume.

Scoring is a ``DocumentStep`` (``ConsiderationScoringStep``) run by ``sdf_pipeline.shared_step_machinery.step_runner.StepRunner`` with
``sdf.max_scoring_attempts`` attempts per run.

Config (``sdf`` section): ``select_enabled`` (default true), ``consideration_scoring_model`` (old name ``select_model``)
(falls back to ``plan_model``, then the global model), ``select_skip_no_stake``
(default true), ``max_scoring_attempts`` (3), ``minimum_assignable_score`` (3),
``max_moral_considerations_per_document`` (3),
``max_epistemic_considerations_per_document`` (3),
``max_considerations_per_document`` (4), ``score_weights`` ({3: 1, 4: 3, 5: 9}),
``extra_consideration_penalty`` (2); the calls use ``workers`` threads.
"""

import random
from dataclasses import dataclass
from sdf_pipeline.shared_step_machinery import considerations_assignment as assignment
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, StepPolicy, StepProgress,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall, line_for_document_left_to_retry, reason_without_error_prefix)
from sdf_pipeline.shared_step_machinery.considerations_catalog import (
    EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST, Consideration, ConsiderationCatalog)
from sdf_pipeline.shared_step_machinery import structured_answers
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import FAILURES_FILENAME, SCORES_FILENAME
from sdf_pipeline.shared_step_machinery.renamed_model_settings import model_setting
from shared.matrix import split_sections

# The shuffle of the entries offered to a document is seeded with this text (with the run seed, the document and
# the list). It is the old stage tag, kept literal: deriving it from a layout's tag would change every request.
SHUFFLE_SEED_NAME = "layer1_select"
DEFAULT_MAX_SCORING_ATTEMPTS = 3
_BASE_MAX_TOKENS = 400
_TOKENS_PER_ENTRY = 120           # label, score and one short sentence; bounds api-backend cost

NO_ENTRIES_MARKER = "(no entries offered)"


class ScoringAnswerError(ValueError):
    """The model's scoring answer was truncated, malformed, or did not score every label exactly once."""


@dataclass(frozen=True)
class DocumentFacts:
    """What the selector is shown about one document (all of it already in the plan record)."""
    document_description: str
    document_type: str
    centrality: str
    sentient_category: str

    @classmethod
    def from_plan(cls, plan: dict) -> "DocumentFacts":
        """The facts of a plan record: its description and the three dealt values the selector shows."""
        variables = plan["variables"]
        return cls(
            document_description=plan["description"],
            document_type=variables.get("document_type", ""),
            centrality=variables.get("centrality", ""),
            sentient_category=variables.get("sentient_category", ""))





def eligible_entries_for_document(catalog: ConsiderationCatalog, variables: dict) -> dict[str, list[Consideration]]:
    """The entries this document may be scored on, per list, in catalog order."""
    is_digital = catalog.is_digital_mind_document(variables)
    return {
        list_name: [e for e in catalog.entries_of(list_name) if is_digital or not e.digital_minds_only]
        for list_name in LIST_NAMES
    }


def shuffled_offer(offered: dict[str, list[Consideration]], seed, prompt_id: str) -> dict[str, list[Consideration]]:
    """Each list in an order shuffled by an RNG seeded from the run seed and the prompt id (deterministic)."""
    shuffled = {}
    for list_name in LIST_NAMES:
        entries = list(offered.get(list_name) or [])
        random.Random(f"{seed}:{SHUFFLE_SEED_NAME}:{prompt_id}:{list_name}").shuffle(entries)
        shuffled[list_name] = entries
    return shuffled


def render_entries(entries: list[Consideration]) -> str:
    return "\n".join(f"- {entry.display}" for entry in entries) if entries else NO_ENTRIES_MARKER


def build_scoring_prompt(
    template: str, facts: DocumentFacts, offered: dict[str, list[Consideration]],
) -> tuple[str | None, str]:
    """Render (system, user) for one document; ``offered`` is in the order to show."""
    rendered = template.format(
        document_type=facts.document_type,
        centrality=facts.centrality,
        sentient_category=facts.sentient_category,
        document_description=facts.document_description.strip(),
        moral_list=render_entries(offered.get(MORAL_LIST, [])),
        epistemic_list=render_entries(offered.get(EPISTEMIC_LIST, [])),
    )
    return split_sections(rendered)


def build_scoring_schema(offered: dict[str, list[Consideration]]) -> dict:
    """The answer schema for one call: for each list that has entries, an array of
    {label, score, where_it_arises} items whose ``label`` enum is the offered labels
    in the shuffled order and whose ``score`` is an integer enum 1-5. A list with no
    entries is left out of the schema."""
    def list_schema(entries: list[Consideration]) -> dict:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": [entry.label for entry in entries]},
                    "score": {"type": "integer", "enum": list(assignment.SCORE_VALUES)},
                    "where_it_arises": {"type": "string"},
                },
                "required": ["label", "score", "where_it_arises"],
                "additionalProperties": False,
            },
        }
    names = [name for name in LIST_NAMES if offered.get(name)]
    return {
        "type": "object",
        "properties": {name: list_schema(offered[name]) for name in names},
        "required": names,
        "additionalProperties": False,
    }


def parse_scoring_answer(raw: str, offered: dict[str, list[Consideration]]) -> dict[str, dict[str, dict]]:
    """{list_name: {label: {"score", "where_it_arises"}}} in the order the model answered.

    Raises ScoringAnswerError unless the answer is one JSON object holding, for every
    offered list, items that name each offered label exactly once with an integer score
    1-5 (and, for a score of 3 or more, a non-empty ``where_it_arises``), and nothing
    for a list that was not offered. ``where_it_arises`` is blanked for scores below 3."""
    answer = structured_answers.parse_json_answer(
        raw, ScoringAnswerError, caught_exceptions=(ValueError,), failure_message="no usable JSON object ({error})")
    if not isinstance(answer, dict):
        raise ScoringAnswerError("answer is not a JSON object")
    parsed: dict[str, dict[str, dict]] = {}
    for list_name in LIST_NAMES:
        entries = offered.get(list_name) or []
        items = answer.get(list_name)
        if not entries:
            if items:
                raise ScoringAnswerError(f"{list_name}: scores given for a list that was not offered")
            parsed[list_name] = {}
            continue
        if not isinstance(items, list):
            raise ScoringAnswerError(f"{list_name}: missing or not an array")
        offered_labels = {entry.label for entry in entries}
        scored: dict[str, dict] = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("label"), str):
                raise ScoringAnswerError(f"{list_name}: malformed item {item!r}")
            label = item["label"].strip()
            if label not in offered_labels:
                raise ScoringAnswerError(f"{list_name}: {item['label']!r} is not on the list shown")
            if label in scored:
                raise ScoringAnswerError(f"{list_name}: {label!r} scored more than once")
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or score not in assignment.SCORE_VALUES:
                raise ScoringAnswerError(f"{list_name}: {label!r} has an invalid score {score!r}")
            where = item.get("where_it_arises")
            where = " ".join(where.split()) if isinstance(where, str) else ""
            if score >= 3 and not where:
                raise ScoringAnswerError(f"{list_name}: {label!r} scored {score} without saying where it arises")
            scored[label] = {"score": score, "where_it_arises": where if score >= 3 else ""}
        missing = sorted(offered_labels - set(scored))
        if missing:
            raise ScoringAnswerError(f"{list_name}: no score for {missing}")
        parsed[list_name] = scored
    return parsed


def scoring_max_tokens(offered: dict[str, list[Consideration]]) -> int:
    """The token limit of one scoring call: a base plus an allowance per offered entry."""
    return _BASE_MAX_TOKENS + _TOKENS_PER_ENTRY * sum(len(entries) for entries in offered.values())


# ---------------------------------------------------------------- the considerations spec



class ConsiderationScoringStep(DocumentStep):
    """Score one plan's offered entries. The document is the plan record (``prompt_id``,
    ``description``, ``variables``). One call per attempt: an unusable answer (truncated,
    malformed, a label missing or repeated, a bad score) is asked again, the same request, up to
    ``sdf.max_scoring_attempts`` times in one run, then left behind with the last problem; a thrown
    call is not asked again. Thrown calls and unusable answers both count towards the systemic
    stop, and any failed document stops the run before the assignment."""

    step_name = "consideration_scoring"
    stage_file_name = SCORES_FILENAME
    stage_file_id_field = "prompt_id"
    writes_stage_file_as_replies_arrive = True   # the scores file is the checkpoint: write each row the moment it arrives
    call_noun = "scoring call"

    def __init__(self, config: dict, catalog: ConsiderationCatalog, template: str):
        sdf = config["sdf"]
        self.catalog = catalog
        self.template = template
        self.model = model_setting(sdf, "consideration_scoring_model") or sdf.get("plan_model")
        self.seed = sdf.get("seed", 0)
        self.maximum_attempts = int(sdf.get("max_scoring_attempts", DEFAULT_MAX_SCORING_ATTEMPTS))
        self.skip_no_stake = bool(sdf.get("select_skip_no_stake", True))
        self.policy = StepPolicy(
            maximum_attempts_in_one_run=self.maximum_attempts,
            # The systemic stop is raised by ``after_documents_are_processed`` (after the summary
            # line, as before), so the runner's own check is switched off.
            failure_kinds_counting_towards_the_systemic_stop=frozenset(),
            failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP,
            errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND
            | WhereAnErrorIsAThrownCall.INTERPRET)

    def offered_entries(self, plan: dict) -> dict[str, list[Consideration]]:
        """The entries offered for the plan, each list in its shuffled order: a pure function of the
        catalog, the plan's variables, the run's seed and the prompt id."""
        return shuffled_offer(
            eligible_entries_for_document(self.catalog, plan["variables"]), self.seed, plan["prompt_id"])

    def applies_to(self, plan: dict) -> bool:
        """A plan with a description, outside the no-welfare-stake arm (when that is skipped)."""
        return bool(plan.get("description")) and not (
            self.skip_no_stake and self.catalog.is_skipped_resolution(plan["variables"]))

    def build_request(self, plan: dict) -> ModelRequest:
        offered = self.offered_entries(plan)
        system, user = build_scoring_prompt(self.template, DocumentFacts.from_plan(plan), offered)
        return self.model_request(
            plan, user, system, self.model,
            max_tokens=scoring_max_tokens(offered),
            json_schema=build_scoring_schema(offered),
        )

    def interpret(self, plan: dict, reply: ModelReply):
        try:
            structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScoringAnswerError)
            scores = parse_scoring_answer(reply.text, self.offered_entries(plan))
        except ScoringAnswerError as error:
            problem = str(error)
            if reply.attempt >= self.maximum_attempts:
                problem = f"{problem} (after {self.maximum_attempts} attempts)"
            return Failed(FailureKind.UNUSABLE_ANSWER, problem)
        return Accepted({"scores": scores, "attempts": reply.attempt})

    def stage_file_record(self, plan: dict, update: dict) -> dict:
        return {"prompt_id": plan["prompt_id"], "scores": update["scores"], "attempts": update["attempts"]}

    @staticmethod
    def _problem_text(failed: Failed) -> str:
        if failed.kind is FailureKind.UNUSABLE_ANSWER:
            return f"unusable answer ({failed.reason})"
        return f"API call failed ({reason_without_error_prefix(failed.reason)})"

    def outcome_records(self, plan: dict, outcome) -> OutcomeRecords:
        """An accepted score row finishes the plan (the scores file is the checkpoint); a failure writes
        its problem to the failures file and finishes nothing."""
        if isinstance(outcome, Failed):
            return OutcomeRecords(((FAILURES_FILENAME, {"prompt_id": plan["prompt_id"], "problem": self._problem_text(outcome)}),))
        return super().outcome_records(plan, outcome)

    def progress_line(self, plan: dict, outcome, progress: StepProgress) -> str | None:
        lines = []
        if isinstance(outcome, Failed):
            lines.append(line_for_document_left_to_retry(
                plan["prompt_id"], self._problem_text(outcome), ending="will be rescored on resume"))
        if progress.clock.is_due():
            elapsed_seconds = progress.clock.mark_printed()
            lines.append(f"  Scoring progress: {progress.answered} answered, {progress.failed} failed of "
                         f"{progress.pending_count} ({elapsed_seconds:.1f}s elapsed)")
        return "\n".join(lines) if lines else None

    def after_documents_are_processed(self, run_result) -> None:
        """The summary line, then the systemic stop when every pending call failed (thrown or
        unusable). Nothing is printed for a run with nothing to score."""
        if not run_result.pending_document_ids:
            return
        progress = run_result.progress
        print(f"  Scoring: {progress.answered} answered, {progress.failed} failed ({progress.clock.seconds_since_start():.1f}s)")
        if len(run_result.failed_document_ids) == len(run_result.pending_document_ids):
            raise SystemExit(self.systemic_stop_message)

    def message_when_failed_documents_stop_the_run(self, run_result) -> str:
        return (f"{self.stage_tag}: {len(run_result.failed_document_ids)} of {len(run_result.pending_document_ids)} "
                "scoring calls failed — stopping before the "
                "assignment, because the assignment is solved over every document together and a "
                "document drafted without scores would be reassigned on resume; fix and --resume.")
