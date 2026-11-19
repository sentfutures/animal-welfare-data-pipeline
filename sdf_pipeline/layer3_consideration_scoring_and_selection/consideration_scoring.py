"""Layer 3 step: the consideration scoring, where the model scores and code assigns (``consideration_selection.py``). One scoring call per eligible document scores every offered entry 1-5 (entries shuffled per
document with a seeded RNG; the model never sees targets and never reproduces ids); a linear programme in ``considerations_assignment.py`` then assigns over all scores.
Rows are appended to consideration_scores.jsonl (that finishes a document); an unusable answer is retried up to ``sdf.max_scoring_attempts``, then recorded in consideration_scoring_failures.jsonl."""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sdf_pipeline.shared_step_machinery import considerations_assignment as assignment
from sdf_pipeline.shared_step_machinery.considerations_catalog import (
    EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST, Consideration, ConsiderationCatalog, load_considerations)
from sdf_pipeline.shared_step_machinery import structured_answers
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import (
    FAILURES_FILENAME, SCORES_FILENAME, load_selection_template)
from sdf_pipeline.shared_step_machinery.step_record import Run, StageFile, Step, loaded_once_per_run
from sdf_pipeline.shared_step_machinery.step_runner import maximum_attempts
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, StepPolicy, StepProgress,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall, line_for_document_left_to_retry, reason_without_error_prefix)
from shared.matrix import split_sections
from shared.sdf_run_layouts import CONSIDERATION_SCORING_AND_SELECTION

# The shuffle of each document's offered entries is seeded with this text (with the run seed, document and list). Kept
# literal: deriving it from a layout's tag would change every document's order.
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


# ---------------------------------------------------------------- eligibility, prompt and schema

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


SCORES = StageFile(CONSIDERATION_SCORING_AND_SELECTION, SCORES_FILENAME, "prompt_id")


@loaded_once_per_run
def _catalog(run: Run) -> ConsiderationCatalog:
    return load_considerations(run.prompts_directory)


@loaded_once_per_run
def _template(run: Run) -> str:
    return load_selection_template(run.prompts_directory)


def offered_entries(plan: dict, run: Run) -> dict[str, list[Consideration]]:
    """The entries offered for the plan, each list in its shuffled order: a pure function of the catalog, the plan's
    variables, the run's seed and the prompt id."""
    return shuffled_offer(
        eligible_entries_for_document(_catalog(run), plan["variables"]), run.config["sdf"].get("seed", 0), plan["prompt_id"])


def applies(plan: dict, run: Run) -> bool:
    """A plan with a description, outside the no-welfare-stake arm (when that is skipped)."""
    skip_no_stake = bool(run.config["sdf"].get("select_skip_no_stake", True))
    return bool(plan.get("description")) and not (skip_no_stake and _catalog(run).is_skipped_resolution(plan["variables"]))


def build_request(documents: list, run: Run) -> ModelRequest:
    """One scoring call for the plan record (``prompt_id``, ``description``, ``variables``)."""
    (plan,) = documents
    offered = offered_entries(plan, run)
    system, user = build_scoring_prompt(_template(run), DocumentFacts.from_plan(plan), offered)
    return STEP.model_request(
        plan, user, system, run, max_tokens=scoring_max_tokens(offered), json_schema=build_scoring_schema(offered))


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """An unusable answer (truncated, malformed, a label missing or repeated, a bad score) is asked again, the same
    request, up to ``sdf.max_scoring_attempts`` times in one run, then left behind with the last problem."""
    (plan,) = documents
    try:
        structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScoringAnswerError)
        scores = parse_scoring_answer(reply.text, offered_entries(plan, run))
    except ScoringAnswerError as error:
        problem = str(error)
        if reply.attempt >= maximum_attempts(STEP, run):
            problem = f"{problem} (after {maximum_attempts(STEP, run)} attempts)"
        return {plan["prompt_id"]: Failed(FailureKind.UNUSABLE_ANSWER, problem)}
    return {plan["prompt_id"]: Accepted({"scores": scores, "attempts": reply.attempt})}


def _problem_text(failed: Failed) -> str:
    if failed.kind is FailureKind.UNUSABLE_ANSWER:
        return f"unusable answer ({failed.reason})"
    return f"API call failed ({reason_without_error_prefix(failed.reason)})"


def records(documents: list, outcomes: dict, run: Run) -> OutcomeRecords:
    """An accepted score row (in the scores file, which finishes the plan) or, for a failure, its problem in the
    failures file."""
    (plan,), (outcome,) = documents, outcomes.values()
    if isinstance(outcome, Failed):
        return OutcomeRecords(((FAILURES_FILENAME, {"prompt_id": plan["prompt_id"], "problem": _problem_text(outcome)}),))
    return OutcomeRecords(((SCORES_FILENAME, {
        "prompt_id": plan["prompt_id"], "scores": outcome.update["scores"], "attempts": outcome.update["attempts"]}),))


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (plan,), (outcome,) = documents, outcomes.values()
    lines = []
    if isinstance(outcome, Failed):
        lines.append(line_for_document_left_to_retry(
            plan["prompt_id"], _problem_text(outcome), ending="will be rescored on resume"))
    if progress.clock.is_due():
        elapsed_seconds = progress.clock.mark_printed()
        lines.append(f"  Scoring progress: {progress.answered} answered, {progress.failed} failed of "
                     f"{progress.pending_count} ({elapsed_seconds:.1f}s elapsed)")
    return "\n".join(lines) if lines else None


def after_run(plans: list, run_result, run: Run) -> None:
    """The summary line, then the systemic stop when every pending call failed (thrown or unusable). Nothing is
    printed for a run with nothing to score."""
    if not run_result.pending_document_ids:
        return
    progress = run_result.progress
    print(f"  Scoring: {progress.answered} answered, {progress.failed} failed ({progress.clock.seconds_since_start():.1f}s)")
    if len(run_result.failed_document_ids) == len(run_result.pending_document_ids):
        raise SystemExit(STEP.systemic_stop_message)


def stop_message(plans: list, run_result, run: Run) -> str:
    return (f"{STEP.stage_tag}: {len(run_result.failed_document_ids)} of {len(run_result.pending_document_ids)} "
            "scoring calls failed — stopping before the "
            "assignment, because the assignment is solved over every document together and a "
            "document drafted without scores would be reassigned on resume; fix and --resume.")


STEP = Step(
    name="consideration_scoring", folder_key=CONSIDERATION_SCORING_AND_SELECTION,
    model_setting_keys=("consideration_scoring_model", "description_write_model"),
    applies=applies, build_request=build_request, read_reply=read_reply, records=records, writes=(SCORES,),
    writes_as_replies_arrive=True,
    policy=StepPolicy(
        maximum_attempts_in_one_run=DEFAULT_MAX_SCORING_ATTEMPTS, maximum_attempts_setting="max_scoring_attempts",
        # ``after_run`` raises the systemic stop (after the summary line), so the runner's own check is switched off.
        failure_kinds_counting_towards_the_systemic_stop=frozenset(),
        failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP,
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND
        | WhereAnErrorIsAThrownCall.INTERPRET),
    call_noun="scoring call", progress_line=progress_line, after_run=after_run, stop_message=stop_message)
