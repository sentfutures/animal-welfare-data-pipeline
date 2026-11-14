"""Layer 1 step: the implausibility screen, an update step that judges the dealt hands before any planning.

This stage shows the model the hands, ten at a time, rendered as the variable lines the planner will see, and
asks which are extremely implausible: no realistic document could satisfy every dealt value, judged against how
the world actually is. It answers with local numbers 1..N, never hand ids; this module maps the numbers back.
Removed hands are never planned (the redeal in ``redeal.py`` deals their cards again).

The screening logic is importable on its own: ``screen_hands`` takes the rendered variable lines of one batch of
hands and returns one judgment per hand, using the same prompt files, answer schema and validation as the
stage, so offline validation scripts can reuse it unchanged. ``build_screen_prompt`` is the matching prompt
builder. The model's answer is schema-constrained JSON, one entry per hand (reasoning first, then the decision);
every entry is kept in screen_judgments.jsonl so keeps are auditable.

``ScreenStep`` is a batch step run by ``BatchStepRunner``. The files of the stage, the deal and the redeal loop
are described in ``sdf_pipeline/layer1_deal_and_screen/run_layer.py``.

Config (``sdf`` section): ``screen_enabled`` (default true), ``screen_model`` (falls back to ``plan_model``, then
the global model), ``screen_batch_size`` (default 10), ``screen_redeal_rounds`` (default 3).
"""

import copy
import random
from dataclasses import dataclass
from pathlib import Path

from shared import utils
from shared.matrix import split_sections
from sdf_pipeline.shared_step_machinery import structured_answers
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.layer1_deal_and_screen.shared_by_this_layers_steps import (
    BATCHES_FILENAME, HAND_BLOCK_FILENAME, SCHEMA_FILENAME, SHUFFLE_SEED_NAME, TEMPLATE_FILENAME)
from sdf_pipeline.shared_step_machinery.step_runner import send_model_request
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, BatchStep, Failed, FailureKind, ModelReply, ModelRequest, Removed, StepPolicy,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall, line_for_document_left_to_retry, reason_without_error_prefix)

_MAX_TOKENS = 8000  # reasoning plus a decision for each of 10 hands; a batch used 1,819 and another hit 2,000 on 2026-10-22


class ScreenAnswerError(ValueError):
    """The model's screen answer was truncated, malformed, or out of range."""


@dataclass(frozen=True)
class ScreenJudgment:
    number: int  # the local 1-based number the model gave
    reasoning: str
    remove: bool
    reason: str  # one-sentence reason; non-empty whenever remove is true

    @property
    def index(self) -> int:
        return self.number - 1


# ---------------------------------------------------------------- prompt and parsing

@dataclass(frozen=True)
class ScreenPrompts:
    """Every model-facing string of the screen, read from prompts/sdf/*.txt (and the
    answer schema from prompts/sdf/layer1_screen_answer_schema.json)."""
    template: str
    hand_block: str
    answer_schema: dict


def load_screen_prompts(prompts_dir: Path) -> ScreenPrompts | None:
    """The screen's prompt files, or None for a run whose snapshot predates the screen."""
    loaded = structured_answers.load_template_and_answer_schema(prompts_dir, TEMPLATE_FILENAME, SCHEMA_FILENAME)
    if loaded is None:
        return None
    template, answer_schema = loaded
    return ScreenPrompts(
        template=template,
        hand_block=(Path(prompts_dir) / HAND_BLOCK_FILENAME).read_text(encoding="utf-8"),
        answer_schema=answer_schema,
    )


def build_answer_schema(prompts: ScreenPrompts, hand_count: int) -> dict:
    """The answer schema with the ``number`` enum filled for this batch (1..N)."""
    schema = copy.deepcopy(prompts.answer_schema)
    schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] = list(
        range(1, hand_count + 1)
    )
    return schema


def build_screen_prompt(prompts: ScreenPrompts, rendered_hands: list[str]) -> tuple[str | None, str]:
    """Render (system, user) for one batch. ``rendered_hands`` holds each hand's
    variable lines as the planner sees them (``compose_prompts.render_variable_lines``)."""
    blocks = [
        prompts.hand_block.format(number=number, variable_lines=lines.strip())
        for number, lines in enumerate(rendered_hands, 1)
    ]
    rendered = prompts.template.format(hand_count=len(rendered_hands), hands="\n\n".join(blocks))
    return split_sections(rendered)


def parse_screen_answer(raw: str, hand_count: int) -> list[ScreenJudgment]:
    """One judgment per hand, ordered by number. Raises ScreenAnswerError unless
    the answer is JSON of the schema's shape with exactly hand_count entries,
    numbers 1..N each exactly once, and a non-empty reason on every entry marked
    for removal."""
    entries = structured_answers.parse_json_answer(raw, ScreenAnswerError, extract=lambda data: data["specifications"])
    if not isinstance(entries, list) or len(entries) != hand_count:
        raise ScreenAnswerError(
            f"expected {hand_count} entries, got "
            f"{len(entries) if isinstance(entries, list) else type(entries).__name__}")
    judgments: dict[int, ScreenJudgment] = {}
    for entry in entries:
        try:
            number, reasoning, remove, reason = (
                entry["number"], entry["reasoning"], entry["remove"], entry["reason"])
        except (KeyError, TypeError) as e:
            raise ScreenAnswerError(f"entry missing a field: {entry!r}") from e
        if (isinstance(number, bool) or not isinstance(number, int)
                or not isinstance(reasoning, str) or not isinstance(remove, bool)
                or not isinstance(reason, str)):
            raise ScreenAnswerError(f"entry has a wrongly typed field: {entry!r}")
        if not 1 <= number <= hand_count:
            raise ScreenAnswerError(f"number {number} is outside 1..{hand_count}")
        if number in judgments:
            raise ScreenAnswerError(f"number {number} listed twice")
        if remove and not reason.strip():
            raise ScreenAnswerError(f"number {number} is marked for removal without a reason")
        judgments[number] = ScreenJudgment(
            number=number, reasoning=reasoning.strip(), remove=remove, reason=reason.strip())
    return [judgments[n] for n in range(1, hand_count + 1)]


def build_screen_request(
    prompts: ScreenPrompts, rendered_hands: list[str], model: str | None, item_id: str | None
) -> ModelRequest:
    """The request for one batch: schema-constrained (number enum 1..N), the global temperature,
    tagged with the screen's stage tag for the cost log."""
    system, user = build_screen_prompt(prompts, rendered_hands)
    return ModelRequest(
        user_message=user,
        system_prompt=system or "",
        model=model,
        stage=ScreenStep.stage,
        item_id=item_id,
        max_tokens=_MAX_TOKENS,
        json_schema=build_answer_schema(prompts, len(rendered_hands)),
    )


def screen_hands(
    rendered_hands: list[str],
    *,
    prompts: ScreenPrompts,
    model: str | None = None,
    item_id: str | None = None,
) -> list[ScreenJudgment]:
    """Screen one batch of rendered hands and return a judgment for every one
    (reasoning included, so keeps are auditable); the removals are the entries
    with ``remove`` true.

    The call is schema-constrained (number enum 1..N). Raises ScreenAnswerError
    for a truncated or invalid answer; API errors propagate. Uses the global
    temperature and tags the call with the screen's stage tag for the cost log."""
    reply = send_model_request(build_screen_request(prompts, rendered_hands, model, item_id))
    structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScreenAnswerError)
    return parse_screen_answer(reply.text, len(rendered_hands))


# ---------------------------------------------------------------- batching and redealing

def shuffled_batch_records(prompt_ids: list[str], batch_size: int, seed: int, round_number: int) -> list[dict]:
    """The batches of one round: a seeded shuffle of the sorted ids (so input order is irrelevant),
    cut into batch_size chunks. The seed string is built from the run seed, the stage tag and the
    round number, exactly as it always was."""
    shuffled = sorted(prompt_ids)
    random.Random(f"{seed}:{SHUFFLE_SEED_NAME}:round{round_number}").shuffle(shuffled)
    return [
        {
            "batch_id": f"screen_round{round_number}_batch{index:04d}",
            "round": round_number,
            "prompt_ids": shuffled[start:start + batch_size],
        }
        for index, start in enumerate(range(0, len(shuffled), batch_size))
    ]


def plan_round_batches(
    output_dir: Path, round_number: int, prompt_ids: list[str], batch_size: int, seed: int,
) -> list[dict]:
    """The batches of one round, persisted before any call of that round.

    A seeded shuffle (of the sorted ids, so input order is irrelevant) cut into
    batch_size chunks. On resume the round's saved batches are reused verbatim when
    together they hold exactly the round's hands; a partial set is planned again."""
    path = output_dir / BATCHES_FILENAME
    every_batch = utils.load_jsonl(path)
    saved = [b for b in every_batch if b["round"] == round_number]
    saved_ids = [pid for b in saved for pid in b["prompt_ids"]]
    if saved and sorted(saved_ids) == sorted(prompt_ids):
        return saved
    if saved:  # an interrupted write left only some of the round's batches; plan the round again
        utils.save_jsonl([b for b in every_batch if b["round"] != round_number], path)
    batches = shuffled_batch_records(prompt_ids, batch_size, seed, round_number)
    for batch in batches:
        utils.append_jsonl(batch, path)
    return batches


@dataclass(frozen=True)
class ScreenBatch:
    """One batch of a round: its hand records (``members``) and the variable lines the model is shown
    for each, in the same order. ``as_record`` is the line screen_batches.jsonl holds."""
    batch_id: str
    round_number: int
    members: tuple
    shown_lines: tuple

    @property
    def document_ids(self) -> list[str]:
        return [member["prompt_id"] for member in self.members]

    def as_record(self) -> dict:
        return {"batch_id": self.batch_id, "round": self.round_number, "prompt_ids": self.document_ids}


class ScreenStep(BatchStep):
    """The implausibility screen: one call per batch of hands, answered with local numbers 1..N.

    ``interpret`` gives each member ``Accepted`` (kept; the update is its judgment) or ``Removed``
    (extremely implausible; the detail is its judgment), or every member ``Failed`` when the answer is
    truncated, malformed, miscounted, duplicated or out of range (``UNUSABLE_ANSWER``). A thrown call
    is made ``Failed`` (``THROWN_CALL``) by the runner. Both kinds count towards the systemic stop, and
    a failed batch stops the run before the redeal."""

    step_name = "screen"
    policy = StepPolicy(
        failure_kinds_counting_towards_the_systemic_stop=frozenset({FailureKind.THROWN_CALL, FailureKind.UNUSABLE_ANSWER}),
        failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP,
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND
        | WhereAnErrorIsAThrownCall.INTERPRET)
    call_noun = "screen call"

    def __init__(self, prompts: ScreenPrompts, model: str | None, batch_size: int, render_variable_lines):
        self.prompts = prompts
        self.model = model
        self.batch_size = batch_size
        self.render_variable_lines = render_variable_lines

    def batch_from_record(self, record: dict, hands_by_id: dict) -> ScreenBatch:
        members = tuple(hands_by_id[prompt_id] for prompt_id in record["prompt_ids"])
        return ScreenBatch(
            batch_id=record["batch_id"], round_number=record["round"], members=members,
            shown_lines=tuple(self.render_variable_lines(member) for member in members))

    def group(self, states: list, seed: int, round_number: int) -> list[ScreenBatch]:
        """Batches of ``batch_size``: a seeded shuffle of the sorted hand ids, seeded from the run
        seed, the stage tag and the round number. Saving and reusing the plan is the driver's job
        (``plan_round_batches``)."""
        records = shuffled_batch_records([state["prompt_id"] for state in states], self.batch_size, seed, round_number)
        hands_by_id = {state["prompt_id"]: state for state in states}
        return [self.batch_from_record(record, hands_by_id) for record in records]

    def build_request(self, batch: ScreenBatch) -> ModelRequest:
        return build_screen_request(self.prompts, list(batch.shown_lines), self.model, batch.batch_id)

    def interpret(self, batch: ScreenBatch, reply: ModelReply) -> dict:
        try:
            structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScreenAnswerError)
            judgments = parse_screen_answer(reply.text, len(batch.members))
        except ScreenAnswerError as error:
            failure = Failed(FailureKind.UNUSABLE_ANSWER, f"unusable answer ({error})")
            return {document_id: failure for document_id in batch.document_ids}
        outcomes: dict = {}
        for member, judgment in zip(batch.members, judgments):
            facts = {"number": judgment.number, "reasoning": judgment.reasoning, "reason": judgment.reason}
            outcomes[member["prompt_id"]] = Removed("extremely_implausible", facts) if judgment.remove else Accepted(facts)
        return outcomes

    def stage_file_records(self, batch: ScreenBatch, outcomes: dict) -> tuple[list[dict], list[dict]]:
        """(the batch's lines of screen_judgments.jsonl, the batch's lines of screen_rejections.jsonl),
        in member order."""
        judgment_records, rejection_records = [], []
        for member, lines in zip(batch.members, batch.shown_lines):
            outcome = outcomes[member["prompt_id"]]
            removed = isinstance(outcome, Removed)
            facts = outcome.detail if removed else outcome.update
            judgment_records.append({
                "prompt_id": member["prompt_id"], "round": batch.round_number,
                "batch_id": batch.batch_id, "number": facts["number"],
                "reasoning": facts["reasoning"], "remove": removed,
                "reason": facts["reason"],
            })
            if removed:
                rejection_records.append({
                    "prompt_id": member["prompt_id"], "round": batch.round_number,
                    "batch_id": batch.batch_id, "number": facts["number"],
                    "reason": facts["reason"], "variables": member["variables"],
                    **value_ids_field_of(member), "variable_lines": lines,
                })
        return judgment_records, rejection_records

    def progress_line(self, batch: ScreenBatch, outcomes: dict, done: int, total: int, seconds: float) -> str:
        removed = sum(isinstance(outcome, Removed) for outcome in outcomes.values())
        return (f"  Screened {batch.batch_id} ({done}/{total}): {len(batch.members)} hands, "
                f"{removed} extremely implausible, {seconds:.1f}s")

    def failed_batch_line(self, batch: ScreenBatch, failure: Failed) -> str:
        if failure.kind is FailureKind.THROWN_CALL:
            problem = f"API call failed ({reason_without_error_prefix(failure.reason)})"
        else:
            problem = failure.reason
        return line_for_document_left_to_retry(batch.batch_id, problem)

    def stop_message_for_failed_batches(self, failed_batch_count: int, failed_batches: list) -> str:
        return (f"{self.stage}: {failed_batch_count} batch(es) of round {failed_batches[0].round_number} failed; the next "
                "round cannot be dealt until the round is complete — --resume retries them.")



def reason_screen_is_inactive(config: dict, prompts_dir: Path) -> str | None:
    """Why the screen (and the deal and redeal it carries) does not run under this configuration and
    prompt snapshot, as the line ``run`` prints; None when it runs."""
    if not config["sdf"].get("screen_enabled", True):
        return "Screen disabled (sdf.screen_enabled: false)"
    if load_screen_prompts(prompts_dir) is None:
        return f"No {TEMPLATE_FILENAME} in this run's prompt snapshot — screen skipped"
    return None
