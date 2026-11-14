"""Layer 1 step: the implausibility screen, which asks the model, ten hands at a time, which are extremely implausible.
The model answers with local numbers 1..N, never hand ids; this module maps them back. ``screen_hands`` and
``build_screen_prompt`` are importable alone and use the stage's prompt files, schema and validation."""

import copy
import random
from dataclasses import dataclass
from pathlib import Path

from shared import utils
from shared.matrix import split_sections
from shared.sdf_run_layouts import DEAL_VARIABLES_AND_SCREEN
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery import structured_answers
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import (
    BATCHES_FILENAME, CHECKPOINT_FILENAME, DEFAULT_BATCH_SIZE, HAND_BLOCK_FILENAME, JUDGMENTS_FILENAME,
    REJECTIONS_FILENAME, SCHEMA_FILENAME, SHUFFLE_SEED_NAME, TEMPLATE_FILENAME)
from sdf_pipeline.shared_step_machinery.step_record import Run, Step, loaded_once_per_run
from sdf_pipeline.shared_step_machinery.step_runner import StepRunResult, send_model_request
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, Removed, StepPolicy, StepProgress,
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
    """One judgment per hand, ordered by number; raises ScreenAnswerError unless the answer has exactly ``hand_count`` entries, numbers 1..N once each, and a reason on every removal."""
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
    """The request for one batch: schema-constrained (number enum 1..N), the global temperature, tagged for the cost log."""
    system, user = build_screen_prompt(prompts, rendered_hands)
    return ModelRequest(
        user_message=user,
        system_prompt=system or "",
        model=model,
        stage=STEP.stage_tag,
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
    """Screen one batch of rendered hands and return a judgment for each (removals have ``remove`` true).
    Raises ScreenAnswerError for a truncated or invalid answer; API errors propagate."""
    reply = send_model_request(build_screen_request(prompts, rendered_hands, model, item_id))
    structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScreenAnswerError)
    return parse_screen_answer(reply.text, len(rendered_hands))


# ---------------------------------------------------------------- batching

def plan_round_batches(
    output_dir: Path, round_number: int, prompt_ids: list[str], batch_size: int, seed: int
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
    shuffled = sorted(prompt_ids)
    random.Random(f"{seed}:{SHUFFLE_SEED_NAME}:round{round_number}").shuffle(shuffled)
    batches = []
    for start in range(0, len(shuffled), batch_size):
        batch = {
            "batch_id": f"screen_round{round_number}_batch{len(batches):04d}",
            "round": round_number,
            "prompt_ids": shuffled[start:start + batch_size],
        }
        batches.append(batch)
        utils.append_jsonl(batch, path)
    return batches


# ---------------------------------------------------------------- the step

@loaded_once_per_run
def _prompts(run: Run) -> ScreenPrompts:
    return load_screen_prompts(run.prompts_directory)


@loaded_once_per_run
def _render_variable_lines(run: Run):
    return cp.make_variable_line_renderer(
        run.prompts_directory, run.config["sdf"].get("entity_pool_seed", cp.DEFAULT_ENTITY_SEED))


def group_hands_into_seeded_batches(hands: list, run: Run) -> dict[str, list]:
    """The batches of the round ``hands`` belong to, {batch id: hands carrying that ``batch_id``}. The round's
    batches are saved before any call, and a resume reuses them (``plan_round_batches``)."""
    if not hands:
        return {}
    sdf = run.config["sdf"]
    batch_records = plan_round_batches(
        run.stage_directories[STEP.folder_key], hands[0]["round"], [hand["prompt_id"] for hand in hands],
        int(sdf.get("screen_batch_size", DEFAULT_BATCH_SIZE)), sdf.get("seed", 0))
    hands_by_id = {hand["prompt_id"]: hand for hand in hands}
    return {record["batch_id"]: [{**hands_by_id[prompt_id], "batch_id": record["batch_id"]} for prompt_id in record["prompt_ids"]]
            for record in batch_records}


def build_request(hands: list, run: Run) -> ModelRequest:
    shown_lines = [_render_variable_lines(run)(hand) for hand in hands]
    return build_screen_request(_prompts(run), shown_lines, STEP.model_for(run.config), hands[0]["batch_id"])


def read_reply(hands: list, reply: ModelReply, run: Run) -> dict:
    """Each hand ``Accepted`` or ``Removed``, or every hand ``Failed`` (``UNUSABLE_ANSWER``) when the answer is truncated, malformed, miscounted, duplicated or out of range."""
    try:
        structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScreenAnswerError)
        judgments = parse_screen_answer(reply.text, len(hands))
    except ScreenAnswerError as error:
        failure = Failed(FailureKind.UNUSABLE_ANSWER, f"unusable answer ({error})")
        return {hand["prompt_id"]: failure for hand in hands}
    outcomes: dict = {}
    for hand, judgment in zip(hands, judgments):
        facts = {"number": judgment.number, "reasoning": judgment.reasoning, "reason": judgment.reason}
        outcomes[hand["prompt_id"]] = Removed("extremely_implausible", facts) if judgment.remove else Accepted(facts)
    return outcomes


def records(hands: list, outcomes: dict, run: Run) -> OutcomeRecords:
    """The batch's lines of screen_judgments.jsonl and screen_rejections.jsonl in hand order, replacing the batch's
    earlier lines; nothing for a failed batch."""
    if any(isinstance(outcome, Failed) for outcome in outcomes.values()):
        return OutcomeRecords()
    batch_id, round_number = hands[0]["batch_id"], hands[0]["round"]
    new_records = []
    for hand in hands:
        outcome = outcomes[hand["prompt_id"]]
        removed = isinstance(outcome, Removed)
        facts = outcome.detail if removed else outcome.update
        new_records.append((JUDGMENTS_FILENAME, {
            "prompt_id": hand["prompt_id"], "round": round_number, "batch_id": batch_id, "number": facts["number"],
            "reasoning": facts["reasoning"], "remove": removed, "reason": facts["reason"]}))
        if removed:
            new_records.append((REJECTIONS_FILENAME, {
                "prompt_id": hand["prompt_id"], "round": round_number, "batch_id": batch_id, "number": facts["number"],
                "reason": facts["reason"], "variables": hand["variables"],
                "variable_lines": _render_variable_lines(run)(hand)}))
    return OutcomeRecords(
        records=tuple(new_records),
        replacing=((JUDGMENTS_FILENAME, "batch_id", batch_id), (REJECTIONS_FILENAME, "batch_id", batch_id)))


def progress_line(hands: list, outcomes: dict, progress: StepProgress, run: Run) -> str:
    batch_id = hands[0]["batch_id"]
    failures = [outcome for outcome in outcomes.values() if isinstance(outcome, Failed)]
    if failures:
        if failures[0].kind is FailureKind.THROWN_CALL:
            problem = f"API call failed ({reason_without_error_prefix(failures[0].reason)})"
        else:
            problem = failures[0].reason
        return line_for_document_left_to_retry(batch_id, problem)
    removed = sum(isinstance(outcome, Removed) for outcome in outcomes.values())
    return (f"  Screened {batch_id} ({progress.groups_seen}/{progress.pending_count}): {len(hands)} hands, "
            f"{removed} extremely implausible, {progress.seconds_of_last_group:.1f}s")


def stop_message(hands: list, result: StepRunResult, run: Run) -> str:
    return (f"{STEP.stage_tag}: {len(result.failed_groups)} batch(es) of round {result.failed_groups[0][0]['round']} "
            "failed; the next round cannot be dealt until the round is complete — --resume retries them.")


STEP = Step(
    name="screen", folder_key=DEAL_VARIABLES_AND_SCREEN, model_setting_keys=("screen_model", "description_write_model"),
    required_prompt_files=lambda config, prompts_directory: [TEMPLATE_FILENAME, HAND_BLOCK_FILENAME, SCHEMA_FILENAME],
    reason_inactive=lambda config, prompts_directory: reason_screen_is_inactive(config, prompts_directory),
    group=group_hands_into_seeded_batches,
    build_request=build_request, read_reply=read_reply, records=records,
    checkpoint_file_name=CHECKPOINT_FILENAME,
    policy=StepPolicy(
        failure_kinds_counting_towards_the_systemic_stop=frozenset({FailureKind.THROWN_CALL, FailureKind.UNUSABLE_ANSWER}),
        failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP,
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND
        | WhereAnErrorIsAThrownCall.INTERPRET),
    call_noun="screen call", unit_noun="batch",
    progress_line=progress_line, stop_message=stop_message)


def reason_screen_is_inactive(config: dict, prompts_dir: Path) -> str | None:
    """Why the screen (and the deal and redeal it carries) does not run under this configuration and
    prompt snapshot, as the line ``run`` prints; None when it runs."""
    if not config["sdf"].get("screen_enabled", True):
        return "Screen disabled (sdf.screen_enabled: false)"
    if load_screen_prompts(prompts_dir) is None:
        return f"No {TEMPLATE_FILENAME} in this run's prompt snapshot — screen skipped"
    return None
