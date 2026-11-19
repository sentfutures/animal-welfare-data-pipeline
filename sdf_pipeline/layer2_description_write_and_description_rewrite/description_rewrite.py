"""Layer 2 step, the description rewrite: one call per planned description checks it for plausibility, dealt-value conformance and organisation naming.
A rewrite keeps ``original_description`` and ``changes``; a dealt-value conflict drops the plan. Results live in description_rewrites.jsonl (plans.jsonl is never modified)
and a malformed answer is not recorded, so ``--resume`` retries it. Use a GPT model for the call: Claude was far too lenient on plausibility (evidence in ``config.yaml``)."""


import copy
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import openai_api, sdf_run_layouts, utils
from shared.api_keys import FIND_DOTENV_FLAG, load_optional_api_key
from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.renamed_model_settings import model_setting
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery import structured_answers
from sdf_pipeline.layer2_description_write_and_description_rewrite.description_write import PLANS
from sdf_pipeline.shared_step_machinery.step_record import Run, StageFile, Step, loaded_once_per_run, description_write_prompt_files
from sdf_pipeline.shared_step_machinery.step_runner import find_pending, run_step, send_model_request
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, Removed, StepPolicy, StepProgress,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall, line_for_document_left_to_retry, reason_without_error_prefix)
from shared.sdf_run_layouts import DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE

TEMPLATE_FILENAME = "layer1_description_rewrite.txt"
SCHEMA_FILENAME = "layer1_description_rewrite_answer_schema.json"
RESULTS_FILENAME = "description_rewrites.jsonl"
REJECTIONS_FILENAME = "description_rewrite_rejections.jsonl"
CHECKPOINT_FILENAME = "_description_rewrite_checkpoint.json"
_MAX_TOKENS = 8000  # for OpenAI models this includes reasoning tokens; reasoning, a description under 300 words and one-line reasons; non-English descriptions and long reasoning overran 3000


class RewriteAnswerError(ValueError):
    """The model's rewrite answer was truncated, malformed, or contradicted itself."""


@dataclass(frozen=True)
class RewriteResult:
    reasoning: str
    rewrite_needed: bool
    rewritten_description: str  # empty unless rewrite_needed
    changes: tuple[dict, ...]   # {"check", "reason"}; empty unless rewrite_needed
    dealt_value_conflict: bool
    conflict_reason: str        # non-empty whenever dealt_value_conflict is true


@dataclass(frozen=True)
class RewritePrompts:
    template: str
    answer_schema: dict


def load_rewrite_prompts(prompts_dir: Path) -> RewritePrompts | None:
    """The stage's prompt files, or None for a run whose snapshot predates the stage."""
    loaded = structured_answers.load_template_and_answer_schema(prompts_dir, TEMPLATE_FILENAME, SCHEMA_FILENAME)
    if loaded is None:
        return None
    template, answer_schema = loaded
    return RewritePrompts(template=template, answer_schema=answer_schema)


def build_rewrite_prompt(
    prompts: RewritePrompts, variable_lines: str, description: str
) -> tuple[str | None, str]:
    """Render (system, user) for one plan."""
    return cp.split_sections(prompts.template.format(
        variable_lines=variable_lines.strip(), description=description.strip()))


def parse_rewrite_answer(raw: str, allowed_checks: list[str]) -> RewriteResult:
    """Validate a rewrite answer; raises RewriteAnswerError unless it is schema-shaped and consistent.
    A dealt-value conflict has a reason and no rewrite, and its ``rewrite_needed`` is stored as false."""
    reasoning, needed, rewritten, changes, conflict, conflict_reason = structured_answers.parse_json_answer(
        raw, RewriteAnswerError,
        extract=lambda data: (
            data["reasoning"], data["rewrite_needed"], data["rewritten_description"],
            data["changes"], data["dealt_value_conflict"], data["conflict_reason"]))
    if (not isinstance(reasoning, str) or not isinstance(needed, bool) or not isinstance(rewritten, str)
            or not isinstance(changes, list) or not isinstance(conflict, bool)
            or not isinstance(conflict_reason, str)):
        raise RewriteAnswerError("answer has a wrongly typed field")
    cleaned = []
    for change in changes:
        if (not isinstance(change, dict) or change.get("check") not in allowed_checks
                or not isinstance(change.get("reason"), str) or not change["reason"].strip()):
            raise RewriteAnswerError(f"unusable change entry: {change!r}")
        cleaned.append({"check": change["check"], "reason": change["reason"].strip()})
    if conflict:
        if rewritten.strip():
            raise RewriteAnswerError("a dealt-value conflict must come with no rewritten description")
        if not conflict_reason.strip():
            raise RewriteAnswerError("a dealt-value conflict needs a conflict reason")
    elif needed:
        if not rewritten.strip():
            raise RewriteAnswerError("rewrite_needed is true but the rewritten description is empty")
        if not cleaned:
            raise RewriteAnswerError("rewrite_needed is true but no change is listed")
    elif rewritten.strip() or cleaned:
        raise RewriteAnswerError("rewrite_needed is false but a rewrite or changes were given")
    return RewriteResult(
        reasoning=reasoning.strip(), rewrite_needed=needed and not conflict,
        rewritten_description=rewritten.strip() if needed and not conflict else "",
        changes=tuple(cleaned) if needed and not conflict else (),
        dealt_value_conflict=conflict, conflict_reason=conflict_reason.strip() if conflict else "",
    )


def check_names(prompts: RewritePrompts) -> list[str]:
    """The checks the answer schema allows (the single list of check names)."""
    return prompts.answer_schema["properties"]["changes"]["items"]["properties"]["check"]["enum"]


def build_rewrite_request(
    prompts: RewritePrompts, variable_lines: str, description: str, model: str | None,
    reasoning_effort: str | None, item_id: str | None,
) -> ModelRequest:
    """The request for one plan: an OpenAI route for a ``gpt-`` model, otherwise Claude; both schema-constrained."""
    system, user = build_rewrite_prompt(prompts, variable_lines, description)
    settings = dict(
        user_message=user, system_prompt=system or "", model=model, stage=STEP.stage_tag, item_id=item_id,
        max_tokens=_MAX_TOKENS, json_schema=copy.deepcopy(prompts.answer_schema))
    if openai_api.is_openai_model(model):
        return ModelRequest(
            **settings, openai_reasoning_effort=reasoning_effort or openai_api.DEFAULT_REASONING_EFFORT,
            openai_schema_name="description_rewrite_answer")
    return ModelRequest(**settings)


def rewrite_description(
    variable_lines: str,
    description: str,
    *,
    prompts: RewritePrompts,
    model: str | None = None,
    reasoning_effort: str | None = None,
    item_id: str | None = None,
) -> RewriteResult:
    """One rewrite call for one plan. Raises RewriteAnswerError for a truncated or invalid answer; API errors propagate."""
    reply = send_model_request(build_rewrite_request(prompts, variable_lines, description, model, reasoning_effort, item_id))
    structured_answers.raise_if_answer_was_truncated(reply.stop_reason, RewriteAnswerError)
    return parse_rewrite_answer(reply.text, check_names(prompts))


# ---------------------------------------------------------------- stage

def load_results(output_dir: Path) -> dict[str, dict]:
    """{prompt_id: stored result}, de-duplicated."""
    return {r["prompt_id"]: r for r in utils.load_jsonl(output_dir / RESULTS_FILENAME)}


def apply_results(plans: list[dict], results: dict[str, dict]) -> list[dict]:
    """Plans with the stored results applied: a dealt-value conflict removes the
    plan; a rewrite swaps in the new description and keeps ``original_description``
    and ``changes``; a decline, and a plan with no description (never sent), pass
    through untouched (same object). A described plan with no stored result was
    never checked (unusable answer or failed call), so it is held out rather than
    passed on unchecked; a resume retries it."""
    applied = []
    for plan in plans:
        result = results.get(plan["prompt_id"])
        if not plan.get("description"):
            applied.append(plan)
        elif result is None:
            continue
        elif result["dealt_value_conflict"]:
            continue
        elif result["rewrite_needed"]:
            applied.append({
                **plan,
                "description": result["rewritten_description"],
                "original_description": plan["description"],
                "changes": result["changes"],
            })
        else:
            applied.append(plan)
    return applied


def enabled(config: dict) -> bool:
    return bool(config["sdf"].get("description_rewrite_enabled", True))


def require_openai_key_if_configured(config: dict) -> None:
    """Exit before any paid call when the rewrite is enabled, routed to an OpenAI model
    (a ``gpt-`` model name) and no ``OPENAI_API_KEY`` is available. Without this, the
    Claude screen and description write calls are paid for before the first rewrite call fails."""
    if not enabled(config):
        return
    sdf = config["sdf"]
    model = sdf.get("description_rewrite_model") or model_setting(sdf, "description_write_model")
    if openai_api.is_openai_model(model) and load_optional_api_key("OPENAI_API_KEY") is None:
        raise SystemExit(
            f"sdf.description_rewrite_model is '{model}', an OpenAI model, so layer "
            f"{sdf_run_layouts.active_layout().layer_number_of_key[sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE]} needs "
            "OPENAI_API_KEY, and the configured key source has none (by default only the repo "
            f"root's .env is read; pass {FIND_DOTENV_FLAG} to search upward and then use the "
            "environment). Set it, or set sdf.description_rewrite_enabled: false to skip the "
            "description rewrite. Stopping before any paid call.")


def load_rewritten_plans(config: dict, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans with saved rewrite results applied, no calls. Used when a run resumes
    past layer 1 and plans are re-read from disk. A run with no stored results at all
    (the stage never ran, e.g. it predates the stage) passes every plan through; once
    any result exists, described plans without one are held out as in ``apply_results``."""
    if not enabled(config):
        return plans
    results = load_results(output_dir)
    if not results:
        return plans
    return apply_results(plans, results)


RESULTS = StageFile(DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, RESULTS_FILENAME, "prompt_id")
REJECTIONS = StageFile(DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, REJECTIONS_FILENAME, "prompt_id")


def _stored_result_record(plan: dict, reasoning: str, rewrite_needed: bool, rewritten_description: str,
                          changes: list, dealt_value_conflict: bool, conflict_reason: str) -> dict:
    return {
        "prompt_id": plan["prompt_id"],
        "reasoning": reasoning,
        "rewrite_needed": rewrite_needed,
        "original_description": plan["description"],
        "rewritten_description": rewritten_description,
        "changes": changes,
        "dealt_value_conflict": dealt_value_conflict,
        "conflict_reason": conflict_reason,
    }


@loaded_once_per_run
def _prompts(run: Run) -> RewritePrompts:
    return load_rewrite_prompts(run.prompts_directory)


@loaded_once_per_run
def _render_variable_lines(run: Run):
    return cp.make_variable_line_renderer(
        run.prompts_directory, run.config["sdf"].get("entity_pool_seed", cp.DEFAULT_ENTITY_SEED))


def build_request(documents: list, run: Run) -> ModelRequest:
    (plan,) = documents
    return build_rewrite_request(
        _prompts(run), _render_variable_lines(run)(plan), plan["description"], STEP.model_for(run.config),
        run.config["sdf"].get("description_rewrite_reasoning_effort"), plan["prompt_id"])


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """Accepted updates are ``RewriteResult``s (a decline or a rewrite); a dealt-value conflict is ``Removed``; a
    truncated or unparseable answer is a ``Failed`` unusable answer."""
    (plan,) = documents
    try:
        structured_answers.raise_if_answer_was_truncated(reply.stop_reason, RewriteAnswerError)
        result = parse_rewrite_answer(reply.text, check_names(_prompts(run)))
    except RewriteAnswerError as error:
        return {plan["prompt_id"]: Failed(FailureKind.UNUSABLE_ANSWER, f"unusable answer ({error})")}
    if result.dealt_value_conflict:
        return {plan["prompt_id"]: Removed(
            "dealt_value_conflict", {"reasoning": result.reasoning, "conflict_reason": result.conflict_reason})}
    return {plan["prompt_id"]: Accepted(result)}


def records(documents: list, outcomes: dict, run: Run) -> OutcomeRecords:
    """An accepted result is one record. A dealt-value conflict is the stored result and then the rejection (in its
    own file), in that order; both finish the plan."""
    (plan,) = documents
    outcome = outcomes[plan["prompt_id"]]
    if isinstance(outcome, Accepted):
        update = outcome.update
        return OutcomeRecords(((RESULTS_FILENAME, _stored_result_record(
            plan, update.reasoning, update.rewrite_needed, update.rewritten_description, list(update.changes),
            update.dealt_value_conflict, update.conflict_reason)),))
    if isinstance(outcome, Removed):
        detail = outcome.detail
        return OutcomeRecords((
            (RESULTS_FILENAME, _stored_result_record(plan, detail["reasoning"], False, "", [], True, detail["conflict_reason"])),
            (REJECTIONS_FILENAME, {
                "prompt_id": plan["prompt_id"], "variables": plan["variables"],
                "conflict_reason": detail["conflict_reason"], "description": plan["description"]}),
        ))
    return OutcomeRecords()


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (plan,), (outcome,) = documents, outcomes.values()
    if isinstance(outcome, Failed):
        if outcome.kind is FailureKind.THROWN_CALL:
            problem = f"API call failed ({reason_without_error_prefix(outcome.reason)})"
        else:
            problem = outcome.reason
        return line_for_document_left_to_retry(plan["prompt_id"], problem)
    if progress.clock.is_due() or progress.groups_seen == progress.pending_count:
        rewritten_count = sum(update.rewrite_needed for update in progress.accepted_updates)
        elapsed_seconds = progress.clock.mark_printed()
        return (f"  Description rewrite {progress.groups_seen}/{progress.pending_count}: {rewritten_count} "
                f"rewritten, {progress.removed} dropped for a dealt-value conflict "
                f"({elapsed_seconds:.1f}s)")
    return None


def count_held_out_plans(plans: list[dict], stage_directory: Path) -> int:
    """How many described plans have no stored result: their check failed or gave an unusable answer."""
    results = load_results(stage_directory)
    return sum(1 for plan in plans if plan.get("description") and plan["prompt_id"] not in results)


def held_out_stop_message(held_out_count: int) -> str:
    return (f"{STEP.stage_tag}: {held_out_count} descriptions held out — stopping before the "
            "stages that assign over the whole plan set; fix and --resume.")


def after_run(plans: list, run_result, run: Run) -> None:
    stage_directory = run.stage_directories[DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE]
    results = load_results(stage_directory)
    described = [plan for plan in plans if plan.get("description")]
    held_out_count = count_held_out_plans(plans, stage_directory)
    if held_out_count:
        print(f"  Description rewrite: {held_out_count} descriptions held out of drafting because their "
              f"check failed or gave an unusable answer — --resume retries them")
    print(f"  Description rewrite: {sum(r['rewrite_needed'] for r in results.values())} of "
          f"{len(described)} descriptions rewritten, "
          f"{sum(r['dealt_value_conflict'] for r in results.values())} dropped for a dealt-value "
          f"conflict ({run_result.progress.clock.seconds_since_start():.1f}s this session)")


def plans_after_description_rewrite(run: Run) -> list[dict]:
    """The plans with the saved rewrite results applied, no calls."""
    return load_rewritten_plans(run.config, run.stage_directories[STEP.folder_key], run.records(PLANS))


STEP = Step(
    name="description_rewrite", folder_key=DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE,
    model_setting_keys=("description_rewrite_model", "description_write_model"),
    required_prompt_files=lambda config, prompts_directory: [
        *description_write_prompt_files(config, prompts_directory), TEMPLATE_FILENAME, SCHEMA_FILENAME],
    reason_inactive=lambda config, prompts_directory: reason_description_rewrite_is_inactive(config, prompts_directory),
    reads=lambda run: run.records(PLANS),
    applies=lambda plan, run: bool(plan.get("description")),
    build_request=build_request, read_reply=read_reply, records=records, writes=(RESULTS, REJECTIONS),
    checkpoint_file_name=CHECKPOINT_FILENAME,
    policy=StepPolicy(
        failure_kinds_counting_towards_the_systemic_stop=frozenset(FailureKind),
        failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP,
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND
        | WhereAnErrorIsAThrownCall.INTERPRET),
    call_noun="call", unit_noun="plan", progress_line=progress_line, after_run=after_run,
    stop_message=lambda plans, run_result, run: held_out_stop_message(count_held_out_plans(plans, run.stage_directories[DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE])))


def reason_description_rewrite_is_inactive(config: dict, prompts_dir: Path) -> str | None:
    """Why the description rewrite does not run, as the line ``run`` prints; None when it runs."""
    if not enabled(config):
        return "Description rewrite disabled (sdf.description_rewrite_enabled: false)"
    if load_rewrite_prompts(prompts_dir) is None:
        return f"No {TEMPLATE_FILENAME} in this run's prompt snapshot — description rewrite skipped"
    return None


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Check every described plan and return the plans after the rewrite; no calls and ``plans`` unchanged when disabled or the snapshot has no template."""
    reason_inactive = reason_description_rewrite_is_inactive(config, prompts_dir)
    if reason_inactive is not None:
        print(f"  {reason_inactive}")
        return plans
    run_of_rewrite = Run(config, prompts_dir, {DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE: output_dir})
    described = [p for p in plans if p.get("description")]
    _, pending = find_pending(STEP, run_of_rewrite, plans)
    print(f"  Description rewrite: {len(described)} descriptions, {len(pending)} to check "
          f"({len(described) - len(pending)} already done)")
    run_step(STEP, run_of_rewrite, plans)  # stops the run itself when a failed document leaves descriptions held out
    held_out_count = count_held_out_plans(plans, output_dir)
    if held_out_count:  # held out with nothing failing in this run
        raise SystemExit(held_out_stop_message(held_out_count))
    return apply_results(plans, load_results(output_dir))


# ---------------------------------------------------------------- reporting

def load_description_rewrite_outputs(run_dir: Path) -> dict | None:
    """The stage's stored results for a run, or None when it never ran."""
    stage_dir = utils.sdf_stage_folder(run_dir, DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE)
    results = load_results(stage_dir)
    if not results:
        return None
    return {"results": results}


def rewrite_share_rows(plans: list[dict], outputs: dict) -> dict:
    """Description-rewrite shares over the checked plans (``plans`` are the plan
    records as planned, with ``variables``).

    Returns {"checked", "rewritten", "declined", "dropped_for_conflict",
    "per_check": {check: plans changed under it}, "per_value": rows of variable,
    value, checked, rewritten, rate, dropped}."""
    results = outputs["results"]
    checked = [p for p in plans if p["prompt_id"] in results]
    per_check: dict[str, int] = {}
    for plan in checked:
        for check in {c["check"] for c in results[plan["prompt_id"]]["changes"]}:
            per_check[check] = per_check.get(check, 0) + 1
    cells: dict[tuple[str, str], list[int]] = {}
    for plan in checked:
        result = results[plan["prompt_id"]]
        for axis, value in (plan.get("variables") or {}).items():
            cell = cells.setdefault((axis, value), [0, 0, 0])
            cell[0] += 1
            cell[1] += result["rewrite_needed"]
            cell[2] += result["dealt_value_conflict"]
    rows = [
        {"variable": axis, "value": value, "checked": n, "rewritten": n_rewritten,
         "rate": n_rewritten / n, "dropped": n_dropped}
        for (axis, value), (n, n_rewritten, n_dropped) in cells.items()
    ]
    rows.sort(key=lambda r: (-r["rate"], -r["dropped"], r["variable"], r["value"]))
    return {
        "checked": len(checked),
        "rewritten": sum(results[p["prompt_id"]]["rewrite_needed"] for p in checked),
        "declined": sum(not results[p["prompt_id"]]["rewrite_needed"]
                        and not results[p["prompt_id"]]["dealt_value_conflict"] for p in checked),
        "dropped_for_conflict": sum(results[p["prompt_id"]]["dealt_value_conflict"] for p in checked),
        "per_check": dict(sorted(per_check.items(), key=lambda kv: -kv[1])),
        "per_value": rows,
    }
