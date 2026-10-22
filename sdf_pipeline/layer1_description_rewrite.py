"""Layer 1 description rewrite: repair a planned description before drafting.

Runs after planning, before the considerations selector and the first draft.
One call per surviving plan is given the dealt variables (rendered as the lines
the planner saw) and the plan's description, and checks it for plausibility
(organisations, species and customs, simulated minds, what the AI can know,
figures and scale), for conformance to every dealt value, and for organisation
naming. The model may decline: when no check fails it answers
``rewrite_needed: false`` and the original description passes through
untouched. When it rewrites, the plan record keeps both versions
(``original_description`` and the ``changes`` list) and ``description`` becomes
the rewritten text, so layers 2-4 use it unchanged. When the only plausible fix
would change a dealt value the model answers ``dealt_value_conflict: true``
without a rewrite and the plan is dropped and logged to
description_rewrite_rejections.jsonl.

The call is schema-constrained (prompts/sdf/layer1_description_rewrite_answer_schema.json)
and the answer is validated in code. plans.jsonl is never modified; every
result is stored in description_rewrites.jsonl and re-applied on resume.

Config (``sdf`` section): ``description_rewrite_enabled`` (default true),
``description_rewrite_model`` (falls back to ``plan_model``, then the global model),
``description_rewrite_reasoning_effort`` (OpenAI models only). A model name starting
with ``gpt-`` routes to OpenAI (shared/openai_api.py, key from the OPENAI_API_KEY
environment variable); anything else goes to Claude.

Why GPT, not Claude (2026-10-22): Claude is far too lenient on plausibility. Given GPT 6.1
Sol's own one-description plausibility prompt on 97 descriptions, Claude Sonnet 5 flagged
1 of 92 and Claude Opus 5 flagged 0 of 95, catching none of the 14 implausible premises
GPT 6.1 Sol flagged; its reasoning notices the problem, then accepts it as "not clearly
impossible" or allowed "within the fictional world". Do not use Claude for plausibility
judgment without re-validating (see config.yaml for the run folder).

Checkpoint semantics (protects paid work): one result per plan, appended to
description_rewrites.jsonl before the plan is checkpointed; a malformed,
truncated or self-contradictory answer is NOT checkpointed, so --resume retries
exactly that plan.
"""

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, openai_api, utils
from sdf_pipeline import compose_prompts as cp

STAGE_TAG = "layer1_description_rewrite"
TEMPLATE_FILENAME = "layer1_description_rewrite.txt"
SCHEMA_FILENAME = "layer1_description_rewrite_answer_schema.json"
RESULTS_FILENAME = "description_rewrites.jsonl"
REJECTIONS_FILENAME = "description_rewrite_rejections.jsonl"
CHECKPOINT_FILENAME = "_description_rewrite_checkpoint.json"
_MAX_TOKENS = 8000  # for OpenAI models this includes reasoning tokens; reasoning, a description under 300 words and one-line reasons; non-English descriptions and long reasoning overran 3000
PROGRESS_INTERVAL_SECONDS = 180


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
    directory = Path(prompts_dir)
    if not (directory / TEMPLATE_FILENAME).is_file():
        return None
    return RewritePrompts(
        template=(directory / TEMPLATE_FILENAME).read_text(encoding="utf-8"),
        answer_schema=json.loads((directory / SCHEMA_FILENAME).read_text(encoding="utf-8")),
    )


def build_rewrite_prompt(
    prompts: RewritePrompts, variable_lines: str, description: str
) -> tuple[str | None, str]:
    """Render (system, user) for one plan."""
    return cp.split_sections(prompts.template.format(
        variable_lines=variable_lines.strip(), description=description.strip()))


def parse_rewrite_answer(raw: str, allowed_checks: list[str]) -> RewriteResult:
    """Validate a rewrite answer. Raises RewriteAnswerError unless it is JSON of the
    schema's shape and consistent: a rewrite carries a non-empty description and at
    least one change with a known check and a reason; a declined rewrite carries
    neither; a dealt-value conflict carries a conflict reason and no rewrite."""
    try:
        data = json.loads(raw or "")
        reasoning, needed, rewritten, changes, conflict, conflict_reason = (
            data["reasoning"], data["rewrite_needed"], data["rewritten_description"],
            data["changes"], data["dealt_value_conflict"], data["conflict_reason"])
    except (ValueError, KeyError, TypeError) as e:
        raise RewriteAnswerError(f"answer is not the expected JSON ({type(e).__name__}: {e})") from e
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
        if needed or rewritten.strip():
            raise RewriteAnswerError("a dealt-value conflict must come with no rewrite")
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


def rewrite_description(
    variable_lines: str,
    description: str,
    *,
    prompts: RewritePrompts,
    model: str | None = None,
    reasoning_effort: str | None = None,
    item_id: str | None = None,
) -> RewriteResult:
    """One rewrite call for one plan. Raises RewriteAnswerError for a truncated, refused or
    invalid answer; API errors propagate. Tags the call ``layer1_description_rewrite``."""
    system, user = build_rewrite_prompt(prompts, variable_lines, description)
    if openai_api.is_openai_model(model):
        raw, stop_reason = openai_api.call_openai(
            user,
            system_prompt=system or "",
            model=model,
            max_output_tokens=_MAX_TOKENS,
            reasoning_effort=reasoning_effort or openai_api.DEFAULT_REASONING_EFFORT,
            stage=STAGE_TAG,
            item_id=item_id,
            json_schema=copy.deepcopy(prompts.answer_schema),
            schema_name="description_rewrite_answer",
        )
    else:
        raw, stop_reason = api.call_claude(
            user,
            system_prompt=system or "",
            model=model,
            max_tokens=_MAX_TOKENS,
            stage=STAGE_TAG,
            item_id=item_id,
            return_stop_reason=True,
            json_schema=copy.deepcopy(prompts.answer_schema),
        )
    if stop_reason != "end_turn":
        raise RewriteAnswerError(f"truncated answer (stop_reason={stop_reason})")
    return parse_rewrite_answer(raw, check_names(prompts))


# ---------------------------------------------------------------- stage

def load_results(output_dir: Path) -> dict[str, dict]:
    """{prompt_id: stored result}, de-duplicated."""
    return {r["prompt_id"]: r for r in utils.load_jsonl(output_dir / RESULTS_FILENAME)}


def apply_results(plans: list[dict], results: dict[str, dict]) -> list[dict]:
    """Plans with the stored results applied: a dealt-value conflict removes the
    plan; a rewrite swaps in the new description and keeps ``original_description``
    and ``changes``; everything else passes through untouched (same object)."""
    applied = []
    for plan in plans:
        result = results.get(plan["prompt_id"])
        if result is None or not plan.get("description"):
            applied.append(plan)
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


def load_rewritten_plans(config: dict, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans with saved rewrite results applied, no calls. Used when a run resumes
    past layer 1 and plans are re-read from disk."""
    if not enabled(config):
        return plans
    return apply_results(plans, load_results(output_dir))


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Check every described plan and return the plans after the rewrite (see the
    module docstring). A disabled stage, or a run whose prompt snapshot has no
    rewrite template, makes no calls and returns ``plans`` unchanged."""
    sdf = config["sdf"]
    if not enabled(config):
        print("  Description rewrite disabled (sdf.description_rewrite_enabled: false)")
        return plans
    prompts = load_rewrite_prompts(prompts_dir)
    if prompts is None:
        print(f"  No {TEMPLATE_FILENAME} in this run's prompt snapshot — description rewrite skipped")
        return plans

    model = sdf.get("description_rewrite_model") or sdf.get("plan_model")
    reasoning_effort = sdf.get("description_rewrite_reasoning_effort")
    render_variable_lines = cp.make_variable_line_renderer(
        prompts_dir, sdf.get("entity_pool_seed", cp.DEFAULT_ENTITY_SEED))
    checkpoint = utils.Checkpoint(output_dir / CHECKPOINT_FILENAME)
    results_path = output_dir / RESULTS_FILENAME
    rejections_path = output_dir / REJECTIONS_FILENAME

    described = [p for p in plans if p.get("description")]
    existing = load_results(output_dir)
    pending = [p for p in described if p["prompt_id"] not in existing and not checkpoint.is_done(p["prompt_id"])]
    print(f"  Description rewrite: {len(described)} descriptions, {len(pending)} to check "
          f"({len(described) - len(pending)} already done)")

    def rewrite_one(plan: dict):
        try:
            started = time.monotonic()
            result = rewrite_description(
                render_variable_lines(plan), plan["description"],
                prompts=prompts, model=model, reasoning_effort=reasoning_effort,
                item_id=plan["prompt_id"])
            return result, time.monotonic() - started, None
        except RewriteAnswerError as e:
            return None, 0.0, f"unusable answer ({e})"
        except Exception as e:  # one failing call must not discard its siblings' work
            return None, 0.0, f"API call failed ({type(e).__name__}: {e})"

    workers = config.get("workers", 1)
    failed = rewritten = conflicts = 0
    started_all = last_progress = time.monotonic()
    for done, (plan, (result, seconds, problem)) in enumerate(
        zip(pending, utils.parallel_map(rewrite_one, pending, workers)), 1
    ):
        pid = plan["prompt_id"]
        if problem:
            failed += 1
            print(f"  {pid}: {problem} — will retry on resume")
            continue
        utils.append_jsonl({
            "prompt_id": pid,
            "reasoning": result.reasoning,
            "rewrite_needed": result.rewrite_needed,
            "original_description": plan["description"],
            "rewritten_description": result.rewritten_description,
            "changes": list(result.changes),
            "dealt_value_conflict": result.dealt_value_conflict,
            "conflict_reason": result.conflict_reason,
        }, results_path)
        if result.dealt_value_conflict:
            conflicts += 1
            utils.append_jsonl({
                "prompt_id": pid, "variables": plan["variables"],
                "conflict_reason": result.conflict_reason, "description": plan["description"],
            }, rejections_path)
        rewritten += result.rewrite_needed
        checkpoint.mark_done(pid)
        if time.monotonic() - last_progress >= PROGRESS_INTERVAL_SECONDS or done == len(pending):
            last_progress = time.monotonic()
            print(f"  Description rewrite {done}/{len(pending)}: {rewritten} rewritten, "
                  f"{conflicts} dropped for a dealt-value conflict ({last_progress - started_all:.1f}s)")

    if pending and failed == len(pending):
        raise SystemExit(
            "layer1_description_rewrite: every pending call failed — this is systemic "
            "(auth, backend, or network), not per-plan; fix and --resume."
        )
    results = load_results(output_dir)
    kept = apply_results(plans, results)
    print(f"  Description rewrite: {sum(r['rewrite_needed'] for r in results.values())} of "
          f"{len(described)} descriptions rewritten, "
          f"{sum(r['dealt_value_conflict'] for r in results.values())} dropped for a dealt-value "
          f"conflict ({time.monotonic() - started_all:.1f}s this session)")
    return kept


# ---------------------------------------------------------------- reporting

def load_description_rewrite_outputs(run_dir: Path) -> dict | None:
    """The stage's stored results for a run, or None when it never ran."""
    stage_dir = utils.sdf_stage_dir(run_dir, 1)
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
