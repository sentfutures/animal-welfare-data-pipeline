"""Layer 1 selector: pick, for each surviving scenario, the considerations it raises.

Runs after the implausibility screen, planning and the description rewrite, and
before drafting. The planner writes a scenario description with no consideration
attached; this stage then reads each (rewritten) description and, from each of two ordered lists (moral and epistemic, from
prompts/sdf/considerations.txt), asks the model for the first consideration
that fits the scenario naturally, or "none". A run-level tracker
(considerations_tracker.py, pure code) counts picks against each entry's target,
orders each list offered to a document by remaining need (target minus picks,
largest first, catalog order breaking ties) and stops offering an entry once its
target is met; the entries that have filled (and that this document could otherwise be
offered and does not already carry) are named in a separate line, with an instruction
to answer none rather than relabel the idea as a neighbouring open entry; later passes re-offer the entries still under target, so a
document can carry more than one consideration.

The model answers through a JSON schema built per call from the entries offered
(``build_selection_schema``): for each list, ``reasoning`` (a brief walk down the
list, logged in model_calls.jsonl only and never passed downstream), ``label``
(an enum of exactly the offered labels plus "none", so a label that was not
offered cannot be produced) and ``where_it_arises`` (one sentence, empty for
none). This module maps labels back to catalog entries and still rejects any
label that was not on the list shown to that call, as a backstop; the tracker
likewise validates a pick against that call's offer, not the current state; a pick
of an entry that reached its target while the call was running is rejected over cap
(never counted, so no entry ends above its target) and that document goes back to the
front of the queue to be asked again for that list with a freshly computed offer. The accepted picks
are assembled in code into ``considerations_spec``, a text block stored on the
plan record next to ``description`` and carried through the draft, rewrite and
final records the same way (empty string when a document has no picks). Each
plan's ``variables`` also gains ``moral_considerations`` and
``epistemic_considerations``: lists of the picked labels, for analysis.

Scheduling is a rolling pool of ``workers`` concurrent calls, not waves: when a
slot frees, the next document's offer is computed from the current counts (under
a lock) and the call is started, so an offer may be slightly out of date by the
time its answer arrives; a pick of an entry that has filled since is rejected over cap and the
document is re-asked in the same pass. Each answer is recorded and appended
to the selections file the moment it completes, under the same lock. Documents
are taken in a seeded shuffle of the sorted ids (so the deal order does not
decide who gets the scarce entries). A row's ``wave`` field is kept for
compatibility and holds the call's start sequence number within its pass.
Documents whose resolution is on the skipped list (the no-welfare-stake arm) get
no considerations.

The selection logic is importable on its own: ``select_for_document`` asks about
one document and returns the parsed choices, using the same template and parser
as the stage, so offline validation scripts can reuse it unchanged.

Config (``sdf`` section): ``select_enabled`` (default true), ``select_model``
(falls back to ``plan_model``, then the global model), ``max_selection_passes``
(default 3), ``select_skip_no_stake`` (default true).

Checkpoint semantics (protects paid work):
- consideration_selections.jsonl gets one row per answered call (every list's
  outcome, including "none", and its pass and start sequence number as ``wave``),
  written as each call completes, and is the source of truth:
  on resume the tracker is rebuilt from it, so paid work is never re-asked.
- A malformed, truncated or unknown-label answer is NOT written, so --resume
  retries exactly that document (and a document that failed is not re-asked
  again in the same session).
- ``over_cap`` rows are written like any other row (they are not counted, and on resume those
  documents are re-asked first), so a rebuild reproduces the state exactly.
- _select_considerations_checkpoint.json marks a pass complete only when none of
  its calls failed, so a resumed run neither skips an unfinished pass nor starts
  another pass the earlier run had already ended.
"""

import json
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline.considerations_catalog import (
    CONSIDERATIONS_FILENAME, EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST,
    Consideration, ConsiderationCatalog, load_considerations,
)
from sdf_pipeline.considerations_tracker import ConsiderationTracker, corpus_coverage
from shared import api, deal_constraints, utils
from shared.matrix import split_sections

STAGE_TAG = "layer1_select"
TEMPLATE_FILENAME = "layer1_select_considerations.txt"
SELECTIONS_FILENAME = "consideration_selections.jsonl"
CHECKPOINT_FILENAME = "_select_considerations_checkpoint.json"
COVERAGE_FILENAME = "consideration_coverage.json"
DEFAULT_MAX_SELECTION_PASSES = 3
PROGRESS_INTERVAL_SECONDS = 180   # progress lines are printed no more often than this
_MAX_TOKENS = 2500  # per-entry reasoning for two lists plus the two short fields; bounds api-backend cost

NO_ENTRIES_MARKER = "(no entries offered)"
NONE_ANSWER = "none"


class SelectionAnswerError(ValueError):
    """The model's selection answer was truncated, malformed, or named a label that was not offered."""


@dataclass(frozen=True)
class DocumentFacts:
    """What the selector is shown about one document (all of it already in the plan record)."""
    document_description: str
    document_type: str
    centrality: str
    sentient_category: str


# ---------------------------------------------------------------- prompt and parsing

def load_selection_template(prompts_dir: Path) -> str | None:
    """The selector template, or None for a run whose snapshot predates the selector."""
    path = Path(prompts_dir) / TEMPLATE_FILENAME
    return path.read_text(encoding="utf-8") if path.is_file() else None


def render_entries(entries: list[Consideration]) -> str:
    return "\n".join(f"- {entry.display}" for entry in entries) if entries else NO_ENTRIES_MARKER


def build_selection_prompt(
    template: str,
    facts: DocumentFacts,
    offered: dict[str, list[Consideration]],
    existing_picks: list[Consideration] = (),
    full_entries: list[Consideration] = (),
) -> tuple[str | None, str]:
    """Render (system, user) for one document. ``offered`` maps each list name to
    the entries shown (an empty or missing list renders as a "no entries" marker);
    ``existing_picks`` are the document's earlier picks, shown only when there are any.
    ``full_entries`` are the entries that have reached their target and would otherwise be
    eligible for this document; their labels are named in a line telling the model not to
    relabel them, shown only when there are any."""
    if not existing_picks:
        template = deal_constraints.drop_template_lines(template, ["existing_picks"])
    if not full_entries:
        template = deal_constraints.drop_template_lines(template, ["full_entries"])
    rendered = template.format(
        document_type=facts.document_type,
        centrality=facts.centrality,
        sentient_category=facts.sentient_category,
        document_description=facts.document_description.strip(),
        moral_list=render_entries(offered.get(MORAL_LIST, [])),
        epistemic_list=render_entries(offered.get(EPISTEMIC_LIST, [])),
        existing_picks="; ".join(pick.label for pick in existing_picks),
        full_entries="; ".join(entry.label for entry in full_entries),
    )
    return split_sections(rendered)


def build_selection_schema(offered: dict[str, list[Consideration]]) -> dict:
    """The answer schema for one call. Properties are ordered reasoning, label,
    where_it_arises so the model walks the list before it chooses. Each list's
    ``label`` enum is the exact offered labels plus "none" (just "none" when the
    list has no entries offered)."""
    def list_schema(list_name: str) -> dict:
        labels = [entry.label for entry in offered.get(list_name) or []]
        return {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "label": {"type": "string", "enum": [*labels, NONE_ANSWER]},
                "where_it_arises": {"type": "string"},
            },
            "required": ["reasoning", "label", "where_it_arises"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {name: list_schema(name) for name in LIST_NAMES},
        "required": list(LIST_NAMES),
        "additionalProperties": False,
    }


def _normalize_label(text: str) -> str:
    return " ".join(text.strip().strip("\"'`").split()).casefold()


def parse_selection_answer(
    raw: str, offered: dict[str, list[Consideration]]
) -> dict[str, tuple[Consideration | None, str]]:
    """{list_name: (entry or None, reason)} for each list that was offered.

    The answer is schema-constrained JSON, so this is a backstop. Raises
    SelectionAnswerError unless the answer is one JSON object with a
    {"label", "where_it_arises"} entry (``reasoning`` is ignored) for every offered list, every label is
    "none" or exactly (up to case and spacing) a label shown in that list, and
    every pick comes with a one-sentence reason. A label from a list that was not
    shown, or one not on the list at all, is rejected."""
    try:
        answer = json.loads(raw or "")
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        raise SelectionAnswerError(f"no usable JSON object ({error})") from error
    if not isinstance(answer, dict):
        raise SelectionAnswerError("answer is not a JSON object")
    choices: dict[str, tuple[Consideration | None, str]] = {}
    for list_name in LIST_NAMES:
        entries = offered.get(list_name) or []
        item = answer.get(list_name)
        if not entries:
            # Nothing was offered here; the model was told to answer none, and anything else is not on the list shown.
            label = item.get("label") if isinstance(item, dict) else None
            if label is not None and _normalize_label(str(label)) not in ("", NONE_ANSWER):
                raise SelectionAnswerError(f"{list_name}: {label!r} is not on the list shown (nothing was offered)")
            continue
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise SelectionAnswerError(f"{list_name}: missing or malformed entry")
        label = _normalize_label(item["label"])
        if label == NONE_ANSWER:
            choices[list_name] = (None, "")
            continue
        by_label = {_normalize_label(entry.label): entry for entry in entries}
        if label not in by_label:
            raise SelectionAnswerError(f"{list_name}: {item['label']!r} is not on the list shown")
        reason = item.get("where_it_arises")
        if not isinstance(reason, str) or not reason.strip():
            raise SelectionAnswerError(f"{list_name}: {item['label']!r} was chosen without saying where it arises")
        choices[list_name] = (by_label[label], " ".join(reason.split()))
    return choices


def select_for_document(
    facts: DocumentFacts,
    offered: dict[str, list[Consideration]],
    existing_picks: list[Consideration] = (),
    full_entries: list[Consideration] = (),
    *,
    template: str,
    model: str | None = None,
    item_id: str | None = None,
) -> dict[str, tuple[Consideration | None, str]]:
    """Ask the model about one document and return its parsed choices.

    Raises SelectionAnswerError for a truncated, malformed, or unknown-label
    answer; API errors propagate. Uses the global temperature and tags the call
    ``layer1_select`` for the cost log."""
    system, user = build_selection_prompt(template, facts, offered, existing_picks, full_entries)
    raw, stop_reason = api.call_claude(
        user,
        system_prompt=system or "",
        model=model,
        max_tokens=_MAX_TOKENS,
        stage=STAGE_TAG,
        item_id=item_id,
        return_stop_reason=True,
        json_schema=build_selection_schema(offered),
    )
    if stop_reason != "end_turn":
        raise SelectionAnswerError(f"truncated answer (stop_reason={stop_reason})")
    return parse_selection_answer(raw, offered)


# ---------------------------------------------------------------- the considerations spec

def assemble_considerations_spec(picks: list) -> str:
    """The text block handed to later prompts: one line per pick, 'Label (gloss) —
    the selector's sentence on where it arises', moral picks first and each list
    in priority order. Empty string when there are no picks."""
    return "\n".join(
        f"{entry.label} ({entry.gloss}) — {reason}" if entry.gloss else f"{entry.label} — {reason}"
        for entry, reason in picks
    )


# ---------------------------------------------------------------- stage

def selection_enabled(config: dict) -> bool:
    return bool(config["sdf"].get("select_enabled", True))


def seeded_document_order(prompt_ids: list[str], seed) -> list[str]:
    """A seeded shuffle of the sorted ids, so plan order does not matter."""
    ordered = sorted(prompt_ids)
    random.Random(f"{seed}:{STAGE_TAG}").shuffle(ordered)
    return ordered


def eligible_plans(config: dict, catalog: ConsiderationCatalog, plans: list[dict]) -> list[dict]:
    """Plans with a description, minus the skipped (no-welfare-stake) resolutions."""
    skip_no_stake = bool(config["sdf"].get("select_skip_no_stake", True))
    return [
        p for p in plans
        if p.get("description") and not (skip_no_stake and catalog.is_skipped_resolution(p["variables"]))
    ]


def _build_tracker(config: dict, catalog: ConsiderationCatalog, plans: list[dict]) -> ConsiderationTracker:
    by_id = {p["prompt_id"]: p for p in eligible_plans(config, catalog, plans)}
    order = seeded_document_order(list(by_id), config["sdf"].get("seed", 0))
    return ConsiderationTracker(
        catalog, [(pid, catalog.is_digital_mind_document(by_id[pid]["variables"])) for pid in order])


def _attach(catalog: ConsiderationCatalog, tracker: ConsiderationTracker, plans: list[dict]) -> list[dict]:
    """Copies of ``plans`` with each described plan's picks attached (plans without a
    description are INCOHERENT and pass through unchanged)."""
    attached = []
    for plan in plans:
        if not plan.get("description"):
            attached.append(plan)
            continue
        picks = tracker.picks_for(plan["prompt_id"])
        entries = sorted(
            ((catalog.entry_by_label(pick.label), pick.reason) for pick in picks),
            key=lambda pair: (LIST_NAMES.index(pair[0].list_name),
                              catalog.entries_of(pair[0].list_name).index(pair[0])),
        )
        variables = {
            **plan["variables"],
            "moral_considerations": [e.label for e, _ in entries if e.list_name == MORAL_LIST],
            "epistemic_considerations": [e.label for e, _ in entries if e.list_name == EPISTEMIC_LIST],
        }
        attached.append({**plan, "variables": variables, "considerations_spec": assemble_considerations_spec(entries)})
    return attached


def attach_selections(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans with the saved selections attached, without any calls. Used when a run
    resumes past layer 1 and plans are re-read from disk. A disabled selector, a
    snapshot without the selector files, or a run with no saved selections returns
    ``plans`` unchanged."""
    if not selection_enabled(config):
        return plans
    if load_selection_template(prompts_dir) is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        return plans
    rows = utils.load_jsonl(Path(output_dir) / SELECTIONS_FILENAME)
    if not rows:
        return plans
    catalog = load_considerations(prompts_dir)
    tracker = _build_tracker(config, catalog, plans)
    tracker.replay(rows)
    return _attach(catalog, tracker, plans)


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Select considerations for the surviving specifications; return ``plans`` with
    each described plan's picks attached. A disabled selector, or a run whose
    prompt snapshot lacks the template or the considerations file, makes no calls
    and returns ``plans`` unchanged."""
    sdf = config["sdf"]
    if not selection_enabled(config):
        print("  Consideration selection disabled (sdf.select_enabled: false)")
        return plans
    template = load_selection_template(prompts_dir)
    if template is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        print(f"  No {TEMPLATE_FILENAME} or {CONSIDERATIONS_FILENAME} in this run's prompt snapshot "
              "— consideration selection skipped")
        return plans

    catalog = load_considerations(prompts_dir)
    model = sdf.get("select_model") or sdf.get("plan_model")
    max_passes = int(sdf.get("max_selection_passes", DEFAULT_MAX_SELECTION_PASSES))
    workers = config.get("workers", 1)
    selections_path = output_dir / SELECTIONS_FILENAME
    checkpoint = utils.Checkpoint(output_dir / CHECKPOINT_FILENAME)

    facts_by_id = {
        p["prompt_id"]: DocumentFacts(
            document_description=p["description"],
            document_type=p["variables"].get("document_type", ""),
            centrality=p["variables"].get("centrality", ""),
            sentient_category=p["variables"].get("sentient_category", ""),
        )
        for p in eligible_plans(config, catalog, plans)
    }
    tracker = _build_tracker(config, catalog, plans)
    tracker.replay(utils.load_jsonl(selections_path))
    print(f"  Selecting considerations for {tracker.eligible_count} specifications "
          f"({tracker.digital_count} on digital minds), up to {max_passes} passes; "
          f"{len(tracker.picks)} picks already saved")

    attempted = failed = 0
    started_all = time.monotonic()
    failed_this_session: set[str] = set()
    lock = threading.Lock()   # guards the tracker, the selections file, the counters and the work queue

    def select_one(call, offered, existing, full, pass_number):
        """The model call only: no shared state is touched, so it runs outside the lock."""
        try:
            started = time.monotonic()
            choices = select_for_document(
                facts_by_id[call.prompt_id], offered, existing, full,
                template=template, model=model, item_id=f"{call.prompt_id}_pass{pass_number}")
            return choices, time.monotonic() - started, None
        except SelectionAnswerError as e:
            return None, 0.0, f"unusable answer ({e})"
        except Exception as e:  # one failing call must not discard its siblings' work
            return None, 0.0, f"API call failed ({type(e).__name__}: {e})"

    def run_pass(pass_number: int) -> int:
        """Ask every document that still needs a call in this pass through a rolling
        pool of ``workers`` threads; return the number of failed calls."""
        nonlocal attempted, failed
        stats = {"failed": 0, "answered": 0, "picks": 0, "slowest": 0.0,
                 "next_start": tracker.next_wave_number(pass_number),
                 "last_progress": time.monotonic()}
        with lock:
            queue = [call.prompt_id for call in tracker.pending_calls(pass_number, frozenset(failed_this_session))]
        pass_started = time.monotonic()

        def take_next():
            """Under the lock: the next document that still needs a call, with the
            offer computed from the counts right now, or None when the queue is done."""
            while queue:
                prompt_id = queue.pop(0)
                if prompt_id in failed_this_session:
                    continue
                call = tracker.pending_call_for(prompt_id, pass_number)
                if call is None:
                    continue    # everything it could be offered has filled since the pass began
                offered = {name: tracker.offered_entries(prompt_id, name) for name in call.list_names}
                existing = [catalog.entry_by_label(label) for label in sorted(tracker.labels_carried_by(prompt_id))]
                full = tracker.full_entries(prompt_id)
                start_number = stats["next_start"]
                stats["next_start"] += 1
                return call, offered, existing, full, start_number
            return None

        def worker_loop():
            nonlocal attempted, failed
            while True:
                with lock:
                    work = take_next()
                if work is None:
                    return
                call, offered, existing, full, start_number = work
                choices, seconds, problem = select_one(call, offered, existing, full, pass_number)
                with lock:
                    attempted += 1
                    if not problem:
                        try:
                            row = tracker.record_answer(
                                call.prompt_id, pass_number, start_number,
                                {name: (entry.label if entry else None, reason)
                                 for name, (entry, reason) in choices.items()},
                                offered_labels={name: {entry.label for entry in entries}
                                                for name, entries in offered.items()})
                            utils.append_jsonl(row, selections_path)
                            if any(o["status"] == "over_cap" for o in row["outcomes"]):
                                queue.insert(0, call.prompt_id)   # re-ask soon, with a fresh offer
                            stats["answered"] += 1
                            stats["slowest"] = max(stats["slowest"], seconds)
                        except ValueError as e:
                            problem = f"answer not accepted ({e})"
                    if problem:
                        failed += 1
                        stats["failed"] += 1
                        failed_this_session.add(call.prompt_id)
                        print(f"  {call.prompt_id} (pass {pass_number}): {problem} — will retry on resume")
                    if time.monotonic() - stats["last_progress"] >= PROGRESS_INTERVAL_SECONDS:
                        stats["last_progress"] = time.monotonic()
                        print(f"  Pass {pass_number} progress: {stats['answered']} calls answered, "
                              f"{stats['failed']} failed, {tracker.accepted_in_pass(pass_number)} picks accepted "
                              f"({time.monotonic() - pass_started:.1f}s elapsed)")

        thread_count = max(1, int(workers))
        threads = [threading.Thread(target=worker_loop, name=f"select-worker-{i}") for i in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(f"  Pass {pass_number}: {stats['answered']} calls answered, {stats['failed']} failed, "
              f"{tracker.accepted_in_pass(pass_number)} picks accepted in total "
              f"({stats['slowest']:.1f}s slowest call, {time.monotonic() - pass_started:.1f}s)")
        return stats["failed"]

    for pass_number in range(1, max_passes + 1):
        pass_key = f"pass_{pass_number}"
        if checkpoint.is_done(pass_key):
            if tracker.accepted_in_pass(pass_number) == 0:
                break  # that pass added nothing, so the earlier run stopped here
            continue
        if tracker.all_targets_met():
            break
        pass_failures = run_pass(pass_number)
        if pass_failures:
            break  # an unfinished pass is retried on resume; do not start the next one
        checkpoint.mark_done(pass_key)
        if tracker.accepted_in_pass(pass_number) == 0:
            break

    if attempted and failed == attempted:
        raise SystemExit(
            "layer1_select: every pending selection call failed — this is systemic "
            "(auth, backend, or network), not per-document; fix and --resume."
        )

    report = tracker.coverage_report()
    utils.ensure_dir(output_dir)
    (output_dir / COVERAGE_FILENAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_coverage_report(report)
    print(f"  Selection took {time.monotonic() - started_all:.1f}s")
    return _attach(catalog, tracker, plans)


# ---------------------------------------------------------------- reporting

def print_coverage_report(report: dict) -> None:
    print(f"  Considerations coverage over {report['eligible_documents']} eligible documents "
          f"({report['digital_mind_documents']} digital-mind):")
    for entry in report["entries"]:
        later = entry.get("from_later_passes")
        suffix = (f", {later} from pass 2+, {entry['rejected_over_cap']} rejected over cap"
                  if later is not None else "")
        print(f"    [{entry['list']}] {entry['label']}: {entry['achieved']}/{entry['target_count']}{suffix}")
    total = report["eligible_documents"]
    print(f"    no consideration: {report['documents_with_no_consideration']}/{total}, "
          f"one list only: {report['documents_with_only_one_list']}/{total}, "
          f"both lists: {report['documents_with_both_lists']}/{total}, "
          f"two or more from one list: {report['documents_with_two_or_more_in_one_list']}")


def load_considerations_outputs(run_dir: Path) -> dict | None:
    """The selector's results for a run, or None when it never ran (older runs).

    Returns {"catalog", "rows", "report"}: the catalog loaded from the run's own
    prompt snapshot, the saved selection rows, and the coverage report written at
    the end of layer 1 (None if the stage never finished)."""
    stage_dir = utils.sdf_stage_dir(run_dir, 1)
    rows = utils.load_jsonl(stage_dir / SELECTIONS_FILENAME)
    snapshot = Path(run_dir) / "inputs" / "prompts"
    if not rows or not (snapshot / CONSIDERATIONS_FILENAME).is_file():
        return None
    report_path = stage_dir / COVERAGE_FILENAME
    return {
        "catalog": load_considerations(snapshot),
        "rows": rows,
        "report": json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None,
    }


def corpus_coverage_report(outputs: dict, corpus: list[dict]) -> dict:
    """The coverage report measured on the final corpus documents."""
    return corpus_coverage(outputs["catalog"], [r["variables"] for r in corpus])
