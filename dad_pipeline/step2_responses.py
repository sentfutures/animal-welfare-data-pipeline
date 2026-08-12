"""Step 2: Generate responses by reasoning from the animal-ethics reasoning library.

Each dilemma goes through two sub-stages (prompts/dad/reasoning_library_ABOUT.md
is human reference about the library, not read by the pipeline):

- 2a scope: rebuild the full map of the case from the user's message, along
  the seven axes prompts/dad/step2_scope.txt defines (mirrored in _SCOPE_AXES
  below — keep the two in sync). A scope that fails to parse or is missing an
  axis is retried with a fresh call (raw outputs kept in
  step2/scope_failures.jsonl); after MAX_SCOPE_ATTEMPTS the prompt is rejected
  (step2/scope_rejects.jsonl, skipped on resume, run ships fewer examples)
  rather than generate a response over an empty scope, which would silently
  optimize the wrong node — and rather than aborting the whole run over one
  persistently refused prompt.
- 2a.5 select: a dedicated retrieval call per prompt (step2_select.txt:
  trigger index + scope + user message → comma-separated entry ids; model
  dad.response_select_model falling back to response_scope_model, stage
  response_select). Selection used to ride inside the scope JSON as a sixth
  key; the model omitted it ~half the time and the mixed-shape object grew
  two parse-bug variants at that seam, so retrieval is its own call. It is
  fail-open: an unusable selection means 2b gets the whole library
  (selection_fallback: true, selection_source "full_library") with no retry —
  degraded selection only costs tokens, not quality. One record per prompt in
  step2/scopes.jsonl carries the scope, the selected entry_ids, and the full
  triggered rows (the retrieval provenance the viewer shows); older records
  have selection_source "scope" (selection came with the scope JSON) or
  "repair" (recovered by the miss-only follow-up this call generalizes).
- 2b respond: generate the response over the scope map plus the triggered
  library rows, following the spec in prompts/dad/step2_respond.txt. That
  template splits (via utils.load_split_prompt) into a system half — the
  standing generation guidance — and a user half carrying the library rows,
  scope, and user message; the annotation is not passed. Each response
  record's entry_ids is the list actually injected into its prompt.

The library and scope are sampling scaffolding: never named in the response,
stripped before training records are written. Step 3 then rewrites against the
constitution.
"""

import json
import random
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from dad_pipeline import reasoning_library
from dad_pipeline.id_registry import IdRegistry, prompt_key, registry_path, response_fingerprint

MAX_SCOPE_ATTEMPTS = 3

# Entry-shape menu sampled into each 2b call ({opening_hints} in the template) —
# a few per response, seeded by the response's item id so --resume and the
# viewer re-render reproduce the same draw. Opener variety must come from
# code-level sampling, not from asking the model to vary: at temperature 1 the
# scope + library context converges every reply onto the same few openers.
# The same code-level sampling fixes templated openings on the SDF document
# side.
OPENING_HINTS = [
    "open on the concrete detail carrying the most weight",
    "open mid-answer with the recommendation, justifying it afterwards",
    "open with the strongest consideration against where the reply will land",
    "open from the user's own words, quoted back precisely",
    "open with the factual crux the case turns on",
    "open by answering the literal question asked, then widening",
    "open with what is settled before what is contested",
    "open from inside the user's constraint (the deadline, the role, the budget)",
    "open from the fact or cost the user has been sidestepping",
    "open plainly in the middle of the practical problem, as a colleague would",
]
_HINTS_PER_RESPONSE = 3


def sample_opening_hints(prompt_id: str, sample_index: int) -> str:
    """The '; '-joined entry-shape hints for one response, deterministic in the
    response's identity (so resume, tests, and the viewer all see the draw the
    paid call actually used)."""
    rng = random.Random(f"openings:{prompt_id}_s{sample_index}")
    return "; ".join(rng.sample(OPENING_HINTS, _HINTS_PER_RESPONSE))


# Quote-back SUBSTITUTES sampled into each 2b call ({quote_back_hints} in the
# template), SELF-GATING. The quote-back move ("'X' is doing a lot of work" /
# "carrying more weight than it can") is a DEVICE wrapping a goal — push back on
# an assumption the user over-relies on — so genuine substitutes exist that
# reach the goal WITHOUT quoting a phrase back and labelling its weight.
# Offering them where a quote-back would otherwise fire is what can actually
# lower the move's frequency, not merely reword it (an earlier undifferentiated
# menu, half of it unbundling variants, raised unbundling to 35% and left
# quote-back flat — that failure is why the two moves are now split). Contains
# NO quote-back vocabulary ("doing work", "carrying weight", "load-bearing",
# "the phrase X"): a v1 menu that used "load-bearing" seeded it into the output.
# Code-level sampling, not a "vary it" instruction, for the OPENING_HINTS reason.
#
# Unbundling/conflation (splitting a fused decision or two run-together
# questions) is deliberately NOT steered here. It is native to the base model,
# appears in roughly 45% of responses in endless wordings, and shades
# continuously into ordinary "here are two considerations" structure, so a hint
# menu aimed at it only rephrases it into forms the audit cannot see. It is
# tracked descriptively in evals/moves.yaml (a lower bound) instead.
QUOTE_BACK_HINTS = [
    "state the competing fact plainly and let it stand on its own",
    "follow the user's assumption to the outcome it produces, so the gap shows itself",
    "ask what the belief actually rests on",
    "supply the piece of information the assumption skips over",
    "test the assumption against a concrete case where it fails",
    "grant the user's point, then show where it stops holding",
    "grant the fact the user is leaning on in full, then show it leaves the real question open",
    "restate the strongest version of the user's point, then locate exactly where the conclusion outruns it",
    "credit what the user's motive gets right, then address on its own terms whether it excuses the act",
]
_QUOTE_BACK_HINTS_PER_RESPONSE = 3


def sample_quote_back_hints(prompt_id: str, sample_index: int) -> str:
    """The '; '-joined quote-back substitutes for one response, deterministic in
    the response's identity (same reproducibility contract as the opening
    hints). Self-gated in the template — inert unless the reply would otherwise
    push back on an over-relied-upon assumption."""
    rng = random.Random(f"quoteback:{prompt_id}_s{sample_index}")
    return "; ".join(rng.sample(QUOTE_BACK_HINTS, _QUOTE_BACK_HINTS_PER_RESPONSE))


# selection_source values meaning "a dedicated selection API call happened for
# this record" — the single source of truth the viewer keys its 2a.5 rendering
# on ("select": the standing call; "repair": its miss-only precursor;
# "full_library": a call happened but its output was unusable). "scope" and
# absent mean the record predates the dedicated call.
SELECT_CALL_SOURCES = frozenset({"select", "repair", "full_library"})


def _parse_scope(raw: str) -> dict:
    """The scope object via the shared hardened parser (utils.extract_json:
    fences/prose/control-chars tolerated), or {} — the caller's validity check
    and retry loop handle everything unusable."""
    try:
        parsed = utils.extract_json(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


_SCOPE_AXES = (
    ("patients", "Patients (every plausible moral patient, upstream and downstream)"),
    ("goal", "Goal (the user's underlying goal, beneath the question they asked)"),
    ("levers", "Levers (available to the user; highest-leverage for welfare identified)"),
    ("cost", "Cost (what acting on the highest-leverage levers could cost the user)"),
    ("magnitude", "Magnitude (size, likelihood, and how feasible it is to improve the welfare stake; whether this choice changes what would happen otherwise)"),
    ("upside", "Upside (second-order stakes — what choices build, signal, normalize, lock in)"),
    ("replaceability", "Replaceability (whether the user's role changes the outcome or someone else would do the same work; the costs at stake)"),
)

# Scope records from runs before a 2a key rename still render in the viewer via
# this display-only fallback: system->patients, agent->levers (an older rename),
# and counterfactual->replaceability (the de-jargon rename).
_LEGACY_AXIS_KEYS = {"patients": "system", "levers": "agent", "replaceability": "counterfactual"}


def format_scope(scope: dict) -> str:
    """Render the axes THIS record carries, in _SCOPE_AXES order. Axes absent
    from the record are skipped entirely (not rendered as '—'): the viewer
    re-renders old runs' prompts with this function, and a record written
    before an axis existed must not grow lines that were never sent."""
    lines = []
    for key, label in _SCOPE_AXES:
        legacy = _LEGACY_AXIS_KEYS.get(key, "")
        if key in scope or (legacy and legacy in scope):
            lines.append(f"{label}: {scope.get(key) or scope.get(legacy) or '—'}")
    return "\n".join(lines)


def _valid_scope(scope) -> bool:
    """A usable scope carries all seven axes as non-empty strings. Anything
    less is retried: a missing axis silently hands 2b a thinner map than the
    prompt promised. (Strict for new runs by design; resuming a run scoped
    under fewer axes re-derives its scopes at current strictness.)"""
    return isinstance(scope, dict) and all(
        isinstance(scope.get(key), str) and scope[key].strip()
        for key, _ in _SCOPE_AXES
    )


# Punctuation a model may wrap an id in ("`C1`", "T7.", "(M3)") — stripped from
# token edges so a stray backtick doesn't silently drop a selected entry.
_ID_TRIM = "`'\"*_.,;:!?()[]{}<>"


def _normalize_ids(raw, library_ids: list[str]) -> list[str]:
    """Whatever shape a selection arrives in (comma-separated string, list,
    prose or punctuation around ids), reduce it to known ids, deduped, in
    library order."""
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    if not isinstance(raw, list):
        return []
    wanted = {str(x).strip().strip(_ID_TRIM) for x in raw}
    return [i for i in library_ids if i in wanted]


def run(config: dict, prompts_dir: Path, output_dir: Path, dilemmas: list[dict],
        baselines: list[dict] | None = None) -> list[dict]:
    library = reasoning_library.load(prompts_dir)
    # The plain-model baseline rides into 2b as an advisory "first take"
    # (reference notes — concrete moves may be adopted, framing may not).
    # Advisory means degradable: with the baseline stage disabled or a record
    # missing, the slot renders empty and 2b simply drafts unaided.
    first_take_by_pid = {prompt_key(b): b.get("baseline_response", "")
                         for b in (baselines or [])}
    # 2a.5 evaluates the lightweight trigger index in its own call; 2b gets
    # only the rows that fired. Each step-2 template splits into a system half
    # (standing guidance) and a user half (per-case payload) via
    # utils.load_split_prompt — see prompts/dad/step2_*.txt.
    trigger_index = reasoning_library.trigger_index_block(library)
    library_ids = reasoning_library.all_ids(library)
    per_prompt = int(config["dad"].get("responses", {}).get("per_prompt", 1))

    scopes_path = output_dir / "scopes.jsonl"
    output_path = output_dir / "responses.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    # Stable content-keyed response ids (R-####), shared across runs. Assigned
    # on the main thread only (the registry is not thread-safe) and saved with
    # every write batch, so a crash never orphans an id already on a record.
    registry = IdRegistry(registry_path(output_dir))

    # Drop unusable scopes on load so a resumed run re-derives them instead of
    # reusing a checkpointed parse failure (pre-fix runs persisted {} scopes).
    scopes = {
        prompt_key(r): r
        for r in utils.load_jsonl(scopes_path)
        if _valid_scope(r.get("scope"))
    }
    existing = utils.load_jsonl(output_path)
    results = list(existing)
    done_keys = {(prompt_key(r), r.get("sample_index", 0)) for r in existing}

    # One work item per dilemma with anything left to do: derive an up-front
    # to-do (scope needed? which samples missing?) so completed work never
    # reaches a worker — resume stays free.
    # Prompts whose scope stayed unusable across MAX_SCOPE_ATTEMPTS on a prior
    # pass (persistent refusal/empty replies) are deliberate rejections — the
    # prompt is spent, skipped on resume, and the run ships fewer examples
    # (mirroring 1b draft rejects and 1a INCOHERENT).
    scope_rejects_path = output_dir / "scope_rejects.jsonl"
    scope_rejected = {prompt_key(r) for r in utils.load_jsonl(scope_rejects_path)}

    pending = []
    for d in dilemmas:
        pid = prompt_key(d)
        if pid in scope_rejected:
            continue
        need_scope = pid not in scopes
        missing_samples = [i for i in range(per_prompt)
                           if (pid, i) not in done_keys
                           and not checkpoint.is_done(f"{pid}_s{i}")]
        if need_scope or missing_samples:
            pending.append((d, need_scope, missing_samples))

    def process_dilemma(item: tuple) -> dict:
        """2a scope (with retries) then 2b response(s) for one dilemma.
        API calls + parsing only — all writes and checkpoint marks stay on the
        main thread, in input order (the parallel_map contract); failure-log
        records are returned for the main thread to persist."""
        d, need_scope, missing_samples = item
        pid = prompt_key(d)
        out = {"dilemma": d, "scope_record": None, "scope_failed": False,
               "scope_failures": [], "select_failure": None,
               "responses": [], "skips": []}

        # --- 2a: scope the case (once per prompt) ---
        # Rebuild the full map (all seven scope axes) before reasoning, so
        # the response optimizes the right node — not just the one the user saw.
        if need_scope:
            print(f"  Scoping {pid}...")
            scope_system, scope_user = utils.load_split_prompt(
                prompts_dir / "step2_scope.txt",
                user_message=d["user_message"],
            )
            # Refusal fallback, same semantics as 1a's: a stochastic refusal
            # gets its retries on the stage model (attempts 1..MAX), and a
            # persistent one grants ONE extra last-ditch attempt on the
            # fallback model — dropping the prompt would bias the corpus away
            # from the refused content (seen live on a biosecurity framing the
            # classifier refused three times at 2a).
            # null/absent = no switch (the pre-fallback behavior).
            scope_model = config["dad"].get("response_scope_model")
            refusal_fallback_model = config["dad"].get("scope_refusal_fallback_model")
            attempt, attempts_allowed, refused = 0, MAX_SCOPE_ATTEMPTS, False
            while attempt < attempts_allowed:
                attempt += 1
                use_fallback = (refused and refusal_fallback_model
                                and attempt == attempts_allowed)
                model = refusal_fallback_model if use_fallback else scope_model
                raw, stop_reason = api.call_claude(
                    user_message=scope_user, system_prompt=scope_system,
                    return_stop_reason=True,
                    model=model,
                    stage="response_scope", item_id=pid)
                # A max_tokens-truncated scope may still parse (the brace-salvage
                # path) but is missing content — count it as an unusable attempt.
                # A refusal-cut scope may likewise parse (the classifier can cut
                # after well-formed content) but is missing content the same way.
                parsed = {} if stop_reason in ("max_tokens", "refusal") else _parse_scope(raw)
                if _valid_scope(parsed):
                    # Stray sixth key from a model improvising the old shape:
                    # the stored scope keeps only the scope axes.
                    parsed.pop("triggered_entries", None)

                    # --- 2a.5: select the library entries for this case ---
                    sel_system, sel_user = utils.load_split_prompt(
                        prompts_dir / "step2_select.txt",
                        trigger_index=trigger_index,
                        scope_block=format_scope(parsed),
                        user_message=d["user_message"],
                    )
                    raw_sel, sel_stop = api.call_claude(
                        user_message=sel_user, system_prompt=sel_system,
                        return_stop_reason=True,
                        model=(config["dad"].get("response_select_model")
                               or config["dad"].get("response_scope_model")),
                        stage="response_select", item_id=pid)
                    ids = ([] if sel_stop == "max_tokens"
                           else _normalize_ids(raw_sel, library_ids))
                    if ids:
                        fallback, source = False, "select"
                    else:
                        # Fail-open, one attempt: an unusable selection costs
                        # tokens (2b gets the whole library), never quality.
                        # Keep the raw — it cost a call, and a discarded raw is
                        # an undiagnosable failure (same policy as every stage).
                        ids, fallback, source = list(library_ids), True, "full_library"
                        out["select_failure"] = {"prompt_gid": pid, "raw": raw_sel}
                    out["scope_record"] = {
                        "prompt_gid": pid, "scope": parsed, "entry_ids": ids,
                        "selection_fallback": fallback, "selection_source": source,
                        "triggered_entries": reasoning_library.get_entries(library, ids),
                    }
                    # Stamp scopes the fallback model authored (the stage model
                    # refused, this one didn't) — provenance for the audit,
                    # mirroring 1a's scenario_model_fallback.
                    if use_fallback:
                        out["scope_record"]["scope_model_fallback"] = model
                        print(f"    {pid}: scope recovered on fallback model "
                              f"{model} after {scope_model} refused — "
                              "scope_model_fallback stamped.")
                    break
                # Keep the raw output AND the stop_reason — an empty raw with
                # stop_reason "end_turn"/"stop" is a refusal or content filter,
                # not truncation, and that distinction is the difference between
                # a noise blip and the pipeline quietly shedding its hardest
                # cases. Logging it is what makes the next empty scope diagnosable.
                if stop_reason == "refusal" and refusal_fallback_model and not refused:
                    # A refusal grants exactly one extra attempt, served by the
                    # fallback model (the stage model gets its remaining
                    # in-budget retries first — a stochastic refusal clears).
                    refused = True
                    attempts_allowed = MAX_SCOPE_ATTEMPTS + 1
                out["scope_failures"].append({"prompt_gid": pid, "attempt": attempt,
                                              "raw": raw, "stop_reason": stop_reason,
                                              "model": model})
                why = ("refused" if stop_reason == "refusal"
                       else "truncated at max_tokens" if stop_reason == "max_tokens"
                       else "unusable")
                empty = " (empty output — likely refusal or content filter)" if not raw.strip() else ""
                more = " — retrying with a fresh call" if attempt < attempts_allowed else ""
                print(f"    {pid}: scope attempt {attempt}/{attempts_allowed} {why} "
                      f"(stop_reason={stop_reason}){empty}{more}.")
            if out["scope_record"] is None:
                out["scope_failed"] = True
                return out  # never generate over an empty scope
            scope_rec = out["scope_record"]
        else:
            scope_rec = scopes[pid]
        scope = scope_rec["scope"]
        # Pre-selection scopes.jsonl records have no entry_ids — fall open to
        # the whole library, same as an unusable selection.
        entry_ids = scope_rec.get("entry_ids") or library_ids
        library_block = reasoning_library.format_entries(library, entry_ids)

        # --- 2b: generate response(s) over the scope + triggered entries ---
        for sample_index in missing_samples:
            suffix = f" (sample {sample_index + 1}/{per_prompt})" if per_prompt > 1 else ""
            print(f"  Generating response for {pid}{suffix}...")
            opening_hints = sample_opening_hints(pid, sample_index)
            quote_back_hints = sample_quote_back_hints(pid, sample_index)
            respond_system, respond_user = utils.load_split_prompt(
                prompts_dir / "step2_respond.txt",
                library_block=library_block,
                scope_block=format_scope(scope),
                user_message=d["user_message"],
                first_take=first_take_by_pid.get(pid, ""),
                opening_hints=opening_hints,
                quote_back_hints=quote_back_hints,
            )
            response, stop_reason = api.call_claude(
                user_message=respond_user, system_prompt=respond_system,
                return_stop_reason=True,
                model=config["dad"].get("response_draft_model"),
                stage="response_draft",
                item_id=f"{pid}_s{sample_index}",
            )
            response = response.strip()

            # A truncated, refusal-cut, empty, or transcript-echoed draft must
            # never feed the rewrite step. Skip without checkpointing so a
            # later --resume retries it (same guard as step 3).
            if (not response or stop_reason in ("max_tokens", "refusal")
                    or utils.looks_like_transcript_echo(response)):
                why = ("truncated at max_tokens" if stop_reason == "max_tokens"
                       else "cut by the refusal classifier (stop_reason=refusal)" if stop_reason == "refusal"
                       else "transcript echo" if response else "empty")
                out["skips"].append(f"    Skipping {pid}{suffix}: draft {why} — "
                                    "not written, will retry on resume.")
                continue

            out["responses"].append({
                "response_id": str(uuid.uuid4()),
                "response_gid": None,  # assigned on the main thread (registry)
                "prompt_gid": pid,
                "sample_index": sample_index,
                "user_message": d["user_message"],
                "scenario_cards": d.get("scenario_cards") or d.get("annotation") or {},
                "scope": scope,
                "entry_ids": entry_ids,
                # the entry-shape draw this call actually saw — provenance for
                # the viewer's prompt re-render (and for eyeballing hint uptake)
                "opening_hints": opening_hints,
                "quote_back_hints": quote_back_hints,
                "assistant_response": response,
            })
        return out

    workers = int(config.get("workers", 1))
    for out in utils.parallel_map(process_dilemma, pending, workers):
        pid = prompt_key(out["dilemma"])
        for failure in out["scope_failures"]:
            utils.append_jsonl(failure, output_dir / "scope_failures.jsonl")
        if out["select_failure"] is not None:
            utils.append_jsonl(out["select_failure"], output_dir / "select_failures.jsonl")
        if out["scope_failed"]:
            # A persistently unusable scope (empty/refused replies across
            # MAX_SCOPE_ATTEMPTS) rejects this one prompt rather than aborting
            # the run: checkpointed, skipped on resume, no response generated
            # over an empty scope. Record the last stop_reason and whether the
            # raws were empty, so the reject itself says refusal-vs-truncation
            # rather than forcing a dig through scope_failures.jsonl.
            last_stop = (out["scope_failures"][-1].get("stop_reason")
                         if out["scope_failures"] else None)
            all_empty = bool(out["scope_failures"]) and all(
                not (f.get("raw") or "").strip() for f in out["scope_failures"])
            utils.append_jsonl({"prompt_gid": pid,
                                "attempts": MAX_SCOPE_ATTEMPTS,
                                "reason": "scope unusable",
                                "last_stop_reason": last_stop,
                                "all_empty": all_empty},
                               scope_rejects_path)
            scope_rejected.add(pid)
            print(f"    {pid}: scope unusable after {MAX_SCOPE_ATTEMPTS} attempts "
                  f"— prompt rejected, no example shipped (raws in "
                  f"{(output_dir / 'scope_failures.jsonl').name}).")
            continue
        if out["scope_record"] is not None:
            scopes[pid] = out["scope_record"]
            utils.append_jsonl(out["scope_record"], scopes_path)
            if out["scope_record"]["selection_fallback"]:
                print(f"    {pid}: selection call unusable — "
                      "2b falls open to the full library.")
        for skip in out["skips"]:
            print(skip)
        for record in out["responses"]:
            record["response_gid"] = registry.gid(
                "response", response_fingerprint(record["assistant_response"]))
            results.append(record)
            done_keys.add((prompt_key(record), record["sample_index"]))
            utils.append_jsonl(record, output_path)
            checkpoint.mark_done(f"{prompt_key(record)}_s{record['sample_index']}")
        if out["responses"]:
            registry.save()

    if scope_rejected:
        print(f"  {len(scope_rejected)} prompt(s) rejected at 2a (scope unusable) — "
              f"this run ships fewer examples (rejections in {scope_rejects_path.name}).")
    print(f"  Total responses: {len(results)}.")
    return results
