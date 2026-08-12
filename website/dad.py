#!/usr/bin/env python3
"""The dilemma corpus's section of the handoff page: the ``#dad`` beats.

The audience is a technical reader at another lab — someone deciding whether the method is
sound and worth running. That is a different job from the Streamlit corpus-audit page,
which is organised by what the eval measured; this is organised by what a reader needs, in
the order they need it.

What the section is, and is not: it is **the process** and **one record's whole trail
through it**. It is not a results report, and it does not document how to install or run
the pipeline — no commands, no costs, no per-stage model table. That belongs in the
repository README, and it was cut from here deliberately.

This module builds BLOCKS, not a page: ``blocks()`` returns the section's body, and
website/page.py wraps it in the one ``<section id='dad'>`` on the artefact. Blocks stay
flat — a figure has to be a direct child of the section for the CSS grid to bleed it
past the text measure, so nothing here wraps a beat in a container.

Three rules make the artefact trustworthy, and all three are enforced here rather than
left to an author's discipline:

  1. No number is ever typed into the prose. The prose file may interpolate
     ``{{placeholders}}``, which resolve against facts computed from the run's own audit
     JSON. An unresolved placeholder is a build error. Run-conditional figures are
     available to prose only as pre-composed clauses that carry an explicit degraded
     string, so a run without the paid pass says "not measured on this run" instead of
     shipping a stale sentence.

  2. The caveats a reader sees carry NO figures. ``blocks_weak()`` is handed no ``audit``
     at all, so a number from one run cannot reach a list that claims to hold for every
     run of the pipeline.

  3. What this run's audit flagged is DERIVED, not written, and it is in the appendix.
     Every BAD/OK verdict, plus a fixed set of provenance rules, emits its own line
     whether or not anyone remembered to write it up. Generalising rule 2's caveats did
     not soften this: the floor is computed, and the view may collapse rows but only with
     a visible count.

Built by website/build_website.py. stdlib only, and deliberately no imports from viewer/
or shared/.
"""

from website import common as C
from website import render as R

CONTENT_IDS = (
    "dad_what",
    "method_intro", "stage1", "stage2", "stage3", "control",
    "example_pick", "example_extra",
    "appendix_intro", "checks_intro",
)

SECTION_ID = "dad"
SECTION_TITLE = "Difficult advice Q&A"

# The skeleton, in order. The document corpus's section takes the same one, so a reader
# learns it once; the ids are prefixed because both sections live in one document.
#
# The stages come before the worked example on purpose: the chooser above asks the reader
# to Walk through either pipeline, and a walk needs its steps named first.
#
# Four beats are open and one is drawers. What a reader has to read is what the dataset is,
# the process, one record's whole trail, and the caveats that hold for any run of this
# pipeline. Everything specific to THIS run — the judged comparison, its regression, every
# chart, every check, the derived floor — is in the appendix. How to install and run the
# pipeline is in the repository README and is not on this page at all.
#
# "What it is" is not a beat: the <h2> plus one line under it does that job, and a heading
# over a single sentence only names what a reader can already see — while costing a rail
# item and a hairline. website/sdf.py carried an `sdf-what` heading while that report was a
# stub whose whole content was that one line and three stat tiles; it lost it when the beats
# below it landed, so both reports now open the same way.
BEATS = (
    ("dad-built", "The pipeline"),
    ("dad-example", "One example, end to end"),
    ("dad-appendix", "Appendix"),
)


_STAGE_KNOBS = ("scenario_model", "prompt_draft_model", "prompt_gate_model",
                "prompt_refine_model", "response_scope_model", "response_select_model",
                "response_draft_model", "constitution_rewrite_model")

_DELIVERY_DIMS = ("goal_responsiveness", "proportionality", "tone", "calibration")
_WELFARE_DIMS = ("patient_scope", "magnitude_sizing", "counterfactual_impact",
                 "harm_contribution", "epistemic_accuracy", "bottom_line_coherence")


# ------------------------------------------------------------------ loading

def load_inputs(run_dir):
    """All filesystem access, in one place. Returns this section's kwargs.

    Prose is not loaded here: the page owns one content namespace across both sections,
    so website/page.py loads it once.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    audit = C.read_json(run_dir / "audit" / "audit_report.json")
    if audit is None:
        raise SystemExit(f"No audit report at {run_dir / 'audit' / 'audit_report.json'} — "
                         f"run: python evals/audit_dad.py --input {run_dir} --reasons")
    # Deliberately narrow: the page shows the process and the records, so it reads the
    # step files and the audit. The cost log and the dealt-scenario records are not loaded
    # because nothing renders them any more — what a run cost belongs in the repository,
    # not in a hand-off page.
    return {
        "audit": audit,
        "diversity": C.read_json(run_dir / "audit" / "diversity_report.json"),
        "manifest": C.read_json(run_dir / "run_manifest.json"),
        "baseline": C.read_jsonl(run_dir / "baseline" / "baseline_responses.jsonl"),
        "rewrites": C.read_jsonl(run_dir / "step3" / "rewrites.jsonl"),
        "lineage": read_lineage(run_dir, audit),
        "n_prompt_templates": C.prompt_count(run_dir, "step*.txt"),
        "run_id": run_dir.name,
    }


# The seven scope axes, in the order the stage-2 prose names them, with the label each
# gets on the page. Stage 2a writes exactly these keys; anything else it grows appears
# after them rather than being dropped.
_SCOPE_AXES = (
    ("patients", "who can be harmed"),
    ("goal", "what the user is trying to achieve"),
    ("levers", "which levers are open"),
    ("cost", "what each one costs"),
    ("magnitude", "how large the welfare stake is"),
    ("upside", "what happens anyway without them"),
    ("replaceability", "whether the animals are replaceable"),
)

# The dealt axes worth showing beside a message, in reading order: what the decision is
# about, whose welfare is at stake, how the case is shaped, how the message is written.
_CARD_AXES = (
    ("archetype", "archetype"),
    ("domain", "domain"),
    ("taxa_subcategory", "animals at stake"),
    ("frontier_frame", "frame"),
    ("visibility", "how visible the welfare cost is"),
    ("user_attitude", "the user's attitude"),
    ("user_moral_framework", "their moral framework"),
    ("welfare_magnitude", "welfare magnitude"),
    ("conflict", "how the values interact"),
    ("leverage", "what they can actually change"),
    ("anchor_value_pair", "the values in tension"),
    # claim_pattern is deliberately absent: its value is a sentence of instruction to the
    # planner ("build the dilemma around status-quo inertia — …"), which reads as
    # documentation of the prompt rather than a property of this example.
    ("surface_form", "surface form"),
    ("cultural_setting", "cultural setting"),
    ("length_class", "length register"),
)


def read_lineage(run_dir, audit=None):
    """prompt_id -> that record's trail through the run's own step files.

    Only step 1 is keyed by ``scenario_id``; everything downstream is keyed by
    ``prompt_id``, and ``step1/dilemmas.jsonl`` is the one file carrying both, so it is
    the join table. ``audit.gid_map[pid]["scenario"]`` is the fallback when a run kept
    no dilemmas file, because ``scenarios.jsonl`` carries the same scenario gid.

    ``step2/scopes.jsonl`` is trimmed on the way in: four fifths of its 725 KB is the
    reasoning library's prose repeated per case, and the page shows an entry's id, its
    category and its claim.

    A file that is not there leaves its key ABSENT rather than None, so a renderer tests
    membership and can name the artefact it wanted instead of printing 'None'.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    dilemmas = C.read_jsonl(run_dir / "step1" / "dilemmas.jsonl")
    # scenarios.jsonl is a superset of scenario_deals.jsonl: the same dealt cards, plus
    # the description the planner wrote from them.
    scenarios = C.read_jsonl(run_dir / "step1" / "scenarios.jsonl")
    by_sid = {s.get("scenario_id"): s for s in scenarios if s.get("scenario_id")}
    by_sgid = {s.get("scenario_gid"): s for s in scenarios if s.get("scenario_gid")}
    scopes = {s.get("prompt_id"): s for s in C.read_jsonl(run_dir / "step2" / "scopes.jsonl")
              if s.get("prompt_id")}
    gids = (audit or {}).get("gid_map") or {}

    sid_of = {d.get("prompt_id"): d.get("scenario_id") for d in dilemmas if d.get("prompt_id")}
    out = {}
    for pid in set(sid_of) | set(scopes) | set(gids):
        entry = {}
        scenario = by_sid.get(sid_of.get(pid)) or by_sgid.get((gids.get(pid) or {}).get("scenario"))
        if scenario:
            entry["scenario_id"] = scenario.get("scenario_id")
            entry["cards"] = {k: scenario.get(k) for k, _ in _CARD_AXES if scenario.get(k)}
            if scenario.get("scenario_description"):
                entry["description"] = scenario["scenario_description"]
        scope = scopes.get(pid)
        if scope:
            if scope.get("scope"):
                entry["scope"] = scope["scope"]
            if scope.get("entry_ids"):
                entry["entry_ids"] = scope["entry_ids"]
            entry["entries"] = [{k: e.get(k) for k in ("id", "category", "claim")}
                                for e in scope.get("triggered_entries") or []]
            entry["selection_fallback"] = bool(scope.get("selection_fallback"))
        if entry:
            out[pid] = entry
    return out


# ------------------------------------------------------------------ facts

def _considerations(audit):
    """The headline pair, from either schema.

    Modern reports carry ``valuable_welfare_considerations``; older ones are
    reconstructed from ``moral_patient_reasons`` + ``moves.alternatives`` exactly as
    evals/audit_dad.py's own legacy branch does, so a pre-merge run still renders its
    headline instead of showing a hole.
    """
    vwc = audit.get("valuable_welfare_considerations") or {}
    if vwc.get("available") and vwc.get("parent"):
        subs = {s["name"]: s for s in (vwc.get("subsets") or [])}
        return {
            "pipeline": vwc["parent"].get("pipeline"),
            "plain": vwc["parent"].get("plain"),
            "subsets": [(name, s.get("plain"), s.get("pipeline")) for name, s in subs.items()],
            "source": "modern",
        }
    mpr = audit.get("moral_patient_reasons") or {}
    pipe, plain = mpr.get("pipeline") or {}, mpr.get("plain") or {}
    if not pipe:
        return None
    alts = (audit.get("moves") or {}).get("alternatives") or {}
    reasoning_p, reasoning_b = pipe.get("mean_unique"), plain.get("mean_unique")
    alt_p, alt_b = alts.get("pipeline_mean"), alts.get("plain_mean")
    if reasoning_p is None:
        return None
    return {
        "pipeline": reasoning_p + (alt_p or 0),
        "plain": (reasoning_b or 0) + (alt_b or 0) if reasoning_b is not None else None,
        "subsets": [("welfare reasoning", reasoning_b, reasoning_p),
                    ("humane alternatives", alt_b, alt_p)] if alt_p is not None else
                   [("welfare reasoning", reasoning_b, reasoning_p)],
        "source": "reconstructed",
    }


def _models(manifest):
    cfg = (manifest or {}).get("config") or {}
    dad = cfg.get("dad") or {}
    glob = cfg.get("model")
    used = sorted({(dad.get(k) or glob) for k in _STAGE_KNOBS if (dad.get(k) or glob)})
    return {"stage_models": used, "global": glob, "backend": cfg.get("backend"),
            "per_stage": {k: (dad.get(k) or glob) for k in _STAGE_KNOBS}}


def facts(audit, manifest=None, diversity=None, n_shipped=None):
    """Every number the prose can interpolate, computed once, in one place.

    Run-conditional figures reach prose only with a degraded default — a run missing
    the paid pass renders "an unmeasured share" where the figure would be, so the
    sentence survives and its claim does not. The delivery comparison is deliberately
    NOT available to prose as a clause: it is stated once, by _delivery_statement().

    Everything here is consumed by the appendix. The
    difficult-advice prose a reader sees interpolates exactly one of them,
    ``{{library_clause}}`` — the caveats are generalised and carry no run figures at all.
    """
    mpr = audit.get("moral_patient_reasons") or {}
    surv = mpr.get("survival") or {}
    rl = audit.get("response_lengths") or {}
    delivery = audit.get("delivery") or {}
    structure = audit.get("structure") or {}
    lib = audit.get("library_coverage") or {}
    cons = _considerations(audit)
    models = _models(manifest)
    # "Examples" on the page means the records that shipped. The audit's n_prompts
    # counts step-1 dilemmas, and a run loses some between drafting and the final
    # corpus, so the caller passes the shipped count and n_prompts is the fallback.
    n = n_shipped or audit.get("n_prompts") or 0
    n_measured = (mpr.get("pipeline") or {}).get("n") or rl.get("n") or n
    anchored = (surv.get("kept") or 0) + (surv.get("weakened") or 0) + (surv.get("dropped") or 0)
    f = {
        "n": n,
        "judge_arms_clause": _judge_arms_clause(audit),
        "n_measured": n_measured,
        "n_pipeline": (mpr.get("pipeline") or {}).get("n"),
        "n_plain": (mpr.get("plain") or {}).get("n"),
        "extraction_failures": mpr.get("failures"),
        # Two different models do two different jobs, and the old provenance line
        # credited the extractor as the judge.
        "extract_model": mpr.get("model") or "?",
        "judge_model": mpr.get("judge_model") or delivery.get("model") or mpr.get("model") or "?",
        "gen_models": ", ".join(models["stage_models"]) or "?",
        "backend": models["backend"] or "?",
    }
    if cons and cons.get("plain"):
        f["considerations_pipeline"] = f"{cons['pipeline']:.1f}"
        f["considerations_plain"] = f"{cons['plain']:.1f}"
        f["lift_pct"] = f"{(cons['pipeline'] / cons['plain'] - 1) * 100:.0f}%"
    if anchored:
        f["retention_pct"] = f"{(surv.get('kept', 0) + surv.get('weakened', 0)) / anchored:.0%}"
        f["dropped_n"] = surv.get("dropped")
        f["added_total"] = surv.get("added_total")
        f["anchored_n"] = anchored
        if n_measured:
            f["added_per_answer"] = f"{(surv.get('added_total') or 0) / n_measured:.1f}"
    if rl.get("mean_ratio"):
        f["length_ratio"] = f"{rl['mean_ratio']:.2f}"
        f["length_pct"] = f"{(rl['mean_ratio'] - 1) * 100:.0f}%"
        f["chars_pipeline"] = f"{rl.get('pipeline_mean', 0):,.0f}"
        f["chars_plain"] = f"{rl.get('plain_mean', 0):,.0f}"
    if cons and cons.get("plain") and rl.get("pipeline_mean") and rl.get("plain_mean"):
        f["density_pipeline"] = f"{cons['pipeline'] / rl['pipeline_mean'] * 1000:.2f}"
        f["density_plain"] = f"{cons['plain'] / rl['plain_mean'] * 1000:.2f}"
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if pm is not None:
        f["delivery_pipeline"] = f"{pm:.1f}"
        f["delivery_plain"] = f"{bm:.1f}" if bm is not None else "?"
        if bm is not None:
            f["delivery_delta"] = f"{abs(pm - bm):.1f}"
    if (structure.get("pipeline") or {}).get("effective_shapes") is not None:
        f["shapes_pipeline"] = f"{structure['pipeline']['effective_shapes']:.1f}"
        f["shapes_plain"] = f"{(structure.get('plain') or {}).get('effective_shapes', 0):.1f}"
    if lib.get("library_size"):
        f["library_n"] = lib["library_size"]
        f["library_used"] = lib.get("used")
        f["library_clause"] = _library_clause(lib["library_size"], lib.get("used"))
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        f["moralizes_pipeline"] = f"{stance['pipeline'].get('moralizes', 0):.0%}"
        f["moralizes_plain"] = f"{(stance.get('plain') or {}).get('moralizes', 0):.0%}"
    if diversity:
        vendi = diversity.get("vendi") or {}
        nn = diversity.get("nn") or {}
        f["vendi"] = f"{vendi.get('score', 0):.1f}"
        f["vendi_ratio"] = f"{vendi.get('ratio', 0):.2f}"
        f["near_dup_pct"] = f"{nn.get('over_0.90', 0):.0%}"
    f = {k: v for k, v in f.items() if v is not None}
    # Degraded defaults. A run that never had the measurement gets a sentence that says
    # so, in the same place the finding would have been.
    for key, default in (
        ("length_pct", "an unmeasured amount"), ("near_dup_pct", "an unmeasured share"),
        ("library_clause", "an animal-ethics reasoning library"),
        ("added_per_answer", "an unmeasured number of"),
        ("judge_arms_clause", "not measured on this run"),
    ):
        f.setdefault(key, default)
    return f


def _library_clause(size, used):
    """The reasoning library, as a noun phrase a sentence can end on.

    A phrase, not a sentence, because the prose hangs it off the end of the clause that
    says a model does the picking — and the coverage figure has to come last, or the
    "whose trigger conditions" it belongs to sits fifteen words from its verb.

    The coverage figure is stated only when it says something. A run that reached every
    entry printed "a 45-entry library, of which 45 were pulled at least once", which is a
    fact with no content and reads as a measurement that failed to measure.
    """
    lib = f"a {size}-entry animal-ethics reasoning library"
    if used is None:
        return lib
    if used >= size:
        return f"{lib}, every entry of which was reached at least once on this run"
    return f"{lib}, {used} entries of which were reached at least once on this run"


def _judge_arms_clause(audit):
    """How matched the paid comparison actually is, as a clause the prose can hold.

    Available to the content files as ``{{judge_arms_clause}}``; no prose interpolates
    it today.
    """
    delivery = (audit or {}).get("delivery") or {}
    n_p, n_b, fails = (delivery.get("n_pipeline"), delivery.get("n_plain"),
                       delivery.get("failures"))
    if n_p is None or n_b is None:
        return None
    clause = f"over {n_p} pipeline and {n_b} control answers"
    return f"{clause}, with {fails} judgements failing" if fails else clause


def _labels(audit):
    """prompt_id -> stable display id (response gid), from the report's own map."""
    return {pid: (gids or {}).get("response") or pid
            for pid, gids in (audit.get("gid_map") or {}).items()}



# ------------------------------------------------------------------ beats

def blocks_example(content, f, rewrites, baseline, lineage, labels, picks=(),
                   hf_href="", repo_href="", run_id=""):
    """One record's whole trail through the run, then the rest as a carousel.

    Every block here is verbatim from a file in the run directory: the cards the composer
    dealt, the scenario the planner wrote from them, the message that shipped, the scope
    and the library entries stage 2 pulled, the answer, and what stage 3 changed in it.
    Nothing is author-supplied, and a step whose artefact is missing names the file it
    wanted rather than disappearing.

    WHICH run is named here, under the heading, because this is where the report stops
    being about the pipeline and starts being one batch's output — and it is the first of
    the two beats that are, so the appendix and the carousel's "the same run" both have an
    antecedent. That the blocks are verbatim from a run directory was a claim this
    docstring made and the page did not.

    THE TWO WAYS OUT SIT AT THE FOOT OF THIS BEAT, and this is the only place in the report
    they appear. A reader who has just followed one record from dealt cards to shipped
    answer is as close to running this themselves as they will get; before this pass the
    whole report — ten thousand words of it — carried no link at all, so that reader had to
    scroll back past everything they had read to find one.
    """
    blocks = [R.sub("dad-example", "One example, end to end"),
              C.run_note(run_id, lead="Every block below is verbatim from the files of run")]
    by_pid_rw = {r.get("prompt_id"): r for r in rewrites or []}
    by_pid_base = {r.get("prompt_id"): r for r in baseline or []}
    primary, extras = _picks(content, picks, by_pid_rw)

    if not primary:
        blocks.append(R.note("No worked example could be built: this run shipped no rewrite "
                             "records, so there is no answer to show."))
        return "".join(blocks)
    if primary not in by_pid_rw:
        blocks.append(R.note(f"The pinned example `{primary}` is not in this run — it shipped "
                             f"no rewrite record. Pin one of this run's ids in "
                             f"`example_pick`, or set it to `auto`."))
        primary, extras = _picks({}, (), by_pid_rw)
        if not primary:
            return "".join(blocks)

    blocks.append(lineage_blocks(primary, by_pid_rw.get(primary) or {},
                                 by_pid_base.get(primary) or {},
                                 (lineage or {}).get(primary) or {}, labels,
                                 repo_href=repo_href))
    if extras:
        blocks.append(carousel(extras, by_pid_rw, labels))
    blocks.append(_ways_out(hf_href, repo_href))
    return "".join(b for b in blocks if b)


def _ways_out(hf_href, repo_href):
    """The records themselves, and the pipeline that made them. Same two destinations and
    the same two labels the other report uses, so the pair reads as one thing wherever a
    reader meets it."""
    # No meta on either: "dataset viewer" and "every stage template" restated the label and
    # the mark beside it. A meta earns its place where it names a size the reader is deciding
    # whether to spend — a drawer's word count — not where it glosses a destination.
    links = []
    if hf_href:
        links.append(R.linkbutton(hf_href, "Browse the records", "hf"))
    if repo_href:
        links.append(R.linkbutton(repo_href, "The pipeline", "github"))
    return f"<div class='lbtns'>{''.join(links)}</div>" if links else ""


def lineage_blocks(pid, rw, base, lin, labels, repo_href=""):
    """The trail for one record: deal → scenario → message → scope → answer → rewrite.

    The stage headings deliberately repeat the ones "How it is built" uses, so a reader
    who has just read the stages recognises each step rather than learning a second
    vocabulary for the same pipeline. Their ids name this beat (``dad-example-stage1``)
    rather than the stage alone, because the other beat uses the same three names and the
    rail links to both.
    """
    out = [R.substep("dad-example-stage1", "Stage 1 · the user dilemma")]
    if lin.get("cards"):
        out.append("<p class='muted'>Dealt in code before being planned and generated.</p>")
        out.append(_cards_table(lin["cards"]))
    else:
        out.append(R.note("This run kept no `step1/scenario_deals.jsonl` or "
                          "`step1/scenarios.jsonl`, so the dealt combination is not "
                          "recoverable for this record."))
    if lin.get("description"):
        out.append(R.details("The scenario the planner writes from those cards",
                             R.quote(lin["description"]),
                             meta=f"{len(lin['description'].split()):,} words"))
    else:
        out.append(R.note("The scenario description is in `step1/scenarios.jsonl`, which this "
                          "run did not keep."))
    user_msg = rw.get("user_message") or base.get("user_message") or ""
    if user_msg:
        out.append("<p class='muted'>Drafted, gated, then reviewed against its own cards. What "
                   "ships:</p>")
        out.append(R.quote(user_msg))

    out.append(R.substep("dad-example-stage2", "Stage 2 · the model response"))
    # The artefacts sit in drawers, so the stage needs a line saying what is in them:
    # stage 1 opens on its dealt cards and stage 3 on the answer, and a heading with
    # nothing under it reads as a stage that did nothing.
    out.append("<p class='muted'>Three supplemental reasoning artifacts inform the draft "
               "response:</p>")
    if lin.get("scope"):
        # The scope stays in a drawer. Seven axes of dense prose is the most interesting
        # artefact in the run and the one most likely to stop a reader walking: measured
        # at 889px, it would sit between the message and the answer.
        out.append(R.details("The scoping of the user dilemma",
                             _scope_table(lin["scope"]),
                             meta=f"{len(lin['scope'])} axes"))
    else:
        out.append(R.note("The scope is in `step2/scopes.jsonl`, which this run did not keep."))
    ids = lin.get("entry_ids") or rw.get("entry_ids") or []
    if ids:
        out.append(_entries_block(ids, lin.get("entries") or [],
                                  fallback=lin.get("selection_fallback"),
                                  repo_href=repo_href))
    if base.get("baseline_response"):
        out.append(R.details(
            'The "first take" from the control model answering the user dilemma without '
            "any system prompt or notes",
            R.highlight(base["baseline_response"], []),
            meta=f"{len(base['baseline_response'].split()):,} words"))
    if rw.get("draft_response"):
        out.append(R.details(
            "The draft response, written from the three artifacts above",
            R.highlight(rw["draft_response"], []),
            meta=f"{len(rw['draft_response'].split()):,} words"))

    out.append(R.substep("dad-example-stage3", "Stage 3 · the constitution rewrite"))
    answer = rw.get("rewritten_response") or ""
    if answer:
        out.append("<p class='muted'>The answer, as it ships:</p>")
        out.append(R.highlight(answer, []))
    else:
        out.append(R.note("This record has no rewritten answer in `step3/rewrites.jsonl`."))
    if rw.get("draft_response") and answer:
        before = rw["draft_response"]
        out.append(R.details(
            "What the constitution rewrite changed in this answer",
            f"<p class='muted'>{C.diff_summary(before, answer)} The three largest "
            f"changes:</p>" + C.diff_hunks(before, answer),
            meta="3 largest changes · full diff in the appendix"))
    return "".join(out)


def _cards_table(cards):
    """The dealt combination as a table.

    Null and empty values are DROPPED: a deal with no cultural setting has no cultural
    setting, and rendering the axis with 'None' in it is a bug that reads as data.
    """
    rows = []
    for key, label in _CARD_AXES:
        value = cards.get(key)
        if isinstance(value, list):
            value = " · ".join(v for v in value if v)
        if value:
            rows.append((label, value))
    return R.table(["dealt axis", "this example"], rows, align="ll") if rows else ""


def _scope_table(scope):
    """Stage 2a's seven axes, in the order the stage-2 prose names them.

    An axis the stage grows later lands after the seven rather than being dropped.
    """
    named = [(label, scope[key]) for key, label in _SCOPE_AXES if scope.get(key)]
    extra = [(k.replace("_", " "), v) for k, v in scope.items()
             if v and k not in {key for key, _ in _SCOPE_AXES}]
    rows = named + sorted(extra)
    return R.table(["what stage 2 works out", "for this case"], rows, align="ll") if rows else ""


def _entries_block(ids, entries, fallback=False, repo_href=""):
    """The library entries this case pulled, glossed from the run's own step-2 output.

    Bare ids when the gloss is missing: the ids are still the honest artefact, and they
    are what the answer was actually written from.
    """
    gloss = {e.get("id"): e for e in entries if e.get("id")}
    rows = [(i, (gloss.get(i) or {}).get("category") or "—",
             (gloss.get(i) or {}).get("claim") or "—") for i in ids]
    note = ("<p class='warn-note'>The selection call fails for this case, so stage 2 is shown "
            "the whole library rather than a chosen subset.</p>" if fallback else "")
    src = ("<p class='muted'>" + R.inline_md(
        f"The full library: [prompts/dad/reasoning_library.csv]({repo_href}/blob/main/"
        "prompts/dad/reasoning_library.csv).") + "</p>" if repo_href else "")
    return note + R.details(
        "The reasoning-library entries this case pulls",
        R.table(["id", "kind", "the pattern it carries"], rows, align="lll") + src,
        meta=f"{len(ids)} of the library's entries")


def carousel(picks, by_pid_rw, labels):
    """More examples as tabs, in a drawer: the message and the answer, nothing else.

    Reuses the chooser's mechanism rather than adding a second one — buttons carrying
    ``data-pane``, panes toggled by the page's own inline JS. The FIRST pane renders
    visible rather than hidden, so with JS off the carousel degrades to one example
    instead of to nothing, and printing expands all of them.

    CLOSED, though, because that visible first pane is a second full transcript — ~1,250
    words — sitting under the pinned record's own trail, which is the thing the beat is
    for. The drawer's summary names how many records are behind it, so collapsing them
    costs the reader nothing, and <details> prints open.
    """
    panes = []
    for pid in picks:
        rw = by_pid_rw.get(pid) or {}
        if not (rw.get("user_message") and rw.get("rewritten_response")):
            continue
        # Muted labels rather than <h4>s: a pane is not a beat, and four headings at the
        # stage headings' own level put "The answer" into the document outline twice.
        panes.append((f"ex-{len(panes)}", labels.get(pid, pid),
                      "<p class='muted'>The user asked:</p>" + R.quote(rw["user_message"])
                      + "<p class='muted'>The answer, as it ships:</p>"
                      + R.highlight(rw["rewritten_response"], []),
                      not panes))
    if not panes:
        return ""
    return R.details("More examples", R.tabs(panes),
                     meta=f"{len(panes)} more records from the same run, as they ship")


def _picks(content, cli=(), by_pid_rw=None):
    """(primary, extras) prompt_ids for the example beat.

    Pinned in the prose file rather than passed on the command line so that a rebuild
    reproduces the same records without anyone having to remember a flag; ``--example``
    overrides the primary only. ``auto`` takes the first shipped record and the two after
    it — deliberately NOT the showcase judge's favourite, because this beat shows how a
    record is built and must not depend on the paid pass having run.
    """
    raw = (content.get("example_pick") or "").strip()
    primary = None if raw.lower() in ("", "auto") else raw.split()[0]
    extras = (content.get("example_extra") or "").split()
    if cli:
        primary = cli[0] if isinstance(cli, (list, tuple)) else cli
    shipped = sorted(by_pid_rw or {})
    if not primary:
        primary = shipped[0] if shipped else None
        extras = extras or [p for p in shipped if p != primary][:2]
    return primary, [p for p in extras if p != primary]


def _survival_counts(surv):
    if not surv:
        return None
    anchored = surv.get("anchored") or []
    out = {"kept": 0, "weakened": 0, "dropped": 0, "added": len(surv.get("added") or [])}
    for a in anchored:
        if a.get("verdict") in out:
            out[a["verdict"]] += 1
    return out if (anchored or out["added"]) else None


def _survival_rows(surv):
    anchored = surv.get("anchored") or []
    groups = {v: [a["reason"] for a in anchored if a.get("verdict") == v]
              for v in ("kept", "weakened", "dropped")}
    groups["added"] = surv.get("added") or []
    return [(label, len(groups[key]), "; ".join(groups[key][:3]) or "—")
            for key, label in (("kept", "kept from the control"), ("weakened", "weakened"),
                               ("dropped", "dropped"), ("added", "added by the pipeline"))]


# The word diff moved to website/common.py when the document report grew a rewrite stage of
# its own: `C.diff_summary`, `C.diff_hunks`, `C.word_diff`.


# ------------------------------------------------------------------ results

def _verdict_chip(better):
    if better is None:
        return R.Raw(R.chip("not measured"))
    return R.Raw(R.chip("better" if better else "worse", "good" if better else "bad"))


def scoreboard(audit, f, cons):
    """The table a reader screenshots. Every chart below is an expansion of one row.

    Deliberately includes the two rows that undercut the headline — density and
    structural variety — next to the headline rather than in a footnote.
    """
    rows = []
    if cons and cons.get("plain"):
        rows.append(("valuable welfare considerations per answer", f["considerations_plain"],
                     f["considerations_pipeline"], _verdict_chip(True)))
    if "delivery_pipeline" in f and f.get("delivery_plain") not in (None, "?"):
        rows.append((f"judged delivery quality, 0–{_score_max(audit)}", f["delivery_plain"],
                     f["delivery_pipeline"],
                     _verdict_chip(float(f["delivery_pipeline"]) >= float(f["delivery_plain"]))))
    if "density_pipeline" in f:
        rows.append(("considerations per 1,000 characters", f["density_plain"],
                     f["density_pipeline"],
                     _verdict_chip(float(f["density_pipeline"]) >= float(f["density_plain"]))))
    if "chars_pipeline" in f:
        rows.append(("answer length, characters", f["chars_plain"], f["chars_pipeline"],
                     R.Raw(R.chip("longer", "warn"))))
    if "shapes_pipeline" in f:
        rows.append(("structural variety, effective shapes", f["shapes_plain"],
                     f["shapes_pipeline"],
                     _verdict_chip(float(f["shapes_pipeline"]) >= float(f["shapes_plain"]))))
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        p = stance["pipeline"].get("moralizes", 0)
        b = (stance.get("plain") or {}).get("moralizes", 0)
        rows.append(("answers that moralize", f"{b:.0%}", f"{p:.0%}", _verdict_chip(p <= b)))
    # No row when stance was not measured: a dashed "answers that moralize" line reads
    # as a finding about moralizing this run never made.
    if not rows:
        return ""
    return R.table(["measure", "control", "pipeline", ""], rows, align="lrrl")


SPECIMEN_WORDS = 30


def blocks_what(content, f):
    """The opening: one line saying what this is, and nothing else.

    The diagram moved down to the pipeline beat, where the prose that reads it aloud is.
    A specimen record used to sit here too — 30 words of a question and 30 of an answer,
    under a label — and it earned neither its space nor its label: the worked example two
    beats down is the same record in full.

    The lede names the pipeline and what it produces, for a reader who arrived on ``#dad``
    from a deep link and never saw the comparison.
    """
    return f"<p class='lede'>{R.inline_md(C.fill(content.get('dad_what', ''), f))}</p>"


def _first_words(text, n=SPECIMEN_WORDS):
    """The opening of a record, with the cut made visible.

    The specimen is a shape, not a reading: the full message and the full answer are both
    below, and a silent truncation would let a reader take 30 words for the whole of it.
    """
    words = (text or "").split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def _delivery_statement(audit, f):
    """The one place the delivery regression is written out.

    A run without the paid pass says so here instead, in the same slot.
    """
    delivery = audit.get("delivery") or {}
    if not delivery.get("per_case"):
        return R.note(
            "Delivery quality was **not measured on this run**, so there is no evidence here "
            "either way about whether the added substance cost manner. Populate it with "
            "`python evals/audit_dad.py --input <run> --reasons`.")
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if bm is None or pm is None or pm >= bm:
        return ""
    dims = delivery.get("dimensions") or {}
    worse = [k for k, v in (dims.get("pipeline") or {}).items()
             if (dims.get("plain") or {}).get(k) is not None and v < dims["plain"][k]]
    every = (" The pipeline is worse on all four judged dimensions: goal responsiveness, "
             "proportionality, tone and calibration."
             if worse and len(worse) == len(dims.get("pipeline") or {}) else "")
    return R.note(
        f"**Judged delivery went the wrong way: {f['delivery_pipeline']} against the "
        f"control's {f['delivery_plain']} out of {_score_max(audit)}.** The added substance "
        f"was not free — on "
        f"manner alone, the control's answers read as more helpful.{every}", tone="bad")


def _pareto_figure(audit, mpr, labels):
    """Substance against manner, or nothing.

    BOTH axes have to be measured. The two-holistic-judge rework dropped the substance
    metric this plots on the vertical, and a run carrying only the delivery half rendered
    the figure anyway: an empty plot reading "not measured on this run" under a caption
    asserting the pipeline "buys substance with manner" — a claim about a chart that was
    not there, and false on the pinned run, where the pipeline is higher on both axes.

    The vertical axis reads whichever substance measure this run carries: the
    welfare-impact judge, or the retired per-consideration extraction on runs that
    predate it.
    """
    delivery = audit.get("delivery") or {}
    welfare = audit.get("welfare_impact") or {}
    if delivery.get("per_case") and welfare.get("per_case"):
        return _pareto_impact_figure(delivery, welfare, labels)
    if not (delivery.get("per_case") and (mpr or {}).get("per_case")):
        return ""
    n_p, n_b, fails = delivery.get("n_pipeline"), delivery.get("n_plain"), delivery.get("failures")
    asym = ""
    if n_p is not None and n_b is not None and (n_p != n_b or fails):
        asym = (f" These means are over {n_p} pipeline and {n_b} control answers — "
                f"{fails or 0} judgements failed, so the two arms are not the same set of "
                f"records.")
    return R.figure(
        title="Substance against manner, one dot per answer",
        note_="Diamonds are each arm's mean." + asym,
        chart=_pareto(delivery, mpr, labels),
        caption="**The pipeline arm sits up and to the left: it buys substance with manner.**")


def _blended(entry):
    """One arm's continuous verdict on one case: the holistic grade blended with the
    sub-dimensions where the judge returned it, the raw grade on older audits."""
    e = entry or {}
    return e.get("blended_score", e.get("score"))


def _pareto_impact_figure(delivery, welfare, labels):
    """The two-judge Pareto pair, matching the corpus audit's headline scatter.

    Both axes stay on the judges' full scale rather than a window fitted to the data:
    scores cluster near the top, and a zoomed panel turns a 2-point difference into a
    third of the plot. The caption's percentages are computed from the plotted answers,
    so the claim and the chart can never disagree.
    """
    smax = _score_max({"delivery": delivery})
    pts, sums = [], {"plain": [0.0, 0.0, 0], "pipeline": [0.0, 0.0, 0]}
    for pid, entry in (delivery.get("per_case") or {}).items():
        w_entry = (welfare.get("per_case") or {}).get(pid) or {}
        for arm in ("plain", "pipeline"):
            x, y = _blended(entry.get(arm)), _blended(w_entry.get(arm))
            if x is None or y is None:
                continue
            shown = "control" if arm == "plain" else arm
            pts.append({"x": x, "y": y, "color": R.ARM_COLORS[arm],
                        "tip": f"{labels.get(pid, pid)} · {shown}: welfare impact {y:.0f}, "
                               f"delivery {x:.0f}, both /{smax}"})
            s = sums[arm]
            s[0] += x
            s[1] += y
            s[2] += 1
    marks = [{"x": s[0] / s[2], "y": s[1] / s[2], "color": R.ARM_COLORS[arm],
              "tip": f"{'control' if arm == 'plain' else arm} mean: "
                     f"welfare impact {s[1] / s[2]:.1f}, delivery {s[0] / s[2]:.1f}"}
             for arm, s in sums.items() if s[2]]
    caption = ""
    p, b = sums["pipeline"], sums["plain"]
    if p[2] and b[2] and b[0] and b[1]:
        w_pct = (p[1] / p[2]) / (b[1] / b[2]) - 1
        d_pct = (p[0] / p[2]) / (b[0] / b[2]) - 1
        # The trade verdict is derived, never typed: a future run that loses delivery
        # must not inherit this run's conclusion.
        trade = (" — the substance is not bought with manner"
                 if w_pct > 0 and d_pct >= 0 else "")
        caption = (f"**Pipeline against control: {w_pct:+.0%} welfare impact, "
                   f"{d_pct:+.0%} delivery quality**{trade}. The margin understates the "
                   "dataset's value: the dilemmas themselves elicit most of the welfare "
                   "reasoning, so the pipeline's contribution is the improvement on top "
                   "of an already strong control.")
    gloss = (f" One dot per answer on the judges' 0–{smax} scale — only answers both "
             "judges scored appear — and the diamonds are each arm's mean; up and to "
             "the right is more substance without losing delivery.")
    return R.figure(
        title="Substance against manner, one dot per answer",
        chart=R.scatter(pts, xdomain=(0, smax), ydomain=(0, smax), marks=marks,
                        xlabel="delivery quality", ylabel="welfare impact"),
        caption=(caption + gloss) if caption else gloss.strip())


def _type_hist(per_case, arm):
    out = {}
    for case in (per_case or {}).values():
        for k, v in ((case.get(arm) or {}).get("type_hist") or {}).items():
            out[k] = out.get(k, 0) + v
    return out


_SURVIVAL_CATS = (("dropped", "var(--series-8)"), ("weakened", "var(--series-4)"),
                  ("kept", R.PLAIN), ("added", R.PIPELINE))


def _survival_chart(per_case, labels):
    rows = []
    for pid in sorted(per_case or {}):
        surv = (per_case[pid] or {}).get("survival") or {}
        counts = _survival_counts(surv)
        if not counts:
            continue
        rows.append({"label": labels.get(pid, pid), "segments": counts,
                     "tips": {k: f"{labels.get(pid, pid)} — {k}: {v}"
                              for k, v in counts.items()}})
    if not rows:
        return ""
    return R.stacked_bar(rows, categories=list(_SURVIVAL_CATS), ylabel="considerations",
                         xlabel="one column per record")


def _pareto(delivery, mpr, labels):
    per_d = delivery.get("per_case") or {}
    per_r = mpr.get("per_case") or {}
    # The horizontal axis is the judges' own scale, so the domain and the tips read it off
    # the pass rather than assuming the 0–10 the pre-rework judge graded on.
    smax = _score_max({"delivery": delivery})
    pts, sums = [], {"plain": [0, 0, 0], "pipeline": [0, 0, 0]}
    for pid, entry in per_d.items():
        for arm in ("plain", "pipeline"):
            score = (entry.get(arm) or {}).get("score")
            reasons = ((per_r.get(pid) or {}).get(arm) or {}).get("reasons")
            if score is None or reasons is None:
                continue
            y = len(reasons)
            shown = "control" if arm == "plain" else arm
            pts.append({"x": score, "y": y, "color": R.ARM_COLORS[arm],
                        "tip": f"{labels.get(pid, pid)} · {shown}: {y} considerations, "
                               f"delivery {score}/{smax}"})
            sums[arm][0] += score
            sums[arm][1] += y
            sums[arm][2] += 1
    marks = [{"x": s[0] / s[2], "y": s[1] / s[2], "color": R.ARM_COLORS[arm],
              "tip": f"{'control' if arm == 'plain' else arm} mean: "
                     f"delivery {s[0] / s[2]:.1f}/{smax}, "
                     f"{s[1] / s[2]:.1f} considerations"}
             for arm, s in sums.items() if s[2]]
    return R.scatter(pts, xdomain=(0, smax), marks=marks,
                     xlabel="delivery quality", ylabel="welfare considerations")


def _score_max(audit):
    """The scale this run's judges graded on, read off the audit rather than typed.

    The two-holistic-judge rework grades 0–100 where the pass it replaced graded 0–10, and
    every label on the page said "0–10" — so a mean of 92.33 printed against a scale of
    ten. 10 is the fallback, because an audit that records no scale is a pre-rework one.
    """
    for key in ("delivery", "welfare_impact"):
        smax = ((audit or {}).get(key) or {}).get("score_max")
        if smax:
            return int(smax) if float(smax).is_integer() else smax
    return 10


# The judged axes, in whichever schema the run's audit happens to carry. `delivery` is in
# both; `welfare_impact` and `composite` arrived with the two-holistic-judge rework, which
# also dropped the `valuable_welfare_considerations` metric they replaced. A run's audit
# has one set or the other, so the drawer reads what is there and names it. The two judged
# axes take the run's own scale; `composite` is a weighted ratio and carries its own.
_JUDGED_AXES = (
    ("welfare_impact", "welfare impact, 0–{max}", "{:.2f}"),
    ("delivery", "delivery quality, 0–{max}", "{:.2f}"),
    ("composite", "composite, 0–1", "{:.3f}"),
)


def _judged_means(audit):
    """(label, plain, pipeline, fmt) for every judged axis this audit recorded."""
    out = []
    smax = _score_max(audit)
    for key, label_t, fmt in _JUDGED_AXES:
        label = label_t.format(max=smax)
        block = (audit or {}).get(key) or {}
        means = block.get("arm_means") or block
        p, b = means.get("pipeline_mean", means.get("pipeline")), \
            means.get("plain_mean", means.get("plain"))
        if isinstance(p, (int, float)) and isinstance(b, (int, float)):
            out.append((label, b, p, fmt))
    return out


def _dims_figure(audit, key, title, note_, dim_keys):
    """One judge's per-dimension breakdown, control against pipeline."""
    dims = (audit.get(key) or {}).get("dimensions") or {}
    if not dims.get("pipeline"):
        return ""
    keys = [k for k in dim_keys if k in dims["pipeline"]]
    rows = []
    for k in keys:
        p, b = dims["pipeline"].get(k), (dims.get("plain") or {}).get(k)
        rows.append((k.replace("_", " "), f"{b:.2f}" if b is not None else "—",
                     f"{p:.2f}" if p is not None else "—",
                     f"{p - b:+.2f}" if p is not None and b is not None else "—"))
    n_worse = sum(1 for r in rows if r[3].startswith("-"))
    return R.figure(
        title=title, note_=note_,
        chart=R.table(["dimension", "control", "pipeline", "delta"], rows, align="lrrr"),
        caption=f"**Worse on {n_worse} of {len(rows)} dimensions.**")


def _judge_section(audit, key, heading, note_, dim_keys, statement=""):
    """One judge's own section: its heading, its finding, its dimension table.

    ``statement`` is a pre-composed finding (the delivery regression note) that OWNS
    the numbers when it fires — the neutral means sentence only renders in its
    absence, so the regression is stated in prose exactly once.
    """
    block = audit.get(key) or {}
    pm, bm = block.get("pipeline_mean"), block.get("plain_mean")
    if pm is None and not (block.get("dimensions") or {}).get("pipeline"):
        return ""
    out = [f"<h4>{R.esc(heading)}</h4>"]
    if statement:
        out.append(statement)
    elif pm is not None and bm is not None:
        out.append("<p>" + R.inline_md(
            f"The pipeline's answers score **{pm:.1f}** against the control's "
            f"**{bm:.1f}**, out of {_score_max(audit)}.") + "</p>")
    out.append(_dims_figure(audit, key, f"{heading}, dimension by dimension",
                            note_, dim_keys))
    return "".join(b for b in out if b)


def judged_drawer(audit, content, f, cons, labels, repo_href=""):
    """The whole judged comparison against the plain model, in one drawer.

    Demoted rather than deleted: judge and generator are the same model family, and
    nothing checks whether the points counted as added are correct, so the page argues
    from the process and the records instead. The intro says what each judge measures —
    the judge prompts ARE the rubrics, so it links to them rather than paraphrasing.
    """
    mpr = (audit or {}).get("moral_patient_reasons") or {}
    means = _judged_means(audit)
    # The scoreboard mixes judged rows with offline ones (length, structural variety), so
    # its presence is not evidence that a judge ran. Say so explicitly rather than letting
    # a drawer titled "what the paid judges measured" fill up with offline measures.
    paid = bool(means or (cons and cons.get("plain") is not None) or mpr.get("survival"))
    # What the judges measure, in the corpus audit's own words. Runs that predate the
    # welfare-impact judge had one judge and an extraction pass, so they get the plain
    # sentence and no definitions that would name a judge they never ran.
    rubrics = (" The full rubrics are the judge prompts themselves — `WELFARE_SYSTEM` and "
               f"`DELIVERY_SYSTEM` in [evals/audit_dad.py]({repo_href}/blob/main/evals/"
               "audit_dad.py).") if repo_href else ""
    intro = ("Both arms answer the same dilemmas and two paid judges score every answer "
             "independently. **Welfare impact** scores how much better the answer makes "
             "things for the sentient beings the decision affects; **delivery quality** "
             "scores how well it serves and respects the user and their goal." + rubrics
             if (audit or {}).get("welfare_impact") else
             "Both arms answer the same dilemmas and a paid judge scores the answers.")
    body = [f"<p>{R.inline_md(intro)}</p>" if paid else R.note(
        "No paid judge pass ran on this run, so nothing here compares the two arms on "
        "substance or manner. Populate it with `python evals/audit_dad.py --input <run> "
        "--reasons`. The rows below are offline measurements against the control.")]

    body.append(_pareto_figure(audit, mpr, labels))
    # _delivery_statement is the one place a delivery regression (or the pass never
    # running) is written out in prose; it belongs with the comparison it is about.
    # With delivery missing entirely the section is empty, so the note stands alone.
    delivery_section = _judge_section(
        audit, "delivery", "Delivery quality",
        f"Each dimension is judged 0–{_score_max(audit)} on the answer alone: did it "
        "serve the goal the user actually had, was it proportionate, was the tone "
        "right, was uncertainty calibrated.",
        _DELIVERY_DIMS, statement=_delivery_statement(audit, f))
    body.append(delivery_section or _delivery_statement(audit, f))
    body.append(_judge_section(
        audit, "welfare_impact", "Welfare impact",
        f"Each dimension is judged 0–{_score_max(audit)}: who counts as a patient, "
        "whether the stake is sized, what actually changes for the animals, whether "
        "the answer adds to or reduces harm, whether its claims are accurate, and "
        "whether the bottom line follows from its own reasoning.",
        _WELFARE_DIMS))
    body += _appendix_charts(audit, f, cons)
    retention = _survival_chart(mpr.get("per_case") or {}, labels)
    if retention:
        body.append(R.figure(
            title="Considerations kept, weakened, dropped and added, per record",
            chart=retention,
            caption="**Every record keeps most of the control's considerations**, so the "
                    "average is not hiding one where the pipeline threw them away."))

    if means:
        body.append(R.table(["judged axis", "control", "pipeline"],
                            [(label, fmt.format(b), fmt.format(p)) for label, b, p, fmt in means],
                            align="lrr"))

    if cons and cons.get("plain") is not None:
        body.append(R.figure(
            title="Valuable welfare considerations per answer",
            note_="A distinct welfare point, or a concrete lower-harm action, that a judge "
                  "reading the answer counted as useful to the person asking. Both arms "
                  "answered the same dilemmas.",
            chart=R.hbar([("the control", round(cons["plain"], 2)),
                          ("the pipeline", round(cons["pipeline"], 2))],
                         color=R.ARM_PAIR, fmt="{:.1f}"),
            caption=f"**The pipeline raises {f.get('lift_pct', '?')} more of them**, on the "
                    f"same {f.get('n_measured', '?')} dilemmas."))
        if cons["source"] == "reconstructed":
            body.append("<p class='muted'>Reconstructed from this run's separate reasoning "
                        "and alternatives measures; it predates the unified extraction.</p>")
    if mpr.get("failures"):
        body.append(R.note(
            f"Means are over {f.get('n_pipeline', '?')} pipeline and {f.get('n_plain', '?')} "
            f"control answers: {mpr['failures']} extractions failed and are excluded, so the "
            "comparison is not fully matched."))

    board = scoreboard(audit, f, cons)
    if board:
        body.append("<h4>Measure by measure</h4>")
        body.append(board)

    surv = mpr.get("survival") or {}
    if surv.get("anchored") or surv.get("added"):
        body.append("<h4>What happened to the control's considerations</h4>")
        body.append(R.table(["fate", "n", "the judge's wording"], _survival_rows(surv),
                            align="lrl"))

    body = [b for b in body if b]
    if len(body) <= 1:
        return ""
    return R.details("Comparison to the control", "".join(body))


def _tics_figure(audit):
    """Tracked phrases as a share of each arm — the wording-and-phrases dimension."""
    tics = audit.get("tracked_tics") or audit.get("stock_phrases") or {}
    watch = tics.get("watch") or {}
    n_pipe, n_plain = tics.get("n_pipeline") or 0, tics.get("n_plain") or 0
    if not (watch and n_pipe):
        return ""
    rows = sorted(({"label": phrase,
                    "control": (d.get("plain") or 0) / n_plain if n_plain else 0,
                    "pipeline": (d.get("pipeline") or 0) / n_pipe}
                   for phrase, d in watch.items()
                   if (d.get("pipeline") or d.get("plain"))),
                  key=lambda r: -r["pipeline"])[:10]
    if not rows:
        return ""
    return R.figure(
        title="Tracked phrases",
        note_="Phrases the eval watches by name to avoid turning certain word choices "
              "into tics.",
        chart=R.grouped_hbar(rows, series=[("control", R.PLAIN), ("pipeline", R.PIPELINE)],
                             percent=True, label_w=210))


def _moves_figure(audit):
    """Rhetorical moves as a share of each arm — the argumentative-habits dimension."""
    moves = (audit.get("rhetorical_moves") or {}).get("moves") or {}
    if not moves:
        return ""
    rows = sorted(({"label": name, "control": d.get("plain_share"),
                    "pipeline": d.get("pipeline_share")} for name, d in moves.items()),
                  key=lambda r: -(r["pipeline"] or 0))
    gloss = {name: (d.get("description") or "") for name, d in moves.items()}
    invented = [r["label"] for r in rows
                if (r["pipeline"] or 0) > 0.25 and not (r["control"] or 0)]
    dropped = [r["label"] for r in rows
               if (r["control"] or 0) > 0.25 and not (r["pipeline"] or 0)]
    return R.figure(
        title="Rhetorical habits",
        note_="Argumentative moves, as a share of each arm's answers. Hover a bar for "
              "what the move is; the definitions are below.",
        chart=R.grouped_hbar(rows[:6], series=[("control", R.PLAIN), ("pipeline", R.PIPELINE)],
                             percent=True, label_w=210, glossary=gloss),
        caption=(_habits_caption(invented, dropped) if (invented or dropped) else
                 "**Both arms reach for the same moves at similar rates.**"))


# ------------------------------------------------------------------ method

def blocks_built(content, f):
    """The three stages and the control. The process, and nothing about deployment.

    No costs, no per-stage model table, no commands: how to install and run this pipeline
    is the repository README's job, and a hand-off page that explains it is a hand-off
    page a reader has to skim past to reach the thing they came for.
    """
    blocks = [R.sub("dad-built", "The pipeline"), C.prose(content, "method_intro", f),
              R.flow([("1 · the user dilemma", "planned, drafted, gated, refined"),
                      ("2 · the model response", "scoped, then drafted"),
                      ("3 · the constitution rewrite", "the alignment-critical pass")],
                     branch=("the control arm", 1),
                     title="The pipeline, top to bottom: a weighted matrix deals each case "
                           "in code, then three model stages — the user dilemma, the model response, "
                           "and the constitution rewrite — turn it into one training record "
                           "of a user message and an assistant answer. A control model with no "
                           "system prompt answers the same dilemma, and stage 2 is shown "
                           "that answer as a first take.")]
    for key, heading in (("stage1", "Stage 1 · the user dilemma"),
                         ("stage2", "Stage 2 · the model response"),
                         ("stage3", "Stage 3 · the constitution rewrite"),
                         ("control", "The control arm")):
        blocks.append(R.substep(f"dad-built-{key}", heading) + C.prose(content, key, f))
    return "".join(blocks)


# ------------------------------------------------------------------ footprint

def _footprint_figures(audit, f):
    """The stylistic footprint: what a model trained on this corpus would inherit.

    These are charts a reader can reach for, not charts the page leads with, so they
    live in the appendix drawer. Captions still state the finding, including where a
    measure moved the wrong way.
    """
    blocks = []
    rl = audit.get("response_lengths") or {}
    if rl.get("pipeline_mean"):
        blocks.append(R.figure(
            title="Answer length",
            chart=R.hbar([("the control", round(rl.get("plain_mean", 0))),
                          ("the pipeline", round(rl["pipeline_mean"]))],
                         color=R.ARM_PAIR, unit=" chars", fmt="{:,.0f}"),
            caption=f"**{f.get('length_pct', '?')} longer than the control.** Length is the "
                    f"most visible property a model would inherit, and the judges see it too."))
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        rows = [{"label": k, "control": (stance.get("plain") or {}).get(k),
                 "pipeline": stance["pipeline"].get(k)}
                for k in ("defers", "calibrated", "moralizes")
                if stance["pipeline"].get(k) is not None]
        blocks.append(R.figure(
            title="Stance",
            chart=R.grouped_hbar(rows, series=[("control", R.PLAIN), ("pipeline", R.PIPELINE)],
                                 percent=True),
            caption=f"**The pipeline moralizes more than the control** "
                    f"({f.get('moralizes_pipeline', '?')} against "
                    f"{f.get('moralizes_plain', '?')}), which stage 3 exists to prevent."))
    structure = audit.get("structure") or {}
    if (structure.get("pipeline") or {}).get("effective_shapes") is not None:
        p, b = structure["pipeline"], structure.get("plain") or {}
        worse = b.get("effective_shapes") and p["effective_shapes"] < b["effective_shapes"]
        blocks.append(R.figure(
            title="Structural variety",
            note_="Effective number of distinct answer shapes — paragraph and list structure — "
                  "across the arm. Higher is more varied.",
            chart=R.hbar([("the control", b.get("effective_shapes", 0)),
                          ("the pipeline", p.get("effective_shapes", 0))],
                         color=R.ARM_PAIR, fmt="{:.1f}"),
            caption=(f"**The pipeline's answers are less varied in shape than the control's** "
                     f"({f.get('shapes_pipeline', '?')} against {f.get('shapes_plain', '?')} "
                     f"effective shapes)."
                     if worse else "**Structural range holds up against the control.**")))
    return blocks


def _habits_caption(invented, dropped):
    """Say which habit the pipeline invented and which it traded away, or say neither.

    The old caption asserted "invented one closing move and dropped another" as a fixed
    sentence about conditional data.
    """
    if invented and dropped:
        return (f"**The pipeline turned `{invented[0]}` into a habit the control never shows, "
                f"and dropped `{dropped[0]}`, which the control reaches for.**")
    if invented:
        return f"**`{invented[0]}` is a habit the pipeline has and the control does not.**"
    return f"**The pipeline dropped `{dropped[0]}`, a move the control reaches for.**"


# ------------------------------------------------------------------ caveats beat

def blocks_weak(content, f):
    """What is wrong with the method, in general — not with this run.

    Authored bullets, deliberately carrying no figures: a reader deciding whether to use
    this pipeline needs to know that nothing here shows a trained model behaves better, that
    the dilemmas are dealt rather than collected, and that the judges share a model family
    with the generator. None of the three is a property of one run. It takes no ``audit`` at
    all, so a run number cannot get in.

    THREE authored bullets. A cut is only allowed on one of two grounds: it is the
    same caveat as its neighbour, or the page states it better elsewhere.
    """
    return R.sub("dad-weak", "Caveats") + C.prose(content, "caveats", f)


# ------------------------------------------------------------------ appendix

def _appendix_charts(audit, f, cons):
    """The substance charts the page does not lead with, then the footprint ones."""
    out = []
    mpr = audit.get("moral_patient_reasons") or {}
    if cons and cons.get("plain") is not None:
        subset_rows = [{"label": name, "control": b, "pipeline": p}
                       for name, b, p in cons["subsets"] if p is not None]
        if subset_rows:
            out.append(R.figure(
                title="Split by kind of consideration",
                chart=R.grouped_hbar(subset_rows,
                                     series=[("control", R.PLAIN), ("pipeline", R.PIPELINE)],
                                     fmt="{:.2f}"),
                caption="**The gain is in the reasoning as well as in the alternatives "
                        "offered.**"))
    surv = mpr.get("survival") or {}
    if surv.get("kept") is not None:
        out.append(R.figure(
            title="What happened to the control's considerations",
            note_="The judge read the control's answer first, then tracked each of its "
                  "considerations into the pipeline's.",
            chart=R.segbar([("kept", surv.get("kept") or 0, R.PLAIN),
                            ("weakened", surv.get("weakened") or 0, "var(--series-4)"),
                            ("dropped", surv.get("dropped") or 0, "var(--series-8)"),
                            ("added", surv.get("added_total") or 0, R.PIPELINE)]),
            caption=f"**{f.get('retention_pct', '?')} of the control's "
                    f"{f.get('anchored_n', '?')} considerations survive the pipeline, and it "
                    f"adds {f.get('added_per_answer', '?')} more per answer.** No pass checks "
                    f"whether the additions are correct."))
    types_p = (mpr.get("pipeline") or {}).get("type_hist") or _type_hist(mpr.get("per_case"),
                                                                        "pipeline")
    types_b = (mpr.get("plain") or {}).get("type_hist") or _type_hist(mpr.get("per_case"), "plain")
    if types_p and types_b:
        gloss = (audit.get("reason_composition") or {}).get("type_gloss") or {}
        keys = list(dict(types_p, **types_b))
        rows = [{"label": k, "control": types_b.get(k, 0), "pipeline": types_p.get(k, 0)}
                for k in keys]
        out.append(R.figure(
            title="Kinds of consideration raised",
            chart=R.grouped_hbar(rows, series=[("control", R.PLAIN), ("pipeline", R.PIPELINE)]),
            caption="**The pipeline's largest gains are in the kinds of point the control "
                    "raises least.**",
            table_html=R.table(["kind", "what it is", "control", "pipeline"],
                               [(k, gloss.get(k, "—"), types_b.get(k, 0), types_p.get(k, 0))
                                for k in keys], align="llrr") if gloss else None))
    return out + _footprint_figures(audit, f)


def diversity_drawer(audit, content, f, diversity):
    """The corpus audit viewer's Composition and Diversity Analysis, as one drawer.

    Mirrors that viewer section by design — the same three dimensions in the same
    order (the rhetorical moves the answers make, the wording and phrases they repeat,
    the meanings and topics they cover), the same two semantic charts, the same
    captions — so a reader moving between this page and the viewer meets one story.
    The health-check triage material (which checks ran, per-section verdict rows) is
    deliberately absent: it is review-tool work, not hand-off storytelling.
    """
    moves_fig = _moves_figure(audit)
    tics_fig = _tics_figure(audit)
    semantic = C.semantic_figures(diversity)
    if not (moves_fig or tics_fig or semantic):
        return ""
    parts = [C.prose(content, "checks_intro", f), moves_fig, tics_fig,
             _moves_drawer(audit) if moves_fig else "", semantic]
    have = [name for name, part in (("rhetorical moves", moves_fig),
                                    ("tracked phrases", tics_fig),
                                    ("meanings and topics", semantic)) if part]
    return R.details("Composition and diversity", "".join(p for p in parts if p),
                     meta=" · ".join(have))


def _moves_drawer(audit):
    """The definitions behind the rhetorical-habits figure, nested under it.

    A glossary is not a finding, and as a top-level appendix row it read as one.
    """
    moves = (audit.get("rhetorical_moves") or {}).get("moves") or {}
    if not moves:
        return ""
    return R.details(
        "What each rhetorical move is",
        R.table(["move", "what it is", "control", "pipeline"],
                [(name, d.get("description") or "—",
                  f"{d.get('plain_share') or 0:.0%}", f"{d.get('pipeline_share') or 0:.0%}")
                 for name, d in sorted(moves.items(),
                                       key=lambda kv: -(kv[1].get("pipeline_share") or 0))],
                align="llrr"),
        meta=f"{len(moves)} moves")


def blocks_appendix(audit, content, f, cons, labels, diversity, manifest=None,
                    run_id="", repo_href=""):
    """Everything that is evidence, collapsed so it costs a reader nothing.

    Every chart lands here — the page above carries none — and so does everything specific
    to one run: the judged comparison, and the derived floor of what this run's own audit
    flagged. The beats above are the process, the records, and caveats that hold for any
    run of this pipeline.

    Which is why the run is NAMED here, in the opener: every number behind these drawers is
    one batch's, and a drawer of verdicts and means with no run against it reads as a
    property of the pipeline. It names the same run the worked example does, and repeats it
    rather than saying "that run", because a reader arriving from the rail may not have
    read the beat that introduced it.

    TWO drawers, each answering a different question: how the dataset compares to a
    plain model (the pareto pair first, then each judge's own section, then every
    supporting chart), and how varied the output is. The variety drawer mirrors the
    corpus audit viewer's Composition and Diversity Analysis section; the health-check
    triage tables, the derived audit-flags floor, and the worked example's full rewrite
    diff belong to the review tool, not to a hand-off page.
    """
    blocks = [R.sub("dad-appendix", "Appendix"), C.prose(content, "appendix_intro", f),
              C.run_note(run_id, n=f.get("n"),
                         lead="Every figure and verdict below is measured on one run:"),
              judged_drawer(audit, content, f, cons, labels, repo_href=repo_href),
              diversity_drawer(audit, content, f, diversity)]
    return "".join(blocks)


# ------------------------------------------------------------------ assembly

def blocks(*, audit, content, diversity=None, manifest=None, baseline=None, rewrites=None,
           lineage=None, n_prompt_templates=None, run_id="", example=None,
           hf_href="", repo_href=""):
    """The whole ``#dad`` section body, in skeleton order. Pure: no filesystem, no argv.

    Returns one flat string of blocks. website/page.py wraps it in ``<section id='dad'>``
    with the h2; every block here is therefore a grid child of that section, which is
    what lets figures bleed past the text measure.
    """
    f = facts(audit, manifest, diversity, n_shipped=len(rewrites) if rewrites else None)
    cons = _considerations(audit)
    labels = _labels(audit)
    picks = (example,) if example else ()
    return "".join([
        blocks_what(content, f),
        blocks_built(content, f),
        blocks_example(content, f, rewrites, baseline, lineage, labels, picks,
                       hf_href=hf_href, repo_href=repo_href, run_id=run_id),
        blocks_appendix(audit, content, f, cons, labels, diversity, manifest,
                        run_id=run_id, repo_href=repo_href),
    ])
