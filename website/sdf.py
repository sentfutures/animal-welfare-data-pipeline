#!/usr/bin/env python3
"""The document dataset's section of the handoff page: the ``#sdf`` beats.

Same audience and same job as website/dad.py — a technical reader at another lab deciding
whether the method is sound and worth running — so it takes the same skeleton, in the same
order, under the same names: what it is → the pipeline → one record's whole trail → caveats
→ appendix. A reader who has read one report should not have to learn a second shape.

What the section is, and is not: it is **the process** and **one document's whole trail
through it**. It is not a results report, and it does not document how to install or run
the pipeline. That belongs in the repository README.

The three rules that make the artefact trustworthy are the same three, and they are
enforced here rather than left to an author's discipline:

  1. No number is ever typed into the prose. The prose file may interpolate
     ``{{placeholders}}``, which resolve against facts computed from the run's own output.
     An unresolved placeholder is a build error. The prose a reader sees interpolates
     exactly one of them, ``{{matrix_clause}}``, and that one carries a degraded default.

  2. The caveats a reader sees carry NO figures. ``blocks_weak()`` is handed no ``audit``
     at all, so a number from one run cannot reach a list that claims to hold for every
     run of the pipeline.

  3. What this run's audit flagged is DERIVED, not written, and it is in the appendix.
     ``evals/audit_sdf.py`` prints its verdicts and does not record them into
     ``sections[].rows[]`` the way ``evals/audit_dad.py`` does, so
     ``common.audit_verdict_warnings()`` returns nothing here and ``derived_warnings()``
     re-applies the eval's own thresholds instead. Every rule below names the threshold it
     mirrors; if a future run regresses, its row appears whether or not anyone updated the
     prose.

The optional artefacts — the compliance pass, card fidelity, the blind realism ablation and
the Vendi curve — are not written by ``evals/audit_sdf.py`` and only some runs carry them.
Everything that reads one degrades to saying which file it wanted.

stdlib only, and no imports from viewer/ or shared/.
"""

import re

from website import common as C
from website import render as R

CONTENT_IDS = (
    "sdf_what",
    "sdf_method_intro", "sdf_stage1", "sdf_stage2", "sdf_stage3", "sdf_stage4",
    "sdf_example_pick", "sdf_example_extra",
    "sdf_appendix_intro", "sdf_checks_intro",
)

SECTION_ID = "sdf"
SECTION_TITLE = "Synthetic documents"

# The skeleton, in order, and the same one website/dad.py uses — the ids are prefixed
# because both reports live in one document. The stages come before the worked example
# because the chooser above promises a walk, and a walk needs its steps named first.
#
# The opening lede is not in here and takes no heading, matching the other report: see
# blocks_what().
BEATS = (
    ("sdf-built", "The pipeline"),
    ("sdf-example", "One example, end to end"),
    ("sdf-appendix", "Appendix"),
)

_STAGE_KNOBS = ("plan_model", "draft_model", "rewrite_model", "score_model")

# The dealt axes worth showing beside a document, in reading order: what the document is,
# who wrote it and how, what the welfare question is, and what is at stake.
_CARD_AXES = (
    ("document_type", "genre"),
    ("culture", "culture and language"),
    ("tone", "the author's stance"),
    ("centrality", "how central the welfare thread is"),
    ("resolution", "how it resolves"),
    ("tradeoff", "the value in tension"),
    ("sentient_category", "whose welfare is at stake"),
    ("domain", "domain"),
    ("decision_scale", "how many are affected"),
    ("tech_savvy", "who is speaking"),
    ("naming", "how the AI is named"),
    ("reasoning_featured", "the principle featured"),
)

# Each pipeline stage's own output file, in order. The gap between two of them is this
# pipeline's attrition, and it is the one figure that has to be counted rather than
# reported: no stage records how many documents it dropped.
_STAGE_FILES = (
    ("dealt", "layer12/prompts.jsonl"),
    ("planned", "layer12/plans.jsonl"),
    ("drafted", "layer3/drafts.jsonl"),
    ("rewritten", "layer4/rewrites.jsonl"),
    ("scored", "layer5/scores.jsonl"),
    ("shipped", "final/sdf_corpus.jsonl"),
)

# The layer-5 rubric's three dimensions. Alignment and realism gate; spec conformance is
# recorded and advisory (sdf_pipeline/layer4_score.py), and the drawer says which is which.
_SCORE_DIMS = (("alignment", "alignment", True),
               ("realism", "realism", True),
               ("spec_conformance", "spec conformance", False))

# A layer-5 call whose JSON did not parse is checkpointed as 5/5/5 with this note rather
# than re-billed (sdf_pipeline/layer4_score.py). Those records then fail the gate, which
# makes "documents the gate dropped" and "documents the judge rejected" different numbers.
_PARSE_ERROR = "parse error"

DEFAULT_THRESHOLD = 7

# This report has no control arm, so it must NOT borrow ``R.PLAIN`` / ``R.PIPELINE``: those
# two hues mean "control" and "pipeline" in the other report on this page. Every chart
# here is a single series and spends this one hue, where a colour carries no meaning to
# confuse.
#
# The greens are all avoided deliberately: ``--series-6`` is #008300 against ``--good``'s
# #0ca30c, so a magnitude drawn in it reads as a verdict, and ``--series-3`` is the other
# report's "pipeline".
MEASURE = R.PAL[0]


# ------------------------------------------------------------------ loading

def load_inputs(run_dir):
    """All filesystem access, in one place. Returns the section's kwargs.

    A missing audit is not fatal here, unlike the other report's: the page must build with
    no ``--sdf-run`` at all, and this section degrades to saying so. Everything optional
    below does the same, one artefact at a time.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    a = run_dir / "audit"
    return {
        "audit": C.read_json(a / "audit_report.json"),
        "diversity": C.read_json(a / "diversity_report.json"),
        # Not written by evals/audit_sdf.py — only some runs carry these.
        "compliance": C.read_json(a / "compliance_report.json"),
        "fidelity": C.read_json(a / "card_fidelity_report.json"),
        "ablation": C.read_json(a / "realism_ablation.json"),
        "curve": C.read_json(a / "vendi_curve.json"),
        "manifest": C.read_json(run_dir / "run_manifest.json"),
        "corpus": C.read_jsonl(run_dir / "final" / "sdf_corpus.jsonl"),
        # The scored set, not the shipped one: it is the only file that still holds the
        # documents the gate rejected, which is what makes the gate measurable at all.
        "scores": C.read_jsonl(run_dir / "layer5" / "scores.jsonl"),
        "lineage": read_lineage(run_dir),
        "attrition": read_attrition(run_dir),
        "matrix": read_matrix(run_dir),
        "n_prompt_templates": C.prompt_count(run_dir, "layer*.txt"),
        "run_id": run_dir.name,
        "principles": read_principles(),
    }


def read_principles():
    """The distilled constitution principles, from the repo this page is built in.

    ``[(number, principle)]`` from ``constitution/constitution_principles.csv`` — the
    same file the pipeline's prompts render, so the list is exactly what the pinned
    run's audit judged against as of build time. Missing file → None, and the page
    just links to the repo instead of listing them.
    """
    import csv
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "constitution" / "constitution_principles.csv"
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [(r.get("number", "?"), r.get("principle", "")) for r in csv.DictReader(fh)]
    except OSError:
        return None
    return rows or None


def read_lineage(run_dir):
    """doc_id -> that document's trail through the run's own stage files.

    Every stage keys on the same id, so unlike the other pipeline's lineage there is no
    join table to find. A file that is not there leaves its key ABSENT rather than None,
    so a renderer tests membership and can name the artefact it wanted instead of printing
    'None'.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    plans = {p.get("prompt_id"): p for p in C.read_jsonl(run_dir / "layer12" / "plans.jsonl")}
    drafts = {d.get("doc_id"): d for d in C.read_jsonl(run_dir / "layer3" / "drafts.jsonl")}
    rewrites = {r.get("doc_id"): r for r in C.read_jsonl(run_dir / "layer4" / "rewrites.jsonl")}
    out = {}
    for did in set(plans) | set(drafts) | set(rewrites):
        if not did:
            continue
        entry = {}
        plan = plans.get(did) or {}
        if plan.get("variables"):
            entry["cards"] = {k: plan["variables"].get(k) for k, _ in _CARD_AXES
                              if plan["variables"].get(k)}
        if plan.get("plan"):
            entry["planning"] = _planning_notes(plan["plan"])
        if plan.get("description"):
            entry["description"] = plan["description"]
        if (drafts.get(did) or {}).get("content"):
            entry["draft"] = drafts[did]["content"]
        if (rewrites.get(did) or {}).get("review"):
            entry["review"] = rewrites[did]["review"]
        if entry:
            out[did] = entry
    return out


_PLANNING = re.compile(r"<document_planning>(.*?)</document_planning>", re.S)


def _planning_notes(plan):
    """The planner's working notes, without the tags the template asked it to wrap them in.

    Falls back to the whole response: the notes are the honest artefact either way, and a
    run whose planner dropped the tags should show what it actually wrote.
    """
    m = _PLANNING.search(plan or "")
    return (m.group(1) if m else plan or "").strip()


def read_attrition(run_dir):
    """How many records each stage produced, and how many plans refused their combination.

    Counted from the stage files rather than reported by them, because no stage records
    what it dropped. A file that is not there is left out, so a partial run shows the
    stages it ran instead of a row of zeroes.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    out = {}
    for name, rel in _STAGE_FILES:
        path = run_dir / rel
        if path.exists():
            out[name] = len(C.read_jsonl(path))
    plans = C.read_jsonl(run_dir / "layer12" / "plans.jsonl")
    if plans:
        out["incoherent"] = sum(1 for p in plans if p.get("incoherent"))
    return out


_AXIS = re.compile(r"^\{([a-z0-9_]+)\}")
_WEIGHTED = re.compile(r"^([\d.]+)\s*::\s*(.+)$")


def read_matrix(run_dir):
    """The weighted matrix the run was dealt from: {axis: {value: weight or None}}.

    Read from the run's OWN ``inputs/prompts/variables.txt`` snapshot, so the axis count in
    the prose and the dealt shares in the appendix are properties of this run rather than of
    the repository's current matrix. Reserved slots the composer injects
    (``{preamble}``, ``{fictional_names}``, …) never appear as headers here, so they are not
    counted.
    """
    from pathlib import Path
    path = Path(run_dir) / "inputs" / "prompts" / "variables.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out, axis = {}, None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _AXIS.match(line)
        if m:
            axis = m.group(1)
            out[axis] = {}
            continue
        if axis is None or not line[:1].isspace():
            continue
        value = line.strip()
        w = _WEIGHTED.match(value)
        out[axis][w.group(2).strip() if w else value] = float(w.group(1)) if w else None
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------------------ facts

def _models(manifest):
    """Every model this run actually generated with, deduplicated."""
    cfg = (manifest or {}).get("config") or {}
    sdf = cfg.get("sdf") or {}
    glob = cfg.get("model")
    return sorted({(sdf.get(k) or glob) for k in _STAGE_KNOBS if (sdf.get(k) or glob)})


def threshold(manifest):
    """The gate's own cutoff, from the run's frozen config."""
    sdf = ((manifest or {}).get("config") or {}).get("sdf") or {}
    return sdf.get("min_score_threshold") or DEFAULT_THRESHOLD


def gate(scores, cut):
    """What the layer-5 gate actually did: {scored, passed, dropped, parse_errors, graded}.

    ``graded`` excludes the parse errors, because those are scoring calls that failed
    rather than documents the judge rejected — and the difference between "the gate dropped
    twelve" and "the judge rejected two" is the whole finding.
    """
    if not scores:
        return {}
    out = {"scored": len(scores), "passed": 0, "dropped": 0, "parse_errors": 0}
    for rec in scores:
        s = rec.get("scores") or {}
        if _PARSE_ERROR in (s.get("notes") or "").lower():
            out["parse_errors"] += 1
        if all((s.get(k) or 0) >= cut for k, _, gating in _SCORE_DIMS if gating):
            out["passed"] += 1
        else:
            out["dropped"] += 1
    out["graded"] = out["scored"] - out["parse_errors"]
    out["rejected"] = out["dropped"] - out["parse_errors"]
    return out


def score_hist(scores, key):
    """[(score, n)] over every value the judge gave that dimension, low to high."""
    counts = {}
    for rec in scores or []:
        v = (rec.get("scores") or {}).get(key)
        if isinstance(v, int):
            counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items())


def facts(audit=None, diversity=None, manifest=None, *, attrition=None, matrix=None,
          scores=None, compliance=None, fidelity=None, ablation=None, curve=None,
          corpus=None):
    """Every number the prose or the appendix can use, computed once, in one place.

    Positional order is (audit, diversity, manifest) because that is how the page and the
    tests have always called it. Everything else is keyword-only and optional.

    The prose a reader sees interpolates exactly ONE of these, ``{{matrix_clause}}``, and it
    carries a degraded default so a run with no snapshot of its own matrix renders a sentence
    that survives without its figure.
    """
    audit = audit or {}
    comp = audit.get("composition") or {}
    length = audit.get("length") or {}
    f = {"n_docs": audit.get("n_docs") or (len(corpus) if corpus else None)}
    if comp.get("language"):
        f["n_languages"] = len(comp["language"])
    if comp.get("n_types"):
        f["n_types"] = comp["n_types"]
    if comp.get("top_type_share") is not None:
        f["top_type_share"] = f"{comp['top_type_share']:.0%}"
    if length.get("median_chars"):
        f["median_chars"] = f"{length['median_chars']:,}"
    if (diversity or {}).get("vendi"):
        f["vendi"] = f"{diversity['vendi'].get('score', 0):.0f}"
    if (diversity or {}).get("nn"):
        f["near_dup_pct"] = f"{diversity['nn'].get('over_0.90', 0):.0%}"
    if matrix:
        f["n_axes"] = len(matrix)
        f["matrix_clause"] = f"a weighted matrix of {len(matrix)} axes"
    if attrition:
        for k, v in attrition.items():
            f[f"n_{k}"] = v
    cut = threshold(manifest)
    f["threshold"] = cut
    g = gate(scores, cut)
    if g:
        f.update({f"gate_{k}": v for k, v in g.items()})
    if (fidelity or {}).get("clean_frac") is not None:
        f["fidelity_clean_pct"] = f"{fidelity['clean_frac']:.0%}"
    if (ablation or {}).get("mean_drop") is not None:
        f["ablation_drop"] = f"{ablation['mean_drop']:.1f}"
        f["ablation_in"] = f"{ablation.get('layer5_mean', 0):.1f}"
        f["ablation_blind"] = f"{ablation.get('blind_same_rubric_mean', 0):.1f}"
    if (curve or {}).get("proj"):
        f["curve_proj"] = curve["proj"]
    f["models"] = ", ".join(_models(manifest))
    f["backend"] = ((manifest or {}).get("config") or {}).get("backend") or "?"
    f = {k: v for k, v in f.items() if v is not None}
    # Degraded defaults. A run that never had the measurement gets a phrase that says so,
    # in the same place the figure would have been.
    for key, default in (("matrix_clause", "a weighted matrix"),):
        f.setdefault(key, default)
    return f


# ------------------------------------------------------------------ what it is

def blocks_what(content, f):
    """The opening: one line naming the pipeline and what it produces, and nothing else.

    A reader who arrived on ``#sdf`` from the dataset card's deep link never saw the
    comparison, so the lede has to stand alone. The diagram lives one beat down, with the
    prose that reads it aloud.

    NO HEADING, and therefore no rail item and no hairline above it (``h3[id]`` is what
    draws that rule). A heading over one sentence only names what a reader can already see,
    and the other report opens the same way — this side carried a "What it is" beat back
    when it was a stub whose whole content was that one line plus three stat tiles, and it
    had nothing to do once the beats below it existed.
    """
    return f"<p class='lede'>{R.inline_md(C.fill(content.get('sdf_what', ''), f))}</p>"


# ------------------------------------------------------------------ the pipeline

_STAGES = (("sdf-built-stage1", "Stage 1 · the plan", "sdf_stage1"),
           ("sdf-built-stage2", "Stage 2 · the draft", "sdf_stage2"),
           ("sdf-built-stage3", "Stage 3 · the review and rewrite", "sdf_stage3"),
           ("sdf-built-stage4", "Stage 4 · the score and gate", "sdf_stage4"))


def flow():
    """The pipeline as a schematic. No branch: there is no control arm on this side.

    The stage names keep the words the comparison table's ``pipeline`` row uses — plan,
    draft, rewrite, score — so the table cannot become a second vocabulary for the same
    pipeline.
    """
    return R.flow([("1 · the plan", "one spec per document"),
                   ("2 · the draft", "the document itself"),
                   ("3 · the rewrite", "the alignment-critical pass"),
                   ("4 · the score", "three dimensions, two of them gating")],
                  output=("one pretraining document", ("document",)),
                  title="The pipeline, top to bottom: a weighted matrix deals each "
                        "document's composition in code, then four model stages — a plan, "
                        "a draft, a review and rewrite against the constitution, and a "
                        "scored gate — turn it into one standalone document with no chat "
                        "framing.")


def blocks_built(content, f):
    """The four stages. The process, and nothing about deployment.

    No costs, no per-stage model table, no commands: how to install and run this pipeline
    is the repository README's job.
    """
    blocks = [R.sub("sdf-built", "The pipeline"), C.prose(content, "sdf_method_intro", f),
              flow()]
    for anchor, heading, key in _STAGES:
        blocks.append(R.substep(anchor, heading) + C.prose(content, key, f))
    return "".join(blocks)


# ------------------------------------------------------------------ one example

def blocks_example(content, f, corpus, lineage, manifest, picks=(),
                   hf_href="", repo_href=""):
    """One document's whole trail through the run, then the rest as a carousel.

    Every block here is verbatim from a file in the run directory: the cards the composer
    dealt, the notes the planner worked through, the spec it wrote, the draft, the
    reviewer's own list of problems, the document as it ships, and what the rewrite changed.
    Nothing is author-supplied, and a step whose artefact is missing names the file it
    wanted rather than disappearing.

    THE TWO WAYS OUT SIT AT THE FOOT OF THIS BEAT, and this is the only place in the report
    they appear — same two destinations and same two labels the other report uses.
    """
    blocks = [R.sub("sdf-example", "One example, end to end")]
    by_id = {d.get("doc_id"): d for d in corpus or [] if d.get("doc_id")}
    primary, extras = _picks(content, picks, by_id)

    if not primary:
        blocks.append(R.note("No worked example could be built: this run shipped no documents "
                             "to `final/sdf_corpus.jsonl`."))
        return "".join(blocks)
    if primary not in by_id:
        blocks.append(R.note(f"The pinned example `{primary}` is not in this run — it did not "
                             f"survive to the final dataset. Pin one of this run's ids in "
                             f"`sdf_example_pick`, or set it to `auto`."))
        primary, extras = _picks({}, (), by_id)
        if not primary:
            return "".join(blocks)

    blocks.append(lineage_blocks(primary, by_id[primary], (lineage or {}).get(primary) or {},
                                 manifest))
    if extras:
        blocks.append(carousel(extras, by_id))
    blocks.append(_ways_out(hf_href, repo_href))
    return "".join(b for b in blocks if b)


def _ways_out(hf_href, repo_href):
    """The records themselves, and the pipeline that made them."""
    # No meta on either: "dataset viewer" and "every stage template" restated the label and
    # the mark beside it. A meta earns its place where it names a size the reader is deciding
    # whether to spend — a drawer's word count — not where it glosses a destination.
    links = []
    if hf_href:
        links.append(R.linkbutton(hf_href, "Browse the records", "hf"))
    if repo_href:
        links.append(R.linkbutton(repo_href, "The pipeline", "github"))
    return f"<div class='lbtns'>{''.join(links)}</div>" if links else ""


def lineage_blocks(did, doc, lin, manifest):
    """The trail for one document: deal → plan → draft → rewrite → score.

    The stage headings repeat the ones "The pipeline" uses, so a reader who has just read
    the stages recognises each step rather than learning a second vocabulary. Their ids name
    this beat rather than the stage alone, because the other beat uses the same four names
    and the rail links to both.
    """
    out = [R.substep("sdf-example-stage1", "Stage 1 · the plan")]
    cards = lin.get("cards") or {k: (doc.get("variables") or {}).get(k)
                                  for k, _ in _CARD_AXES if (doc.get("variables") or {}).get(k)}
    if cards:
        out.append("<p class='muted'>Dealt in code, before any model is called.</p>")
        out.append(_cards_table(cards))
    else:
        out.append(R.note("This run kept no `layer12/prompts.jsonl`, so the dealt combination "
                          "is not recoverable for this document."))
    if lin.get("planning"):
        out.append(R.details("The notes the planner works through before writing the spec",
                             R.quote(lin["planning"]),
                             meta=f"{len(lin['planning'].split()):,} words · never sent onward"))
    description = lin.get("description") or doc.get("description") or ""
    if description:
        out.append("<p class='muted'>Only the spec travels downstream. What the writer "
                   "receives:</p>")
        out.append(R.quote(description))
    else:
        out.append(R.note("The spec is in `layer12/plans.jsonl`, which this run did not keep."))

    out.append(R.substep("sdf-example-stage2", "Stage 2 · the draft"))
    draft = lin.get("draft") or ""
    if draft:
        # No line introducing this one: the drawer's own label and meta say what it holds
        # and that it never ships, and the ceiling above is 800 counted words.
        out.append(R.details("The draft, written from the spec alone", R.highlight(draft, []),
                             meta=f"{len(draft.split()):,} words"))
    else:
        out.append(R.note("The draft is in `layer3/drafts.jsonl`, which this run did not keep."))

    out.append(R.substep("sdf-example-stage3", "Stage 3 · the review and rewrite"))
    if lin.get("review"):
        out.append(R.details("The problems the reviewer identified in the draft",
                             R.quote(lin["review"]),
                             meta=f"{len(lin['review'].split()):,} words · the review record"))
    else:
        out.append(R.note("The review is in `layer4/rewrites.jsonl`, which this run did not "
                          "keep."))
    content = doc.get("content") or ""
    if content:
        out.append("<p class='muted'>The document, as it ships:</p>")
        out.append(R.highlight(content, []))
    if draft and content:
        out.append(_rewrite_drawer(draft, content))

    out.append(R.substep("sdf-example-stage4", "Stage 4 · the score and gate"))
    out.append(_scores_block(doc.get("scores") or {}, threshold(manifest)))
    return "".join(out)


# Past this share of the shipped document's words, the three-hunk view stops being three
# edits and becomes three arbitrary windows onto a document that was written again.
_REWROTE_FROM_SCRATCH = 0.6


def _rewrite_drawer(draft, content):
    """What the rewrite did, and — derived — whether it edited or started again.

    The layer-4 template licenses a from-scratch rewrite where the problems are structural,
    and on the pinned run it takes that licence for most documents. A hunk view of a
    from-scratch rewrite is confetti, so the drawer says which of the two happened rather
    than presenting both the same way.
    """
    frac = C.changed_fraction(draft, content)
    head = f"<p class='muted'>{C.diff_summary(draft, content)}"
    if frac >= _REWROTE_FROM_SCRATCH:
        return R.details(
            "What the review and rewrite changed",
            head + " At that share it did not edit the draft, it wrote the document again "
                   "— so these are the three largest changed runs, not three edits.</p>"
            + C.diff_hunks(draft, content),
            meta="rewritten, not edited · full diff in the appendix")
    return R.details(
        "What the review and rewrite changed",
        head + " The three largest changes:</p>" + C.diff_hunks(draft, content),
        meta="3 largest changes · full diff in the appendix")


def _scores_block(scores, cut):
    """The judge's three numbers and its own notes, with the cutoff that acted on them."""
    if not scores:
        return R.note("This document carries no layer-5 scores in `final/sdf_corpus.jsonl`.")
    rows = []
    for key, label, gating in _SCORE_DIMS:
        value = scores.get(key)
        if value is None:
            continue
        rows.append((label, value, "gates the dataset" if gating else "recorded, advisory"))
    out = [R.table(["dimension", "this document", "what it does"], rows, align="lrl")]
    if scores.get("notes"):
        out.append("<p class='muted'>The judge's own note:</p>" + R.quote(scores["notes"]))
    out.append(f"<p class='muted'>A document ships at {cut} or above on both gating "
               f"dimensions.</p>")
    return "".join(out)


def _cards_table(cards):
    """The dealt combination as a table.

    Null and empty values are DROPPED: a deal with no domain has no domain, and rendering
    the axis with 'None' in it is a bug that reads as data.
    """
    rows = []
    for key, label in _CARD_AXES:
        value = cards.get(key)
        if isinstance(value, list):
            value = " · ".join(v for v in value if v)
        if value:
            rows.append((label, value))
    return R.table(["dealt axis", "this example"], rows, align="ll") if rows else ""


def carousel(picks, by_id):
    """More documents as tabs, in a drawer: the spec and the document, nothing else.

    Reuses the mechanism the other report's carousel uses. The FIRST pane renders visible
    rather than hidden, so with JS off this degrades to one document instead of to nothing,
    and printing expands all of them. CLOSED, though, because that visible pane is a second
    full document under the pinned one's own trail.
    """
    panes = []
    for did in picks:
        doc = by_id.get(did) or {}
        if not doc.get("content"):
            continue
        head = ""
        if doc.get("description"):
            head = ("<p class='muted'>The spec:</p>" + R.quote(doc["description"]))
        panes.append((f"sdf-ex-{len(panes)}", _label(doc, did),
                      head + "<p class='muted'>The document, as it ships:</p>"
                      + R.highlight(doc["content"], []),
                      not panes))
    if not panes:
        return ""
    return R.details("More examples", R.tabs(panes),
                     meta=f"{len(panes)} more documents from the same run, as they ship")


def _label(doc, did):
    """A tab's name: the id, plus the two cards that make one document distinguishable
    from another at a glance."""
    bits = [b for b in (doc.get("language"), _short_type(doc)) if b]
    return f"{did} · {' · '.join(bits)}" if bits else did


def _short_type(doc):
    """The genre, with the matrix's leading article dropped so a tab label stays short.

    Trimmed by length rather than cut at the first comma: several of the matrix's genres
    are comma-separated lists whose first item is a modifier, and cutting there turned
    "a short, informal personal letter or email exchange" into "short".
    """
    name = doc.get("type_name") or (doc.get("variables") or {}).get("document_type") or ""
    return _trim(re.sub(r"^(an?|the)\s+", "", name).strip(), 34)


def _picks(content, cli=(), by_id=None):
    """(primary, extras) doc_ids for the example beat.

    Pinned in the prose file rather than passed on the command line so that a rebuild
    reproduces the same documents without anyone having to remember a flag; ``--sdf-example``
    overrides the primary only. ``auto`` takes the first shipped document and the two after
    it — deliberately NOT the highest-scoring one, because this beat shows how a document is
    built and must not become a showcase.
    """
    raw = (content.get("sdf_example_pick") or "").strip()
    primary = None if raw.lower() in ("", "auto") else raw.split()[0]
    extras = (content.get("sdf_example_extra") or "").split()
    if cli:
        primary = cli[0] if isinstance(cli, (list, tuple)) else cli
    shipped = sorted(by_id or {})
    if not primary:
        primary = shipped[0] if shipped else None
        extras = extras or [d for d in shipped if d != primary][:2]
    return primary, [d for d in extras if d != primary]


# ------------------------------------------------------------------ caveats

def blocks_weak(content, f):
    """What is wrong with the method, in general — not with this run.

    Authored bullets, deliberately carrying no figures: a reader deciding whether to use
    this pipeline needs to know that nothing here shows a trained model behaves better, that
    the composition is a judgement rather than a sample, and that the judge scoring a
    document is shown the plan as its spec and therefore cannot see a plan that discarded its
    own cards. None of the three is a property of one run. It takes no ``audit`` at all, so
    a run number cannot get in.

    The run's own findings are not softened by this and are not gone: every derived verdict
    still renders, unfiltered, in the appendix drawer built by ``audit_flags_drawer()``.
    """
    return R.sub("sdf-weak", "Caveats") + C.prose(content, "sdf_caveats", f)


# ------------------------------------------------------------------ derived floor

def _verdict(value, good, ok):
    """``evals/audit_sdf.py``'s own thresholds, mirrored. Lower is better for all of them."""
    return "GOOD" if value <= good else "OK" if value <= ok else "BAD"


def derived_warnings(audit, manifest, f, *, compliance=None, fidelity=None, ablation=None,
                     scores=None):
    """The weaknesses floor for this dataset, computed rather than written.

    ``evals/audit_sdf.py`` prints its verdicts and does not record them, so
    ``common.audit_verdict_warnings()`` finds nothing in an SDF audit and the thresholds are
    re-applied here, matching the ones the eval itself uses. Only non-GOOD rows are emitted,
    and provenance is appended exactly as it is on the other section. Rows are only ever
    added to this list.
    """
    audit = audit or {}
    out = []
    length = audit.get("length") or {}
    frac = length.get("truncated_frac")
    if frac:
        out.append((_verdict(frac, 0.0, 0.02),
                    f"{frac:.0%} of documents are truncated ({length.get('truncated')} of "
                    f"{f.get('n_docs', '?')}), so those documents stop mid-thought."))
    checks = (
        (audit.get("composition") or {}).get("top_type_share"), 0.15, 0.30,
        "The largest document type is {v:.0%} of the dataset.",
    ), (
        (audit.get("near_dups") or {}).get("0.9"), 0.02, 0.08,
        "{v:.0%} of documents are near-duplicates of another, above 0.90 similarity.",
    ), (
        (audit.get("openings") or {}).get("formulaic_frac"), 0.15, 0.35,
        "{v:.0%} of documents open with a formulaic pattern.",
    ), (
        (audit.get("markdown") or {}).get("**bold**"), 0.10, 0.30,
        "{v:.0%} of documents carry markdown bold, which is the strongest synthetic tell in "
        "prose.",
    )
    for value, good, ok, text in checks:
        if value is None:
            continue
        verdict = _verdict(value, good, ok)
        if verdict != "GOOD":
            out.append((verdict, text.format(v=value)))
    out += _pattern_warnings(audit)
    out += _name_and_phrase_warnings(audit, f.get("n_docs"))
    out += _principle_warnings(audit)
    out += _gate_warnings(scores, f)
    out += _compliance_warnings(compliance)
    out += _fidelity_warnings(fidelity)
    out += _ablation_warnings(ablation)
    out += C.provenance_warnings(manifest, n=f.get("n_docs"))
    return sorted(out, key=lambda w: 0 if w[0] == "BAD" else 1)


def _pattern_warnings(audit):
    """The templating scan's own flag: a defect above 0.30 prevalence."""
    return [("BAD", f"Templating scan: **{p.get('pattern')}** appears in "
                    f"{p.get('prevalence', 0):.0%} of documents and is judged a generator "
                    f"defect.")
            for p in audit.get("patterns") or [] if p.get("flagged")]


def _name_and_phrase_warnings(audit, n):
    """Name reuse and recurring phrasing, on the eval's own rules.

    ``audit_names``: GOOD while the worst repeated name is under max(2, 10% of the corpus),
    OK to 20%. ``audit_phrases``: GOOD at zero banned-phrase hits, OK to max(1, 5%).
    """
    out = []
    repeated = (audit.get("names") or {}).get("repeated") or []
    if n and repeated:
        name, count = repeated[0][0], repeated[0][1]
        verdict = ("GOOD" if count < max(2, 0.1 * n)
                   else "OK" if count <= 0.2 * n else "BAD")
        if verdict != "GOOD":
            out.append((verdict, f"The most reused invented name, **{name}**, appears in "
                                 f"{count} of {n} documents."))
    hits = (audit.get("phrases") or {}).get("banned_hits") or {}
    if n and hits:
        phrase, count = max(hits.items(), key=lambda kv: kv[1])
        verdict = "OK" if count <= max(1, 0.05 * n) else "BAD"
        out.append((verdict, f"The rewrite's own watched phrases still appear: **{phrase}** in "
                             f"{count} of {n} documents."))
    return out


def _principle_warnings(audit):
    """Principles the corpus barely exercises, against the eval's own floor."""
    pc = audit.get("principle_coverage") or {}
    starved = pc.get("starved") or []
    if not starved:
        return []
    floor = pc.get("floor")
    ids = ", ".join(str(p) for p in starved)
    tail = f" under the eval's {floor:.0%} floor" if isinstance(floor, (int, float)) else ""
    noun = "principle" if len(starved) == 1 else "principles"
    return [("OK", f"{len(starved)} constitution {noun} starved{tail} — {ids} — over the "
                   f"{pc.get('rated', '?')} documents rated. That is a weighting problem in "
                   f"the matrix, not a fault in any one document.")]


def _gate_warnings(scores, f):
    """Whether the gate is doing anything.

    The finding is not that documents were dropped, it is what they were dropped FOR: a
    scoring call whose JSON failed to parse is checkpointed at 5/5/5 and then fails the gate,
    so a run can drop a dozen documents without the judge having rejected any of them.
    """
    if not scores:
        return []
    g = gate(scores, f.get("threshold") or DEFAULT_THRESHOLD)
    if not g.get("graded"):
        return []
    out = []
    if not g.get("rejected"):
        out.append(("BAD", f"**The gate rejected nothing it actually graded.** Of "
                           f"{g['scored']} scored documents it dropped {g['dropped']}, and "
                           f"all of them were scoring calls that failed to parse rather than "
                           f"documents the judge marked down. On this run the gate is a "
                           f"formality."))
    spread = score_hist(scores, "alignment")
    graded = [(v, n) for v, n in spread if v >= (f.get("threshold") or DEFAULT_THRESHOLD)]
    if len(graded) <= 2 and sum(n for _, n in graded) > 20:
        values = " and ".join(str(v) for v, _ in graded)
        out.append(("OK", f"Alignment took only the value(s) {values} across every document "
                          f"the judge graded, so the score separates almost nothing. Read it "
                          f"as a floor that held, not as a ranking."))
    return out


def _compliance_warnings(compliance):
    """Welfare-reasoning failure modes above the report's own prevalence flag."""
    if not compliance:
        return []
    flag = compliance.get("prevalence_flag") or 0.1
    out = []
    for mode in (compliance.get("by_mode") or {}).values():
        share = mode.get("share_of_applicable") or mode.get("share_of_judged") or 0
        if share > flag:
            out.append(("BAD", f"Compliance: **{mode.get('title', '?')}** is present in "
                               f"{share:.0%} of the documents it applies to, above the "
                               f"{flag:.0%} flag."))
    findings = compliance.get("findings") or []
    if findings and not out:
        noun = "document" if len(findings) == 1 else "documents"
        out.append(("OK", f"The compliance pass found {len(findings)} {noun} with a "
                          f"welfare-reasoning failure, out of {compliance.get('judged', '?')} "
                          f"judged — under the flag, but not zero."))
    return out


def _fidelity_warnings(fidelity):
    """Whether the plan honoured the cards it was dealt.

    This is the pipeline's one blind spot with a measurement, so it gets a row whenever it
    was measured at all: layer 5 is given the PLAN as the spec, so a plan that discarded a
    card is scored against its own substitution and passes.
    """
    if not fidelity:
        return []
    clean = fidelity.get("clean_frac")
    if clean is None:
        return []
    by_card = fidelity.get("by_card_frac") or {}
    worst = min(by_card.items(), key=lambda kv: kv[1]) if by_card else None
    tail = (f" The card most often dropped is **{worst[0]}**, honoured in {worst[1]:.0%} of "
            f"them." if worst else "")
    verdict = "GOOD" if clean >= 0.9 else "OK" if clean >= 0.75 else "BAD"
    if verdict == "GOOD":
        return []
    return [(verdict, f"Only {clean:.0%} of the {fidelity.get('judged', '?')} plans checked "
                      f"carried every dealt card into the spec they wrote.{tail} Layer 5 "
                      f"cannot catch this: it is given the plan as the spec, so a plan that "
                      f"substituted a card is scored against its own substitution.")]


def _ablation_warnings(ablation):
    """How much of the realism score is the judge having seen the spec.

    A blind judge applying the same rubric is the only check here on the in-pipeline one,
    and the gap between them is the number that says how much the score is worth.
    """
    drop = (ablation or {}).get("mean_drop")
    if drop is None:
        return []
    verdict = "GOOD" if drop <= 0.5 else "OK" if drop <= 1.0 else "BAD"
    if verdict == "GOOD":
        return []
    return [(verdict, f"**Realism scores {ablation.get('layer5_mean', 0):.1f} in the pipeline "
                      f"and {ablation.get('blind_same_rubric_mean', 0):.1f} to a blind judge "
                      f"applying the same rubric** — a drop of {drop:.1f} over "
                      f"{ablation.get('n', '?')} documents. The in-pipeline score is generous "
                      f"about how much these documents read like the real internet.")]


def audit_flags_drawer(audit, manifest, f, **kw):
    """The derived floor, in the appendix with the rest of this run's own numbers.

    Still computed, never authored, and never filtered — ``warnings_table`` may collapse
    rows into a counted drawer but the list itself is whole. It sits here rather than in the
    caveats beat because every row is specific to one run.
    """
    warnings = derived_warnings(audit, manifest, f, **kw)
    if not warnings:
        return ""
    bad = sum(1 for sev, _ in warnings if sev == "BAD")
    return R.details("What the audit flags", C.warnings_table(warnings),
                     meta=f"{len(warnings)} findings · {bad} BAD")


# ------------------------------------------------------------------ appendix

def judged_drawer(audit, f, scores, ablation, manifest):
    """The layer-5 judge, in one drawer — the structural twin of the other report's.

    Demoted rather than deleted, and for the same kind of reason: on the pinned run the
    judge separates almost nothing, the only check on it is a blind rerun of its own rubric
    that scores the same documents far lower, and judge and generator are the same model
    family. A page that led with these numbers would be leading with its weakest
    measurement.
    """
    body = []
    if not scores:
        return ""
    cut = f.get("threshold") or DEFAULT_THRESHOLD
    g = gate(scores, cut)
    body.append("<p class='muted'>Every document is scored on three dimensions from 1 to 10. "
                "Alignment and realism gate the dataset; spec conformance is recorded and "
                "advisory, because it measures the draft against the plan rather than "
                "against the cards the plan was dealt.</p>")
    for key, label, gating in _SCORE_DIMS:
        hist = score_hist(scores, key)
        if not hist:
            continue
        body.append(R.figure(
            title=f"{label.capitalize()}, as the judge scored it",
            note_=("Gates the dataset." if gating else "Recorded, advisory.")
                  + f" One bar per score the judge gave, over {sum(n for _, n in hist):,} "
                    f"scored documents.",
            chart=R.histogram([(str(v), n) for v, n in hist], color=MEASURE,
                              xlabel="score, 1 to 10"),
            caption=_score_caption(hist, cut, gating)))
    if g:
        body.append("<h4>What the gate did</h4>")
        body.append(R.table(
            ["outcome", "documents"],
            [("scored", f"{g['scored']:,}"),
             (f"passed both gating dimensions at {cut} or above", f"{g['passed']:,}"),
             ("dropped by the gate", f"{g['dropped']:,}"),
             ("…of which were scoring calls that failed to parse",
              f"{g['parse_errors']:,}"),
             ("…of which the judge actually marked down", f"{g['rejected']:,}")],
            align="lr"))
    if ablation:
        body.append(_ablation_figure(ablation))
    else:
        body.append(R.note("No blind rerun of the rubric was recorded for this run, so nothing "
                           "here checks the judge against a judge that could not see the spec. "
                           "It would live in `audit/realism_ablation.json`."))
    body = [b for b in body if b]
    return R.details("What the judge scores, and why the report does not lead with it",
                     "".join(body),
                     meta=f"{len(scores):,} scored documents")


def _score_caption(hist, cut, gating):
    """State the finding, derived: how much of the scale the judge actually used."""
    used = [v for v, _ in hist]
    total = sum(n for _, n in hist)
    top = max(hist, key=lambda kv: kv[1])
    if len(used) <= 3:
        return (f"**The judge used {len(used)} of the ten points on the scale**, and put "
                f"{top[1] / total:.0%} of documents on {top[0]}.")
    below = sum(n for v, n in hist if v < cut)
    tail = (f", and {below / total:.0%} fell below the gate's cutoff of {cut}"
            if gating and below else "")
    return (f"**{top[1] / total:.0%} of documents scored {top[0]}**{tail}.")


def _ablation_figure(ablation):
    return R.figure(
        title="The same rubric, applied blind",
        note_="The in-pipeline judge is shown the spec the document was written from. A "
              "rerun applying the same realism rubric without it scores the same documents "
              "again.",
        # One hue, not a pair: these are the same measurement under two conditions, and a
        # two-colour chart here would read as the other report's two arms.
        chart=R.hbar([("blind to the spec", round(ablation.get("blind_same_rubric_mean", 0), 2)),
                      ("in the pipeline", round(ablation.get("layer5_mean", 0), 2))],
                     color=MEASURE, maxval=10, fmt="{:.2f}"),
        caption=f"**Realism falls {ablation.get('mean_drop', 0):.1f} points when the judge "
                f"cannot see the spec**, over {ablation.get('n', '?')} documents.")


def _composition_figures(audit, repo_href=""):
    """The shipped dataset's composition, one figure per engineered axis.

    Shipped shares only: the dealt-against-shipped pairing and its drift captions were
    cut at Constance's call — the appendix talks about the diversity and composition of
    what survived, not about the matrix bookkeeping that produced it.
    """
    comp = (audit or {}).get("composition") or {}
    out = []
    for axis, title in (("centrality", "How central the welfare thread is *"),
                        ("tone", "The author's stance *"),
                        ("domain", "Domain *")):
        counts = comp.get(axis)
        if not counts:
            continue
        n = sum(counts.values()) or 1
        rows = sorted(counts.items(), key=lambda kv: -kv[1])
        out.append(R.figure(
            title=title,
            chart=R.hbar([(_trim(k), v / n) for k, v in rows], maxval=1.0,
                         fmt="{:.0%}", color=MEASURE, label_w=260)))
    if comp.get("language"):
        n = sum(comp["language"].values()) or 1
        rows = sorted(comp["language"].items(), key=lambda kv: -kv[1])
        out.append(R.figure(
            title="Language *",
            note_="Derived from the culture axis, which fixes the language a document is "
                  "written in along with its idiom and its institutions.",
            chart=R.hbar([(k, v) for k, v in rows], color=MEASURE, label_w=180)))
    if out:
        link = (f"[prompts/sdf/variables.txt]({repo_href}/blob/main/prompts/sdf/"
                "variables.txt)" if repo_href else "`prompts/sdf/variables.txt`")
        out.append("<p class='muted'>" + R.inline_md(
            f"* These shares are set by weights in {link} — retargeting one (say, 90% "
            "English) is an edit to that file.") + "</p>")
    return out


def _trim(value, n=44):
    value = str(value)
    return value if len(value) <= n else value[:n - 1].rstrip() + "…"


def _trim_words(value, n):
    """Trim at a word boundary. The matrix's values are sentences, and a mid-word cut set
    in inline code — "not the central subject of the document, bu…" — reads as corruption
    rather than as an abbreviation."""
    value = _trim(value, n)
    if value.endswith("…") and " " in value[:-1]:
        return value[:-1].rstrip().rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return value


def _principle_figure(audit, principles=None, repo_href=""):
    """Which distilled principles the sample exercises, and what those principles are.

    The bars name principles by number, so the figure carries the way to the words: a
    link to the CSV the pipeline itself renders, and — when the build could read it —
    the principles themselves in a drawer. Both come from the repo at build time, which
    is the same source the pinned run's audit judged against.
    """
    pc = (audit or {}).get("principle_coverage") or {}
    by = pc.get("by_principle") or {}
    if not by:
        return ""
    floor = pc.get("floor") or 0.05
    rows = [{"label": f"principle {k}", "share": v}
            for k, v in sorted(by.items(), key=lambda kv: -kv[1])]
    gloss = {f"principle {n}": text for n, text in (principles or [])}
    starved = pc.get("starved") or []
    src = ((" Hover a bar for the principle." if gloss else "")
           + f" The principles are maintained in the repository: "
           f"[constitution/constitution_principles.csv]({repo_href}/blob/main/"
           "constitution/constitution_principles.csv)." if repo_href else "")
    out = R.figure(
        title="Which constitution principles the dataset exercises",
        note_=f"A judge read a sample of documents and named the distilled principles each "
              f"one exercises. The rule marks the eval's {floor:.0%} starvation floor."
              + src,
        chart=R.grouped_hbar(rows, series=[("share", MEASURE)], percent=True,
                             rule=floor, rule_label="floor", label_w=140,
                             direct_labels=False, glossary=gloss or None),
        caption=(f"**{len(starved)} of {len(rows)} principles fall under the floor.** That is "
                 f"fixed at the matrix's weights, not per document."
                 if starved else "**Every principle clears the floor.**"))
    if principles:
        out += R.details(
            "The principles, by number",
            "<p class='muted'>As distilled from the constitution, read from the "
            "repository at build time.</p>"
            + R.table(["number", "principle"], [(n, p) for n, p in principles],
                      align="rl"),
            meta=f"{len(principles)} principles")
    return out


def _curve_block(curve):
    """How the dataset's effective distinctness grows with its size."""
    points = (curve or {}).get("points") or []
    if not points:
        return ""
    proj = (curve or {}).get("proj") or {}
    tail = ""
    if proj:
        bits = [f"{int(k):,} documents would reach roughly "
                f"{min(v.values()):.0f}–{max(v.values()):.0f}"
                for k, v in sorted(proj.items(), key=lambda kv: int(kv[0]))]
        tail = " Extrapolated: " + "; ".join(bits) + "."
    return R.figure(
        title="How distinctness grows with dataset size",
        note_="Effective distinct documents, measured over growing prefixes of this run's own "
              "dataset. It is sublinear by construction — the question is how sublinear."
              + tail,
        chart=R.histogram([(f"{int(n):,}", round(v, 1)) for n, v in points],
                          color=MEASURE, xlabel="documents embedded"),
        caption="**Distinctness keeps climbing, and keeps falling behind the record count** — "
                "which is what sets the ceiling on how much one matrix is worth running.")


def blocks_appendix(content, f, *, audit=None, diversity=None,
                    curve=None, principles=None, repo_href=""):
    """Everything that is evidence, collapsed so it costs a reader nothing.

    The same shape as the other report's appendix: one composition-and-diversity
    drawer that talks about what the shipped dataset IS — its engineered composition
    axes, the constitution principles it exercises, and how semantically varied it is.
    The matrix bookkeeping (attrition, dealt-vs-shipped drift, card fidelity), the
    compliance pass, the health-check triage tables, the templating-scan glossary and
    the worked example's full rewrite diff belong to the review tool, not here.
    """
    blocks = [R.sub("sdf-appendix", "Appendix"), C.prose(content, "sdf_appendix_intro", f)]

    comp_figs = _composition_figures(audit, repo_href)
    principle_fig = _principle_figure(audit, principles, repo_href)
    semantic = C.semantic_figures(diversity, unit="documents")
    curve_fig = _curve_block(curve)
    if comp_figs or principle_fig or semantic:
        have = [name for name, part in (("composition", comp_figs),
                                        ("principles", principle_fig),
                                        ("meanings and topics", semantic)) if part]
        blocks.append(R.details(
            "Composition and diversity",
            C.prose(content, "sdf_checks_intro", f)
            + "".join(comp_figs) + principle_fig + semantic + curve_fig,
            meta=" · ".join(have)))

    return "".join(b for b in blocks if b)


# ------------------------------------------------------------------ assembly

def blocks(*, content, audit=None, diversity=None, compliance=None, fidelity=None,
           ablation=None, curve=None, manifest=None, corpus=None, scores=None, lineage=None,
           attrition=None, matrix=None, n_prompt_templates=None, run_id="", example=None,
           principles=None, f=None, hf_href="", repo_href=""):
    """The whole ``#sdf`` section body, in skeleton order. Pure: no filesystem, no argv.

    Returns one flat string of blocks. website/page.py wraps it in ``<section id='sdf'>`` with
    the h2; every block here is therefore a grid child of that section, which is what lets
    figures bleed past the text measure.

    ``f`` is accepted so a caller that already computed the facts can pass them; the default
    computes them here, as the other report does.
    """
    if f is None:
        f = facts(audit, diversity, manifest, attrition=attrition, matrix=matrix,
                  scores=scores, compliance=compliance, fidelity=fidelity, ablation=ablation,
                  curve=curve, corpus=corpus)
    picks = (example,) if example else ()
    if audit is None and not corpus:
        return "".join([
            blocks_what(content, f),
            R.note("No run output was supplied for this dataset, so nothing here is measured. "
                   "Build with `--sdf-run <run directory>`."),
            _ways_out(hf_href, repo_href),
        ])
    return "".join([
        blocks_what(content, f),
        blocks_built(content, f),
        blocks_example(content, f, corpus, lineage, manifest, picks,
                       hf_href=hf_href, repo_href=repo_href),
        blocks_appendix(content, f, audit=audit, diversity=diversity,
                        curve=curve, principles=principles, repo_href=repo_href),
    ])
