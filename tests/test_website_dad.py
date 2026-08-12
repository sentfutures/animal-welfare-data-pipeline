"""Tests for website/dad.py — the dilemma corpus's section of the handoff page.

The section never renders alone any more, so every test here builds the whole page
around it (website/page.py owns the shell) and asserts on the ``#dad`` beats.

Six things carry real risk here and get most of the coverage:

  * **Degradation.** Not every committed run has the paid delivery/showcase keys or a
    full set of step files, so the generator must render a complete section from a
    partial run and say what is missing rather than quietly omitting it.
  * **Self-containment.** The artefact's whole format exists so it can be opened
    offline from a filesystem. One external asset reference breaks that.
  * **Candour.** The caveats a reader sees are authored and general, so they carry no run
    figures at all; what the run's own audit flagged is derived, and asserted to survive
    into the appendix even with the caveats prose emptied. The view may collapse rows but
    only with a visible count.
  * **Not leading with the judge, and not documenting deployment.** The report is the
    process and the records: no judged figure appears outside the appendix, the delivery
    regression is stated in prose exactly once, beside the comparison it qualifies, and
    nothing on the page explains how to run the pipeline. Demotion is not deletion — the
    numbers are all still on the page, in drawers.
  * **The lineage.** The worked example is assembled from the run's own step files, so
    each step either renders or names the artefact it wanted.
  * **Colour integrity.** Arm colours must follow the arm rather than the row order,
    and a series hue must never double as the page's "good".

Fully offline — the generator touches no network and no API, so no stubs beyond the
suite's autouse guards are needed.
"""

import json
import re

import pytest

from website import common as C
from website import dad as D
from website import page as P
from website import render as R
from website import sdf as S

# --- fixtures, shaped like the real audit JSON --------------------------------

PER_CASE = {
    "AW-0001": {
        "pipeline": {"reasons": ["a", "b", "c"], "chars": 4000,
                     "type_hist": {"direct": 2, "sentience": 1}},
        "plain": {"reasons": ["a", "b"], "chars": 2500, "type_hist": {"direct": 2}},
        "survival": {"anchored": [{"reason": "a", "verdict": "kept"},
                                  {"reason": "b", "verdict": "dropped"}],
                     "added": ["c"]},
        "response_gid": "R-0201", "example_gid": "E-0172",
    },
    "AW-0002": {
        "pipeline": {"reasons": ["d", "e"], "chars": 3800, "type_hist": {"direct": 2}},
        "plain": {"reasons": ["d"], "chars": 2400, "type_hist": {"direct": 1}},
        "survival": {"anchored": [{"reason": "d", "verdict": "weakened"}], "added": ["e"]},
        "response_gid": "R-0202", "example_gid": "E-0173",
    },
}

AUDIT_FULL = {
    "n_prompts": 2,
    "gid_map": {"AW-0001": {"response": "R-0201", "example": "E-0172"},
                "AW-0002": {"response": "R-0202", "example": "E-0173"}},
    "sections": [
        {"title": "Response stance (LLM)", "group": "paid",
         "rows": [{"label": "moralizes", "value": "pipeline 40% / plain 0%",
                   "verdict": "BAD", "note": "(fault — lower is better)"},
                  {"label": "defers", "value": "100%", "verdict": "GOOD", "note": ""}]},
        {"title": "Locale / taxa plausibility", "group": "prompt",
         "rows": [{"label": "implausible", "value": "0", "verdict": "GOOD", "note": ""}]},
    ],
    "moral_patient_reasons": {
        "n": 2, "failures": 1, "model": "claude-sonnet-5", "judge_model": "claude-opus-5",
        "pipeline": {"n": 2, "mean_unique": 2.5},
        "plain": {"n": 2, "mean_unique": 1.5},
        "survival": {"kept": 1, "weakened": 1, "dropped": 1, "added_total": 2},
        "per_case": PER_CASE,
    },
    "moves": {
        "alternatives": {"pipeline_mean": 3.0, "plain_mean": 2.0},
        "stance": {"pipeline": {"defers": 1.0, "calibrated": 0.97, "moralizes": 0.4},
                   "plain": {"defers": 1.0, "calibrated": 1.0, "moralizes": 0.0}},
    },
    "delivery": {
        "pipeline_mean": 8.2, "plain_mean": 7.9, "n_pipeline": 2, "n_plain": 2, "failures": 0,
        "dimensions": {"pipeline": {"tone": 8.0, "calibration": 9.0},
                       "plain": {"tone": 8.5, "calibration": 8.0}},
        "per_case": {"AW-0001": {"pipeline": {"score": 8}, "plain": {"score": 7}},
                     "AW-0002": {"pipeline": {"score": 9}, "plain": {"score": 8}}},
    },
    "showcase": {
        "model": "claude-opus-5",
        "examples": [{"prompt_id": "AW-0001", "label": "Welfare reasoning added",
                      "summary": "The pipeline **surfaced** a point plain missed.",
                      "user_message": "Should I do the thing?",
                      "plain_response": "Maybe.", "pipeline_response": "Consider the animals here.",
                      "highlights": ["the animals"], "fit": 9}],
    },
    "response_lengths": {"n": 2, "pipeline_mean": 4659.0, "plain_mean": 2988.0,
                         "mean_ratio": 1.56, "per_case": {}},
    "tracked_tics": {"n_pipeline": 2, "n_plain": 2,
                     "watch": {"cuts both ways": {"origin": "pipeline-origin",
                                                  "pipeline": 1, "plain": 0}}},
    "rhetorical_moves": {"moves": {"unbundling": {"description": "splits a bundled choice",
                                                  "pipeline_share": 0.28, "plain_share": 0.28},
                                   "autonomy-coda": {"description": "hands the call back",
                                                     "pipeline_share": 0.38, "plain_share": 0.0}}},
    "structure": {"pipeline": {"effective_shapes": 9.44},
                  "plain": {"effective_shapes": 13.88}},
    "library_coverage": {"n_cases": 2, "library_size": 44, "used": 37},
    "reason_composition": {"type_gloss": {"direct": "the animal's own experience"}},
}

DIVERSITY = {"n_records": 2, "embed_model": "gemini-embedding-001",
             "vendi": {"score": 5.15, "ratio": 0.132},
             "nn": {"over_0.90": 0.0, "over_0.80": 0.33},
             "scopes": {"combined": {
                 "n": 2, "nn_sims": [0.75, 0.85], "vendi_ratio": 0.50,
                 "over": {"0.90": 0.0, "0.80": 0.33},
                 "clusters": {"k": 2, "evenness": 0.875, "largest_share": 0.33,
                              "sizes": [1, 1],
                              "detail": [{"size": 1, "rep_id": "R-0201",
                                          "rep": "Should I do the thing?",
                                          "ids": ["AW-0001"]}]}}}}

MANIFEST = {"run_id": "2026-07-20_20-51_bedrock-40", "created_at": "2026-07-20T20:51:58",
            "git_commit": "abc12345", "git_dirty": True,
            "config": {"backend": "bedrock", "model": "claude-sonnet-5",
                       "dad": {"scenario_model": "claude-opus-4-8",
                               "constitution_rewrite_model": "claude-opus-4-8"}}}

COSTS = [{"stage": "prompt_draft", "cost_usd": 0.5, "model": "claude-opus-4-8"},
         {"stage": "constitution_rewrite", "cost_usd": 1.5, "model": "claude-opus-4-8"}]

BASELINE = [{"prompt_id": "AW-0001", "user_message": "Should I do the thing?",
             "baseline_response": "Maybe."}]
REWRITES = [{"prompt_id": "AW-0001", "user_message": "Should I do the thing?",
             "draft_response": "Consider the animals.",
             "rewritten_response": "Consider the animals here.",
             "entry_ids": ["C2", "T13"]},
            {"prompt_id": "AW-0002", "user_message": "And this other thing?",
             "draft_response": "Perhaps not.", "rewritten_response": "Weigh the birds."}]
DEALS = [{"scenario_id": "S-001", "domain": ["public policy / law"],
          "taxa_category": "farmed animals",
          "cultural_setting": "Brazil, written in Portuguese"}]
DILEMMAS = [{"prompt_id": "AW-0001", "scenario_id": "S-001"},
            {"prompt_id": "AW-0002", "scenario_id": "S-002"}]
# taxa_subcategory is deliberately None: a deal with no value on an axis must drop the
# row rather than render the word "None" as if it were data.
SCENARIOS = [{"scenario_id": "S-001", "scenario_gid": "S-0138",
              "scenario_description": "A county fair contract is up for renewal.",
              "domain": ["public policy / law"], "taxa_subcategory": None,
              "user_attitude": "unaware", "cultural_setting": None},
             {"scenario_id": "S-002", "scenario_gid": "S-0139",
              "scenario_description": "A supplier list needs vetting.",
              "domain": ["procurement"], "user_attitude": "skeptical / dismissive"}]
SCOPES = [{"prompt_id": "AW-0001", "entry_ids": ["C2", "T13"], "selection_fallback": False,
           "scope": {"patients": "the fair's ponies", "goal": "a defensible decision",
                     "levers": "the contract terms"},
           "triggered_entries": [{"id": "C2", "category": "Conduct", "claim": "Surface it.",
                                  "reasoning": "long text", "trigger_condition": "x"},
                                 {"id": "T13", "category": "Topic", "claim": "Heat matters.",
                                  "reasoning": "long text", "trigger_condition": "y"}]}]

# What read_lineage() builds out of the four files above, for the tests that build a page
# without touching a run directory. AW-0002 carries only its cards, so the beat's
# per-artefact degradation is exercised by the default fixture rather than only by the
# tests that go looking for it.
LINEAGE = {
    "AW-0001": {"scenario_id": "S-001",
                "cards": {"domain": ["public policy / law"], "user_attitude": "unaware"},
                "description": "A county fair contract is up for renewal.",
                "scope": SCOPES[0]["scope"], "entry_ids": ["C2", "T13"],
                "entries": [{"id": "C2", "category": "Conduct", "claim": "Surface it."},
                            {"id": "T13", "category": "Topic", "claim": "Heat matters."}],
                "selection_fallback": False},
    "AW-0002": {"scenario_id": "S-002",
                "cards": {"domain": ["procurement"], "user_attitude": "skeptical / dismissive"}},
}

CORPUS = [{"record_id": "AW-0001", "messages": []}, {"record_id": "AW-0002", "messages": []}]

CONTENT = {k: f"Prose for {k}." for k in P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS}
CONTENT["title"] = "Test report"
CONTENT["example_pick"] = "auto"
CONTENT["example_extra"] = "AW-0002"
CONTENT["dad_what"] = "A {{n}}-example run, {{near_dup_pct}} near-duplicated."


def content(**overrides):
    return {**CONTENT, **overrides}


def build(**kwargs):
    """Build the whole page around this DAD run. The section never renders alone."""
    kwargs.setdefault("audit", AUDIT_FULL)
    page_content = kwargs.pop("content", None) or content()
    example = kwargs.pop("example", None)
    return P.build(content=page_content, dad_inputs=kwargs, example=example)


def dad_section(html):
    """Just the #dad panel, for assertions that must not be satisfied elsewhere. It is
    the last section on the page: synthetic documents comes first throughout."""
    return html[html.index("<section id='dad'"):]


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


# The run's own text, which the page quotes verbatim. Substring assertions about the
# PAGE have to exclude it, or a dilemma that happens to contain "None" or "url(" fails a
# test about the generator. Deliberately narrower than common._STRIP_BLOCKS, which also
# strips <style> — the one place a real url() would hide.
_CORPUS_TEXT = re.compile(r"<blockquote\b.*?</blockquote>|<div class='resp'>.*?</div>", re.S)


def without_corpus_text(html):
    return _CORPUS_TEXT.sub(" ", html)


def beat(html, anchor):
    """One beat's body: after its own <h3> and before the next one.

    Slicing on ``index("id='dad-weak'")`` looks right and is not: it keeps the tail of its
    own opening tag and the head of the next beat's ``<h3``, and that stray ``3`` breaks
    any assertion about digits in a beat.
    """
    section = dad_section(html)
    start = section.index(f"<h3 id='{anchor}'")
    body = section[section.index(">", start) + 1:]
    nxt = body.find("<h3 id=")
    return body if nxt == -1 else body[:nxt]


def make_run_dir(tmp_path, audit=None, diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS):
    run_dir = tmp_path / "runs" / "2026-07-20_20-51_bedrock-40"
    (run_dir / "audit").mkdir(parents=True)
    (run_dir / "final").mkdir()
    (run_dir / "baseline").mkdir()
    (run_dir / "step1").mkdir()
    (run_dir / "step2").mkdir()
    (run_dir / "step3").mkdir()
    (run_dir / "audit" / "audit_report.json").write_text(
        json.dumps(audit if audit is not None else AUDIT_FULL), encoding="utf-8")
    if diversity is not None:
        (run_dir / "audit" / "diversity_report.json").write_text(json.dumps(diversity),
                                                                encoding="utf-8")
    if manifest is not None:
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if costs is not None:
        (run_dir / "cost_log.jsonl").write_text(
            "\n".join(json.dumps(c) for c in costs), encoding="utf-8")
    (run_dir / "baseline" / "baseline_responses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in BASELINE), encoding="utf-8")
    (run_dir / "step3" / "rewrites.jsonl").write_text(
        "\n".join(json.dumps(r) for r in REWRITES), encoding="utf-8")
    (run_dir / "step1" / "scenario_deals.jsonl").write_text(
        "\n".join(json.dumps(d) for d in DEALS), encoding="utf-8")
    (run_dir / "step1" / "dilemmas.jsonl").write_text(
        "\n".join(json.dumps(d) for d in DILEMMAS), encoding="utf-8")
    (run_dir / "step1" / "scenarios.jsonl").write_text(
        "\n".join(json.dumps(s) for s in SCENARIOS), encoding="utf-8")
    (run_dir / "step2" / "scopes.jsonl").write_text(
        "\n".join(json.dumps(s) for s in SCOPES), encoding="utf-8")
    (run_dir / "final" / "dad_corpus.jsonl").write_text(
        json.dumps({"record_id": "AW-0001", "messages": []}), encoding="utf-8")
    content_file = tmp_path / "content_all.md"
    content_file.write_text("".join(f"<!-- id: {k} -->\n{v}\n\n" for k, v in CONTENT.items()),
                            encoding="utf-8")
    return run_dir, content_file


class TestFacts:
    def test_reconstructs_considerations_from_legacy_schema(self):
        cons = D._considerations(AUDIT_FULL)
        assert cons["source"] == "reconstructed"
        assert cons["pipeline"] == pytest.approx(5.5)  # 2.5 reasoning + 3.0 alternatives
        assert cons["plain"] == pytest.approx(3.5)

    def test_prefers_modern_schema_when_present(self):
        audit = dict(AUDIT_FULL, valuable_welfare_considerations={
            "available": True, "parent": {"pipeline": 9.0, "plain": 6.0},
            "subsets": [{"name": "welfare reasoning", "pipeline": 5.0, "plain": 4.0}]})
        cons = D._considerations(audit)
        assert cons["source"] == "modern"
        assert cons["pipeline"] == 9.0

    def test_facts_are_read_from_the_data_not_hardcoded(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["response_lengths"]["mean_ratio"] = 2.5
        assert D.facts(audit)["length_pct"] == "150%"

    def test_dealt_and_measured_counts_are_distinguished(self):
        """40 dilemmas dealt and 39 measured is the normal case, and reporting the
        first as if it were the second is the kind of thing a reader spots in
        thirty seconds."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["n_prompts"] = 40
        f = D.facts(audit)
        assert f["n"] == 40 and f["n_measured"] == 2

    def test_the_extractor_is_not_credited_as_the_judge(self):
        f = D.facts(AUDIT_FULL)
        assert f["extract_model"] == "claude-sonnet-5"
        assert f["judge_model"] == "claude-opus-5"

    def test_the_delivery_comparison_is_not_available_to_prose(self):
        """It is written once, by _delivery_statement(). A clause in facts() is an
        invitation to write it a second time in a prose file."""
        assert "delivery_clause" not in D.facts(AUDIT_FULL)
        assert "substance_clause" not in D.facts(AUDIT_FULL)

    def test_the_footprint_regressions_fact_went_with_its_drawer(self):
        """It existed only for the "Every chart" drawer's meta line; the charts now
        live inside the comparison drawer, and a fact nothing renders is a fact that
        drifts."""
        assert "footprint_regressions" not in D.facts(AUDIT_FULL)

    def test_facts_carry_no_cost_or_scale_figures(self):
        """Cost per example and the dealt spread came off the page with the descriptive
        tiles, and their facts came off with them: a fact nothing renders is a fact that
        will end up in prose."""
        f = D.facts(AUDIT_FULL, MANIFEST)
        for gone in ("cost_total", "cost_per_example", "n_shipped", "records_clause",
                     "spread_clause"):
            assert gone not in f, gone


class TestBuildSection:
    def test_builds_every_beat(self):
        html = build(diversity=DIVERSITY, manifest=MANIFEST,
                     baseline=BASELINE, rewrites=REWRITES, lineage=LINEAGE, run_id="run-x")
        for anchor, label in D.BEATS:
            assert f"<h3 id='{anchor}'>{label}</h3>" in html

    def test_the_beats_render_in_the_declared_order(self):
        """The chooser above asks the reader to walk through a generation, so the stages
        come before the example that walks through them."""
        section = dad_section(build(lineage=LINEAGE, rewrites=REWRITES))
        assert re.findall(r"<h3 id='(dad-[^']+)'", section) == [a for a, _ in D.BEATS]

    def test_the_measured_beat_is_gone(self):
        """This report is not a results report. Its measurements are either a
        descriptive tile or an appendix drawer."""
        html = build(diversity=DIVERSITY)
        assert "dad-measured" not in html
        assert "What we measured" not in strip_tags(html)

    def test_the_beats_are_flat_children_of_one_section(self):
        """A figure has to be a direct child of the section for the CSS grid to bleed
        it past the text measure, so no beat may wrap itself in a container — and the
        panel IS that section rather than a wrapper around one."""
        section = dad_section(build(diversity=DIVERSITY))
        assert section.count("<section") == 1
        assert "class='panel'" in section.split(">", 1)[0]

    def test_is_self_contained(self):
        """``url(`` and ``@import`` are checked outside the run's own text: the page now
        carries three records verbatim, and a dilemma that happens to quote a CSS
        snippet would fail a bare substring search on the whole document. What is being
        tested is the page's markup, not the dataset's prose."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST, lineage=LINEAGE,
                     rewrites=REWRITES, baseline=BASELINE)
        assert not re.search(r"<iframe\b", html)
        for tag in re.findall(r"<link\b[^>]*>", html):        # only the tab icon; see
            assert re.fullmatch(                              # test_website_page.py
                r"<link rel='icon' sizes='\d+x\d+' href='data:image/png;base64,[^']+'>",
                tag), tag
        assert not re.search(r"<script[^>]*\ssrc=", html)
        markup = without_corpus_text(html)
        assert "@import" not in markup and "url(" not in markup
        refs = re.findall(r"(?:src|href)='([^']+)'", html)
        assert refs and all(r.startswith(("data:", "#", "https://")) for r in refs)

    def test_prose_hyperlinks_are_allowed(self):
        html = build(content=content(dad_what="See [the post](https://x.test/y)."))
        assert "href='https://x.test/y'" in html

    def test_is_light_mode_only(self):
        html = build()
        assert "color-scheme:only light" in html
        assert "content='only light'" in html
        assert "prefers-color-scheme" not in html
        assert "data-theme" not in html

    def test_placeholders_are_resolved(self):
        html = build()
        assert "{{" not in html
        assert "A 2-example run" in html

    @pytest.mark.parametrize("where", ["answer", "message", "scenario", "card", "scope",
                                       "library"])
    def test_escapes_hostile_run_text(self, where):
        """Every path by which a run's own text reaches the page. The lineage added four
        of them, and an unescaped one is a stored XSS in a file people email around."""
        evil = "<script>alert(1)</script>"
        rewrites = json.loads(json.dumps(REWRITES))
        lineage = json.loads(json.dumps(LINEAGE))
        if where == "answer":
            rewrites[0]["rewritten_response"] = evil
        elif where == "message":
            rewrites[0]["user_message"] = evil
        elif where == "scenario":
            lineage["AW-0001"]["description"] = evil
        elif where == "card":
            lineage["AW-0001"]["cards"]["domain"] = [evil]
        elif where == "scope":
            lineage["AW-0001"]["scope"] = {"patients": evil}
        else:
            lineage["AW-0001"]["entries"] = [{"id": "C2", "category": "Conduct", "claim": evil}]
        html = build(rewrites=rewrites, lineage=lineage, baseline=BASELINE)
        assert evil not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_the_report_is_titled_for_what_it_teaches(self):
        """"The dilemma corpus" told a reader nothing they could act on, and "corpora"
        was the wrong register for the whole page."""
        html = build()
        assert f"<h2>{R.esc(D.SECTION_TITLE)}</h2>" in html
        assert D.SECTION_TITLE == "Difficult advice Q&A"
        headings = re.findall(r"<h2>([^<]*)</h2>", html)
        assert headings and not any(h[0].isdigit() for h in headings)
        assert "corpus" not in strip_tags(html).lower().replace("dad_corpus.jsonl", "")

    def test_no_eyebrow(self):
        """The uppercase kicker over the title read as generated; it is gone, and the
        page's only uppercase treatment is now the chip."""
        assert "eyebrow" not in build()

    def test_every_stage_heading_is_its_own_anchor(self):
        """The rail links to the stages, not only the beats, and an ``<h4>`` becomes a rail
        item by having an id. The ids carry their beat's name because the same three stage
        names are written twice — once in "How it is built", once in the worked example —
        and two headings sharing an id is a link that lands on whichever came first."""
        html = build(rewrites=REWRITES, lineage=LINEAGE, baseline=BASELINE,
                     content=content(example_pick="AW-0001"))
        for key in ("stage1", "stage2", "stage3"):
            assert f"<h4 id='dad-built-{key}'>" in html
            assert f"<h4 id='dad-example-{key}'>" in html
        assert "<h4 id='dad-built-control'>" in html
        ids = re.findall(r"<h4 id='([^']+)'>", html)
        assert len(ids) == len(set(ids)), ids

    def test_the_appendix_headings_stay_unanchored(self):
        """They live inside closed drawers, so a rail link to one would scroll to a heading
        the reader cannot see. Having no id is what keeps them out of the rail."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST, rewrites=REWRITES,
                     lineage=LINEAGE, baseline=BASELINE)
        appendix = html[html.index("id='dad-appendix'"):]
        assert "<h4>Measure by measure</h4>" in appendix
        assert "<h4 id=" not in appendix

    def test_anchored_beats_land_with_headroom(self):
        """A link used to drop the heading flush against the top of the viewport.
        Sub-beats are link targets too — the chooser opens a panel from #dad-weak."""
        html = build()
        assert "scroll-behavior:smooth" in html
        assert re.search(r"section\{[^}]*scroll-margin-top:[\d.]+rem", html)
        assert re.search(r"h3\[id\]\{[^}]*scroll-margin-top:[\d.]+rem", html)
        reduced = html[html.find("@media (prefers-reduced-motion:reduce)"):][:120]
        assert "scroll-behavior:auto" in reduced


class TestJudgedComparison:
    """The comparison against the plain model is demoted to one appendix drawer.

    Demoted, not deleted: the numbers are all still on the page, and these tests exist to
    keep both halves of that true.
    """

    def test_reports_the_measures_that_undercut_the_headline(self):
        """Density and structural variety both move the wrong way while the substance
        measure moves the right way. They belong in the same table, not in a footnote."""
        html = D.scoreboard(AUDIT_FULL, D.facts(AUDIT_FULL), D._considerations(AUDIT_FULL))
        text = strip_tags(html)
        assert "considerations per 1,000 characters" in text
        assert "structural variety" in text
        assert "answer length" in text

    def test_unmeasured_stance_adds_no_moralizing_row(self):
        """A dashed "answers that moralize" row reads as a finding about moralizing on
        a run that never measured stance, so the row appears only when stance ran."""
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "moves"}
        html = D.scoreboard(audit, D.facts(audit), D._considerations(audit))
        assert "moralize" not in strip_tags(html)

    def test_a_worse_number_gets_the_bad_chip(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"]["pipeline_mean"] = 7.0
        audit["delivery"]["plain_mean"] = 7.9
        html = D.scoreboard(audit, D.facts(audit), D._considerations(audit))
        assert "chip bad'>worse" in html

    def test_the_whole_comparison_lives_inside_the_appendix(self):
        section = dad_section(build(diversity=DIVERSITY, lineage=LINEAGE, rewrites=REWRITES))
        appendix = section[section.index("id='dad-appendix'"):]
        for marker in ("considerations per 1,000 characters", "Substance against manner",
                       "Valuable welfare considerations per answer", "judged axis"):
            assert marker in appendix, marker
            assert marker not in section[:section.index("id='dad-appendix'")], marker

    def test_the_drawer_leads_with_what_is_compared_not_a_caveat(self):
        """The drawer is "Comparison to the control" and opens on what the judges score;
        the arm asymmetry renders once, with the run's own numbers, in the audit-flags
        drawer rather than as a caveat repeated here."""
        html = build(diversity=DIVERSITY)
        assert "Comparison to the control" in html
        assert "Improving on the control" not in html
        assert "least sound measurement" not in html
        assert "Both arms answer the same dilemmas" in strip_tags(html)

    def test_the_two_judge_intro_defines_both_axes_and_links_the_rubrics(self):
        """A run with the welfare-impact judge names both judges in the corpus audit's
        words and points at the judge prompts, which are the rubrics."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["welfare_impact"] = {"pipeline_mean": 92.0, "plain_mean": 83.0,
                                   "score_max": 100, "per_case": {}}
        html = build(audit=audit)
        text = strip_tags(html)
        assert "Welfare impact" in text and "delivery quality" in text
        assert "evals/audit_dad.py" in text

    def test_the_pareto_plots_both_judges_and_computes_its_caption(self):
        """With both judges' per-case scores the figure renders on their own scale, and
        the caption's percentages come from the plotted answers — a delivery loss must
        not inherit the "not bought with manner" verdict."""
        delivery = {"score_max": 100, "per_case": {
            "AW-0001": {"pipeline": {"score": 91, "blended_score": 91.0},
                        "plain": {"score": 93, "blended_score": 93.0}}}}
        welfare = {"score_max": 100, "per_case": {
            "AW-0001": {"pipeline": {"blended_score": 85.0},
                        "plain": {"blended_score": 60.0}}}}
        fig = D._pareto_figure({"delivery": delivery, "welfare_impact": welfare}, {}, {})
        assert "Substance against manner" in fig
        assert "+42% welfare impact" in fig and "-2% delivery quality" in fig
        assert "not bought with manner" not in fig
        gained = json.loads(json.dumps(welfare))
        gained["per_case"]["AW-0001"]["plain"]["blended_score"] = 85.0
        gained["per_case"]["AW-0001"]["pipeline"]["blended_score"] = 92.0
        flipped = D._pareto_figure(
            {"delivery": {"score_max": 100, "per_case": {
                "AW-0001": {"pipeline": {"blended_score": 93.0},
                            "plain": {"blended_score": 91.0}}}},
             "welfare_impact": gained}, {}, {})
        assert "not bought with manner" in flipped

    def test_the_example_count_is_the_shipped_corpus_not_the_step1_deck(self):
        """The audit's n_prompts counts step-1 dilemmas; the page's "examples" is the
        records that shipped."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["n_prompts"] = 7
        html = build(audit=audit, rewrites=REWRITES)
        assert "A 2-example run" in strip_tags(html)

    def test_no_paid_pass_says_so_instead_of_showing_a_hole(self):
        audit = {k: v for k, v in AUDIT_FULL.items()
                 if k not in ("delivery", "moral_patient_reasons", "moves",
                              "valuable_welfare_considerations")}
        text = strip_tags(build(audit=audit))
        assert "No paid judge pass ran on this run" in text

    def test_reads_the_two_holistic_judge_schema(self):
        """PR #107 replaced the considerations metric upstream with welfare_impact and
        composite. A run audited after it must still render its judged means."""
        audit = {k: v for k, v in AUDIT_FULL.items()
                 if k not in ("moral_patient_reasons", "valuable_welfare_considerations")}
        audit["welfare_impact"] = {"pipeline_mean": 6.4, "plain_mean": 5.1}
        audit["composite"] = {"arm_means": {"pipeline": 0.62, "plain": 0.55}}
        text = strip_tags(build(audit=audit))
        assert "welfare impact" in text and "6.40" in text
        assert "composite" in text and "0.620" in text


class TestSayingItOnce:
    """The delivery regression was stated in four places, which reads as hedging."""

    @staticmethod
    def _regressed():
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"]["pipeline_mean"] = 7.0
        audit["delivery"]["plain_mean"] = 7.8
        return audit

    def test_prose_states_it_exactly_once(self):
        """Tables and the derived weakness carry the same number as DATA, which is not
        the same as saying it again."""
        html = build(audit=self._regressed(), manifest=MANIFEST)
        # Strip inline SVG first: path data is full of decimals, and a paragraph that
        # merely contains an icon is not a paragraph that states a finding. (\b on the
        # p as well, or the regex matches <path> too.)
        text = re.sub(r"<svg\b.*?</svg>", " ", html, flags=re.S)
        prose = re.findall(r"<p\b[^>]*>(.*?)</p>", text, re.S)
        said = [p for p in prose if "7.0" in p and "7.8" in p]
        assert len(said) == 1, said
        assert "bad-note" in html[:html.index(said[0])].rsplit("<p", 1)[-1]

    def test_it_is_stated_beside_the_comparison_it_is_about(self):
        """Inside the appendix's judged drawer. The caveats beat is generalised — it holds
        for any run — so a figure from this one cannot live there."""
        section = dad_section(build(audit=self._regressed(), manifest=MANIFEST))
        assert section.index("id='dad-appendix'") < section.index("went the wrong way")
        assert "wrong way" not in section[:section.index("id='dad-appendix'")]

    def test_the_number_still_reaches_the_appendix(self):
        """Demotion is not deletion. It used to reach the caveats beat as well; that beat
        was cut, so the appendix is the one place left and the number has to be there."""
        section = dad_section(build(audit=self._regressed(), manifest=MANIFEST,
                                   diversity=DIVERSITY))
        appendix = section[section.index("id='dad-appendix'"):]
        assert "wrong way" in strip_tags(appendix)
        assert section.index("chip bad'>worse") > section.index("id='dad-appendix'")


class TestChartsAreEvidence:
    """No chart leads. Every one is in the appendix, where a reader goes for evidence."""

    def test_no_figure_appears_outside_the_appendix(self):
        """The restructure, in one assertion. The report above the appendix is prose, the
        dealt cards, and one worked example — a chart there would be arguing a result."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST, baseline=BASELINE,
                     lineage=LINEAGE, rewrites=REWRITES)
        section = dad_section(html)
        lead = section[:section.index("id='dad-appendix'")]
        assert re.findall(r"<figcaption class='fig-t'>([^<]*)</figcaption>", lead) == []

    def test_every_chart_is_still_on_the_page(self):
        """Moved, not dropped: the "Every chart" drawer folded into the comparison
        drawer, so every chart lives with the comparison it supports."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST, baseline=BASELINE)
        appendix = dad_section(html)[dad_section(html).index("id='dad-appendix'"):]
        for title in ("Answer length", "Stance", "Structural variety",
                      "What happened to the control's considerations",
                      "Delivery quality, dimension by dimension",
                      "Valuable welfare considerations per answer",
                      "Substance against manner"):
            assert title in appendix, title
        assert "Every chart" not in html

    def test_the_pareto_leads_and_each_judge_gets_a_section(self):
        """The scatter is the drawer's first figure, right after the intro paragraph,
        and each judge's dimension table sits under its own heading."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["welfare_impact"] = {
            "pipeline_mean": 92.0, "plain_mean": 83.0, "score_max": 100,
            "dimensions": {"pipeline": {"patient_scope": 82.0},
                           "plain": {"patient_scope": 45.0}},
            "per_case": {"AW-0001": {"pipeline": {"blended_score": 85.0},
                                     "plain": {"blended_score": 60.0}}}}
        section = dad_section(build(audit=audit))
        drawer = section[section.index("Comparison to the control"):]
        assert (drawer.index("Substance against manner")
                < drawer.index("Delivery quality, dimension by dimension")
                < drawer.index("Welfare impact, dimension by dimension")
                < drawer.index("Answer length"))

    def test_the_report_opens_on_a_lede_and_measures_nothing_there(self):
        """The report opens on a bare lede — one sentence saying what this is.

        It briefly opened on a "What it is" beat carrying the flow and a specimen record;
        both moved (the diagram to the pipeline beat whose prose reads it aloud, the record
        to the worked example that shows it in full), and a heading over a single sentence
        only names what a reader can already see. `test_neither_report_puts_a_heading_over
        _its_opening_line` in test_website_page.py holds both reports to that.

        What did NOT change is the half of the rule that was about measurement: nothing
        above the pipeline beat carries a figure, a tile, a chip or a score. A chart there
        would argue a result before the reader knows what the data is.
        """
        section = dad_section(build(diversity=DIVERSITY, manifest=MANIFEST,
                                    rewrites=REWRITES, lineage=LINEAGE))
        head = section[:section.index("<h3 id='dad-built'")]
        assert "class='lede'" in head          # the line still says what the dataset is
        assert "<h3" not in head               # and it carries no heading of its own
        for banned in ("/10", "chip", "class='tiles'", "<figure"):
            assert banned not in head, banned


class TestDegradation:
    def test_offline_only_audit_still_builds(self):
        audit = {k: v for k, v in AUDIT_FULL.items()
                 if k not in ("delivery", "showcase", "moves", "moral_patient_reasons")}
        html = build(audit=audit, lineage=LINEAGE, rewrites=REWRITES)
        assert "id='dad-built'" in html
        # Outside the run's own text: a shipped answer may legitimately contain the word.
        assert not re.search(r"\bNone\b", strip_tags(without_corpus_text(html)))

    def test_missing_delivery_says_so_rather_than_omitting(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        section = dad_section(build(audit=audit))
        text = strip_tags(section[section.index("id='dad-appendix'"):])
        assert "not measured on this run" in text
        assert "--reasons" in text

    def test_missing_delivery_drops_the_scatter_without_a_hole(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        html = build(audit=audit)
        assert "Substance against manner" not in html
        assert "id='dad-example'" in html

    def test_delivery_present_renders_the_pareto_in_the_appendix(self):
        section = dad_section(build())
        assert "Substance against manner" in section
        assert "<circle" in section
        assert section.index("Substance against manner") > section.index("id='dad-appendix'")

    def test_delivery_without_the_substance_measure_drops_the_pareto(self):
        """BOTH axes have to be measured. The two-holistic-judge rework dropped the
        considerations metric this plots on the vertical, and a run carrying only the
        delivery half rendered the figure anyway: an empty plot reading "not measured on
        this run" under a caption asserting the pipeline buys substance with manner — a
        claim about a chart that was not there, and false on the pinned run, where the
        pipeline is higher on both axes."""
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "moral_patient_reasons"}
        section = dad_section(build(audit=audit))
        assert "Substance against manner" not in section
        assert "buys substance with manner" not in section
        assert "id='dad-appendix'" in section              # the rest of the drawer survives

    def test_bare_audit_still_carries_the_narrative(self):
        """The process and the caveats are authored, so they survive an audit with nothing
        in it — which is the point of keeping figures out of them."""
        html = build(audit={"n_prompts": 3})
        assert "Prose for method_intro." in html

    def test_the_page_does_not_explain_how_to_run_the_pipeline(self):
        """That is the repository README's job. A hand-off page that documents installation
        is a page a reader skims past to reach what they came for."""
        html = build(manifest=MANIFEST)
        text = strip_tags(html)
        assert "<pre>" not in html
        for gone in ("Running it yourself", "config.yaml", "per example", "Per-stage cost",
                     "dad_pipeline/run.py"):
            assert gone not in text, gone

    def test_missing_manifest_diversity_and_costs(self):
        """With no diversity pass the distinctness tile is ABSENT, not 0.0 — the trap
        that `.get("score", 0)` walks straight into."""
        html = build(manifest=None, diversity=None)
        assert "id='dad-built'" in html
        assert "effectively distinct records" not in html
        assert "0.0" not in strip_tags(dad_section(html))

    def test_missing_gid_map_falls_back_to_prompt_ids(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "gid_map"}
        assert "AW-0001" in build(audit=audit)


class TestTheFlow:
    """The pipeline schematic, which lives in the pipeline beat.

    It used to open the report, in a `dad-what` beat with a 30-word specimen record beside
    it. Both went: the diagram moved down to the beat whose prose reads it aloud, and the
    specimen was cut because the worked example two beats below is the same record in full.
    What survived the move is what this class keeps — the diagram owes an accessible name,
    because SVG text is not read as prose, and it owes the chart palette a wide berth,
    because it measures nothing.
    """

    @staticmethod
    def _built(**kwargs):
        kwargs.setdefault("rewrites", REWRITES)
        kwargs.setdefault("lineage", LINEAGE)
        section = dad_section(build(**kwargs))
        return section[section.index("id='dad-built'"):section.index("id='dad-example'")]

    def test_the_flow_is_parseable_svg_with_an_accessible_name(self):
        """SVG text is not read as prose, so the diagram owes a name that says what the
        whole thing is — not just labels a sighted reader can assemble."""
        import xml.etree.ElementTree as ET
        svg = re.search(r"<svg[^>]*class='flow'.*?</svg>", self._built(), re.S).group(0)
        el = ET.fromstring(svg)
        assert el.get("role") == "img"
        assert el.find("title") is not None
        assert "weighted matrix" in el.get("aria-label")

    def test_the_flow_names_the_stages_the_report_goes_on_to_explain(self):
        """The diagram is a map of the stages beside it. Three vocabularies for three views
        of one pipeline is how a reader stops believing it is one pipeline."""
        section = dad_section(build(rewrites=REWRITES, lineage=LINEAGE))
        built = section[section.index("id='dad-built'"):section.index("id='dad-example'")]
        flow = re.search(r"<svg[^>]*class='flow'.*?</svg>", section, re.S).group(0)
        for stage in ("the user dilemma", "the model response", "the constitution rewrite"):
            assert stage in flow and stage in built

    def test_the_flow_is_a_schematic_so_it_carries_no_series_or_status_colour(self):
        """Nothing in it is proportional to a measurement. Drawn in the chart palette it
        would read as a result, which is the one thing the report does not lead with."""
        flow = re.search(r"<svg[^>]*class='flow'.*?</svg>", self._built(), re.S).group(0)
        for reserved in ("--series-", "--good", "--warn", "--bad", "--accent"):
            assert reserved not in flow, reserved

    def test_the_prose_says_what_the_diagram_shows(self):
        """The prose beside the flow is not a caption. If the diagram does not render —
        print, a stripped mail client, a screen reader — the reader still learns the shape.
        Read against the SHIPPED prose, because the fixture's placeholder cannot prove it.
        """
        from pathlib import Path
        # Off the module's own __file__ rather than a spelled-out directory name.
        prose = (Path(D.__file__).resolve().parent
                 / "content_dad.md").read_text(encoding="utf-8")
        said = prose[prose.index("id: method_intro"):prose.index("id: example_pick")]
        for shown in ("weighted matrix", "stage 1", "stage 2", "stage 3"):
            assert shown in said, shown
        # NOTE: the diagram's stage names and this prose no longer share a vocabulary —
        # the flow says "the constitution rewrite" and the copy pass rewrote the prose to
        # say "your alignment documents". That is a copy decision, not a test one, so this
        # asserts the source and the count a reader can check rather than pinning either
        # wording until it is settled.

    def test_the_pipeline_beat_does_not_lead_on_a_measurement(self):
        """A chart here would argue a result before the reader knows what the data is. The
        report's figures live in the appendix, and this beat is the process."""
        built = self._built(diversity=DIVERSITY, manifest=MANIFEST)
        for banned in ("/10", "class='tiles'", "<figure"):
            assert banned not in built, banned

class TestLineage:
    """The worked example is one record's whole trail, assembled from the run's own step
    files. Nothing in it is author-supplied, and nothing in it depends on the paid pass."""

    @staticmethod
    def _example(**kwargs):
        kwargs.setdefault("lineage", LINEAGE)
        kwargs.setdefault("rewrites", REWRITES)
        kwargs.setdefault("baseline", BASELINE)
        section = dad_section(build(**kwargs))
        return section[section.index("id='dad-example'"):section.index("id='dad-appendix'")]

    def test_the_stages_render_in_pipeline_order(self):
        ex = self._example()
        marks = [ex.index(m) for m in (
            "Stage 1 · the user dilemma", "dealt axis", "the planner writes",
            "Should I do the thing?", "Stage 2 · the model response",
            "what stage 2 works out", "Stage 3 · the constitution rewrite")]
        assert marks == sorted(marks), marks

    def test_the_stage_headings_match_the_ones_the_method_beat_uses(self):
        """A reader who has just read "How it is built" should recognise each step, not
        learn a second vocabulary for the same pipeline."""
        section = dad_section(build(lineage=LINEAGE, rewrites=REWRITES))
        built = section[section.index("id='dad-built'"):section.index("id='dad-example'")]
        for heading in ("Stage 1 · the user dilemma", "Stage 2 · the model response",
                        "Stage 3 · the constitution rewrite"):
            assert heading in built and heading in self._example()

    def test_the_shipped_answer_is_inline_and_verbatim(self):
        """The answer is the artefact: it is not behind a drawer. The control's first take
        IS, because it is context for stage 2 rather than the thing being shown."""
        ex = self._example()
        answer = ex[ex.index("Stage 3 · the constitution rewrite"):]
        assert "Consider the animals here." in strip_tags(answer.split("<details")[0])
        # The control's answer is inside a <summary>...</summary> drawer label's details
        # block, and the label says what it is for.
        take = ex[ex.index("first take"):]
        assert "control model answering the user dilemma" in take[:200]
        assert "Maybe." not in strip_tags(ex[:ex.index("first take")])

    def test_the_dealt_cards_drop_null_axes(self):
        """A deal with no cultural setting has no cultural setting. Rendering the axis
        with "None" in it is a bug that reads as data."""
        lineage = json.loads(json.dumps(LINEAGE))
        lineage["AW-0001"]["cards"] = {"domain": ["marketing"], "cultural_setting": None,
                                       "archetype": ""}
        ex = self._example(lineage=lineage)
        assert "marketing" in ex
        assert "cultural setting" not in strip_tags(ex)
        assert not re.search(r"\bNone\b", strip_tags(without_corpus_text(ex)))

    def test_the_library_entries_are_glossed_from_the_run(self):
        ex = self._example()
        assert "C2" in ex and "Surface it." in ex
        assert "reasoning_library.csv" in ex

    def test_bare_ids_when_the_gloss_is_missing(self):
        lineage = json.loads(json.dumps(LINEAGE))
        lineage["AW-0001"].pop("entries")
        ex = self._example(lineage=lineage)
        assert "C2" in ex and "T13" in ex

    @pytest.mark.parametrize("artefact,wanted", [
        ("cards", "scenario_deals.jsonl"), ("description", "step1/scenarios.jsonl"),
        ("scope", "step2/scopes.jsonl")])
    def test_a_missing_artefact_names_the_file_it_wanted(self, artefact, wanted):
        """A step that silently disappears reads as a step the pipeline does not have."""
        lineage = json.loads(json.dumps(LINEAGE))
        lineage["AW-0001"].pop(artefact)
        assert wanted in self._example(lineage=lineage)

    def test_no_lineage_still_shows_the_message_and_the_answer(self):
        """The rewrite record alone carries both, so the beat survives a run that kept no
        step-1 or step-2 files at all."""
        ex = self._example(lineage=None)
        assert "Should I do the thing?" in strip_tags(ex)
        assert "Consider the animals here." in strip_tags(ex)
        assert "step1/scenarios.jsonl" in ex

    def test_examples_are_pinned_in_the_prose_file(self):
        """The prose file's picks decide which record is walked and which sit behind the
        carousel. Asserted on the records themselves: the beat prints the extra's gid on its
        carousel tab, but the PRIMARY record's gid appears nowhere — it and the "pinned in
        the prose file" note were both dropped in the copy pass, so the featured record is
        currently not locatable in the dataset viewer by id."""
        ex = self._example(content=content(example_pick="AW-0001", example_extra="AW-0002"))
        assert "Should I do the thing?" in strip_tags(ex)     # AW-0001 is the one walked
        assert "R-0202" in ex                                 # AW-0002's carousel tab

    def test_the_cli_overrides_the_primary_only(self):
        ex = self._example(content=content(example_pick="AW-0001", example_extra="AW-0002"),
                           example="AW-0002")
        # The primary is the CLI's record, not the prose file's: its own message is the one
        # quoted in stage 1. (The prose file's only extra IS this record, so it is spent as
        # the primary and no carousel is left over — which is why this asserts the trail
        # rather than a tab.)
        first = ex.index("Stage 1 · the user dilemma")
        assert "And this other thing?" in strip_tags(ex[first:ex.index("Stage 2 ·")])
        assert "Should I do the thing?" not in strip_tags(ex)

    def test_auto_picks_the_first_shipped_record(self):
        """Deliberately not the showcase judge's favourite: this beat shows how a record
        is built, and must not depend on the paid pass having run."""
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "showcase"}
        ex = self._example(audit=audit, content=content(example_pick="auto"))
        assert "Should I do the thing?" in strip_tags(ex)      # the first shipped record

    def test_a_pinned_id_this_run_never_shipped_says_so(self):
        ex = self._example(content=content(example_pick="AW-9999"))
        assert "is not in this run" in strip_tags(ex)
        # and it falls back to the first shipped record rather than rendering nothing
        assert "Should I do the thing?" in strip_tags(ex)

    def test_the_extra_examples_are_a_carousel(self):
        ex = self._example(content=content(example_pick="AW-0001", example_extra="AW-0002"))
        assert "class='carousel'" in ex
        assert "And this other thing?" in strip_tags(ex)

    def test_the_extra_examples_are_collapsed_and_say_how_many(self):
        """The carousel's first pane is a second full transcript — measured at ~1,250
        words on the pinned run — sitting under the pinned record's own trail, which is
        what the beat is for. It is behind a closed drawer that names what is in it."""
        ex = self._example(content=content(example_pick="AW-0001", example_extra="AW-0002"))
        drawer = re.search(r"<details(?! open)>(<summary>.*?</summary>)", ex, re.S)
        assert drawer, ex[-600:]
        summaries = re.findall(r"<summary>(.*?)</summary>", ex, re.S)
        more = [s for s in summaries if "More examples" in s]
        assert more and "1 more record" in strip_tags(more[0])
        assert ex.index("<details><summary>More examples") < ex.index("class='carousel'")

    def test_the_first_carousel_pane_survives_javascript_being_off(self):
        """One pane renders visible in the markup, so the carousel degrades to a single
        example rather than to nothing once the drawer is opened."""
        ex = self._example(content=content(example_pick="AW-0001",
                                           example_extra="AW-0002 AW-0001"))
        panes = re.findall(r"<div class='pane-x' id='(ex-\d)'[^>]*>", ex)
        assert panes, ex[:400]
        assert "hidden" not in re.search(r"<div class='pane-x' id='ex-0'([^>]*)>", ex).group(1)

    def test_rewrite_diff_shows_hunks_inline_and_only_inline(self):
        """The worked example carries the three largest hunks; the full-diff drawer that
        duplicated the whole answer into the appendix was cut."""
        html = build(content=content(example_pick="AW-0001"), baseline=BASELINE,
                     rewrites=REWRITES, lineage=LINEAGE)
        assert "<ins>" in html
        assert "3 largest changes" in html
        appendix = html[html.find("id='dad-appendix'"):]
        assert "full stage-3 rewrite diff" not in appendix.lower()

    def test_diff_summary_reports_how_much_changed(self):
        assert "%" in C.diff_summary("a b c d", "a b c e")

    def test_no_example_data_is_reported_not_crashed(self):
        assert "No worked example" in strip_tags(build(audit={"n_prompts": 1}))

    def test_the_report_carries_a_way_to_the_records_and_the_pipeline(self):
        """The whole report — ten thousand words of it — carried no link at all, so a
        reader who had just followed one record end to end, which is the moment they are
        most likely to want the data, had to scroll back past everything they had read to
        find one. The pair sits at the foot of the worked example, where that happens."""
        html = build(content=content(example_pick="AW-0001"), baseline=BASELINE,
                     rewrites=REWRITES, lineage=LINEAGE)
        panel = html[html.index("<section id='dad'"):html.index("<footer")]
        assert panel.count("class='lbtn'") == 2, "two destinations, and only two"
        # esc()d, because the config name carries an "&": the href is written &amp;, which is
        # what makes it a valid attribute value.
        assert R.esc(P.HF_DAD) in panel and P.REPO_URL in panel
        example = panel[panel.index("id='dad-example'"):panel.index("id='dad-appendix'")]
        assert "class='lbtns'" in example, "not in the appendix, and not before the trail"
        assert example.index("class='lbtns'") > example.index("Stage 3")

    def test_a_report_built_without_destinations_carries_no_empty_button_row(self):
        assert "class='lbtns'" not in D.blocks(audit=AUDIT_FULL, content=content(),
                                               rewrites=REWRITES, baseline=BASELINE,
                                               lineage=LINEAGE)


class TestColourIntegrity:
    def test_status_colors_are_not_series_colors(self):
        """--good used to be byte-identical to --series-3, the pipeline's own hue, so
        the palette quietly editorialised 'pipeline = good'."""
        series = set(re.findall(r"--series-\d:(#[0-9a-f]{6})", R.CSS))
        status = set(re.findall(r"--(?:good|warn|bad):(#[0-9a-f]{6})", R.CSS))
        assert not (series & status)

    def test_arm_colors_follow_the_arm_not_the_row_order(self):
        """hbar(color=None) falls back to PAL[i], which painted the considerations
        chart's pipeline bar in the control's colour while every other chart used green.
        The chart is in the appendix now; the invariant is unchanged."""
        section = dad_section(build(diversity=DIVERSITY))
        appendix = section[section.index("id='dad-appendix'"):]
        chart = appendix[appendix.index("Valuable welfare considerations per answer"):]
        chart = chart[:chart.find("</svg>")]
        fills = re.findall(r"fill='(var\(--series-\d\))'", chart)
        assert fills == [R.PLAIN, R.PIPELINE]

    def test_every_chart_carries_an_accessible_name(self):
        """Charts are named; the button icons are decorative and marked aria-hidden,
        which is the correct treatment for a mark that repeats its own label."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST,
                     baseline=BASELINE, rewrites=REWRITES)
        for svg in re.findall(r"<svg\b.*?</svg>", html, flags=re.S):
            assert "<title>" in svg or "aria-hidden='true'" in svg


class TestJudgedScale:
    """The judges' scale is the run's, read off the audit rather than typed."""

    def test_a_hundred_point_pass_is_labelled_as_one(self):
        """The two-holistic-judge rework grades 0–100 where the pass it replaced graded
        0–10, and six places typed "0–10" — so a mean of 92.33 printed against a scale of
        ten."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"].update(score_max=100, pipeline_mean=92.33, plain_mean=83.01)
        text = strip_tags(dad_section(build(audit=audit)))
        assert "judged delivery quality, 0–100" in text
        assert "judged 0–100" in text                      # and the checks table with it
        assert "0–10" not in text.replace("0–100", "")      # nowhere is the old scale left

    def test_a_pass_that_records_no_scale_is_a_ten_point_one(self):
        """Which is what every pre-rework run looks like, and they are still on the page."""
        assert "judged delivery quality, 0–10" in strip_tags(dad_section(build()))


class TestWhichRun:
    """A report is about a pipeline. Its worked example and its appendix are one run, and
    the page used to leave that to be inferred — the carousel said "the same run" of a run
    it had never introduced."""

    RUN = "2026-07-29_12-26_archetype200"

    def test_the_example_and_the_appendix_both_name_the_run(self):
        """Both, and the id repeated rather than "that run": a reader arriving from the
        rail lands in the appendix without having read the beat that introduced it."""
        section = dad_section(build(run_id=self.RUN, rewrites=REWRITES, lineage=LINEAGE))
        example = section[section.index("id='dad-example'"):section.index("id='dad-appendix'")]
        appendix = section[section.index("id='dad-appendix'"):]
        assert self.RUN in example
        assert self.RUN in appendix

    def test_the_appendix_says_how_big_the_run_was(self):
        section = dad_section(build(run_id=self.RUN))
        assert "2 examples" in strip_tags(section[section.index("id='dad-appendix'"):])

    def test_the_run_is_named_where_the_report_stops_being_general(self):
        """Not in the pipeline beat and not in the caveats: those hold for any run, and a
        run id in them would say the opposite."""
        section = dad_section(build(run_id=self.RUN, rewrites=REWRITES, lineage=LINEAGE))
        general = section[section.index("id='dad-built'"):section.index("id='dad-example'")]
        assert self.RUN not in general

    def test_a_build_with_no_run_id_ships_no_dangling_sentence(self):
        assert "measured on one run" not in strip_tags(build()).lower()


class TestCandour:
    """What candour the page carries: disclosed asymmetries and unhidden regressions.

    The derived weaknesses floor (``derived_warnings`` + the "What the audit flags"
    drawer) was cut at Constance's call — review-tool triage, not hand-off
    storytelling — so candour now lives with the measurements it qualifies.
    """

    def test_moralizing_regression_is_shown_in_both_arms(self):
        text = strip_tags(build())
        assert "40%" in text and "0%" in text

    def test_the_audit_flags_drawer_is_gone_machinery_and_all(self):
        html = build(diversity=DIVERSITY, manifest=MANIFEST)
        assert "What the audit flags" not in html
        assert not hasattr(D, "derived_warnings")
        assert not hasattr(D, "audit_flags_drawer")

    def test_extraction_failures_produce_an_asymmetry_note(self):
        """It moved into the judged drawer with the comparison it qualifies."""
        section = dad_section(build())
        appendix = section[section.index("id='dad-appendix'"):]
        assert "not fully matched" in strip_tags(appendix)

    def test_the_health_check_triage_tables_do_not_render(self):
        """The variety drawer mirrors the corpus audit viewer's diversity section; the
        audit's per-section verdict dump and the which-checks-ran table are review-tool
        triage and stay out of the hand-off page."""
        html = build(diversity=DIVERSITY)
        appendix = html[html.find("id='dad-appendix'"):]
        assert "Composition and diversity" in appendix
        assert "Locale / taxa plausibility" not in appendix
        assert "As the audit records them" not in appendix
        assert "not run on this run" not in appendix

    def test_the_variety_drawer_mirrors_the_viewer_charts_and_captions(self):
        """The viewer's two semantic charts and their captions, with the Vendi count as
        a sentence — computed from the same scopes.combined data the viewer reads."""
        html = build(diversity=DIVERSITY)
        text = strip_tags(html)
        assert "Redundancy — how close each record sits to its nearest neighbour" in text
        assert "Topic spread — the records grouped into meaning clusters" in text
        assert "0% near-duplicate (similarity above 0.90), 33% similar (above 0.80)" in text
        assert "Evenness 0.875 across 2 clusters" in text
        assert "1.0 of 2 records effectively distinct in meaning (Vendi ratio 0.50)" in text
        assert "What each cluster is" in text

    def test_no_per_record_diversity_data_drops_the_semantic_block_only(self):
        """A diversity report without scopes.combined.nn_sims (pre-field runs) keeps the
        moves and phrases figures but renders no meanings-and-topics charts."""
        html = build(diversity={"n_records": 2, "vendi": {"score": 5.0, "ratio": 0.1}})
        appendix = html[html.find("id='dad-appendix'"):]
        assert "Composition and diversity" in appendix
        assert "Meanings and topics" not in appendix
        assert "Rhetorical habits" in appendix

    def test_charts_emit_parseable_svg(self):
        import xml.etree.ElementTree as ET

        ET.fromstring(R.hbar([("a", 1), ("b<script>", 2)]))
        ET.fromstring(R.grouped_hbar([{"label": "x", "p": 1, "q": 2}],
                                     series=[("p", "red"), ("q", "blue")]).split("<div")[0])
        ET.fromstring(R.stacked_bar([{"label": "r", "segments": {"kept": 2}}],
                                    categories=[("kept", "red")]).split("<div")[0])
        ET.fromstring(R.scatter([{"x": 1, "y": 2, "color": "red", "tip": "t"}]))
        ET.fromstring(R.segbar([("kept", 2, "red"), ("added", 1, "blue")]).split("<div")[0])
        ET.fromstring(R.histogram([("7", 2), ("8", 5)]))

    def test_empty_data_is_a_note_not_a_broken_chart(self):
        assert "no" in R.hbar([]).lower()
        assert "<svg" not in R.grouped_hbar([], series=[("a", "red")])
        assert "<svg" not in R.segbar([("kept", 0, "red")])

    def test_segbar_labels_live_in_the_legend_not_on_the_fill(self):
        """Surface-coloured text on the arm fills was 2.5:1 on the green — a fail on
        cream, and already a fail on white."""
        html = R.segbar([("kept", 439, R.PLAIN), ("added", 260, R.PIPELINE)])
        svg = html[:html.find("</svg>")]
        assert "<text" not in svg
        assert "kept · 439" in html and "added · 260" in html

    def test_zero_values_do_not_divide_by_zero(self):
        assert "<svg" in R.hbar([("a", 0), ("b", 0)])

    def test_hbar_takes_one_color_or_a_sequence(self):
        assert R.hbar([("a", 1), ("b", 2)], color="red").count("fill='red'") == 2
        both = R.hbar([("a", 1), ("b", 2)], color=("red", "blue"))
        assert "fill='red'" in both and "fill='blue'" in both

    def test_table_escapes_cells_but_passes_raw_through(self):
        html = R.table(["h"], [("<b>x</b>",), (R.Raw("<b>y</b>"),)])
        assert "&lt;b&gt;x&lt;/b&gt;" in html
        assert "<b>y</b>" in html

    def test_table_right_aligns_the_columns_it_is_told_to(self):
        html = R.table(["a", "n"], [("x", "1")], align="lr")
        assert "<td class='num'>1</td>" in html
        assert "<td>x</td>" in html

    def test_inline_md_escapes_before_formatting(self):
        assert R.inline_md("**a** <b>") == "<b>a</b> &lt;b&gt;"

    def test_deks_come_from_a_prose_convention(self):
        assert R.paragraphs("> the finding\n\nbody") == \
            "<p class='dek'>the finding</p><p>body</p>"

    def test_highlight_is_fail_open(self):
        assert R.highlight("hello", ["nope"]) == "<div class='resp'>hello</div>"
        assert "<mark>ell</mark>" in R.highlight("hello", ["ell"])

    def test_figure_names_the_chart_and_states_the_finding(self):
        html = R.figure(title="T", chart="<svg viewBox='0 0 1 1'></svg>", caption="**F.**")
        assert "<title>T</title>" in html
        assert "<figcaption class='fig-c'><b>F.</b></figcaption>" in html

    def test_at_most_one_hero_tile(self):
        with pytest.raises(ValueError, match="one hero"):
            R.tiles([R.stat("1", "a", tone="hero"), R.stat("2", "b", tone="hero")])

    def test_a_tile_carries_direction_as_a_chip_not_a_colored_numeral(self):
        html = R.stat("7.0", "delivery", flag="regression", tone="bad")
        assert "chip bad'>regression" in html
        assert "class='tile-v'>7.0" in html

    def test_a_subheading_is_a_deep_link_target(self):
        assert R.sub("dad-weak", "Where it is weak") == \
            "<h3 id='dad-weak'>Where it is weak</h3>"

    def test_the_illustration_is_a_data_uri_or_an_honest_hole(self):
        """What fills the slot has to travel inside the file, so the primitive takes a
        data URI and refuses anything else."""
        empty = R.illustration()
        assert "TODO" in empty and "src=" not in empty
        filled = R.illustration("data:image/png;base64,AAAA", alt="a butterfly")
        assert "<img src='data:image/png;base64,AAAA' alt='a butterfly'>" in filled
        assert "TODO" not in filled
        with pytest.raises(ValueError, match="data: URI"):
            R.illustration("../assets/hero.png")


    def test_drawer_summaries_name_their_payload_size(self):
        assert "1,010 words" in R.details("Full answer", "x", meta="1,010 words")

    def test_tabs_leave_one_pane_visible(self):
        """No-JS is the case that matters: the markup itself has to show a record."""
        html = R.tabs([("a", "R-1", "one", True), ("b", "R-2", "two", False)])
        assert "<div class='pane-x' id='a' role='tabpanel' aria-labelledby='tab-a'>" in html
        assert "<div class='pane-x' id='b' role='tabpanel' aria-labelledby='tab-b' hidden>" in html
        assert html.count("class='tab'") == 2
        assert "aria-selected='true'" in html and "aria-selected='false'" in html

    def test_every_pane_is_named_by_the_button_that_opens_it(self):
        """A tabpanel with no accessible name is announced as a bare group — which here
        means an unlabelled 1,200-word transcript."""
        html = R.tabs([("a", "R-1", "one", True), ("b", "R-2", "two", False)])
        for pid in ("a", "b"):
            assert f"id='tab-{pid}'" in html
            assert f"aria-labelledby='tab-{pid}'" in html

    def test_the_carousel_is_a_tab_set_and_finishes_the_pattern(self):
        """This one really is a tab set — exactly one pane is open at all times — so it
        owes the rest of role='tab': one tab in the tab order at a time, and the arrow
        keys. The page carried no keydown handler at all before this."""
        html = R.tabs([("a", "R-1", "one", True), ("b", "R-2", "two", False)])
        assert "tabindex='0'" in html and "tabindex='-1'" in html
        assert html.count("tabindex='0'") == 1
        js = R.JS
        assert "keydown" in js
        for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
            assert key in js
        assert "o.setAttribute('tabindex',on?'0':'-1')" in js

    def test_tabs_of_nothing_render_nothing(self):
        assert R.tabs([]) == ""

    def test_tabs_escape_their_labels(self):
        assert "&lt;b&gt;" in R.tabs([("a", "<b>", "body", True)])

    def test_print_rules_keep_figures_and_rows_whole(self):
        block = R.CSS[R.CSS.find("@media print"):]
        assert "figure" in block and "break-inside:avoid-page" in block
        assert "thead{display:table-header-group}" in block

    def test_print_expands_every_carousel_pane(self):
        """A printed page is not a page anyone can click, so all the examples print."""
        block = R.CSS[R.CSS.find("@media print"):]
        assert ".pane-x[hidden]" in block and "display:block!important" in block


class TestCLI:
    def _argv(self, run_dir, content_file, out_dir, sdf_run=None):
        argv = ["build_website.py", "--dad-run", str(run_dir),
                "--content", str(content_file), "--out-dir", str(out_dir)]
        return argv + (["--sdf-run", str(sdf_run)] if sdf_run else [])

    def test_writes_one_file(self, tmp_path, monkeypatch):
        """One page, named index.html so it publishes to Pages as it stands."""
        from website import build_website as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        assert not (tmp_path / "dad.html").exists()
        out = tmp_path / "index.html"
        assert "<section id='dad' class='panel'" in out.read_text(encoding="utf-8")
        assert "<section id='sdf' class='panel'" in out.read_text(encoding="utf-8")

    def test_rebuild_overwrites_cleanly(self, tmp_path, monkeypatch):
        from website import build_website as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        first = (tmp_path / "index.html").read_text(encoding="utf-8")
        B.main()
        assert (tmp_path / "index.html").read_text(encoding="utf-8") == first

    def test_a_dad_run_alone_is_enough(self, tmp_path, monkeypatch):
        from website import build_website as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        assert "not published yet" in (tmp_path / "index.html").read_text(encoding="utf-8")

    def test_a_hosted_build_carries_its_card_image_out(self, tmp_path, monkeypatch):
        """The one file that travels beside the page.

        `og:image` is fetched over the network by whoever renders the link, so it cannot be
        a data URI like everything else here — which makes it the one thing a deploy can
        leave behind. Naming the site is enough to get both files.
        """
        from website import build_website as B
        run_dir, content_file = make_run_dir(tmp_path)
        out = tmp_path / "site"
        out.mkdir()
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, out)
                            + ["--site-url", "https://x.test/"])
        B.main()
        assert (out / "preview.png").read_bytes() == B.PREVIEW.read_bytes()
        html = (out / "index.html").read_text(encoding="utf-8")
        assert '<meta property="og:image" content="https://x.test/preview.png">' in html

    def test_an_image_hosted_elsewhere_is_not_copied_out(self, tmp_path, monkeypatch):
        """--preview-url points at someone else's file, so shipping ours would be litter."""
        from website import build_website as B
        run_dir, content_file = make_run_dir(tmp_path)
        out = tmp_path / "site"
        out.mkdir()
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, out)
                            + ["--site-url", "https://x.test/",
                               "--preview-url", "https://cdn.test/card.png"])
        B.main()
        assert not (out / "preview.png").exists()
        assert '"https://cdn.test/card.png"' in (out / "index.html").read_text(encoding="utf-8")

    def test_a_local_build_ships_nothing_beside_the_page(self, tmp_path, monkeypatch):
        """No site URL, no deploy: the file that opens from disk stands alone.

        Including its tab icon — the icons are inlined, not copied out, so adding them did
        not add a second file to carry. That is the half of this test that would break if
        someone ever "fixed" the favicon by writing one next to the page.
        """
        from website import build_website as B
        run_dir, content_file = make_run_dir(tmp_path)
        out = tmp_path / "local"
        out.mkdir()
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, out))
        B.main()
        assert [p.name for p in out.iterdir()] == ["index.html"]
        html = (out / "index.html").read_text(encoding="utf-8")
        assert html.count("<link rel='icon'") == len(B.FAVICONS)

    def test_no_dad_run_exits_with_guidance(self, tmp_path, monkeypatch):
        from website import build_website as B
        monkeypatch.setattr("sys.argv", ["build_website.py", "--out-dir", str(tmp_path)])
        with pytest.raises(SystemExit, match="--dad-run"):
            B.main()

    def test_missing_audit_report_exits_with_guidance(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path)
        (run_dir / "audit" / "audit_report.json").unlink()
        with pytest.raises(SystemExit, match="audit_dad.py"):
            D.load_inputs(run_dir)

    def test_loads_real_run_shaped_inputs(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path)
        kwargs = D.load_inputs(run_dir)
        assert kwargs["audit"]["n_prompts"] == 2
        assert kwargs["baseline"][0]["prompt_id"] == "AW-0001"
        assert kwargs["diversity"]["vendi"]["score"] == 5.15
        assert "content" not in kwargs  # the page owns one content namespace
        # The cost log and the dealt-scenario file are deliberately not read: nothing
        # renders them since the cost figures came off the page.
        assert "costs" not in kwargs and "deals" not in kwargs and "corpus" not in kwargs

    def test_every_loaded_key_is_a_blocks_parameter(self, tmp_path):
        """page.py splats these straight into dad.blocks(), so a key added to the loader
        without a matching parameter is a TypeError at build time."""
        import inspect
        run_dir, _ = make_run_dir(tmp_path)
        params = set(inspect.signature(D.blocks).parameters)
        assert set(D.load_inputs(run_dir)) <= params

    def test_the_lineage_is_joined_and_trimmed(self, tmp_path):
        """step1 is keyed by scenario_id and everything after it by prompt_id, so
        dilemmas.jsonl is the join. And the library gloss is trimmed on the way in:
        scopes.jsonl repeats the whole reasoning library per case."""
        run_dir, _ = make_run_dir(tmp_path)
        lin = D.load_inputs(run_dir)["lineage"]["AW-0001"]
        assert lin["scenario_id"] == "S-001"
        assert lin["description"] == "A county fair contract is up for renewal."
        assert lin["cards"]["user_attitude"] == "unaware"
        assert "taxa_subcategory" not in lin["cards"]  # it was null in the deal
        assert lin["scope"]["patients"] == "the fair's ponies"
        assert lin["entries"][0] == {"id": "C2", "category": "Conduct", "claim": "Surface it."}

    def test_the_lineage_falls_back_to_the_audits_scenario_gid(self, tmp_path):
        """A run that kept no dilemmas file can still be joined: the audit's gid_map
        carries the scenario gid, and scenarios.jsonl carries it too."""
        run_dir, _ = make_run_dir(tmp_path)
        (run_dir / "step1" / "dilemmas.jsonl").unlink()
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["gid_map"]["AW-0001"]["scenario"] = "S-0138"
        (run_dir / "audit" / "audit_report.json").write_text(json.dumps(audit), encoding="utf-8")
        lin = D.load_inputs(run_dir)["lineage"]["AW-0001"]
        assert lin["description"] == "A county fair contract is up for renewal."

    def test_a_run_without_step_files_yields_an_empty_lineage(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path)
        (run_dir / "step1" / "dilemmas.jsonl").unlink()
        (run_dir / "step1" / "scenarios.jsonl").unlink()
        (run_dir / "step2" / "scopes.jsonl").unlink()
        lin = D.load_inputs(run_dir)["lineage"]
        assert all(not v for v in lin.values()) or lin == {}
