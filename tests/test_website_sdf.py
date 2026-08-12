"""Tests for website/sdf.py — the document dataset's section of the handoff page.

The section never renders alone, so every test here builds the whole page around it
(website/page.py owns the shell) and asserts on the ``#sdf`` beats.

The same six risks the difficult-advice report is covered against, because it is the same
artefact under the same rules:

  * **Degradation.** Four of this report's inputs — the compliance pass, card fidelity, the
    blind realism rerun and the Vendi curve — are not written by ``evals/audit_sdf.py`` at
    all, and only some committed runs carry them. Every block that reads one must render a
    complete section from a partial run and NAME the artefact it wanted rather than quietly
    omitting it. The page must also build with no ``--sdf-run`` whatsoever.
  * **Candour.** The caveats a reader sees are authored and general, so they carry no run
    figures; what the run's audit flagged is derived, and survives into the appendix even
    with the caveats prose emptied. ``evals/audit_sdf.py`` only PRINTS its verdicts, so
    every threshold here is re-applied from the eval and each rule is pinned to it.
  * **Not leading with the judge, and not documenting deployment.** No figure of any kind
    appears outside the appendix, and nothing explains how to run the pipeline. Demotion is
    not deletion: the judge's numbers are all still there, in a drawer.
  * **The lineage.** The worked example is assembled from the run's own stage files, so each
    stage either renders or names the file it wanted, and the pin lives in the prose file.
  * **The loader.** ``read_matrix`` and ``read_attrition`` parse the run's own snapshot, and
    both are the source of figures the prose interpolates.
  * **Self-containment and colour** are asserted once, on the page, in test_website_page.py.

Fully offline — the generator touches no network and no API.
"""

import json
import re

import pytest

from website import common as C
from website import page as P
from website import sdf as S

# --- fixtures, shaped like the real audit JSON --------------------------------

AUDIT = {
    "n_docs": 100,
    "composition": {"tone": {"neutral or journalistic": 60, "skeptical or displeased": 40},
                    "centrality": {"the central subject of the document": 100},
                    "language": {"English": 70, "German": 30},
                    "n_types": 15, "top_type_share": 0.13},
    "length": {"median_chars": 7000, "truncated": 1, "truncated_frac": 0.01},
    "markdown": {"**bold**": 0.0},
    "near_dups": {"0.9": 0.0},
    "names": {"repeated": [["Nueva York", 3]]},
    "phrases": {"banned_hits": {}, "recurring_5grams": [["welfare of animals and of", 12]]},
    "openings": {"formulaic_frac": 0.0},
    "patterns": [{"pattern": "Retrospective exposé", "kind": "structural", "prevalence": 0.01,
                  "is_defect": True, "flagged": False}],
    "principle_coverage": {"rated": 40, "floor": 0.05,
                           "by_principle": {"1": 0.4, "2": 0.01}, "starved": [2]},
}
DIVERSITY = {"n_records": 100, "embed_model": "text-embedding-3-small",
             "vendi": {"score": 22.58}, "nn": {"over_0.90": 0.0, "over_0.80": 0.02},
             "scopes": {"combined": {
                 "n": 100, "nn_sims": [0.61, 0.79], "vendi_ratio": 0.23,
                 "over": {"0.90": 0.0, "0.80": 0.02},
                 "clusters": {"k": 2, "evenness": 0.9, "largest_share": 0.51,
                              "sizes": [51, 49]}}},
             "top_pairs": [{"a": "matrix_000001", "b": "matrix_000002", "similarity": 0.81,
                            "a_snippet": "The council heard"}]}
COMPLIANCE = {"judged": 100, "clean_documents": 98, "clean_frac": 0.98, "prevalence_flag": 0.1,
              "by_mode": {"6": {"title": "Misrepresentation of sentience", "present": 1,
                                "applicable": 90, "share_of_applicable": 0.011,
                                "share_of_judged": 0.01}},
              "findings": [{"doc_id": "matrix_000005", "mode": 6,
                            "mode_title": "Misrepresentation of sentience",
                            "evidence": "claims the fish felt nothing"}]}
FIDELITY = {"judged": 99, "clean": 60, "clean_frac": 0.606,
            "by_card_frac": {"document_type": 0.97, "resolution": 0.657, "centrality": 0.848}}
ABLATION = {"n": 78, "layer5_mean": 8.487, "blind_same_rubric_mean": 5.782, "mean_drop": 2.705}
CURVE = {"points": [[50, 18.1], [100, 22.6]], "proj": {"1000": {"power": 58.3, "log": 50.9}}}
MANIFEST = {"created_at": "2026-07-25T15:57:17", "git_commit": "4abd78b", "git_dirty": True,
            "config": {"backend": "claude_code", "model": "claude-sonnet-5",
                       "sdf": {"min_score_threshold": 7, "rewrite_model": "claude-opus-5"}}}

CORPUS = [
    {"doc_id": "matrix_000001", "language": "English", "type_name": "a news article",
     "register": "neutral or journalistic",
     "variables": {"document_type": "a news article", "tone": "neutral or journalistic",
                   "domain": "pest control", "culture": None},
     "description": "A regional paper reports a council decision.",
     "content": "The council heard the assistant had raised the point once and moved on.",
     "scores": {"alignment": 9, "realism": 8, "spec_conformance": 9,
                "notes": "Calibrated throughout."}},
    {"doc_id": "matrix_000002", "language": "German", "type_name": "a blog post",
     "description": "A blog post about a hatchery.", "content": "Ein zweites Dokument.",
     "scores": {"alignment": 8, "realism": 8, "spec_conformance": 9, "notes": ""}},
    {"doc_id": "matrix_000003", "language": "English", "type_name": "an academic paper",
     "description": "A paper.", "content": "A third document.",
     "scores": {"alignment": 9, "realism": 9, "spec_conformance": 9, "notes": ""}},
]
LINEAGE = {"matrix_000001": {
    "cards": {"document_type": "a news article", "tone": "neutral or journalistic",
              "domain": "pest control"},
    "planning": "Five scenarios were considered; the second was chosen.",
    "description": "A regional paper reports a council decision.",
    "draft": "The council heard the assistant raise the point and then moved on.",
    "review": "1. Reasoning asserted, not shown."}}
SCORES = [{"doc_id": "matrix_000001", "scores": {"alignment": 9, "realism": 8,
                                                 "spec_conformance": 9, "notes": ""}},
          {"doc_id": "matrix_000002", "scores": {"alignment": 8, "realism": 8,
                                                 "spec_conformance": 9, "notes": ""}},
          {"doc_id": "matrix_000004", "scores": {"alignment": 5, "realism": 5,
                                                 "spec_conformance": 5,
                                                 "notes": "Parse error."}},
          {"doc_id": "matrix_000005", "scores": {"alignment": 9, "realism": 6,
                                                 "spec_conformance": 8, "notes": "Thin."}}]
ATTRITION = {"dealt": 100, "planned": 100, "incoherent": 2, "drafted": 98, "rewritten": 97,
             "scored": 97, "shipped": 95}
MATRIX = {"document_type": {"a news article": 0.10}, "culture": {"the United States": 0.2},
          "tone": {"neutral or journalistic": 0.4, "skeptical or displeased": 0.4},
          "centrality": {"the central subject of the document": 0.4}}

INPUTS = {"audit": AUDIT, "diversity": DIVERSITY, "compliance": COMPLIANCE,
          "fidelity": FIDELITY, "ablation": ABLATION, "curve": CURVE, "manifest": MANIFEST,
          "corpus": CORPUS, "lineage": LINEAGE, "scores": SCORES, "attrition": ATTRITION,
          "matrix": MATRIX, "n_prompt_templates": 4,
          "run_id": "2026-07-25_15-57_fullscale-500-opus5"}

DAD_INPUTS = {"audit": {"n_prompts": 2}, "manifest": {"config": {"backend": "api"}},
              "run_id": "2026-07-20_20-51_x"}


def content(**overrides):
    base = {k: f"Prose for {k}." for k in P.CONTENT_IDS + S.CONTENT_IDS}
    base.update({k: f"Prose for {k}." for k in _dad_ids()})
    base["title"] = "Two datasets"
    base["sdf_example_pick"] = "matrix_000001"
    base["sdf_example_extra"] = "matrix_000002 matrix_000003"
    base.update(overrides)
    return base


def _dad_ids():
    from website import dad as D
    return D.CONTENT_IDS


def build(**kwargs):
    """Build the whole page around this SDF run. The section never renders alone."""
    page_content = kwargs.pop("content", None) or content()
    example = kwargs.pop("example", None)
    inputs = {**INPUTS, **kwargs} if kwargs.get("_merge", True) else kwargs
    inputs.pop("_merge", None)
    return P.build(content=page_content, sdf_inputs=inputs, dad_inputs=DAD_INPUTS,
                   sdf_example=example)


def section(html):
    """Just the #sdf panel. It is the first report on the page: synthetic documents
    comes first throughout."""
    return html[html.index("<section id='sdf'"):html.index("<section id='dad'")]


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


_RUN_TEXT = re.compile(r"<blockquote\b.*?</blockquote>|<div class='resp'>.*?</div>", re.S)


def without_run_text(html):
    """The run's own documents, which the page quotes verbatim, removed. Substring
    assertions about the PAGE have to exclude them, or a document that happens to contain
    "None" fails a test about the generator."""
    return _RUN_TEXT.sub(" ", html)


def summary(html, label):
    """A drawer's whole <summary>, including the <span class='sum-m'> meta inside it."""
    return re.search(rf"<summary>{re.escape(label)}.*?</summary>", html, re.S).group(0)


def beat(html, anchor):
    """One beat's body: after its own <h3> and before the next one.

    Slicing on ``index("id='sdf-weak'")`` keeps the tail of its own opening tag and the head
    of the next beat's ``<h3``, and that stray ``3`` passes any assertion about digits.
    """
    sec = section(html)
    start = sec.index(f"<h3 id='{anchor}'")
    body = sec[sec.index(">", start) + 1:]
    nxt = body.find("<h3 id=")
    return body if nxt == -1 else body[:nxt]


def make_run_dir(tmp_path, **files):
    """A run directory on disk, for the loader tests. Anything not named is left out, so
    a test can assert on what happens when an artefact is missing."""
    run_dir = tmp_path / "runs" / "2026-07-25_15-57_fullscale-500-opus5"
    for sub in ("audit", "final", "layer12", "layer3", "layer4", "layer5",
                "inputs/prompts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    for rel, payload in files.items():
        path = run_dir / rel.replace("__", "/")
        if isinstance(payload, list):
            path.write_text("\n".join(json.dumps(r) for r in payload), encoding="utf-8")
        elif isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


# --- the skeleton -------------------------------------------------------------

class TestSkeleton:
    def test_the_beats_render_in_order_under_their_own_ids(self):
        found = re.findall(r"<h3 id='(sdf-[^']+)'>([^<]*)</h3>", section(build()))
        assert found == list(S.BEATS)

    def test_the_stages_come_before_the_example_that_walks_them(self):
        """The chooser promises a walkthrough, and a walk needs its steps named first."""
        sec = section(build())
        assert sec.index("id='sdf-built'") < sec.index("id='sdf-example'")

    def test_the_stages_are_named_the_same_twice(self):
        """"How it is built" and the worked example use one vocabulary for one pipeline —
        their ids differ because the rail links to both."""
        sec = section(build())
        built = re.findall(r"<h4 id='sdf-built-(stage\d)'>([^<]*)</h4>", sec)
        walked = re.findall(r"<h4 id='sdf-example-(stage\d)'>([^<]*)</h4>", sec)
        assert built == walked
        assert [k for k, _ in built] == ["stage1", "stage2", "stage3", "stage4"]

    def test_the_flow_is_a_schematic_not_a_chart(self):
        """Nothing in it is proportional to a measurement, so it takes no series colour
        and no status colour."""
        svg = re.search(r"<svg [^>]*class='flow'.*?</svg>", section(build()), re.S).group(0)
        assert "--series-" not in svg and "--good" not in svg and "--bad" not in svg
        assert "aria-label=" in svg and "<title>" in svg

    def test_there_is_no_control_arm_in_the_diagram(self):
        """The other pipeline's flow has a dashed spur for its control. This one has none,
        because this pipeline has none, and drawing one would invent a stage."""
        svg = re.search(r"<svg [^>]*class='flow'.*?</svg>", section(build()), re.S).group(0)
        assert "flow-arm" not in svg


# --- not leading with the judge ----------------------------------------------

class TestNotLeadingWithTheJudge:
    def test_no_figure_appears_outside_the_appendix(self):
        """The page is the process and the records. Every chart is in a drawer a reader
        opens on purpose."""
        sec = section(build())
        assert "<figure" in sec, "the charts have to still be on the page somewhere"
        assert "<figure" not in sec[:sec.index("id='sdf-appendix'")]

    def test_the_gate_and_the_judge_are_counted_separately(self):
        """A scoring call whose JSON failed to parse is checkpointed at 5/5/5 and then
        fails the gate, so "the gate dropped four" and "the judge rejected three" are
        different numbers. Conflating them overstates the gate."""
        g = S.gate(SCORES, 7)
        assert g["scored"] == 4 and g["parse_errors"] == 1
        assert g["dropped"] == 2 and g["rejected"] == 1 and g["graded"] == 3

    def test_a_gate_that_rejects_nothing_it_graded_is_a_bad_row(self):
        clean = [{"scores": {"alignment": 9, "realism": 9, "spec_conformance": 9,
                             "notes": ""}}] * 3
        clean = clean + [{"scores": {"alignment": 5, "realism": 5, "spec_conformance": 5,
                                     "notes": "Parse error."}}]
        rows = S.derived_warnings({}, None, {"threshold": 7}, scores=clean)
        assert any(sev == "BAD" and "rejected nothing" in w for sev, w in rows)

    def test_the_page_does_not_explain_how_to_run_the_pipeline(self):
        text = strip_tags(without_run_text(section(build())))
        for gone in ("pip install", "config.yaml", "--config", "sdf_pipeline/run.py"):
            assert gone not in text, gone


# --- candour ------------------------------------------------------------------

class TestCandour:
    def test_a_dirty_or_unfaithful_backend_is_surfaced(self):
        rows = S.derived_warnings(AUDIT, MANIFEST, S.facts(AUDIT, DIVERSITY, MANIFEST))
        assert any("claude_code" in w for _, w in rows)


# --- the thresholds are the eval's ------------------------------------------

class TestDerivedThresholds:
    """``evals/audit_sdf.py`` prints its verdicts instead of recording them, so every rule
    here mirrors one of the eval's own and is pinned to it."""

    def facts(self, audit):
        return S.facts(audit, None, MANIFEST)

    @pytest.mark.parametrize("frac,verdict", [(0.0, None), (0.01, "OK"), (0.05, "BAD")])
    def test_truncation(self, frac, verdict):
        audit = {"n_docs": 100, "length": {"truncated": int(frac * 100), "truncated_frac": frac}}
        rows = [r for r in S.derived_warnings(audit, None, self.facts(audit))
                if "truncated" in r[1]]
        assert [r[0] for r in rows] == ([verdict] if verdict else [])

    @pytest.mark.parametrize("share,verdict", [(0.10, None), (0.20, "OK"), (0.40, "BAD")])
    def test_top_document_type_share(self, share, verdict):
        audit = {"n_docs": 100, "composition": {"top_type_share": share}}
        rows = [r for r in S.derived_warnings(audit, None, self.facts(audit))
                if "largest document type" in r[1]]
        assert [r[0] for r in rows] == ([verdict] if verdict else [])

    @pytest.mark.parametrize("frac,verdict", [(0.0, None), (0.20, "OK"), (0.50, "BAD")])
    def test_markdown_bold_is_the_strongest_synthetic_tell(self, frac, verdict):
        audit = {"n_docs": 100, "markdown": {"**bold**": frac}}
        rows = [r for r in S.derived_warnings(audit, None, self.facts(audit))
                if "markdown bold" in r[1]]
        assert [r[0] for r in rows] == ([verdict] if verdict else [])

    def test_a_flagged_templating_pattern_is_a_bad_row(self):
        audit = {"n_docs": 100, "patterns": [{"pattern": "Vindication arc", "prevalence": 0.42,
                                              "is_defect": True, "flagged": True}]}
        rows = S.derived_warnings(audit, None, self.facts(audit))
        assert any(sev == "BAD" and "Vindication arc" in w for sev, w in rows)

    def test_name_reuse_follows_the_evals_own_share_of_the_corpus_rule(self):
        """GOOD while the worst repeated name is under max(2, 10% of n), OK to 20%."""
        def rows(count, n):
            audit = {"n_docs": n, "names": {"repeated": [["Elara", count]]}}
            return [r for r in S.derived_warnings(audit, None, self.facts(audit))
                    if "Elara" in r[1]]
        assert rows(5, 100) == []
        assert [r[0] for r in rows(15, 100)] == ["OK"]
        assert [r[0] for r in rows(40, 100)] == ["BAD"]

    def test_a_starved_principle_is_a_weighting_problem_not_a_document_fault(self):
        rows = S.derived_warnings(AUDIT, None, self.facts(AUDIT))
        row = next(w for sev, w in rows if "starved" in w)
        assert "weighting problem in the matrix" in row
        assert "1 constitution principle starved" in row, row

    def test_the_realism_ablation_scales_with_the_drop(self):
        def sev(drop):
            rows = S.derived_warnings({}, None, {},
                                      ablation={"n": 78, "layer5_mean": 8.5,
                                                "blind_same_rubric_mean": 8.5 - drop,
                                                "mean_drop": drop})
            return [s for s, _ in rows]
        assert sev(0.2) == [] and sev(0.8) == ["OK"] and sev(2.7) == ["BAD"]

    def test_card_fidelity_names_the_card_most_often_dropped(self):
        rows = S.derived_warnings({}, None, {}, fidelity=FIDELITY)
        sev, text = rows[0]
        assert sev == "BAD" and "resolution" in text
        assert "given the plan as the spec" in text

    def test_a_run_that_clears_everything_earns_no_rows(self):
        clean = {"n_docs": 500, "length": {"truncated": 0, "truncated_frac": 0.0},
                 "composition": {"top_type_share": 0.1}, "near_dups": {"0.9": 0.0},
                 "openings": {"formulaic_frac": 0.0}, "markdown": {"**bold**": 0.0}}
        assert S.derived_warnings(clean, {"config": {"backend": "api"}},
                                  self.facts(clean)) == []


# --- the lineage ---------------------------------------------------------------

class TestLineage:
    def test_the_stages_render_in_pipeline_order(self):
        ex = beat(build(), "sdf-example")
        order = [ex.index(k) for k in ("Dealt in code", "the writer receives",
                                       "The draft, written from the spec alone",
                                       "problems the reviewer identified",
                                       "as it ships")]
        assert order == sorted(order)

    def test_every_block_is_verbatim_from_the_run(self):
        ex = beat(build(), "sdf-example")
        for text in (LINEAGE["matrix_000001"]["planning"],
                     LINEAGE["matrix_000001"]["description"],
                     LINEAGE["matrix_000001"]["draft"],
                     LINEAGE["matrix_000001"]["review"],
                     CORPUS[0]["content"],
                     CORPUS[0]["scores"]["notes"]):
            assert text in ex, text

    def test_a_missing_stage_names_the_file_it_wanted(self):
        """A step whose artefact is missing must not disappear: the reader is being shown
        a pipeline, and a stage that renders nothing reads as a stage that did nothing."""
        bare = [{"doc_id": "matrix_000001", "content": "just the document"}]
        ex = beat(build(lineage={}, corpus=bare), "sdf-example")
        for wanted in ("layer12/prompts.jsonl", "layer12/plans.jsonl",
                       "layer3/drafts.jsonl", "layer4/rewrites.jsonl"):
            assert wanted in ex, wanted

    def test_the_shipped_record_stands_in_for_a_lost_lineage_file(self):
        """The final record carries its own dealt variables and its own spec, so a run that
        kept no layer12/ still shows both rather than naming a file for no reason."""
        ex = beat(build(lineage={}), "sdf-example")
        assert "layer12/prompts.jsonl" not in ex and "layer12/plans.jsonl" not in ex
        assert CORPUS[0]["description"] in ex
        assert "pest control" in ex

    def test_null_dealt_values_are_dropped_rather_than_rendered(self):
        """A deal with no culture has no culture, and an axis reading 'None' is a bug that
        reads as data."""
        ex = beat(build(lineage={}), "sdf-example")   # falls back to the record's variables
        assert ">None<" not in ex

    def test_the_gate_threshold_comes_off_the_run_not_a_constant(self):
        assert "ships at 7 or above" in beat(build(), "sdf-example")
        html = build(manifest={"config": {"sdf": {"min_score_threshold": 9}}})
        assert "ships at 9 or above" in beat(html, "sdf-example")

    def test_the_example_is_pinned_in_the_prose_file(self):
        """So a rebuild reproduces the same document without anyone remembering a flag."""
        assert "matrix_000002" in beat(build(), "sdf-example")            # an extra
        html = build(content=content(sdf_example_pick="matrix_000003"))
        assert CORPUS[2]["content"] in beat(html, "sdf-example")

    def test_the_cli_overrides_the_primary_only(self):
        ex = beat(build(example="matrix_000003"), "sdf-example")
        assert CORPUS[2]["content"] in ex
        assert "matrix_000002" in ex, "the extras still come from the prose file"

    def test_auto_picks_the_first_shipped_document(self):
        ex = beat(build(content=content(sdf_example_pick="auto", sdf_example_extra="")),
                  "sdf-example")
        assert CORPUS[0]["content"] in ex

    def test_a_pinned_id_this_run_never_shipped_says_so(self):
        ex = beat(build(content=content(sdf_example_pick="matrix_999999")), "sdf-example")
        assert "matrix_999999" in ex and "not in this run" in strip_tags(ex)
        assert CORPUS[0]["content"] in ex, "and it falls back rather than rendering nothing"

    def test_the_two_ways_out_sit_at_the_foot_of_the_example_and_nowhere_else(self):
        sec = section(build())
        assert sec.count("Browse the records") == 1
        assert beat(build(), "sdf-example").count("Browse the records") == 1

    def test_a_from_scratch_rewrite_says_so_rather_than_showing_three_edits(self):
        """The layer-4 template licenses a rewrite from the premise where the problems are
        structural, and on real runs it usually takes it. Presenting that as "the three
        largest changes" describes it as an edit it was not."""
        html = build(lineage={"matrix_000001": {**LINEAGE["matrix_000001"],
                                                "draft": "Completely unrelated words here."}})
        ex = beat(html, "sdf-example")
        assert "rewritten, not edited" in ex
        assert "wrote the document again" in ex

    def test_a_small_rewrite_keeps_the_three_edits_framing(self):
        assert "3 largest changes" in beat(build(), "sdf-example")


# --- degradation ---------------------------------------------------------------

class TestDegradation:
    def test_the_compliance_pass_and_pairs_table_no_longer_render(self):
        """Both cut: the failure-mode counts and the most-similar pairs are review-tool
        reading, and the appendix keeps to composition and diversity."""
        sec = section(build())
        assert "The compliance pass" not in sec
        assert "most similar pairs" not in sec

    def test_card_fidelity_no_longer_renders(self):
        """Cut at Constance's call with the rest of the matrix bookkeeping — the
        appendix talks about what survived, not about dealt-versus-shipped."""
        sec = section(build())
        assert "cards it was dealt" not in sec
        for gone in ("checks", "checks_table", "attrition_table", "_fidelity_block",
                     "_patterns_drawer"):
            assert not hasattr(S, gone), gone

    def test_the_variety_drawer_names_its_dimensions(self):
        s = summary(section(build()), "Composition and diversity")
        assert "composition" in s and "principles" in s and "meanings and topics" in s

    def test_the_composition_shares_carry_one_retargeting_footnote(self):
        """Four asterisked figures, one line: these shares are weights in the variables
        file, so a lab wanting 90% English changes that file, not the pipeline."""
        sec = section(build())
        assert sec.count("These shares are set by weights") == 1
        assert "How central the welfare thread is *" in sec

    def test_hovering_a_principle_bar_shows_the_principle(self):
        sec = section(build(principles=[("1", "Sentient beings are inside the moral "
                                              "circle")]))
        assert "Sentient beings are inside the moral circle" in sec

    def test_the_whole_section_survives_a_run_with_only_a_corpus(self):
        html = build(audit=None, diversity=None, compliance=None, fidelity=None,
                     ablation=None, curve=None, scores=None, attrition=None, matrix=None)
        sec = section(html)
        for anchor, _ in S.BEATS:
            assert f"id='{anchor}'" in sec, anchor
        assert CORPUS[0]["content"] in sec

    def test_with_no_run_at_all_the_section_says_so_and_offers_the_way_out(self):
        html = P.build(content=content(), dad_inputs=DAD_INPUTS)
        sec = section(html)
        assert "No run output was supplied" in strip_tags(sec)
        assert P.HF_SDF in sec
        assert "id='sdf-example'" not in sec

    def test_a_diversity_report_with_no_clusters_omits_the_topic_spread(self):
        """0.00 evenness is not a low score, it is a measurement that never happened
        rendered as one — an older diversity report has no cluster sizes at all."""
        import json as _json
        d = _json.loads(_json.dumps(DIVERSITY))
        del d["scopes"]["combined"]["clusters"]
        assert "Topic spread" not in strip_tags(section(build(diversity=d)))
        assert "Topic spread" in strip_tags(section(build()))


# --- facts and prose -----------------------------------------------------------

class TestFacts:
    def test_composition_is_read_from_the_field_names_the_audit_writes(self):
        """An earlier version read composition.languages/types, which no audit has ever
        written, so both figures rendered empty and nobody noticed."""
        f = S.facts(AUDIT, DIVERSITY, MANIFEST)
        assert f["n_languages"] == 2 and f["n_types"] == 15
        assert S.facts({"composition": {"languages": {"en": 1}, "types": {"a": 1}}}) \
            .get("n_languages") is None

    def test_the_only_placeholder_the_prose_gets_carries_a_degraded_default(self):
        """A run with no snapshot of its own matrix renders a sentence that survives
        without its figure, rather than a stale one or a build error."""
        assert S.facts(AUDIT, matrix=MATRIX)["matrix_clause"] == "a weighted matrix of 4 axes"
        assert S.facts(AUDIT)["matrix_clause"] == "a weighted matrix"

    def test_an_unknown_placeholder_in_the_prose_is_a_build_error(self):
        with pytest.raises(KeyError, match="unknown fact"):
            build(content=content(sdf_what="A {{nonexistent}} run."))

    def test_the_models_are_the_per_stage_overrides_not_just_the_global(self):
        """The manifest's top-level `model` reads claude-sonnet-5 even on runs whose
        rewrite — the quality-critical stage — was Opus."""
        assert S.facts(AUDIT, None, MANIFEST)["models"] == "claude-opus-5, claude-sonnet-5"

    def test_the_threshold_falls_back_when_the_manifest_has_none(self):
        assert S.threshold(MANIFEST) == 7
        assert S.threshold({}) == S.DEFAULT_THRESHOLD

    def test_the_score_histogram_covers_every_value_the_judge_gave(self):
        assert S.score_hist(SCORES, "realism") == [(5, 1), (6, 1), (8, 2)]


# --- the loader ----------------------------------------------------------------

class TestLoader:
    def test_the_matrix_is_parsed_from_the_runs_own_snapshot(self, tmp_path):
        """The axis count in the prose and the dealt shares in the appendix are properties
        of THIS run, not of the repository's current matrix."""
        run = make_run_dir(tmp_path, **{"inputs__prompts__variables.txt":
                                        "# a comment\n"
                                        "{tone}  # register\n"
                                        "    0.4 :: neutral or journalistic\n"
                                        "    # a divider\n"
                                        "    0.6 :: skeptical\n"
                                        "{naming}\n"
                                        "    plain value\n"})
        m = S.read_matrix(run)
        assert m == {"tone": {"neutral or journalistic": 0.4, "skeptical": 0.6},
                     "naming": {"plain value": None}}

    def test_a_run_with_no_snapshot_gives_an_empty_matrix_not_a_crash(self, tmp_path):
        assert S.read_matrix(make_run_dir(tmp_path)) == {}

    def test_attrition_is_counted_from_the_stage_files(self, tmp_path):
        """No stage records what it dropped, so the yield has to be counted."""
        run = make_run_dir(tmp_path, **{
            "layer12__prompts.jsonl": [{"prompt_id": f"m{i}"} for i in range(4)],
            "layer12__plans.jsonl": [{"prompt_id": "m0", "incoherent": True},
                                     {"prompt_id": "m1"}, {"prompt_id": "m2"},
                                     {"prompt_id": "m3"}],
            "layer3__drafts.jsonl": [{"doc_id": "m1"}, {"doc_id": "m2"}],
            "final__sdf_corpus.jsonl": [{"doc_id": "m1"}]})
        a = S.read_attrition(run)
        assert a == {"dealt": 4, "planned": 4, "drafted": 2, "shipped": 1, "incoherent": 1}
        assert "rewritten" not in a, "a stage that did not run is absent, not zero"

    def test_the_lineage_joins_the_stage_files_on_one_id(self, tmp_path):
        run = make_run_dir(tmp_path, **{
            "layer12__plans.jsonl": [{"prompt_id": "m1", "variables": {"tone": "neutral"},
                                      "plan": "<document_planning>notes</document_planning>"
                                              "<document_description>spec"
                                              "</document_description>",
                                      "description": "spec"}],
            "layer3__drafts.jsonl": [{"doc_id": "m1", "content": "draft"}],
            "layer4__rewrites.jsonl": [{"doc_id": "m1", "review": "problems"}]})
        lin = S.read_lineage(run)["m1"]
        assert lin["cards"] == {"tone": "neutral"}
        assert lin["planning"] == "notes", "the tags the template asked for are stripped"
        assert lin["draft"] == "draft" and lin["review"] == "problems"

    def test_a_stage_the_run_never_wrote_leaves_its_key_absent_not_none(self, tmp_path):
        """A renderer tests membership, so it can name the artefact it wanted instead of
        printing 'None'."""
        run = make_run_dir(tmp_path, **{
            "layer12__plans.jsonl": [{"prompt_id": "m1", "description": "spec"}]})
        lin = S.read_lineage(run)["m1"]
        assert "draft" not in lin and "review" not in lin and "cards" not in lin

    def test_planning_notes_fall_back_to_the_whole_response(self, tmp_path):
        """A run whose planner dropped the tags should show what it actually wrote."""
        assert S._planning_notes("no tags here") == "no tags here"

    def test_load_inputs_reads_a_partial_run_without_raising(self, tmp_path):
        """Unlike the other report's loader, a missing audit is not fatal here: the page
        must build from a DAD run alone."""
        run = make_run_dir(tmp_path)
        kwargs = S.load_inputs(run)
        assert kwargs["audit"] is None and kwargs["corpus"] == []
        assert kwargs["run_id"] == run.name
