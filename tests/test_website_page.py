"""Tests for website/page.py — the handoff page that carries both corpora.

This replaces test_report_hub.py: there is no landing page and no second file any more,
so the risks that page had (a link to a report nobody built, a card whose numbers
disagree with the report it points at) are gone. What replaces them:

  * **The choice.** Neither report is open on load; ``#dad`` / ``#sdf`` in the URL
    opens one, which is what the dataset card's deep links depend on, and printing
    expands both.
  * **Degradation.** The page must build from a DAD run alone, with the synthetic
    documents' column and report saying so.
  * **Candour before evidence.** The five shared caveats, including the licence TODO,
    render above both report sections.
  * **Brevity.** The page exists because a reader has forty seconds. Deks are rationed
    and the prose has a ceiling, both asserted here.

Fully offline.
"""

import inspect
import re

import pytest

from website import common as C
from website import dad as D
from website import page as P
from website import render as R
from website import sdf as S

CONTENT = {k: f"Prose for {k}." for k in P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS}
CONTENT["title"] = "Two corpora"
CONTENT["example_pick"] = "auto"

DAD_AUDIT = {
    "n_prompts": 40,
    "valuable_welfare_considerations": {"available": True,
                                        "parent": {"pipeline": 17.07, "plain": 12.54}},
    "moral_patient_reasons": {"n": 39, "model": "claude-sonnet-5", "judge_model": "claude-opus-5",
                              "pipeline": {"n": 39}, "plain": {"n": 39}},
    "delivery": {"pipeline_mean": 7.03, "plain_mean": 7.85,
                 "per_case": {"AW-0001": {"pipeline": {"score": 7}, "plain": {"score": 8}}}},
    "response_lengths": {"n": 39, "mean_ratio": 1.56},
}
DAD_MANIFEST = {"created_at": "2026-07-20T20:51:58", "git_commit": "326e4567", "git_dirty": True,
                "config": {"backend": "bedrock", "model": "claude-sonnet-5",
                           "dad": {"constitution_rewrite_model": "claude-opus-4-8"}}}
# No costs, corpus or deals: the report shows the process and the records, so the loader
# stopped reading the cost log and the dealt-scenario file when the cost tiles and the
# per-stage cost drawer came off the page.
DAD_INPUTS = {"audit": DAD_AUDIT, "manifest": DAD_MANIFEST,
              "n_prompt_templates": 8, "run_id": "2026-07-20_20-51_bedrock-40"}

SDF_AUDIT = {"n_docs": 100,
             "composition": {"language": {"English": 29, "Mandarin Chinese": 12},
                             "n_types": 15, "top_type_share": 0.13},
             "length": {"truncated": 12, "truncated_frac": 0.12},
             "near_dups": {"0.9": 0.0},
             "openings": {"formulaic_frac": 0.0},
             "patterns": [{"pattern": "Refuse-then-alternative", "prevalence": 0.013,
                           "is_defect": True, "flagged": False}]}
SDF_MANIFEST = {"created_at": "2026-07-11T20:06:36", "git_commit": "18ede291", "git_dirty": True,
                "config": {"backend": "claude_code", "model": "claude-sonnet-5",
                           "sdf": {"rewrite_model": "claude-fable-5"}}}
SDF_DIVERSITY = {"n_records": 100, "vendi": {"score": 22.58}, "nn": {"over_0.90": 0.0},
                 "embed_model": "text-embedding-3-small",
                 "scopes": {"combined": {
                     "n": 100, "nn_sims": [0.61, 0.79], "vendi_ratio": 0.23,
                     "over": {"0.90": 0.0, "0.80": 0.02},
                     "clusters": {"k": 2, "evenness": 0.9, "largest_share": 0.51,
                                  "sizes": [51, 49]}}}}
# One shipped document with its whole trail, which is what the worked example needs. The
# draft differs from the shipped content so the rewrite drawer has something to diff.
SDF_CORPUS = [{"doc_id": "matrix_000001", "language": "English", "type_name": "a news article",
               "register": "neutral or journalistic",
               "variables": {"document_type": "a news article", "tone": "neutral or journalistic",
                             "domain": "pest control"},
               "description": "A regional paper reports a council decision.",
               "content": "The council heard the assistant had raised one point and moved on.",
               "scores": {"alignment": 9, "realism": 8, "spec_conformance": 9,
                          "notes": "Calibrated throughout."}},
              {"doc_id": "matrix_000002", "language": "German", "type_name": "a blog post",
               "content": "Ein zweites Dokument.", "description": "A blog post.",
               "scores": {"alignment": 8, "realism": 8, "spec_conformance": 9, "notes": ""}}]
SDF_LINEAGE = {"matrix_000001": {
    "cards": {"document_type": "a news article", "tone": "neutral or journalistic",
              "domain": "pest control"},
    "planning": "Five scenarios considered; the second was chosen.",
    "description": "A regional paper reports a council decision.",
    "draft": "The council heard the assistant raise a point.",
    "review": "1. Reasoning asserted, not shown."}}
SDF_SCORES = [{"doc_id": "matrix_000001", "scores": {"alignment": 9, "realism": 8,
                                                     "spec_conformance": 9, "notes": ""}},
              {"doc_id": "matrix_000003", "scores": {"alignment": 5, "realism": 5,
                                                     "spec_conformance": 5,
                                                     "notes": "Parse error."}}]
SDF_INPUTS = {"audit": SDF_AUDIT, "manifest": SDF_MANIFEST, "diversity": SDF_DIVERSITY,
              "corpus": SDF_CORPUS, "lineage": SDF_LINEAGE, "scores": SDF_SCORES,
              "attrition": {"dealt": 100, "planned": 100, "drafted": 100, "rewritten": 100,
                            "scored": 100, "shipped": 100},
              "matrix": {"tone": {"neutral or journalistic": 0.4}},
              "n_prompt_templates": 4,
              "run_id": "2026-07-11_20-06_matrix100-cli"}


def content(**overrides):
    return {**CONTENT, **overrides}


def shipped_content():
    """The prose files this repository actually publishes.

    The brevity tests measure these rather than the fixtures, because prose growing
    back is the regression they exist to catch. Loading them also pins the id contract:
    a section renamed in a module and not in its prose file fails here.
    """
    from pathlib import Path
    # Off the package's own __file__, not a path spelled out here: the directory was
    # called report/ until it was renamed, and this was one of two places outside it
    # that had to be found and changed.
    website_dir = Path(P.__file__).resolve().parent
    return C.load_content([website_dir / "content_page.md", website_dir / "content_dad.md",
                           website_dir / "content_sdf.md"],
                          P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS)


def build(**kwargs):
    kwargs.setdefault("content", content())
    kwargs.setdefault("dad_inputs", DAD_INPUTS)
    return P.build(**kwargs)


def beat(section, anchor):
    """One beat's body: after its own <h3> and before the next one.

    Slicing on ``index("id='sdf-weak'")`` looks right and is not: it keeps the tail of its
    own opening tag and the head of the next beat's ``<h3``, and that stray ``3`` passes
    any assertion about digits in a beat. Same helper as test_website_dad.py's.
    """
    start = section.index(f"<h3 id='{anchor}'")
    body = section[section.index(">", start) + 1:]
    nxt = body.find("<h3 id=")
    return body if nxt == -1 else body[:nxt]


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def _bar_rem(html, t):
    """The chooser bar's height in rem loose (t=0) or tight (t=1), from the CSS.

    Derived rather than pinned: the bar is six tokens and five interpolations, and what
    matters about it — that a deep-linked beat clears it, that the rail sits under it, that
    tightening it actually tightens it — has to follow the declarations rather than a
    number typed in here.
    """
    bar = re.search(r"\.choicebar\{[^}]*\}", html).group(0)
    btn = re.search(r"\.choice\{[^}]*\}", html).group(0)
    tok = {k: float(re.search(rf"--{k}:([\d.]+)rem", bar).group(1))
           for k in ("pad", "btn-y", "label")}
    pad_f = float(re.search(r"var\(--pad\)\*\(1 - ([\d.]+)\*", bar).group(1))
    btn_f = float(re.search(r"var\(--btn-y\)\*\(1 - ([\d.]+)\*", btn).group(1))
    label_f = float(re.search(r"var\(--label\)\*\(1 - ([\d.]+)\*", btn).group(1))
    line_h = float(re.search(r"font:650 var\(--label\)/([\d.]+)", btn).group(1))
    return (2 * tok["pad"] * (1 - pad_f * t)
            + 2 * tok["btn-y"] * (1 - btn_f * t)
            + tok["label"] * (1 - label_f * t) * line_h
            + 2 / 16)  # the buttons' 1px borders


def _rail_top_rem(html, t):
    """Where the rail pins, loose (t=0) or tight (t=1) — one interpolation off --t, so it
    tracks the bar rather than leaving a gap that grows when the bar shrinks."""
    rule = re.search(r"\.rail\{[^}]*\}", html).group(0)
    base, factor = re.search(r"top:calc\(([\d.]+)rem - ([\d.]+)rem\*var\(--t\)\)",
                             rule).groups()
    return float(base) - float(factor) * t


class TestShape:
    def test_every_comment_in_the_stylesheet_closes_the_one_it_opened(self):
        """The stylesheet is two thirds prose, and an unbalanced delimiter is silent.

        This shipped for one build: a rationale added above `#datasets::before` left a stray
        `*/` in front of the rule, so the parser read the whole thing as one invalid selector
        and DROPPED the declaration — the divider and the 3rem of air above the comparison
        both vanished, and nothing in the built HTML looked wrong. Nothing else here can
        catch that, because every other test reads the CSS as text.
        """
        css = re.search(r"<style>(.*?)</style>", build(), re.S).group(1)
        depth, i = 0, 0
        while i < len(css):
            if css.startswith("/*", i):
                assert depth == 0, f"nested /* at {css[i:i + 60]!r}"
                depth, i = 1, i + 2
            elif css.startswith("*/", i):
                assert depth == 1, f"unopened */ before {css[i + 2:i + 62]!r}"
                depth, i = 0, i + 2
            else:
                i += 1
        assert depth == 0, "a comment is left open"

    def test_the_page_is_three_sections_and_two_reports(self):
        html = build(sdf_inputs=SDF_INPUTS)
        ids = re.findall(r"<section id='([^']+)'", html)
        assert ids == ["datasets", "explore", "sdf", "dad"]
        assert re.findall(r"<h2>([^<]*)</h2>", html) == [
            "Walk through either pipeline", R.esc(S.SECTION_TITLE), R.esc(D.SECTION_TITLE)]
        # The comparison is titled by its own two mastheads on screen — and by a heading
        # a screen reader can find, because heading navigation skipped it entirely.
        assert re.match(r"<section id='datasets'><h2 class='vh'>[^<]+</h2><div class='cmp-wrap'>",
                        re.search(r"<section id='datasets'>.*", html).group(0))

    def test_the_comparison_has_a_heading_that_is_heard_and_not_seen(self):
        """Pressing H went from the page title to the chooser, past both datasets — the
        one section a reader is meant to read in a single pass had no heading at all.
        It has one now, and it is off screen: the mastheads stay the visible title."""
        html = build(sdf_inputs=SDF_INPUTS)
        head = re.search(r"<section id='datasets'>(<h2[^>]*>[^<]*</h2>)", html).group(1)
        assert "class='vh'" in head and strip_tags(head).strip()
        rule = re.search(r"\.vh\{[^}]*\}", html).group(0)
        assert "position:absolute" in rule and "clip-path:inset(50%)" in rule
        assert "display:none" not in rule, "display:none would hide it from a screen reader too"

    def test_the_page_itself_has_no_rail_but_a_report_does(self):
        """The page's own navigation is still one choice — a rail of five page-wide links
        beside a hero and a comparison is furniture, and it stays gone.

        This replaces test_there_is_no_contents_rail. What changed is the thing being
        navigated: a report is 4,000 words of records, and from inside one a reader could
        see neither its shape nor a way past the worked example. So a rail exists, scoped
        to one report, inside #explore, appearing only once that report is open.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        assert "counter(sec)" not in html
        assert "Contents" not in strip_tags(html)
        before = html[:html.index("<section id='explore'>")]
        assert "class='rail'" not in before, "a rail outside #explore is the page-wide one"
        rails = re.findall(r"<nav class='rail' data-rail='([^']+)'", html)
        assert rails == ["sdf", "dad"]

    def test_both_reports_take_the_same_skeleton(self):
        """A reader learns the shape once. Both reports carry the same beats, in the same
        order, under the same names.

        The Caveats beat was cut from BOTH sides in the external-readiness copy pass — the
        bullets restated what the pipeline openly does rather than conceding anything, so
        they read as filler where self-criticism belonged. Its absence is asserted on both
        sides rather than merely unmentioned, so restoring it to one report alone fails
        here instead of quietly splitting the skeleton.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        beats = dict(re.findall(r"<h3 id='([^']+)'>([^<]*)</h3>", html))
        assert "sdf-weak" not in beats and "dad-weak" not in beats
        for anchor, _ in D.BEATS + S.BEATS:
            assert anchor in beats, anchor
        # Same names in the same order.
        assert [t for _, t in S.BEATS] == [t for _, t in D.BEATS]

    def test_neither_report_puts_a_heading_over_its_opening_line(self):
        """Both open on a bare lede. A heading over one sentence only names what a reader
        can already see, and it costs a rail item and a hairline for nothing."""
        html = build(sdf_inputs=SDF_INPUTS)
        for pid in ("sdf", "dad"):
            panel = html[html.index(f"<section id='{pid}'"):]
            head = panel[:panel.index(f"<h3 id='{pid}-built'")]
            assert "class='lede'" in head
            assert "<h3" not in head, f"#{pid} opens on a heading"

    # A `test_the_difficult_advice_report_opens_on_what_it_is` sat here, requiring
    # `<h3 id='dad-what'>` above the lede. It was the exact opposite of
    # test_neither_report_puts_a_heading_over_its_opening_line above, which is the current
    # rule and covers the same ground — both cannot hold, and the beat is gone.

    def test_the_hero_is_the_image_the_title_and_the_lines_that_follow(self):
        """Image, title, intro, centred, and nothing else. A lede, a meta line or a set
        of tiles in here is the masthead this page was rebuilt to get rid of, and an
        "Intro" heading over the paragraph only names what a reader can already see."""
        html = build(sdf_inputs=SDF_INPUTS)
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert hero.index("class='illo") < hero.index("<h1>") < hero.index("id='intro'")
        assert "class='lede'" not in hero and "class='meta'" not in hero
        assert "class='tiles'" not in hero
        assert "<h2>Intro</h2>" not in html
        assert re.search(r"\.hero\{[^}]*text-align:center", html)
        assert "min-height" not in re.search(r"\.hero\{[^}]*\}", html).group(0)

    def test_the_hero_image_is_cropped_to_the_ink(self):
        """The artwork is 1536x1024 with its drawing in a 1318x425 band — a third of the
        file is transparent above it and a third below, so uncropped the hero spends
        ~340px on nothing and every gap under it looks wrong. Cropped in CSS, so the
        asset stays exactly as supplied."""
        rule = re.search(r"\.hero \.illo\.art img\{[^}]*\}", build()).group(0)
        assert "aspect-ratio:1318/425" in rule
        assert "object-fit:cover" in rule and "object-position:50% 48.5%" in rule

    def test_the_intro_stops_after_five_paragraphs(self):
        """Five paragraphs of intro prose and it stops: the problem, the finding, the
        sentence that introduces the pair, then what we built on it in two. A bulleted list
        here is the two DATASETS listed a second time within a screen of the comparison's
        mastheads, and `<ul>` staying out is what stops that coming back.

        The two TECHNIQUES are a different thing from the two datasets, which is why an
        `<ol>` is allowed where a `<ul>` is not — and it is a FIGURE between the second
        paragraph and the third, not a list inside one, so the count is of the intro's own
        top-level paragraphs and the pair's own blocks are not paragraphs at all."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert "<ul>" not in hero
        assert hero.count("<ol class='npair'>") == 1 and hero.count("<li>") == 2
        # Three before the pair, two after, and nothing inside it: the body and tie lines are
        # divs, so a paragraph in here is authored prose and can be counted as such.
        before, after = hero.split("<ol class='npair'>")
        assert before.count("<p>") == 3 and after.count("<p>") == 2
        assert "<p>" not in after[:after.index("</ol>")]

    def test_the_two_techniques_read_in_the_page_s_own_order(self):
        """Synthetic documents first, here as in the comparison, the chooser and the
        panels. The hero is where a reader meets the pair, so an order that disagreed with
        the rest of the page would teach the wrong one first."""
        hero = self.hero(build(content=shipped_content(), sdf_inputs=SDF_INPUTS))
        pair = re.search(r"<ol class='npair'>.*?</ol>", hero, re.S).group(0)
        assert pair.index("Synthetic document finetuning") < pair.index("Difficult advice")

    def test_the_pair_carries_no_eyebrow_over_either_name(self):
        """A small uppercase "Technique 1" over each name was tried and cut: an eyebrow
        names what the heading under it already says, and the ordinal is not information a
        reader needs. The <ol> keeps the semantics without drawing a marker."""
        hero = self.hero(build(content=shipped_content(), sdf_inputs=SDF_INPUTS))
        assert "Technique 1" not in hero and "npair-i" not in hero

    def test_the_technique_s_name_is_its_own_heading_not_the_start_of_its_sentence(self):
        """The prose file's convention: a technique block's leading bold run is its name.
        Promoted to the column's heading, so it is not also the first two words of the
        sentence under it."""
        hero = self.hero(build(content=shipped_content(), sdf_inputs=SDF_INPUTS))
        assert "<span class='npair-h'>Synthetic document finetuning</span>" in hero
        body = re.search(r"<span class='npair-h'>Synthetic document finetuning</span>"
                         r"<div class='npair-b'>(.*?)</div>", hero, re.S).group(1)
        assert "Synthetic document finetuning" not in body
        assert "<b>" not in body                        # the bold run was consumed, not kept

    def test_the_pair_is_two_bordered_boxes(self):
        """Two columns, each in a hairline box: border all the way round, 4px, 24px of
        padding. This ran as a hairline over each column and nothing else, on the reasoning
        that a box here would be the first on a page that has none and that 4px belongs to
        things you press — the two techniques are the one place the page names a pair of
        objects rather than making an argument, and they are boxed deliberately.

        FLAT, still: no fill and no shadow, so the border is the whole of it. And still an
        <ol>, so it is heard as a list of two ordered items."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        rules = "".join(re.findall(r"\.npair[^{]*\{[^}]*\}", html))
        assert rules
        for banned in ("box-shadow", "background"):
            assert banned not in rules, banned
        assert "border:1px solid var(--hairline)" in rules
        assert "border-radius:4px" in rules
        assert "padding:24px" in rules
        assert "list-style:none" in rules
        # Never a bare 1fr: a wide child would grow the track past the page.
        assert "grid-template-columns:repeat(2,minmax(0,1fr))" in rules

    def test_the_pair_stacks_on_a_phone(self):
        """Two columns inside a 390px viewport are ~16 characters each. It becomes one
        column at the same breakpoint the type scale restates at."""
        html = build()
        narrow = html[html.index("@media (max-width:620px)"):]
        assert re.search(r"\.npair\{grid-template-columns:minmax\(0,1fr\)", narrow)

    def test_stacked_it_centres_and_stays_a_pair_of_boxes(self):
        """Flush left is a property of the two-column form, not of the pair.

        Centring is wrong across two columns — it leaves four ragged edges — and that is the
        reason the pair goes flush left. Stacked there is one column with two edges, inside a
        hero whose title, both paragraphs and closing lines are all centred, so flush left
        made the stack read as a different kind of block rather than the same one narrower.
        It also comes off the screen edges, which is the air the centred prose above it has at
        the ends of its lines. The box comes with it: a stacked technique is the same object
        narrower, so the border stays and only the inner padding gives ground.
        """
        html = build()
        wide = html[:html.index("@media (max-width:620px)")]
        narrow = html[html.index("@media (max-width:620px)"):]
        assert "text-align:left" in re.search(r"\.npair\{[^}]*\}", wide).group(0)
        stacked = re.search(r"\.npair\{[^}]*\}", narrow).group(0)
        assert "text-align:center" in stacked
        assert re.search(r"padding:0 [\d.]+rem", stacked), stacked      # off the edges
        # The box is not swapped for a rule when it stacks, and nothing turns the border off.
        assert not re.search(r"\.npair[^{]*::before", html)
        assert "border" not in re.search(r"\.npair>li\{[^}]*\}", narrow).group(0)
        assert "border:1px solid var(--hairline)" in re.search(r"\.npair>li\{[^}]*\}",
                                                               wide).group(0)

    def test_the_two_rules_the_page_draws_are_both_inside_the_intro(self):
        """Under the opening claim, and under the pair of techniques. Same 240px and the
        same 48px either side, so a reader meets one rule twice rather than two rules.

        Nowhere else: one above the comparison and one above the chooser were both tried and
        both cut, because a page that rules every seam between its parts reads as ruled
        sections rather than as one piece. Each removal has to pay for the gap it carried —
        the comparison's rule was the ENTIRE space between the intro and the table, since the
        hero's bottom padding was zero — so the hero carries a bottom padding and the chooser
        carries a top margin, or each section rides up against the one above it."""
        html = build()
        wide = html[:html.index("@media (max-width:620px)")]
        assert "#datasets::before" not in html and "#explore::before" not in html
        rules = re.findall(r"\.hero-intro>[^{]*::before\{[^}]*\}", wide)
        assert len(rules) == 1, rules              # the two rules share one declaration
        assert "width:min(100%,240px)" in rules[0]
        assert "margin:48px auto" in rules[0]
        assert "border-top:1px solid var(--hairline)" in rules[0]
        assert "#explore{margin-top:6rem}" in wide
        hero = re.search(r"\n\.hero\{[^}]*\}", html).group(0)
        assert re.search(r"padding:96px 28px [\d.]+rem", hero), hero

    def test_the_rule_under_the_pair_clears_the_boxes_own_edge(self):
        """16px below the pair on top of the rule's own 48px: the boxes have a visible
        bottom edge now, and at 48px flat that edge and the rule read as a pair of lines.

        The paragraph carrying the rule is a flow-root, and that is what makes the 16px
        exist at all — the rule is its first child, so an unconstrained top margin collapses
        through it and out, and collapsed margins take the larger of the two, not the sum."""
        html = build()
        wide = html[:html.index("@media (max-width:620px)")]
        assert "display:flow-root" in re.search(r"\.hero-intro>ol\+p\{[^}]*\}", wide).group(0)
        assert re.search(r"\.npair\{[^}]*margin:[\d.]+rem 0 16px", wide)

    def test_a_bold_run_in_the_intro_takes_the_colour_of_its_paragraph(self):
        """Two emphasised phrases, marked by WEIGHT alone. A colour on them was tried and
        cut: a bold run that also changes colour reads as a link that has lost its
        underline, in a paragraph that carries a real link two lines below it."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        assert not re.search(r"\.hero-intro[^{]*\bb\{", html)
        assert self.hero(html).count("<b>") == 2

    def test_the_intro_carries_two_measures(self):
        """The paragraphs keep the measure a centred line can be read at; the container is
        wider because the pair is two columns inside it, and two columns inside 60ch are
        ~24 characters each. Widening the container must not widen the prose with it."""
        html = build()
        assert re.search(r"\.hero-intro\{max-width:min\(100%,4[0-9]rem\)", html)
        assert re.search(r"\.hero-intro>p\{max-width:60ch", html)

    @staticmethod
    def hero(html):
        return re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)

    def test_the_footer_carries_no_provenance(self):
        """Who made it and where to go, and nothing else.

        One line per run — id, commit, dirty flag, backend — was here twice, and was
        removed twice. It was restored on the argument that provenance otherwise "appeared
        nowhere", which is not so: `common.run_note()` names the run inside the report,
        where the reader who wants it is. What the footer added on top of that was a commit
        sha, a dirty flag and a backend name, none of which a reader can act on, and
        "+ uncommitted changes" on the last line of a handoff page reads as a draft.
        """
        html = build(sdf_inputs=SDF_INPUTS, dad_inputs=DAD_INPUTS)
        foot = re.search(r"<footer class='foot'>.*?</footer>", html, re.S).group(0)
        assert f"A project by <a href='{P.MAKER_URL}'" in foot
        assert P.MAKER in foot
        assert "class='foot-run'" not in foot
        text = strip_tags(foot)
        for gone in (SDF_INPUTS["run_id"], DAD_INPUTS["run_id"], "git ", "backend"):
            assert gone not in text, gone

    def test_the_only_marks_in_the_footer_name_a_destination(self):
        """The maker's own mark is gone, and so is every rule and argument that fed it.

        A 15px squircle in front of "Sentient Futures" is a picture of the name printed
        beside it — a third link idiom in a footer that had two, and the only saturated
        colour on the page's least important line. The two that stay identify a place the
        reader has not been yet.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        foot = re.search(r"<footer class='foot'>.*?</footer>", html, re.S).group(0)
        assert "<img" not in foot and "ico-img" not in foot, foot
        assert foot.count("class='ilink'") == 2
        for dead in (".ico-img{", ".maker{"):
            assert dead not in html, dead
        assert "maker_icon" not in inspect.signature(P.body).parameters

    def test_the_run_is_still_named_where_the_reader_needs_it(self):
        """The footer carrying nothing is only correct because the report carries it. If
        both stopped, the page's figures would come off a batch nobody could identify."""
        html = build(sdf_inputs=SDF_INPUTS, dad_inputs=DAD_INPUTS)
        panel = html[html.index("<section id='dad'"):]
        assert DAD_INPUTS["run_id"] in panel

    def test_the_footer_is_the_credit_then_a_split_row(self):
        """Two rows: the byline, then who made it left and where to go right.

        The split lives on `.foot-row`, which has exactly two children, and NOT on the
        footer: on the footer it had four, so the byline took a full-width line, the
        colophon and the maker split the next and the two destinations wrapped alone onto a
        third — three rows with three different alignments. The destinations are links and
        not buttons, there being nothing to press down here, only somewhere to go.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        foot = re.search(r"<footer class='foot'>.*?</footer>", html, re.S).group(0)
        assert "class='lbtn'" not in foot          # the buttons live in the comparison
        assert ">Datasets</span>" in foot and ">Pipelines</span>" in foot
        order = [foot.index(s) for s in ("foot-by", "foot-row", "foot-colophon", "foot-links")]
        assert order == sorted(order), foot
        rule = re.search(r"footer\.foot\{[^}]*\}", html).group(0)
        assert "justify-content" not in rule and "text-align" not in rule, rule
        row = re.search(r"\.foot-row\{[^}]*\}", html).group(0)
        assert "justify-content:space-between" in row, row

    def test_the_page_asks_not_to_be_indexed(self):
        """It is handed to a reader, not found.

        Unconditional, and not left to a robots.txt: a URL that is disallowed but linked
        can still be indexed by reference, and the file is also served from disk and by
        email, where there is no robots.txt to serve.
        """
        assert '<meta name="robots" content="noindex,nofollow">' in build()

    def test_the_page_carries_a_description_as_flat_text(self):
        """The one authored line that never renders in the document.

        A link preview is the whole first impression of a page nothing links to, and
        markdown in a `content` attribute is asterisks in someone's Slack.
        """
        html = build(content=content(description="Two **open** [datasets](https://x.test)."))
        assert '<meta name="description" content="Two open datasets.">' in html

    def test_the_preview_tags_need_the_hosted_url(self):
        """Where the page lives is not a property of the page — it is a property of a
        deployment, so the copy that opens from disk says nothing about it."""
        assert "og:" not in build()
        html = build(site_url="https://x.test/", preview_url="https://x.test/p.png")
        assert '<meta property="og:url" content="https://x.test/">' in html
        assert '<meta property="og:image" content="https://x.test/p.png">' in html
        assert '<meta name="twitter:card" content="summary_large_image">' in html

    def test_a_card_with_no_image_does_not_promise_one(self):
        """og:image cannot be a data URI — it is fetched out of band — so with no hosted
        image the card declares the size it can actually fill."""
        html = build(site_url="https://x.test/")
        assert "og:image" not in html
        assert '<meta name="twitter:card" content="summary">' in html

    def test_is_self_contained(self):
        """One file. Every reference in it either stays on the page (an anchor), is
        carried inside it (a data URI), or is prose pointing at the web — never a
        relative path to something that has to travel alongside.

        `<link>` used to be banned outright. The tab icon is the one exception, because a
        favicon has no other spelling — a browser will not read one out of a `<meta>` —
        so the rule is now shape-checked instead of absent: every link on the page must be
        an icon with a `data:` href, which lets nothing else (a stylesheet, a preload, a
        font) in behind it.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        assert not re.search(r"<iframe\b", html)
        for tag in re.findall(r"<link\b[^>]*>", html):
            assert re.fullmatch(
                r"<link rel='icon' sizes='\d+x\d+' href='data:image/png;base64,[^']+'>",
                tag), tag
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert "@import" not in html and "url(" not in html
        refs = re.findall(r"(?:src|href)='([^']+)'", html)
        assert refs and all(r.startswith(("data:", "#", "https://")) for r in refs)

    def test_the_hero_illustration_is_carried_inside_the_page(self):
        html = build(sdf_inputs=SDF_INPUTS, illustration="data:image/png;base64,AAAA")
        assert "<img src='data:image/png;base64,AAAA'" in html
        assert "TODO: hero illustration" not in html
        assert re.search(r"<img[^>]+alt='[^']+'", html)  # it is a picture, so it needs one

    def test_an_illustration_that_would_have_to_travel_is_refused(self):
        """A relative path renders fine in the repo and 404s the moment the file is
        emailed or published on its own, which is the whole failure this format avoids."""
        with pytest.raises(ValueError, match="data: URI"):
            build(illustration="assets/hero.png")

    def test_the_tab_icon_is_carried_inside_the_page(self):
        """Inlined for the same reason the hero is: the copy that opens from disk or
        arrives by email keeps its icon, and no new file has to travel beside the page.

        Each size is declared, because each PNG is decimated for the size it names — the
        art is hairline pencil work, and letting the browser scale one image down averages
        the ink away. `sizes=` is what stops it.
        """
        html = build(icons=[(16, "data:image/png;base64,AAAA"),
                            (32, "data:image/png;base64,BBBB")])
        assert "<link rel='icon' sizes='16x16' href='data:image/png;base64,AAAA'>" in html
        assert "<link rel='icon' sizes='32x32' href='data:image/png;base64,BBBB'>" in html

    def test_a_page_with_no_icon_asks_for_nothing(self):
        """A missing asset degrades to no tag, never to a tag pointing at nothing — the
        same shape as the hero's empty slot. A `<link rel='icon'>` with an empty href is a
        request for the page's own URL, which is not a picture."""
        assert "rel='icon'" not in build()
        assert "rel='icon'" not in build(icons=[(16, "")])

    def test_a_tab_icon_that_would_have_to_travel_is_refused(self):
        """Same guard as the hero, for the same failure."""
        with pytest.raises(ValueError, match="data: URI"):
            build(icons=[(16, "assets/favicon-16.png")])

    def test_is_light_mode_only(self):
        html = build()
        assert "color-scheme:only light" in html
        assert "prefers-color-scheme" not in html

    def test_links_out_only_to_the_repo_and_the_dataset(self):
        html = build(sdf_inputs=SDF_INPUTS)
        origins = {re.match(r"https://[^/]+", u).group(0)
                   for u in re.findall(r"href='(https?://[^']+)'", html)}
        assert origins <= {"https://github.com", "https://huggingface.co",
                           "https://alignment.anthropic.com", P.MAKER_URL}
        assert R.esc(P.HF_DAD) in html and P.HF_SDF in html and P.REPO_URL in html

    def test_every_link_that_leaves_the_page_says_so(self):
        """The arrow is the only signal a reader gets that a click ends the page."""
        html = build(sdf_inputs=SDF_INPUTS)
        for m in re.finditer(r"<a [^>]*href='(https?://[^']+)'[^>]*>(.*?)</a>", html, re.S):
            assert "class='ext'" in m.group(2), m.group(1)
        for m in re.finditer(r"<a [^>]*href='#[^']+'[^>]*>(.*?)</a>", html, re.S):
            assert "class='ext'" not in m.group(1)

    def test_selected_text_takes_the_accent(self):
        """The page's one piece of interaction colour, rather than the browser's blue."""
        html = build()
        assert re.search(r"::selection\{background:var\(--accent\);color:var\(--surface-0\)\}",
                         html)

    def test_a_link_is_marked_never_re_faced(self):
        """A link takes the typography of the text around it; the mark is the accent and the
        2px accent underline, and the rule sets no face, size or weight at all.

        It was mono 600 at .92em, on the reasoning that mono carries identity and a link is a
        thing you go and fetch. That conflated two kinds of content: a run id or a path IS a
        literal string and mono is right for it, while a work's title is language, and setting
        it in mono changed x-height and letterfit mid-sentence in every paragraph of both
        reports. Mono means a literal string now, and nothing else.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\na\{[^}]*\}", html).group(0)
        for want in ("color:var(--accent)", "text-decoration-thickness:2px",
                     "font-weight:600"):
            assert want in rule, rule
        # WEIGHT IS PART OF THE MARK — 600 against whatever the surrounding text is. FACE and
        # SIZE are what a link must never set: those come from the context it sits in.
        for gone in ("font-family", "font-size", "letter-spacing"):
            assert gone not in rule, f"{gone} re-faces the link: {rule}"
        # No per-section link style either: one link object, everywhere.
        assert ".hero-intro a{" not in html

    def test_the_buttons_are_not_dragged_along_with_the_links(self):
        """.lbtn, .choice and .tab are actions, and each sets its own font shorthand so a bare
        `a` rule cannot reach them. This matters MORE now that the `a` rule sets no face: a
        control that forgot to declare one would silently inherit whatever it sat in."""
        html = build(sdf_inputs=SDF_INPUTS)
        for cls in ("\n.lbtn{", ".choice{", ".tab{"):
            start = html.index(cls)
            rule = html[start:html.index("}", start)]
            assert "font:" in rule, rule          # its own shorthand beats the `a` rule

    def test_a_control_declares_the_serif_and_only_an_id_stays_mono(self):
        """One face for the things you press. The chooser was serif and everything else was
        mono, which is the drift this removes — the chooser becomes the model.

        `.tab` is the single exception and it is about CONTENT, not about being a control: a
        carousel tab's label is a record id, so it is set in the face this page uses for every
        other literal string.
        """
        html = build(sdf_inputs=SDF_INPUTS)

        def face(cls):
            start = html.index(cls)
            return re.search(r"font:[^;}]*", html[start:html.index("}", start)]).group(0)

        for cls in ("\n.lbtn{", ".choice{"):
            assert "var(--serif)" in face(cls), (cls, face(cls))
        assert "var(--mono)" in face(".tab{")            # a record id, not a title
        # .ilink is NOT a control — nothing to press in the footer, only somewhere to go — so
        # it declares no face at all and takes the footer's own sans, like the maker link
        # beside it. Declaring serif here left a serif 600 link next to a sans 400 one.
        start = html.index(".ilink{")
        assert "font" not in html[start:html.index("}", start)]

    def test_paper_has_no_indigo_on_it(self):
        """Print turns every accent object to ink — including the UNDERLINE and the buttons.

        Overriding `color` alone is not enough and the page did exactly that: the base `a` rule
        sets `text-decoration-color` to the accent, so a printed link was black text under an
        indigo rule, `.lbtn` declares its own accent colour so it printed indigo outright, and
        `.cite-n sup` is not an anchor, so a bare `a` override never reached it. Invisible on
        screen, which is why it is asserted here.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        printed = html[html.index("@media print"):]
        rule = re.search(r"a,[^{]*\{[^}]*color:var\(--text-primary\)[^}]*\}", printed).group(0)
        for sel in (".lbtn", ".cite-n sup"):
            assert sel in rule, f"{sel} still prints in the accent: {rule}"
        assert "text-decoration-color:currentColor" in rule

    def test_the_palette_is_closed_even_in_the_print_block(self):
        """Every colour is a token. The printed URL's own colour was a raw `#555` — the one
        hex on the page outside the token block, and `--text-muted` is what it meant."""
        html = build()
        css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
        # Token declarations are where hexes live; anything else is a colour typed by hand.
        css = re.sub(r":root\{[^}]*\}", "", css)
        assert not re.findall(r"[:\s]#[0-9a-fA-F]{3,8}\b", css), re.findall(
            r".{30}#[0-9a-fA-F]{3,8}", css)

    def test_the_outbound_arrow_is_drawn_not_typed(self):
        """As a glyph U+2197 is a hairline in most faces and a different shape in every
        one, on a page that gets printed and screenshotted."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "&#8599;" not in html and "\u2197" not in html
        assert "<svg class='ext'" in html
        arrow = re.search(r"<svg class='ext'.*?</svg>", html, re.S).group(0)
        assert "stroke-width='2'" in arrow and "currentColor" in arrow
        assert "aria-hidden='true'" in arrow

    def test_leaving_the_page_leaves_it_in_a_new_tab(self):
        """The chooser's state lives in the URL, so a reader who follows a link out and
        comes back with the back button lands on a page that has closed itself."""
        html = build(sdf_inputs=SDF_INPUTS)
        for m in re.finditer(r"<a [^>]*href='(https?://[^']+)'([^>]*)>", html):
            assert "target='_blank'" in m.group(2), m.group(1)
            assert "rel='noopener noreferrer'" in m.group(2), m.group(1)
        for m in re.finditer(r"<a href='#[^']+'([^>]*)>", html):
            assert "target=" not in m.group(1)

    def test_the_comparison_links_at_the_pipeline_not_just_the_records(self):
        """The reader is here to run the pipeline, so each column offers the code before
        its dataset viewer. Both columns point at the repository root rather than at each
        pipeline's own prompts directory: one repository holds both, and a reader who
        wants a specific template is already inside it by then."""
        table = re.search(r"<section id='datasets'>.*?</section>",
                          build(sdf_inputs=SDF_INPUTS), re.S).group(0)
        assert table.count(">Pipeline</span>") == 2
        assert table.index(P.REPO_URL) < table.index(P.HF_SDF)  # pipeline, then data

    def test_the_buttons_are_accent_outlines_not_cream_panels(self):
        """One filled surface was doing duty as a button, a card and a code block at
        once. The controls are the accent now; the cream is just paper."""
        html = build(sdf_inputs=SDF_INPUTS)
        for cls in ("\n.lbtn{", ".choice{"):
            start = html.index(cls)
            rule = html[start:html.index("}", start)]
            assert "background:none" in rule and "var(--accent-edge)" in rule
            assert "border-radius:4px" in rule
            assert "var(--surface-1)" not in rule and "var(--surface-2)" not in rule

    def test_the_two_destinations_are_buttons_with_their_own_mark(self):
        html = build(sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        assert table.count("class='lbtn'") == 4  # a dataset and a prompts link per column
        assert R.ICONS["github"][3][:40] in html   # the published silhouette
        assert R.ICONS["hf"][3][:40] in html        # and the real Hugging Face mark
        for svg in re.findall(r"<svg class='ico'.*?</svg>", html, re.S):
            assert "aria-hidden='true'" in svg and "currentColor" in svg


class TestChooser:
    """The page asks which dataset you want and shows that one. The risks are a panel
    that cannot be reached, and a panel that cannot be found."""

    def test_nothing_is_open_until_something_is_chosen(self):
        html = build(sdf_inputs=SDF_INPUTS)
        for pid in ("dad", "sdf"):
            panel = re.search(rf"<section id='{pid}' class='panel'[^>]*>", html).group(0)
            assert "hidden" in panel
            assert f"aria-labelledby='choose-{pid}'" in panel
        assert html.count("aria-expanded='false'") == 2
        assert "aria-expanded='true'" not in html

    def test_the_buttons_are_a_disclosure_pair_not_a_tab_set(self):
        """This was `role='tablist'` with two `role='tab'`s, which promises what the
        control cannot do: a tablist always has exactly one selected tab, and nothing here
        is selected on load — that is the whole point of the chooser. A screen reader
        announced "tab, 1 of 2, not selected" twice, and the arrow keys the pattern
        requires did nothing. Two buttons that expand a region say what actually happens.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        choices = re.search(r"<div class='choices'[^>]*>.*?</div>", html, re.S).group(0)
        assert "role='tablist'" not in choices and "role='tab'" not in choices
        for pid, label in (("dad", R.esc(D.SECTION_TITLE)), ("sdf", S.SECTION_TITLE)):
            assert f"aria-controls='{pid}'" in choices
            assert f"id='choose-{pid}'" in choices
            assert label in choices
        assert choices.count("aria-expanded=") == 2
        # and the open state is driven by the same attribute the markup ships
        assert ".choice[aria-expanded=true]{background:var(--accent)" in html
        assert "b.setAttribute('aria-expanded',on?'true':'false')" in html

    def test_the_page_has_a_visible_focus_ring_on_its_buttons(self):
        """The chooser and the carousel are the only controls on the page, and `button`
        was the one selector missing from the focus rule — so they fell back to the UA
        ring, the one focus treatment nobody here designed."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"a:focus-visible[^{]*\{[^}]*\}", html).group(0)
        assert "button:focus-visible" in rule
        assert "outline:2px solid var(--accent)" in rule

    def test_the_buttons_are_two_names_and_nothing_else(self):
        """What each dataset is and how big it is are in the comparison directly above.
        Repeating both under each button made them hard to read as buttons.

        Sliced from the tablist rather than from #explore: both reports live inside
        #explore now, which is what gives the sticky bar its travel."""
        choices = re.search(r"<div class='choices'[^>]*>.*?</div>",
                            build(sdf_inputs=SDF_INPUTS), re.S).group(0)
        assert strip_tags(choices).split() == [*S.SECTION_TITLE.split(), "&darr;",
                                               *R.esc(D.SECTION_TITLE).split(), "&darr;"]

    def test_the_choice_lines_up_with_what_is_being_chosen(self):
        """At rest, 40rem centred is exactly the two dataset columns above (2 x 20rem), so
        each button sits under its own column instead of off to the left with the prose.
        It narrows from there as the bar tightens, which only moves the pair inwards."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.choicebar\{[^}]*--w:40rem", html)
        assert re.search(r"\.choices\{[^}]*width:min\(100%,calc\(var\(--w\)", html)
        assert re.search(r"\.choices\{[^}]*margin:0 auto", html)
        assert re.search(r"#explore>h2\{[^}]*text-align:center", html)
        # A descendant selector here would centre and stretch both report titles too:
        # every panel opens with its own <h2>, and the panels are inside #explore.
        assert "#explore h2{" not in html

    def test_the_instruction_does_not_set_level_with_the_reports_it_points_at(self):
        """"Walk through either pipeline" is the one h2 on the page that is not a name. At
        the h2's own 2rem it was the same size as "Synthetic documents" and "Difficult
        advice" — both h2s INSIDE this section — so the label was as loud as the thing, and
        with #datasets' heading visually hidden it was the only visible h2 before a report
        opened. It stays an <h2>, so the outline and the H key are unchanged."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"#explore>h2\{[^}]*\}", html).group(0)
        head = float(re.search(r"font-size:([\d.]+)rem", rule).group(1))
        h2 = float(re.search(r"(?m)^h2\{font:600 ([\d.]+)rem", html).group(1))
        h3 = float(re.search(r"(?m)^h3\{font:600 ([\d.]+)rem", html).group(1))
        assert head < h2 and head <= h3
        assert "<h2>Walk through either pipeline</h2>" in html

    def test_a_report_ends_where_its_content_ends(self):
        """There was a filled button here offering the other dataset, from when the
        chooser scrolled away behind the reader. The bar is pinned now, so the way across
        is on screen throughout and a second one at the foot of every report was a button
        the page did not need."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "panel-cta" not in html
        assert "class='cta'" not in html and ".cta{" not in html
        for title in (S.SECTION_TITLE, D.SECTION_TITLE):
            assert f"{title} example" not in html
        # Every button that opens a report is a tab, and both live in the bar.
        bar = re.search(r"<div class='choicebar'>.*?</div></div>", html, re.S).group(0)
        assert html.count("data-panel=") == bar.count("data-panel=") == 2

    def test_hiding_a_panel_actually_hides_it(self):
        """A panel is a <section>, and section{display:grid} beats the browser's own
        [hidden] rule, so the override has to be written down."""
        assert ".panel[hidden]{display:none}" in build()

    def test_a_printed_page_carries_both_reports(self):
        """Whichever is open on screen, a PDF of this page is the whole thing — and that
        now covers the example carousel's panes as well as the two report panels, so the
        rule is matched by selector rather than as one exact string."""
        block = build()[build().find("@media print"):]
        rule = re.search(r"([^{}]*\.panel\[hidden\][^{}]*)\{([^}]*)\}", block)
        assert rule, block[:400]
        assert "display:block!important" in rule.group(2)
        assert ".pane-x[hidden]" in rule.group(1)
        assert ".choicebar" in block  # the bar, sticky and all, does not print

    def test_the_chooser_reads_the_url_so_deep_links_survive(self):
        """The dataset card links to #dad and #sdf. Without this the link lands on a
        page with both reports closed."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "hashchange" in html
        assert "closest('.panel')" in html  # #dad-weak opens the report it lives in

    def test_the_deep_link_waits_for_the_page_to_finish_laying_out(self):
        """The hero is a multi-megabyte data URI. Scrolling to a deep-linked beat at
        parse time put the reader ~2,200px away from it once the image claimed its
        space; measured in Chromium, and fixed by deferring to load."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "window.addEventListener('load'" in html
        assert "readyState==='complete'" in html


class TestStickyBar:
    """The buttons stay on screen for as long as a report is being read, and choosing one
    puts them at the top of it. The risks are a bar with nowhere to travel, a bar that
    hides the beat a deep link just landed on, and a phone screen full of chrome."""

    def test_the_bar_and_the_reports_are_one_block_so_the_bar_can_stick(self):
        """position:sticky travels only inside its containing block, and the containing
        block of a grid item is its own grid area — one row, as tall as the buttons. The
        bar and both reports share one plain block, and that block is the travel."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "<div class='explore-body'><div class='choicebar'><div class='choices'" in html
        explore = html[html.index("<section id='explore'>"):html.index("<footer")]
        assert "<section id='sdf'" in explore and "<section id='dad'" in explore
        assert re.search(r"section>[^{]*\.explore-body[^{]*\{grid-column:text-start/full-end\}",
                         html)

    def test_the_bar_is_paper_and_the_tooltip_still_clears_it(self):
        """Its background is the page's own, so a report scrolls under it and out of
        sight rather than through it."""
        html = build()
        rule = re.search(r"\.choicebar\{[^}]*\}", html).group(0)
        for want in ("position:sticky", "top:0", "background:var(--surface-0)", "z-index:5"):
            assert want in rule, rule
        assert re.search(r"body\{[^}]*background:var\(--surface-0\)", html)
        assert re.search(r"#tip\{[^}]*z-index:9", html)  # 9 > the bar's 5

    def test_choosing_a_report_puts_the_bar_at_the_top_of_the_screen(self):
        """Measured from .explore-body, never from the bar: once sticky takes hold, the
        bar's own rect and offsetTop report where it is painted, not where it sits."""
        html = build(sdf_inputs=SDF_INPUTS)
        script = html[html.index("<script>"):]
        assert "flow.getBoundingClientRect()" in script
        assert "past=-flow.getBoundingClientRect().top" in script
        # NOTHING measures the bar — not the trigger, and not the rail's current-item pass,
        # which takes its line from each heading's own scroll-margin-top instead.
        assert "querySelector('.choicebar')" not in script
        assert "open(id,true);mark(id)" in script  # pressing a tab scrolls and marks

    def test_the_page_owns_the_smoothness_not_the_script(self):
        """An explicit behavior:'smooth' beats the CSS, so prefers-reduced-motion could
        not turn it off — which is what the old scroll-out-of-frame handler did."""
        html = build()
        assert "behavior:'smooth'" not in html
        assert "scroll-behavior:smooth" in html
        assert "scroll-behavior:auto" in html[html.index("prefers-reduced-motion"):]

    def test_a_deep_linked_beat_lands_clear_of_the_pinned_bar(self):
        """Recomputed from the tokens the bar is built out of, so retuning the buttons
        without revisiting the headroom fails here rather than in a browser. Measured
        against the bar at REST, which is the taller of its two states."""
        html = build()
        bar = _bar_rem(html, 0)
        # h4[id] is in the list because the rail links to the stages, not only the beats.
        for target in (r"h3\[id\]\{scroll-margin-top:([\d.]+)rem",
                       r"h4\[id\]\{scroll-margin-top:([\d.]+)rem",
                       r"\.panel\{[^}]*scroll-margin-top:([\d.]+)rem"):
            head = float(re.search(target, html).group(1))
            assert head > bar, f"a beat lands under the {bar:.2f}rem bar"
        # And the rail pins under the bar rather than behind it, in both of its sizes.
        for t in (0, 1):
            assert _rail_top_rem(html, t) > _bar_rem(html, t), f"the rail is under the bar"

    def test_the_bar_has_two_sizes_and_the_tight_one_is_smaller(self):
        """Loose it is right; pinned over a report it is heavy. Both states are one set of
        numbers — every dimension is an interpolation off --t — and each factor can only
        make the bar smaller. An inverted sign here would grow it over the reading column."""
        html = build()
        # --t lives on the wrapper, not the bar: the rail's top reads it too, so a
        # tightening bar does not leave a growing gap above the contents beside the report.
        assert re.search(r"\.explore-body\{--t:0;", html)   # loose by default
        assert ".explore-body.tight{--t:1}" in html         # the one thing the script sets
        for name, rule in (("bar-pad", r"\.choicebar\{[^}]*\}"), ("pair", r"\.choices\{[^}]*\}"),
                           ("button", r"\.choice\{[^}]*\}"), ("arrow", r"\.choice-a\{[^}]*\}")):
            block = re.search(rule, html).group(0)
            found = re.findall(r"\(1 - ([\d.]+)\*var\(--t\)\)", block)
            assert found, f"{name} does not interpolate off --t: {block}"
            for f in found:
                assert 0 < float(f) < 1, f"{name} factor {f} does not shrink the bar"
        # Tight has to be lighter than loose, or the change is not worth animating — but the
        # RANGE is deliberately narrow: ~72px -> ~52px. It was 83px -> 52px, and the pinned
        # size is the one measured to sit beside prose, so a resting bar 61% taller than it
        # was oversized on arrival by the design's own evidence, and the collapse read as a
        # layout event rather than the bar settling. The floor is what matters and it has not
        # moved: the coefficients were re-derived to hold the pinned size where it was.
        assert _bar_rem(html, 1) < 0.8 * _bar_rem(html, 0)
        assert 3.1 < _bar_rem(html, 1) < 3.4, "the pinned size is the measured one"
        # The pair narrows too, and its floor is measured rather than chosen: below 27.5rem
        # "Synthetic documents" wraps and the tight bar ends up TALLER than the loose one
        # (measured at 202px per button in Chromium).
        rest = float(re.search(r"--w:([\d.]+)rem", html).group(1))
        w_f = float(re.search(r"width:min\(100%,calc\(var\(--w\)\*\(1 - ([\d.]+)\*var\(--t\)",
                              html).group(1))
        assert rest * (1 - w_f) >= 27.5, f"the pair shrinks to {rest * (1 - w_f)}rem and wraps"

    def test_the_bar_crosses_once_rather_than_tracking_the_scroll(self):
        """A size that followed the scroll meant the bar moved whenever the page did, which
        reads as distraction beside prose. It crosses a trigger instead — and two of them,
        because one threshold lets a reader parked on the boundary flip a layout change back
        and forth. The script sets a flag and nothing else; the sizes are CSS."""
        script = build()[build().index("<script>"):]
        assert "querySelector('.explore-body')" in script
        assert "classList.toggle('tight'" in script
        assert "setProperty" not in script  # no per-frame value written into the bar
        tight = int(re.search(r"TIGHT=(\d+)", script).group(1))
        loose = int(re.search(r"LOOSE=(\d+)", script).group(1))
        assert 0 < loose < tight, f"the thresholds do not hold apart: {loose} then {tight}"
        assert "requestAnimationFrame(onScroll)" in script
        assert "addEventListener('scroll'" in script and "addEventListener('resize'" in script

    def test_the_animation_is_a_transition_so_reduced_motion_can_stop_it(self):
        """The animated properties are the concrete ones, not --t: a custom property is
        discrete unless it is registered, and putting the transition on padding, width and
        font-size is also what lets the page's own prefers-reduced-motion rule turn it off
        with the transition:none it applies to everything."""
        html = build()
        for rule, prop in ((r"\.choicebar\{[^}]*\}", "transition:padding"),
                           (r"\.choices\{[^}]*\}", "transition:width"),
                           (r"\.choice\{[^}]*\}", "transition:padding"),
                           (r"\.choice-a\{[^}]*\}", "transition:font-size")):
            block = re.search(rule, html).group(0)
            assert prop in block, block
        reduced = html[html.index("@media (prefers-reduced-motion:reduce){"):]
        assert "transition:none!important" in reduced[:120]

    def test_the_browser_does_not_correct_the_scroll_the_shrink_causes(self):
        """Measured: with scroll anchoring left on, tightening the pinned bar moves the
        report, the browser compensates by moving the scroll, and that moves the very
        element the trigger is measured from — so the bar settled 31px below the top of the
        screen, or bounced between its two sizes depending on where the reader stopped."""
        assert re.search(r"\.explore-body\{[^}]*overflow-anchor:none", build())

    def test_the_bar_stays_one_row_on_a_phone(self):
        """Stacked, the two buttons are ~10rem of permanently pinned chrome — a quarter
        of a small screen. Two columns and tighter type instead."""
        html = build()
        small = html[html.index("@media (max-width:760px)"):html.index("@media (max-width:620px)")]
        assert ".choices{grid-template-columns:1fr}" not in small
        # Tokens, not sizes: the breakpoint restates them and the shrink still works.
        assert re.search(r"\.choicebar\{[^}]*--label:1rem", small)
        assert re.search(r"\.choicebar\{[^}]*--btn-y:[\d.]+rem", small)

    def test_nothing_hangs_off_the_bottom_of_the_bar(self):
        """The bar carried a second row of section links for one revision and it read as
        clutter on the control. The bar is the choice; the contents are the rail."""
        html = build(sdf_inputs=SDF_INPUTS)
        bar = re.search(r"<div class='choicebar'>.*?<div class='railcol'>", html, re.S).group(0)
        assert bar.count("<a ") == 0, bar[:300]
        assert "class='beats'" not in html


class TestNarrowLayout:
    """Below 760px the page is one column. The risk is the one that actually shipped: a
    media block that collapses the grid and then re-places its children with a selector
    too weak to do it, so the page reads as one word per line on a phone."""

    def _small(self):
        html = build(sdf_inputs=SDF_INPUTS)
        return html[html.index("@media (max-width:760px)"):html.index("@media (max-width:620px)")]

    def test_the_single_column_keeps_the_named_grid_lines(self):
        """Every bleed rule on the page places its item BY NAME. Collapsing the grid to a
        bare minmax(0,1fr) deletes those names, and a placement against a name that does
        not exist resolves into a 0px implicit track."""
        rule = re.search(r"section\{grid-template-columns:([^}]*)\}", self._small()).group(1)
        for line in ("[text-start]", "text-end", "full-end"):
            assert line in rule, rule

    def test_no_bare_width_is_wider_than_a_phone(self):
        """A fixed width wide enough to overflow the narrowest track is the bug that shipped:
        the hairline above the comparison was `width:30rem`, 480px in a ~358px track on a
        390px viewport, so the whole document scrolled 122px to the right with blank paper
        beside every section — and a border-top is invisible past the edge, so nothing on
        screen said what was doing it.

        Written as the invariant rather than against that one rule, and deliberately blind to
        which block a declaration sits in: a media query is no defence, since the widest
        viewport a phone rule applies to is still a phone. Anything this wide has to be
        wrapped — `min()`, `clamp()`, a percentage — which is what every other width on the
        page already is. 320px is the narrowest viewport the page is expected to survive.
        """
        css = re.search(r"<style>(.*?)</style>", build(sdf_inputs=SDF_INPUTS), re.S).group(1)
        css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
        for value, unit in re.findall(r"[^-]width:\s*([\d.]+)(rem|px)\b", css):
            px = float(value) * (16 if unit == "rem" else 1)
            assert px < 320, f"width:{value}{unit} cannot fit a phone — cap it with min()"

    def test_the_narrow_block_does_not_re_place_children_with_a_weaker_selector(self):
        """`section>*{grid-column:1}` is (0,0,1). It loses to `section>figure` (0,0,2) and
        to `section>.explore-body` (0,1,1) — the chooser, both rails and both reports —
        which keep pointing at the names this block used to delete. Keeping the names is
        what makes the re-placement unnecessary, and !important unnecessary with it."""
        small = self._small()
        assert "section>*{grid-column:1}" not in small
        assert "!important" not in small


class TestContentsRail:
    """One report's contents, beside it, for as long as it is being read.

    A report is ~2,700 visible words of records with four beats and seven stages in it, and
    from inside one a reader could see neither its shape nor a way past the worked example.
    The risks: a link naming a heading the run never rendered, contents belonging to the
    other report, a rail with nowhere to travel, and a rail pinned behind the bar.
    """

    def rail(self, html, pid):
        return re.search(rf"<nav class='rail' data-rail='{pid}'.*?</nav>", html, re.S).group(0)

    def test_each_report_gets_its_own_contents(self):
        html = build(sdf_inputs=SDF_INPUTS)
        markup = html[:html.index("<script>")]
        assert re.findall(r"data-rail='([^']+)'", markup) == ["sdf", "dad"]
        rail = self.rail(html, "dad")
        # Three beats, the report's own. "What it is" and "Caveats" were both cut — the
        # report opens on a bare lede with no heading, so it earns no rail item, and
        # test_both_reports_take_the_same_skeleton holds the caveats beat gone on both sides.
        assert [t for _, t in re.findall(r"class='r-b' href='#([^']+)'>([^<]+)<", rail)] == [
            "The pipeline", "One example, end to end", "Appendix"]

    def test_the_stages_are_sub_items_under_the_beat_they_belong_to(self):
        """The point of the sub-items: a report's stages are where a reader is going, and
        the rail nests them under their beat rather than flattening the outline.

        This fixture ships no rewrite records, so the worked example has no stages to
        list — which is the other half of the contract, and why the beats with no anchored
        stage under them get no sub-items rather than borrowing the ones above.
        (tests/test_website_dad.py checks the example's three against a run that has them.)
        """
        rail = self.rail(build(sdf_inputs=SDF_INPUTS), "dad")
        by_beat, current = {}, None
        for kind, target in re.findall(r"class='(r-[bs])' href='#([^']+)'", rail):
            if kind == "r-b":
                current = target
                by_beat[current] = []
            else:
                by_beat[current].append(target)
        assert by_beat["dad-built"] == ["dad-built-stage1", "dad-built-stage2",
                                       "dad-built-stage3", "dad-built-control"]
        assert by_beat["dad-appendix"] == []
        # The two cut beats are not in the rail because they are not in the report: a rail
        # item naming a beat that did not render is the failure render.outline() exists to
        # prevent, so their absence is asserted here rather than assumed.
        assert "dad-weak" not in by_beat and "dad-what" not in by_beat

    def test_the_appendix_drawers_are_not_in_the_rail(self):
        """An <h4> becomes a rail item by having an id, and the appendix's headings live
        inside closed drawers — a link that scrolls to a collapsed heading goes nowhere,
        so they carry none.

        Stated as the invariant rather than against one named heading: which drawers an
        appendix renders depends on which eval artefacts the run carried, so a test that
        pins a title passes or fails on the fixture rather than on the rule.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        # Synthetic documents comes first, so its panel has to be cut at the next
        # section — an unbounded slice runs on into difficult advice's own headings.
        for pid, ends_at in (("sdf", "<section id='dad'"), ("dad", None)):
            panel = html[html.index(f"<section id='{pid}'"):]
            if ends_at:
                panel = panel[:panel.index(ends_at)]
            appendix = panel[panel.index(f"id='{pid}-appendix'"):]
            assert "<h4 id=" not in appendix          # an id is what puts it in the rail
            rail = strip_tags(self.rail(html, pid))
            for title in re.findall(r"<summary>([^<]+)", appendix):
                assert title.strip() not in rail, title

    def test_every_rail_link_lands_on_a_heading_that_rendered(self):
        """The rail is read back off the built panel rather than taken from a BEATS list,
        because the beats are conditional — the document report only earns ``sdf-weak``
        when its run's audit flagged something. A run with a clean audit must not advertise
        a beat it did not render."""
        clean = {**SDF_INPUTS, "audit": {**SDF_AUDIT, "length": {}, "composition": {}}}
        for inputs in (SDF_INPUTS, clean):
            html = build(sdf_inputs=inputs)
            for pid in ("sdf", "dad"):
                rail = self.rail(html, pid)
                panel = html[html.index(f"<section id='{pid}'"):]
                panel = panel[:panel.index("</section>")]
                for target in re.findall(r"href='#([^']+)'", rail):
                    assert (f"<h3 id='{target}'>" in panel
                            or f"<h4 id='{target}'>" in panel), target

    def test_the_contents_are_hidden_until_their_report_is_opened(self):
        """Nothing is open on load, and what shows must be the contents of what is being
        read — so the rails are toggled by the same handler that opens a panel."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert html.count("<nav class='rail' data-rail=") == 2
        assert html.count("aria-label='Sections of this report' hidden>") == 2
        script = html[html.index("<script>"):]
        assert "querySelectorAll('[data-rail]')" in script
        assert "r.getAttribute('data-rail')===id" in script

    def test_the_rail_has_somewhere_to_travel(self):
        """position:sticky moves only inside its containing block. The rail and the
        panels are one grid row each, so the rail's column stretches to the height of the
        open report and it pins for the length of it."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert ("<div class='railcol'>" in html
                and "</nav></div><div class='panels'><section id='sdf'" in html)
        assert re.search(r"\.explore-body\{[^}]*display:grid", html)
        assert re.search(r"\.railcol\{grid-column:rail-col", html)
        assert re.search(r"\.panels\{grid-column:read-col", html)
        rule = re.search(r"\.rail\{[^}]*\}", html).group(0)
        assert "position:sticky" in rule
        assert "max-height:calc(100vh" in rule and "overflow-y:auto" in rule

    def test_the_rail_moves_with_the_bar_rather_than_leaving_a_gap(self):
        """Both are pinned, so the rail's top is one interpolation off the same --t the
        bar's six sizes come from, animated the same way."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\.rail\{[^}]*\}", html).group(0)
        assert "var(--t)" in rule and "transition:top" in rule
        assert _rail_top_rem(html, 1) < _rail_top_rem(html, 0)

    def test_the_room_for_the_rail_did_not_come_out_of_the_report(self):
        """The shell widened instead, and the gutter comes out of its left margin. A rail or
        a gutter taken out of the reading column would have narrowed the 38rem measure or
        shrunk the figure track, and every chart is drawn at 800px — an 11px label in a
        narrower track is no longer 11px."""
        html = build(sdf_inputs=SDF_INPUTS)
        shell = float(re.search(r"\.shell\{max-width:([\d.]+)rem", html).group(1))
        rail = float(re.search(r"\.explore-body\{[^}]*--rail:([\d.]+)rem", html).group(1))
        gap = float(re.search(r"\.explore-body\{[^}]*column-gap:([\d.]+)rem", html).group(1))
        pull = float(re.search(r"\.explore-body\{[^}]*--pull:min\(([\d.]+)rem", html).group(1))
        prose = float(re.search(r"minmax\(0,([\d.]+)rem\) \[text-end\]", html).group(1))
        assert prose == 38
        # 3.5rem is .shell's own 28px padding either side; the pull is width the block takes
        # back from the margin, so it counts on the reading side.
        assert shell - 3.5 - rail - gap - prose + pull >= 11.5, "the figure track shrank"

    def test_the_contents_start_level_with_the_report_s_first_line(self):
        """Not with its <h2>: levelled with the title, a .8rem sans link shares a band with
        2rem serif and reads as a second heading. The datum is the lede, and every term of it
        is derived — the panel's margin, the h2's line box and its margin place the lede's
        box; the two half-leadings are the optical correction, so it is cap to cap rather
        than box to box. Hardcoding the sum goes stale the moment any of them is retuned."""
        html = build(sdf_inputs=SDF_INPUTS)
        panel = float(re.search(r"\.panel\{margin-top:([\d.]+)rem", html).group(1))
        h2 = re.search(r"(?m)^h2\{font:\d+ ([\d.]+)rem/([\d.]+)", html)
        size, lh = (float(g) for g in h2.groups())
        # The panel's own h2 margin, not the global .5rem it overrides — deriving from the
        # wrong one of the two put the rail 22px above the line it is meant to meet.
        below = float(re.search(r"\.panel>h2\{margin-bottom:([\d.]+)rem", html).group(1))
        lede = re.search(r"\.lede\{font:([\d.]+)rem/([\d.]+)", html)
        link = re.search(r"\.rail a\{[^}]*padding:([\d.]+)rem", html)
        beat = re.search(r"\.rail \.r-b\{font:\d+ ([\d.]+)rem/([\d.]+)", html)
        half = lambda m: (float(m.group(2)) - 1) * float(m.group(1)) / 2
        want = panel + size * lh + below + half(lede) - (float(link.group(1)) + half(beat))
        col = float(re.search(r"\.railcol\{[^}]*padding-top:([\d.]+)rem", html).group(1))
        rail = float(re.search(r"\.rail\{[^}]*padding:([\d.]+)rem", html).group(1))
        assert abs(col + rail - want) <= 0.15, f"{col} + {rail} is not the lede's {want}rem"
        # Below 900px the contents sit above the report, so there is no line to line up with.
        small = html[html.index("@media (max-width:900px)"):html.index("@media (max-width:760px)")]
        assert re.search(r"\.railcol\{[^}]*padding-top:0", small)

    def test_the_contents_hang_into_the_margin_only_where_there_is_one(self):
        """The pull is what keeps the reading column still while the gutter grows, and it is
        clamped against the shell's own width: on a viewport with no margin outside the
        shell it is 0 and the gutter narrows the reading column instead, rather than pulling
        the contents off the left of the page. The chooser cancels it so its centred buttons
        stay on the page's centre line."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\.explore-body\{[^}]*\}", html).group(0)
        shell = re.search(r"\.shell\{max-width:([\d.]+)rem", html).group(1)
        assert f"max(0px,(100vw - {shell}rem)/2)" in rule, rule
        assert "margin-left:calc(-1*var(--pull))" in rule
        assert "margin-left:var(--pull)" in re.search(r"\.choicebar\{[^}]*\}", html).group(0)

    def test_where_the_reader_is_takes_ink_and_an_edge_not_a_fill(self):
        """An accent FILL on this page means selected — the open tab, the open pane — and
        the reader did not press this."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\.rail a\[aria-current=true\]\{[^}]*\}", html).group(0)
        assert "border-left-color:var(--accent)" in rule
        assert "color:var(--text-primary)" in rule
        assert "background" not in rule
        script = html[html.index("<script>"):]
        # The current item is the last heading the reader has arrived at, and the line for
        # that is the heading's own scroll-margin-top — the CSS already says how far below
        # the top of the screen a linked heading lands, so the same number decides whether
        # it has been reached, and nothing has to measure the bar.
        assert "querySelectorAll('h3[id],h4[id]')" in script
        assert "getComputedStyle(el).scrollMarginTop" in script
        assert "setAttribute('aria-current','true')" in script

    def test_the_last_beat_is_current_at_the_bottom_of_the_page(self):
        """The appendix cannot reach the line the other beats reach, so the bottom decides.

        The current beat is the last heading whose top has crossed its own scroll-margin-top —
        112px. The appendix is the last beat and its drawers are closed, so there is less page
        under it than there is screen: measured at 1440x900, the difficult-advice appendix would
        need the page scrolled to 7,047px and 6,917px is as far as it goes — 130px short, and
        220px short on the documents side. The rail therefore marked a stage inside the worked
        example while the reader was looking at the appendix.

        Corrected at the bottom rather than by shortening the line, because the line is the
        CSS's own headroom for a linked heading and is right everywhere else. Measured after:
        "Appendix" at the bottom in both reports, at 900px and 1400px of viewport and on a
        phone, with the drawers closed and again with every drawer opened.
        """
        script = build(sdf_inputs=SDF_INPUTS)
        script = script[script.index("<script>"):]
        assert "document.documentElement.scrollHeight" in script
        assert "heads[heads.length-1].el.id" in script
        # It reads the live viewport and scroll, so a tall window and a phone both get it.
        assert "innerHeight+scrollY" in script

    def test_the_rail_links_are_a_control_not_the_page_s_link_treatment(self):
        """Every other link on the page is mono, bold and underlined in the accent. The
        rail measures the document rather than arguing in it, so it takes the sans — with
        its own font shorthand, which is what beats the bare a{} rule."""
        html = build(sdf_inputs=SDF_INPUTS)
        base = re.search(r"\.rail a\{[^}]*\}", html).group(0)
        assert "text-decoration:none" in base and "var(--mono)" not in base
        for cls in ("r-b", "r-s"):
            rule = re.search(rf"\.rail \.{cls}\{{[^}}]*\}}", html).group(0)
            assert "var(--sans)" in rule, rule

    def test_below_the_width_it_fits_the_rail_goes_to_the_top_of_the_report(self):
        """There is no beside on a narrow screen. It becomes a static block at the head of
        the report — where its reader is about to start — rather than a row under the bar,
        which is the thing this replaced.

        A rule under it is the one separator the contents get, and only here: beside the
        report there is none, because a fixed column of sans links is already not the prose
        next to it. Wrapped across the head of the document it needs the line."""
        html = build(sdf_inputs=SDF_INPUTS)
        small = html[html.index("@media (max-width:900px)"):html.index("@media (max-width:760px)")]
        assert ".explore-body{grid-template-columns:minmax(0,1fr)}" in small
        assert re.search(r"\.rail\{[^}]*position:static", small)
        assert re.search(r"\.rail\{[^}]*flex-wrap:wrap", small)
        assert re.search(r"\.rail\{[^}]*border-bottom:1px solid var\(--hairline\)", small)
        assert "border-right" not in re.search(r"\.railcol\{[^}]*\}", html).group(0)

    def test_the_separator_belongs_to_the_contents_and_not_to_their_column(self):
        """Nothing is drawn for contents that are not there.

        The rule and the 3.6rem above it were on ``.railcol``, which is in the markup whether
        or not a rail is inside it — and on load neither report is open, so both rails are
        hidden. Measured at 390px: a hairline right across the page under the two buttons,
        above the footer, separating nothing, on narrow screens only. ``.rail[hidden]`` is
        ``display:none``, so on the rail itself they arrive with the contents they belong to.
        """
        html = build(sdf_inputs=SDF_INPUTS)
        small = html[html.index("@media (max-width:900px)"):html.index("@media (max-width:760px)")]
        col = re.search(r"\.railcol\{[^}]*\}", small).group(0)
        for drawn in ("border", "margin", "padding-bottom"):
            assert drawn not in col, col
        assert re.search(r"\.rail\{[^}]*margin-top:", small)
        assert "[hidden]{display:none}" in re.search(r"\.rail\[hidden\]\{[^}]*\}", html).group(0)

    def test_neither_control_prints(self):
        """Paper has nothing to press and no links to follow."""
        html = build(sdf_inputs=SDF_INPUTS)
        printed = html[html.index("@media print{"):]
        assert re.search(r"#tip,\.skip,\.choicebar,\.railcol\{display:none\}", printed)


class TestTypeScale:
    def test_the_type_scale_steps_down(self):
        """There was no scale: h3 (a beat, the biggest unit in a report) was 1.1rem
        against a 1.0625rem body, and h4 (a stage) was .82rem — SMALLER than the prose
        under it — so a report read as one undifferentiated column. Each level now sits
        clear of the next, and of the body text."""
        html = build(sdf_inputs=SDF_INPUTS)

        def size(sel):
            rule = re.search(rf"(?m)^{re.escape(sel)}\{{[^}}]*\}}", html).group(0)
            return float(re.search(r"(?:font(?:-size)?:[^;}]*?)([\d.]+)rem", rule).group(1))

        body = float(re.search(r"(?m)^body\{[^}]*font:([\d.]+)rem", html).group(1))
        sizes = [size(s) for s in ("h1", "h2", "h3", "h4")]
        assert sizes == sorted(sizes, reverse=True), sizes
        for level, px in zip(("h1", "h2", "h3", "h4"), sizes):
            assert px > body, f"{level} is not larger than the prose under it"
        # A beat is also chunked off the one before it, which is what makes four beats
        # read as four rather than as one scroll.
        assert re.search(r"h3\[id\]\{[^}]*border-top:1px solid", html)

    def test_the_side_by_side_labels_stay_small_sans(self):
        """h4 became a document subhead, but in one place it is a label over a block —
        the two halves of a control-vs-pipeline pair. That keeps the old treatment."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"h4\.pane-h\{font:650 [\d.]+rem/[\d.]+ var\(--sans\)", html)

    def test_the_scale_is_restated_where_the_page_narrows(self):
        html = build(sdf_inputs=SDF_INPUTS)
        small = html[html.index("@media (max-width:620px)"):]
        for level in ("h1", "h2", "h3", "h4"):
            assert re.search(rf"[;{{}}\n]{level}\{{font-size:[\d.]+rem\}}", small), level


class TestNamedPair:
    """The component itself, apart from the hero that is its only caller today."""

    def test_nothing_to_pair_renders_nothing(self):
        assert R.named_pair([]) == ""

    def test_the_name_is_escaped_and_the_body_is_not(self):
        """The body arrives already rendered — it went through `inline_md`, so it carries
        the markup prose is allowed. The name is a plain string."""
        html = R.named_pair([("A & B", "<i>already</i> markup")])
        assert "A &amp; B" in html
        assert "<i>already</i> markup" in html


class TestComparisonTable:
    def test_the_rows_are_what_a_lab_needs_to_run_it(self):
        """Five rows, in one pass, and each says which side of the line it is on: three
        describe the output, two link out. Dates, model ids and the composition spread went
        to the report that goes into them: a reader here is deciding whether to run the
        pipeline, not shopping for a dataset.

        The `pipeline` row was cut: the two walkthroughs below ARE the pipeline, and a
        one-line chain above them was a summary met before it could mean anything."""
        html = build(sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        labels = re.findall(r"<th class='cmp-k' scope='row'>([^<]*)</th>", table)
        assert labels == ["output", "output format", "what it is for",
                          "pipeline", "example dataset"]  # the code before the data
        text = strip_tags(table)
        for gone in ("July 2026", "claude-", "domains", "taxa groups", "languages",
                     "licence"):
            assert gone not in text, gone

    def test_the_labels_are_one_line_each_flush_against_the_pair(self):
        """An index down the side of the comparison; one that wraps, or that floats away
        from the columns it indexes, stops reading as one. Both properties have to
        out-specify `.cmp th`, which sets the alignment and padding for every cell."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.cmp th\.cmp-k\{[^}]*text-align:right", html)
        assert re.search(r"\.cmp-k\{[^}]*white-space:nowrap", html)

    def test_the_last_two_rows_are_the_way_out_and_carry_no_figure(self):
        """Both link rows are button-only now. The prompt-template count used to sit in
        the `pipeline` row; it was cut with the row's deep links, so neither row states a
        number and their buttons line up under each other."""
        html = build(sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        rows = self._rows(table)
        for label in ("pipeline", "example dataset"):
            assert not re.search(r"\d", strip_tags(rows[label])), label
            # The empty figure slot stays, so the two rows' buttons stay aligned.
            assert rows[label].count("<span class='cmp-fig'><span></span>") == 2
        assert P.REPO_URL in rows["pipeline"]
        assert P.HF_SDF in rows["example dataset"]
        assert R.esc(P.HF_DAD) in rows["example dataset"]
        assert "<tfoot>" not in table

    def test_the_comparison_carries_no_record_count(self):
        """How many records a run made is a property of THAT run, and this section
        describes the pipelines; the counts belong in each report's appendix, beside the
        run they came off. The DAD figure was wrong as well as out of place: it showed
        the 40 dilemmas dealt, while the dataset behind the button beside it shipped 39.
        """
        html = build(dad_inputs=DAD_INPUTS, sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        text = strip_tags(table)
        assert "100" not in text and "40" not in text and "39" not in text
        # The row that carried them is still the way to the data, button only: the
        # figure's flex item stays, empty, so the buttons line up with the row above.
        row = self._rows(table)["example dataset"]
        assert row.count("<span class='cmp-fig'><span></span><a class='lbtn'") == 2
        assert not hasattr(P, "_records")

    @staticmethod
    def _rows(table):
        return dict(re.findall(r"<th class='cmp-k' scope='row'>([^<]*)</th>(.*?)</tr>",
                               table, re.S))

    def test_the_comparison_states_no_template_count_either_way(self):
        """It used to show one, and "—" when a run kept no prompts snapshot. The count
        went with the `pipeline` row's deep links, so neither case renders a figure."""
        for n in (8, None):
            html = build(dad_inputs={**DAD_INPUTS, "n_prompt_templates": n})
            table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
            assert "templates" not in strip_tags(table).lower()

    def test_synthetic_documents_comes_first_everywhere(self):
        """One order for the whole page: the comparison, the chooser and the panels."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        assert re.findall(r"<span class='cmp-name'>([^<]*)</span>", html) == [
            S.SECTION_TITLE, R.esc(D.SECTION_TITLE)]
        assert re.findall(r"data-panel='(\w+)' id='choose", html) == ["sdf", "dad"]
        assert html.index("<section id='sdf'") < html.index("<section id='dad'")

    def test_the_mastheads_are_the_names_and_what_each_one_is_is_a_row(self):
        """A masthead is a name, not a filename — and not a subtitle either. What each
        dataset IS used to hang under the name, unlabelled, in a table whose every other
        line said what it was answering; it is the `output` row now, so a reader can tell
        a claim about the data from a claim about the process by reading down the side."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        head = re.search(r"<thead>.*?</thead>", table, re.S).group(0)
        assert f"<span class='cmp-name'>{R.esc(D.SECTION_TITLE)}</span>" in head
        assert f"<span class='cmp-name'>{S.SECTION_TITLE}</span>" in head
        assert "cmp-d" not in html  # the subtitle slot, and its rule, are gone
        # Derived from the prose file, not typed here: this line is edited, and a hardcoded
        # copy of it fails the next time someone rewords it rather than the next time
        # someone breaks the table.
        row = strip_tags(self._rows(table)["output"])
        for key in ("dad_desc", "sdf_desc"):
            assert strip_tags(shipped_content()[key]).strip() in row
        assert "<code>dad</code>" not in head and "<code>sdf</code>" not in head

    def test_the_two_columns_are_centred_on_the_page_not_on_the_prose(self):
        """The comparison centres itself with left:50% + translateX(-50%), which is
        50% OF ITS GRID AREA. In the default `section>*` track that area is the 38rem
        prose column, and the whole table lands ~5.75rem left of the hero centred above
        it; only the full-bleed track spans the main column, whose centre is the page's.
        This is the rule that was missing."""
        html = build(sdf_inputs=SDF_INPUTS)
        bleed = re.search(r"section>figure,[^{]*\{grid-column:text-start/full-end\}", html)
        assert bleed and "section>.cmp-wrap" in bleed.group(0)
        wrap = re.search(r"\n\.cmp-wrap\{[^}]*\}", html).group(0)  # the rule, not the selector list
        assert "left:50%" in wrap and "translateX(-50%)" in wrap

    def test_the_labels_hang_off_the_left_of_the_centred_pair(self):
        """What is centred is the PAIR, not the table: the table is pushed right by
        half the wrapper minus one column minus the labels, which puts the pair's
        midpoint on the wrapper's midpoint and leaves the labels outside it, to the
        left. Stated as arithmetic rather than left to flex free space or auto margins
        to arrive at — two earlier attempts at this centred the whole table instead."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\n\.cmp\{[^}]*\}", html).group(0)
        assert "margin-left:calc(50% - var(--cmp-col) - var(--cmp-label))" in rule
        assert "width:calc(var(--cmp-label) + 2*var(--cmp-col))" in rule
        assert "margin-left:-" not in rule and "margin:0 auto" not in rule
        wrap = re.search(r"\n\.cmp-wrap\{[^}]*\}", html).group(0)
        assert "display:flex" not in wrap  # the pair's position must not depend on it

    def test_the_three_column_widths_are_set_where_fixed_layout_reads_them(self):
        """table-layout:fixed takes its widths from the first row only, so the corner
        cell has to carry the label width; .cmp-k sits in the body rows, which fixed
        layout never consults."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.cmp \.cmp-corner\{[^}]*width:var\(--cmp-label\)", html)
        assert re.search(r"\.cmp thead th\{[^}]*width:var\(--cmp-col\)", html)
        assert re.search(r"\.cmp-wrap\{--cmp-label:[\d.]+rem;--cmp-col:[\d.]+rem", html)


    def test_the_label_column_carries_no_rules(self):
        """The row rules belong to the two columns being compared; the labels are an
        index down the side, not a third column. `.cmp th` sets the border, so the
        override has to out-specify it rather than merely follow it."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.cmp th\.cmp-k\{[^}]*border-bottom:0", html)

    def test_no_sdf_run_leaves_an_honest_column(self):
        """A cell that quietly shows nothing reads as a dataset with no properties."""
        html = build()
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        assert "not published yet" in strip_tags(table)
        assert P.HF_SDF in table  # the viewer link still works

    def test_the_comparison_does_not_report_a_dealt_spread(self):
        """The dealt spread came off the page with the descriptive tiles, and the loader
        stopped reading scenario_deals.jsonl with it. The comparison's job is what a
        record is and how many there are; how wide the matrix runs is the pipeline's
        documentation, not a cell in a table read in one pass."""
        assert "domains" not in strip_tags(build(dad_inputs=DAD_INPUTS))
        assert not hasattr(D, "spread")


class TestSdfReport:
    """The document report, from the page's side. Its own risks are in
    test_website_sdf.py; what is here is the part the page is responsible for."""

    def section(self, html):
        return html[html.index("<section id='sdf'"):html.index("<section id='dad'")]

    def test_headline_figures_come_from_the_run(self):
        """No figure on this report is typed. The diversity numbers are the ones a reader
        is most likely to quote, and they arrive from the run's own report."""
        text = strip_tags(self.section(build(sdf_inputs=SDF_INPUTS)))
        assert "23.0 of 100 documents effectively distinct" in text
        assert "Vendi ratio 0.23" in text

    def test_its_weaknesses_are_derived_too(self):
        """audit_sdf.py prints its verdicts instead of recording them, so this report
        re-applies the eval's own thresholds. 12% truncated is BAD on them."""
        html = build(sdf_inputs=SDF_INPUTS)
        section = html[html.index("<section id='sdf'"):]
        rows = S.derived_warnings(SDF_INPUTS["audit"], SDF_INPUTS["manifest"],
                                  S.facts(SDF_INPUTS["audit"], SDF_INPUTS.get("diversity"),
                                          SDF_INPUTS["manifest"]))
        text = " ".join(w for _, w in rows)
        assert "12% of documents are truncated" in strip_tags(text)
        assert "claude_code" in text  # the backend provenance rule
        assert "id='sdf-appendix'" in section

    def test_a_clean_run_earns_no_flagged_rows(self):
        """A run that clears every threshold gets no drawer, rather than an empty one."""
        clean = {**SDF_INPUTS,
                 "audit": {"n_docs": 500, "length": {"truncated": 0, "truncated_frac": 0.0},
                           "composition": {"top_type_share": 0.1},
                           "near_dups": {"0.9": 0.0}, "openings": {"formulaic_frac": 0.0}},
                 "manifest": {"config": {"backend": "api"}}, "scores": None}
        assert "What the audit flags" not in self.section(build(sdf_inputs=clean))

    def test_a_flagged_templating_pattern_is_a_bad_row(self):
        audit = {**SDF_AUDIT, "patterns": [{"pattern": "Refuse-then-alternative",
                                            "prevalence": 0.42, "is_defect": True,
                                            "flagged": True}]}
        warnings = S.derived_warnings(audit, SDF_MANIFEST, S.facts(audit))
        assert any(sev == "BAD" and "Refuse-then-alternative" in w for sev, w in warnings)

    def test_composition_is_read_from_the_field_names_the_audit_writes(self):
        """An earlier version read composition.languages/types, which no audit has ever
        written, so both figures rendered empty and nobody noticed."""
        f = S.facts(SDF_AUDIT, SDF_DIVERSITY, SDF_MANIFEST)
        assert f["n_languages"] == 2 and f["n_types"] == 15
        assert S.facts({"composition": {"languages": {"en": 1}, "types": {"a": 1}}}) \
            .get("n_languages") is None

    def test_without_a_run_the_section_says_so(self):
        """The page must build from a DAD run alone. What survives is the lede, a line
        saying nothing here is measured, and the two ways out — never a beat structure
        with holes in it."""
        html = build()
        section = html[html.index("<section id='sdf'"):]
        assert "No run output was supplied" in strip_tags(section)
        assert P.HF_SDF in section
        assert "id='sdf-example'" not in section, "no worked example without a run to walk"

    def test_the_comparison_no_longer_marks_this_column_as_a_stub(self):
        """The chip said "Report in preparation" while this report was ~200 words against
        the other's ~10,000. It is written now, so the flag and the chip both went — a
        state that is no longer true must not survive as a string."""
        html = build(sdf_inputs=SDF_INPUTS, dad_inputs=DAD_INPUTS)
        heads = re.search(r"<thead>.*?</thead>", html, re.S).group(0)
        assert S.SECTION_TITLE in heads
        assert "class='cmp-s'" not in html
        assert "in preparation" not in strip_tags(html).lower()
        assert not hasattr(S, "IS_PLACEHOLDER")


class TestBrevity:
    """The page's whole brief is a reader with forty seconds, so these read the
    SHIPPED prose files rather than the fixtures."""

    def page(self):
        # The document report's worked example is overridden onto the fixture's own
        # record: the shipped prose pins a real run's doc_id, and against a fixture run
        # that renders a "not in this run" note whose 30-odd words would be counted as
        # prose and blame the ceiling for a mismatch in the fixture.
        return build(content=shipped_content(), sdf_inputs=SDF_INPUTS,
                     sdf_example=SDF_CORPUS[0]["doc_id"])

    def test_at_most_two_deks(self):
        """Every aphoristic two-beat line under a heading came out. Two is the
        allowance, so adding a third means taking one away."""
        assert self.page().count("class='dek'") <= 2

    def test_the_prose_has_a_ceiling(self):
        """The two pages this replaced carried ~3,400 words of authored prose between
        them. A regression here is prose growing back, which is the failure mode this
        page was rebuilt to fix.

        It was 1,800 while the document report was a stub. Writing that report is what
        raised it, and it is raised by one report's worth and no more — the per-report
        ceilings below are what actually hold the line, and this one only stops a THIRD
        body of prose appearing somewhere neither of them measures.
        """
        assert C.editorial_words(self.page()) < 3000

    @pytest.mark.parametrize("pid", ["dad", "sdf"])
    def test_each_report_a_reader_reads_has_its_own_ceiling(self, pid):
        """The whole-page count above is dominated by the appendices, which are closed
        drawers a reader opens on purpose. What has to stay short is the part that is
        open: each report's beats before its appendix. Two rounds of cutting took the
        difficult-advice one from 1,199 words to under 800 — dropping the results
        narrative, then the cost tiles, the commands and the run-specific caveats — and
        this is the assertion that stops them coming back one caption at a time.

        BOTH reports answer to the same number, because a reader opens one of them, not
        the page. The ceiling carries a sixth of headroom over the measured value, not the
        third the whole-page one carries, because this is the number being defended.
        """
        html = self.page()
        section = html[html.index(f"<section id='{pid}'"):]
        read_first = section[:section.index(f"id='{pid}-appendix'")]
        assert C.editorial_words(read_first) < 800

    def test_the_method_is_credited_once_where_the_reader_starts(self):
        """The Teaching Claude Why grounding was on both of the pages this replaces, and
        again inside the report. It belongs in the intro, once."""
        html = self.page()
        assert html.count("alignment.anthropic.com") == 1
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert "alignment.anthropic.com" in hero

    def test_no_link_on_the_page_has_a_number_for_its_accessible_name(self):
        """A link's ACCESSIBLE name is the name of the thing it points at.

        The visible mark may be a number — the intro's two sources are raised citation
        markers, which is what keeps the claim they hang off readable. What may not be a
        number is the name the link announces: `[1](url)` gives a screen reader "link, 1" and
        a links list a row that says nothing (WCAG 2.4.4). So a marker carries its work's name
        in `aria-label`, and that is what this checks — across the whole page, because the
        same shortcut is available anywhere prose cites something.
        """
        html = self.page()
        links = re.findall(r"<a\b([^>]*)>(.*?)</a>", html, re.S)
        assert links, "no links found — the selector is wrong, not the page"
        for attrs, inner in links:
            aria = re.search(r"aria-label='([^']*)'", attrs)
            name = aria.group(1) if aria else strip_tags(inner).strip()
            assert not re.fullmatch(r"[\d\W]*", name), f"link named {name!r}"

    def test_a_citation_marker_is_raised_numbered_and_named(self):
        """The author writes the work's name, the renderer draws the number.

        Three things have to hold together or the form is worse than the words it replaced:
        the visible mark is a numeral (so the sentence reads uninterrupted), the accessible
        name is the work (so it is not "link, 1"), and consecutive markers are separated (or
        1 and 2 side by side read as "12").
        """
        html = self.page()
        markers = re.findall(r"<a class='cite-n'[^>]*>(.*?)</a>", html, re.S)
        assert len(markers) == 2
        assert [strip_tags(m).strip() for m in markers] == ["1", "2"]   # counted, not authored
        assert "aria-label='Teaching Claude Why'" in html
        assert all("<sup>" in m for m in markers)
        # Underlined like every other link, but thinner: the brand's 2px under a ~9px numeral
        # is proportionally what 4px would be under body text.
        # ...and drawn on the <sup>, not the anchor: a decoration is positioned from its own
        # element's baseline, and the anchor's is the PARAGRAPH's, which put the underline below
        # and left of the raised numeral (measured at 6x).
        assert "text-decoration:none" in re.search(r"\.cite-n\{[^}]*\}", html).group(0)
        marker_rule = re.search(r"\.cite-n sup\{[^}]*\}", html).group(0)
        assert "text-decoration:underline" in marker_rule
        thick = float(re.search(r"text-decoration-thickness:([\d.]+)px", marker_rule).group(1))
        base = float(re.search(r"\na\{[^}]*text-decoration-thickness:([\d.]+)px",
                               html, re.S).group(1))
        assert 0 < thick < base, (thick, base)
        # The raise is stated ONCE, on the <sup>, not also on the anchor: <sup> carries the
        # UA's own super + smaller, so doing it twice shifted the digits clear of the comma.
        sup = re.search(r"\.cite-n sup\{[^}]*\}", html).group(0)
        assert "vertical-align:super" in sup and "line-height:0" in sup
        assert "vertical-align" not in re.search(r"\.cite-n\{[^}]*\}", html).group(0)
        # Each marker says it leaves the page, with a smaller-drawn version of the same
        # arrow every other outbound link carries — which is also what stops a raised numeral
        # promising a footnote that is not there.
        assert all("class='ext-c'" in m for m in markers)
        assert "stroke-width='2.4'" in html                        # heavier, to hold at 7px
        # Consecutive markers are separated, or 1 and 2 side by side read as "12". A raised
        # comma did that until the arrows arrived; they separate them now.
        assert re.search(r"\.cite-n\+\.cite-n\{margin-left:", html)
        assert "content:','" not in html

    def test_two_markers_cannot_be_split_across_two_lines(self):
        """Separated is not the same as breakable, and both markers are one word's worth.

        The arrow inside each marker is `display:inline-block` — an atomic inline, which
        UAX#14 lets a line break either side of. Measured at 390px: the intro's first
        paragraph ended on marker 1 and opened the next line with marker 2. Two halves fix
        it, one per break opportunity: a word joiner before each marker (so it cannot start
        a line, which glues it both to its word and to the marker before it) and nowrap
        inside the anchor (so the numeral cannot part from its own arrow).
        """
        html = self.page()
        assert html.count(f"{R.WORD_JOINER}<a class='cite-n'") == 2
        assert "white-space:nowrap" in re.search(r"\.cite-n\{[^}]*\}", html).group(0)

    def test_the_page_does_not_argue_a_third_route(self):
        """The belief-implantation comparison is a decision record, not something a
        reader deciding whether to use the data needs."""
        text = strip_tags(self.page()).lower()
        assert "belief implantation" not in text
        assert "third route" not in text

    def test_the_shipped_prose_files_satisfy_the_id_contract(self):
        """A section renamed in a module and not in its prose file, or a prose block
        left behind after a rename, is a build error rather than a silent hole."""
        assert set(shipped_content()) == set(P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS)

    def test_the_shipped_prose_does_not_explain_how_to_run_anything(self):
        """Installation and invocation are the repository README's job. This page is the
        process and the records."""
        prose = " ".join(shipped_content().values())
        for gone in ("pip install", "python ", "config.yaml", "--config", "$ "):
            assert gone not in prose, gone


class TestFacts:
    def test_unknown_placeholder_in_page_prose_is_a_build_error(self):
        with pytest.raises(KeyError, match="unknown fact"):
            build(content=content(intro="A {{n}}-example run."))

    def test_the_page_itself_interpolates_nothing(self):
        """Every figure on the page is rendered by a section from its run's facts, so
        the page's own prose has nothing to interpolate and must not try."""
        assert P.PAGE_FACTS == {}

    def test_a_date_survives_a_manifest_that_has_none(self):
        assert P._date({}) == "—"
        assert P._date({"created_at": "not-a-date"}) == "not-a-date"
        assert P._date({"created_at": "2026-07-01T09:00:00"}) == "1 July 2026"


class TestByline:
    """The author list and its affiliation key.

    The numbering is derived from ``P.AUTHORS`` by first appearance, so what these pin is
    the derivation, not a typed result: four of the seven authors share one institution,
    and a hand-kept key is what silently breaks when someone is added.
    """

    def _byline(self, html=None):
        html = html or build(sdf_inputs=SDF_INPUTS)
        return re.search(r"<div class='foot-by'>.*?</div>", html, re.S).group(0)

    def test_every_author_is_named(self):
        by = strip_tags(self._byline())
        for name, _ in P.AUTHORS:
            assert name in by, name

    def test_one_number_per_institution_by_first_appearance(self):
        by = self._byline()
        names = re.search(r"<p class='foot-authors'>(.*?)</p>", by, re.S).group(1)
        marked = re.findall(r"([A-Z][^,<]*)<sup title='([^']*)'>(\d+)</sup>", names)
        assert len(marked) == len(P.AUTHORS)
        # Same institution -> same number, every time it recurs.
        by_inst = {}
        for _, inst, n in marked:
            by_inst.setdefault(inst, set()).add(n)
        assert all(len(v) == 1 for v in by_inst.values()), by_inst
        # Numbered 1..N in order of first appearance, with no gaps.
        first_seen, order = [], []
        for _, inst, n in marked:
            if inst not in first_seen:
                first_seen.append(inst)
                order.append(int(n))
        assert order == list(range(1, len(first_seen) + 1))

    def test_the_key_lists_each_institution_exactly_once(self):
        key = re.search(r"<p class='foot-affil'>(.*?)</p>", self._byline(), re.S).group(1)
        listed = re.findall(r"<sup>(\d+)</sup>([^<]*)", key)
        institutions = [inst for _, inst in (*P.AUTHORS, *P.CONTRIBUTORS)]
        expected = list(dict.fromkeys(institutions))       # dedup, order kept
        assert [inst for _, inst in listed] == expected
        assert [n for n, _ in listed] == [str(i + 1) for i in range(len(expected))]
        # Many people, few institutions: the key is shorter than the author list.
        assert len(listed) < len(P.AUTHORS)

    def test_a_shared_institution_is_not_repeated_in_the_key(self):
        """The specific failure a typed key invites."""
        key = re.search(r"<p class='foot-affil'>(.*?)</p>", self._byline(), re.S).group(1)
        assert key.count("Sentient Futures") == 1

    def test_the_key_follows_the_names_so_it_reads_in_order(self):
        """A bare superscript digit means nothing alone, so the key has to come after the
        names in DOM order — that is what a screen reader reads, in that sequence."""
        by = self._byline()
        assert by.index("foot-authors") < by.index("foot-affil")

    def test_each_marker_names_its_institution_for_a_hover(self):
        by = self._byline()
        for _, inst in P.AUTHORS:
            assert f"<sup title='{inst}'>" in by, inst

    def test_the_byline_sits_above_the_maker_line_not_in_the_hero(self):
        """Credit belongs with the other "who made this" furniture. The hero is the
        illustration, the title and the intro, and nothing else."""
        html = build(sdf_inputs=SDF_INPUTS)
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert "foot-by" not in hero
        assert P.AUTHORS[0][0] not in hero
        foot = re.search(r"<footer class='foot'>.*?</footer>", html, re.S).group(0)
        assert foot.index("foot-by") < foot.index("A project by")

    def test_names_and_institutions_are_escaped(self):
        out = P.byline((("Ada <script>", "Inst & Co"),))
        assert "<script>" not in out and "&lt;script&gt;" in out
        assert "Inst &amp; Co" in out
