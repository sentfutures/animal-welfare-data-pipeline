#!/usr/bin/env python3
"""The handoff page: both datasets, one file.

Written for one reader — someone who runs midtraining at another lab, has no context on
this project, and has about forty seconds to decide whether to keep reading. So the page
opens as one image and a sentence that runs straight on into what this is, puts the two
datasets side by side in a table they can read in one pass, and then asks them to choose
which one they want to read about.

Structure:

    hero        the illustration, the title, and #intro — the finding, the two techniques
                it names (as two columns), and what we built on them
    #datasets   the two datasets, compared row by row
    #explore    Walk through either pipeline — two buttons
      #sdf      Synthetic documents  (website/sdf.py, hidden until chosen)
      #dad      Difficult advice     (website/dad.py, hidden until chosen)
    footer      repo and both viewers, and nothing else

Nothing is open on load; ``#dad`` or ``#sdf`` in the URL opens that report, so the
dataset card's deep links land where they say they will, and printing expands both. The
cost of the chooser is that a closed report is invisible to Cmd-F — the tradeoff was
made deliberately, in favour of a reader who is choosing rather than scrolling.

stdlib only, and no imports from viewer/ or shared/.
"""

import datetime
import re

from website import common as C
from website import dad
from website import render as R
from website import sdf

CONTENT_IDS = ("title", "description", "intro", "sdf_technique", "dad_technique",
               "intro_close",
               "sdf_desc", "sdf_use", "sdf_unit",
               "dad_desc", "dad_use", "dad_unit")


REPO_URL = "https://github.com/sentfutures/animal-welfare-data-pipeline"
# Deep links to the prompts each pipeline runs. The reader this page is written for
# wants the templates, not the records, so the comparison links straight at them.
PROMPTS_DAD = f"{REPO_URL}/tree/main/prompts/dad"
PROMPTS_SDF = f"{REPO_URL}/tree/main/prompts/sdf"
HF_URL = ("https://huggingface.co/datasets/sentientfutures/"
          "animal-welfare-training-claude")
# The config names as they exist on the Hub, percent-encoded: the dataset's two configs are
# named for the datasets rather than for the pipeline directories, so these are not `sdf` and
# `dad`. The `&` in the difficult-advice one is escaped to `&amp;` by `esc()` when the href is
# written, which is what makes it a valid attribute — so a test comparing against these
# constants has to escape them too.
#
# These names are HAND-MAINTAINED, in the `configs:` block of the dataset card's
# frontmatter, which is edited on the Hub. Nothing in this repository generates
# them or can check them: rename a config there and these two links 404 in
# silence. See "The dataset card" in evals/README.md.
HF_DAD = f"{HF_URL}/viewer/difficult%20advice%20Q&A"
HF_SDF = f"{HF_URL}/viewer/synthetic%20documents"

HERO_ALT = "A line drawing of a butterfly at the end of a looping dashed flight path."
# Inferred from the team's own domain; one constant to change if it is wrong.
MAKER, MAKER_URL = "Sentient Futures", "https://sentientfutures.ai"


def load_inputs(content_paths, dad_run=None, sdf_run=None):
    """All filesystem access, in one place. Returns build() kwargs."""
    ids = CONTENT_IDS + dad.CONTENT_IDS + sdf.CONTENT_IDS
    out = {"content": C.load_content(content_paths, ids)}
    if dad_run:
        out["dad_inputs"] = dad.load_inputs(dad_run)
    if sdf_run:
        out["sdf_inputs"] = sdf.load_inputs(sdf_run)
    return out


# ------------------------------------------------------------------ facts

# The page's own prose interpolates NOTHING: every figure on it is rendered by a section
# from a run's facts, so a {{placeholder}} in content_page.md is a build error. (The two
# model-name facts that used to live here existed only for the caveats strip.)
PAGE_FACTS = {}


def _date(manifest):
    """A run's generation date, in prose. Falls back to whatever the manifest holds."""
    raw = str((manifest or {}).get("created_at") or "")
    try:
        day = datetime.date.fromisoformat(raw[:10])
    except ValueError:
        return raw or "—"
    return f"{day.day} {day:%B %Y}"


# ------------------------------------------------------------------ sections

def section_datasets(content, f, dad_kwargs, sdf_kwargs):
    """The two datasets, side by side. Their names are this section's heading.

    Five rows: three describe the OUTPUT — what the dataset is, what one record is, what
    it is for — before the two that link out to the templates and to a made example.

    A ``pipeline`` row naming each chain of stages used to sit between them. It was cut:
    the two walkthroughs below ARE the pipeline, at length and with a diagram each, and a
    one-line chain above them was a summary the reader met before it could mean anything.

    ``output`` was the masthead's subtitle. It reads as a row because it is one: the
    sentence saying what each dataset is was the only unlabelled claim in the comparison.

    How MANY records is not a row — that is a property of a run, and this section
    describes the pipelines; the counts live in each report's appendix, beside the run
    they came off. Dates, model ids and the composition spread belong there too, not in a
    table meant to be read in one pass.
    """
    rows = [
        ("output", _cell(content, "sdf_desc", f), _cell(content, "dad_desc", f)),
        ("output format", _cell(content, "sdf_unit", f), _cell(content, "dad_unit", f)),
        ("what it is for", _cell(content, "sdf_use", f), _cell(content, "dad_use", f)),
        ("pipeline", _with_button("", REPO_URL, "Pipeline", "github"),
         _with_button("", REPO_URL, "Pipeline", "github")),
        ("example dataset",
         _with_button("" if sdf_kwargs else "not published yet", HF_SDF,
                      "Example dataset", "hf"),
         _with_button("", HF_DAD, "Example dataset", "hf")),
    ]
    columns = [(sdf.SECTION_TITLE,), (dad.SECTION_TITLE,)]
    # The heading is heard, not seen: the two mastheads are the heading on screen.
    return C.section("datasets", "The two datasets", R.compare(columns, rows),
                     heading_class="vh")


def _with_button(value, href, label, icon):
    """A cell's figure at the left of its column, and the way to the thing it counts at
    the right. An empty value still leaves its flex item behind, so a button-only cell
    lines its button up under the one in the row above."""
    return R.Raw(f"<span class='cmp-fig'><span>{R.esc(value)}</span>"
                 f"{R.linkbutton(href, label, icon)}</span>")



def _cell(content, key, f):
    return R.Raw(R.inline_md(C.fill(content.get(key, ""), f)))


_LEAD_BOLD = re.compile(r"^\s*\*\*(.+?)\*\*\s*")


def _technique(content, key, f):
    """One technique, as ``named_pair()`` wants it: (name, sentence).

    The prose block's leading ``**bold**`` run is the technique's name — the convention the
    intro's list already used, so the copy moves across without being rewritten.
    """
    text = C.fill(content.get(key, ""), f).strip()
    m = _LEAD_BOLD.match(text)
    name, body = (m.group(1), text[m.end():]) if m else ("", text)
    return name, R.inline_md(body)






def section_explore(panels, outlines):
    """The choice, both reports under it, and each one's contents beside it.

    Two names on the buttons, nothing else: what each dataset is and how big it is are in
    the comparison directly above, and repeating them here only made the buttons hard to
    read as buttons. The report's own beats and stages go in the rail beside it, read back
    off the panel that was built, so a rail link cannot name a beat the report did not
    render.

    The panels are nested here rather than left as siblings in ``<main>`` because the
    buttons and the rail stay on screen while a report is read, and a sticky box travels
    only inside its containing block — see ``render.explore_body``.
    """
    rails = "".join(R.rail(pid, outlines.get(pid, ())) for pid in ("sdf", "dad"))
    return C.section("explore", "Walk through either pipeline",
                     R.explore_body(
                         R.chooser([("sdf", sdf.SECTION_TITLE),
                                    ("dad", dad.SECTION_TITLE)]),
                         rails, panels))


# The people behind the page, in credit order, each with their institution.
#
# The affiliation NUMBERS are derived from this list by first appearance, never typed:
# most of the names share one institution, so a hand-kept numbering is exactly the sort
# of thing that goes quietly wrong the first time someone is added or the order changes.
# Add a name here and the key renumbers itself.
AUTHORS = (
    ("Constance Li", "Sentient Futures"),
    ("Aidan Kankyoku", "Anima International"),
    ("Oscar Horta", "University of Santiago de Compostela"),
    ("Declan McKenna", "Sentient Futures"),
    ("Andrew Blackwood", "Sentient Futures"),
    ("Allen Lu", "NYU Center for Mind, Ethics, and Policy"),
    ("Thomas Giovinazzo", "Sentient Futures"),
    ("Arda Enfiyeci", "Sentient Futures"),
)

# Technical contributors, credited under the authors with the same numbered-affiliation
# treatment. Separate tuple, separate line: a contribution is not an authorship claim.
CONTRIBUTORS = (
    ("Jasmine Brazilek", "Compassion Aligned Machine Learning"),
    ("Miles Tidmarsh", "Compassion Aligned Machine Learning"),
)


def byline(authors=AUTHORS, contributors=CONTRIBUTORS):
    """The author list and its numbered affiliation key, paper-style.

    One number per institution, assigned by first appearance — the convention a reader
    arriving from a paper already knows, and the one in the screenshot this was specified
    from. Both halves come off ``AUTHORS``.

    The key follows the names in DOM order, so a screen reader reads "Constance Li 1,
    ... 1 Sentient Futures" — the same sequence, in the same order, that a sighted reader
    gets. Bare superscript digits carry no meaning on their own, so each marker also
    names its institution in a ``title`` for a hover.
    """
    seen = []
    for _, inst in (*authors, *contributors):
        if inst not in seen:
            seen.append(inst)
    num = {inst: i + 1 for i, inst in enumerate(seen)}
    names = ", ".join(f"{R.esc(name)}<sup title='{R.esc(inst)}'>{num[inst]}</sup>"
                      for name, inst in authors)
    key = "".join(f"<span><sup>{num[inst]}</sup>{R.esc(inst)}</span>" for inst in seen)
    contrib = ""
    if contributors:
        c_names = ", ".join(f"{R.esc(name)}<sup title='{R.esc(inst)}'>{num[inst]}</sup>"
                            for name, inst in contributors)
        contrib = f"<p class='foot-authors'>with technical contributions from {c_names}</p>"
    return (f"<div class='foot-by'><p class='foot-authors'>{names}</p>"
            f"{contrib}<p class='foot-affil'>{key}</p></div>")


def footer():
    """Two rows: the credit, then who made it on the left and where to go on the right.

    TWO ROWS, NOT FOUR THINGS IN ONE. The footer used to be a single ``space-between`` row
    that four children had outgrown — the byline claimed a full-width line, the feedback
    sentence and the maker's name split the next, and the two destinations wrapped alone
    onto a third — so it read as three rows with three different alignments. The split is
    kept and given exactly two items a side, inside its own ``.foot-row``; the byline, which
    belongs to neither half, has the line above to itself.

    No run ids, no commits, no dirty flag, no backend. This was restored once on the
    grounds that provenance had otherwise "appeared nowhere" — untrue: ``common.run_note()``
    names the run inside the report, twice, where the reader who wants it is. What the
    footer added on top was a commit sha, a dirty flag and a backend name, none of which a
    reader can act on, and "+ uncommitted changes" on the last line of a handoff page reads
    as an unfinished draft.
    """
    return (byline()
            + "<div class='foot-row'>"
            # Two spans and no glyph between them: the separating is column-gap's job here
            # exactly as it is in the affiliation key, so nothing a screen reader has to
            # read out sits between two links.
            + f"<p class='foot-colophon'><span>A project by "
            f"<a href='{MAKER_URL}'{R.NEW_TAB}>{R.esc(MAKER)}{R.EXT_ARROW}</a></span>"
            f"<span><a href='{REPO_URL}/issues'{R.NEW_TAB}>Give feedback{R.EXT_ARROW}</a>"
            f"</span></p>"
            f"<p class='foot-links'>{R.iconlink(HF_URL, 'Datasets', 'hf')}"
            f"{R.iconlink(REPO_URL, 'Pipelines', 'github')}</p></div>")


# ------------------------------------------------------------------ assembly

def body(*, content, dad_inputs=None, sdf_inputs=None, example=None, sdf_example=None,
         illustration="", icons=(), site_url="", preview_url=""):
    """The masthead and the sections. Pure: no filesystem, no argv."""
    dad_kwargs, sdf_kwargs = dad_inputs or {}, sdf_inputs or {}
    f = dict(PAGE_FACTS)
    title = C.fill(content["title"], f).strip()

    # Synthetic documents first, throughout: the comparison, the chooser and the panels
    # all read in one order. Each report's contents come back off its own built markup, so
    # the rail is the outline of the report that was actually rendered.
    bodies = [(sdf.SECTION_ID,
               f"<h2>{R.esc(sdf.SECTION_TITLE)}</h2>"
               + sdf.blocks(content=content, example=sdf_example, hf_href=HF_SDF,
                            repo_href=REPO_URL, **sdf_kwargs))]
    if dad_kwargs:
        bodies.append((dad.SECTION_ID,
                       f"<h2>{R.esc(dad.SECTION_TITLE)}</h2>"
                       + dad.blocks(content=content, example=example,
                                    hf_href=HF_DAD, repo_href=REPO_URL, **dad_kwargs)))
    panels = "".join(R.panel(pid, html) for pid, html in bodies)
    outlines = {pid: R.outline(html) for pid, html in bodies}
    sections = [
        section_datasets(content, f, dad_kwargs, sdf_kwargs),
        section_explore(panels, outlines),
    ]
    # The intro is four blocks, not one: the finding and its two sources, the sentence that
    # introduces the pair, the two techniques as two columns, then what we built on them.
    # The pair is a figure between two runs of prose rather than a list inside one, because
    # the two techniques ARE the two datasets below — same things, same order — and ~90 words
    # of definitional prose with two digits in front of it did not say so.
    intro = (C.prose(content, "intro", f)
             + R.named_pair([_technique(content, "sdf_technique", f),
                             _technique(content, "dad_technique", f)])
             + C.prose(content, "intro_close", f))
    head = {
        "title": title,
        # The one sentence a link preview gets, authored beside the title rather than typed
        # into code — same rule as every other word on the page. It never renders in the
        # document, so it is stripped to plain text: markdown in a `content` attribute is
        # asterisks in someone's Slack.
        "description": R.plain_md(C.fill(content["description"], f)),
        "site_url": site_url,
        "preview_url": preview_url,
        "icons": icons,
        "masthead": R.hero(title, R.illustration(illustration, alt=HERO_ALT), intro=intro),
        "footer": footer(),
    }
    return "".join(sections), head


def build(**kwargs):
    body_html, head = body(**kwargs)
    return R.document(body=body_html, **head)
