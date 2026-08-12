# `website/` — the handoff page

One self-contained HTML file covering both datasets, written for one reader: someone who
runs midtraining at another lab, has no context on this project, and has about forty
seconds to decide whether to keep reading. It publishes to GitHub Pages as it stands,
emails, and opens offline from the filesystem.

The two datasets are **Difficult advice** (`dad`) and **Synthetic documents** (`sdf`) —
"corpus" and "corpora" are not words this page uses.

This is **not** the Streamlit corpus-audit page. That is an internal review tool
organised by what the eval measured; this is organised by what a reader needs, in order.

## Build

```bash
python website/build_website.py \
  --dad-run outputs/dad/runs/2026-07-29_12-26_archetype200 \
  --sdf-run outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5
# -> website/index.html
```

Those two runs are the pinned ones behind the current build. `--run` still works as an
alias for `--dad-run`. `--content` (repeatable) overrides the prose files, `--example` and
`--sdf-example` override each report's worked example, `--out-dir` writes elsewhere.

**The page does not document how to run the pipeline.** No install, no invocation, no
costs, no per-stage model table — that is this repository's own README and `CLAUDE.md`.
What the page carries is the process, one record's whole trail through it, and caveats.

`--sdf-run` is optional. Without it the synthetic documents' column says "not published
yet" and its report keeps its lede, says no run output was supplied, and offers the two
ways out — the page still builds, and carries no dead links.

Four of the document report's inputs — `audit/compliance_report.json`,
`audit/card_fidelity_report.json`, `audit/realism_ablation.json` and
`audit/vendi_curve.json` — are **not** written by `evals/audit_sdf.py`, and only
`fullscale-500-opus5` carries them. Every block that reads one degrades to naming the file
it wanted, and the checks table marks it "not run on this run". Building against
`2026-07-11_20-06_matrix100-cli` is the cheapest way to exercise that path.

The paid audit pass only affects the appendix. Without it, the judged drawer says no paid
pass ran and the derived-flags drawer gains a BAD row; the four beats above the appendix are
unchanged, because none of them depends on a judge — the overview's flow is authored and its
specimen comes off `step3/rewrites.jsonl`:

```bash
python evals/audit_dad.py --input outputs/dad/runs/<run_id> --reasons
python evals/diversity.py --input outputs/dad/runs/<run_id>
```

To run the evals on the shared AWS Bedrock credits instead of an Anthropic API key, add
`--config config.bedrock.yaml` (identical to `config.yaml` but `backend: bedrock`, which
reads `CHAD_AWS_BEDROCK_KEY`).

## Hosting

The page is one file with **one** file beside it, so serving it is an upload: nothing in it
refers to its own URL apart from the preview tags, every other asset is inlined and every
outbound link is absolute. It stays that way — a hosted copy is never a second,
un-self-contained build.

`.github/workflows/pages.yml` is the deploy. It publishes to `reasoning.sentientfutures.ai`
on a push to `main` that touches the built page or the card image, and it does **not** run
the builder: it copies `website/index.html` and `website/assets/preview.png` into the site
root and hands them to Pages. So the committed `index.html` is what is live, and rebuilding
it is a commit like any other.

**It is unlisted, and that is a meta tag rather than a `robots.txt`.** Every build carries
`<meta name="robots" content="noindex,nofollow">`, unconditionally. The two files do
different jobs and are easy to mix up: `robots.txt` governs **crawling**, the tag governs
**indexing**. `noindex` is not a `robots.txt` directive at all — Google dropped support for
the unofficial one in 2019 — so a `Disallow` cannot say it.

**Do not `Disallow` this page.** A crawler that is refused the file never reads the tag
asking it not to index, and a URL someone links to can still be indexed with no content
behind it — the `Disallow` makes the outcome *worse*, not better. Let it be crawled and let
it say `noindex`. (A host that sets `X-Robots-Tag: noindex` as a header does the same job for
consumers that only read headers; it is a belt to the tag's braces, unlike a `Disallow`.)
Removing the tag is one line in `render.head_meta()` if the page is ever announced.

Because a pasted link is then the *only* way anyone arrives, a hosted build carries preview
tags. They need to know where the page lives, so naming the site is what turns them on:

```bash
python website/build_website.py --dad-run <run> --sdf-run <run> \
    --site-url    https://reasoning.sentientfutures.ai/ \
    --preview-url https://reasoning.sentientfutures.ai/preview.png
# -> website/index.html   (the workflow serves the card image from assets/)
```

`--site-url` adds `og:title` / `og:url` / `og:description` / `twitter:card` and points
`og:image` at `preview.png` beside the page, because a card renderer fetches the image out
of band and a data URI is no use to it. On its own it also **copies** that file into the
output directory. The command above passes `--preview-url` naming the same URL, which
suppresses the copy: the deploy already stages the image from `website/assets/`, so a second
one committed next to the page would be a drifting duplicate. (`.gitignore` covers it if
someone rebuilds without the flag.) `--preview-url` is otherwise for an image hosted
somewhere else entirely; with no image at all the card declares `summary` rather than
promising a large image it has not got. With neither flag the build says nothing about where
it lives, which is right for the copy that opens from disk or arrives attached to an email;
the build line prints `preview=no`.

**The card image and the tab icons are the hero.** `python website/make_preview.py` redraws
all three from `assets/hero.png` and their output is committed; it needs Pillow, which is
why it is a separate script and not part of the stdlib-only `build_website.py`.

- `assets/preview.png` — the butterfly trimmed to its own bounds and centred on the page's
  paper at 1200×630. No crop through the drawing, no filter, no text baked over it.
- `assets/favicon-16.png`, `assets/favicon-32.png` — the butterfly *alone*, without the
  dashed trail, squared up on the same paper. These are inlined as data URIs, so they are
  not among the files a deploy carries, and they are the one `<link>` on the page: a
  favicon has no other spelling, and `test_is_self_contained` allows that shape and no
  other.

Two sizes rather than one, and the ink is thickened before each is shrunk. Both fall out of
the same fact — the hero is hairline pencil work. A straight resize to 16px leaves 14 of 256
pixels carrying any ink, the darkest at 3.9:1 on the paper, which is a blank cream square in
a tab; a darkest-pixel-wins pass first is the ordinary way to decimate line art and holds the
strokes at full strength. That is a resampling choice and changes nothing about `hero.png`.
It is also why `sizes=` is declared: hand a browser one 32 and it scales to 16 itself,
averaging the ink straight back out. The two filter radii in `ICONS` were tuned by eye
against a contact sheet — lower reads washed out, higher smears the wing veins into a blob.

The `description` those tags use is prose, authored in `content_page.md` under the
`description` id like every other word on the page. It is the one id that never renders in
the document, so it is flat text — `render.plain_md()` strips the markdown subset.

The datasets' licence is **not** on the page: CC0-1.0 is declared by hand in the Hugging
Face card's frontmatter, beside the files it governs. (The pipeline code is Apache-2.0 —
a separate licence, on a separate thing.)

Deploy from `main` rather than from a laptop, so the repo stays the source of truth and the
hosted copy is never edited in place.

## The page

| Anchor      | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hero        | The illustration, the title, and the lines that follow from it — centred, and carrying the `#intro` id. Four blocks: the finding (_Teaching Claude Why_ and the SDF post, the page's one credit to both), the sentence that introduces the pair, **the two techniques as two columns** (`render.named_pair()`, synthetic documents first), then what we built on them, in two paragraphs. Four paragraphs of prose and it stops. The pair is a figure between the second paragraph and the third, not a list inside one: those two techniques _are_ the two datasets below, in the same order, and ~90 words of definitional prose with two digits in front of it did not say so. It carries **no index over either name** — an eyebrow ("Technique 1") names what the heading under it already says — and no tie line naming the dataset each produced, which was tried and read as awkward against names that barely differ. Nothing else: no lede, no provenance, no tiles, and no "Intro" heading over a paragraph that needs no introducing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `#datasets` | The comparison. No heading over it: the two column mastheads (the name in serif, and nothing else) are the heading. Five rows, and **each one says whether it describes the data or the process that makes it** — `output`, `output format`, `what it is for`, then `prompt templates` and `example dataset`. (A `pipeline` row of stage chains was cut: the two walkthroughs below _are_ the pipeline, and a one-line chain above them was a summary met before it could mean anything.) This is where the page draws that line first, because everything after it is two long pipeline walkthroughs and the reader came for the datasets. What each dataset _is_ used to be the masthead's subtitle; it is the `output` row now, because it was the one unlabelled claim in a table whose every other line said what it was answering. **The record count is deliberately not here**: how many records exist is a property of one run, and this section describes the pipelines. Dates, model ids, the composition spread and the counts all live in the report that goes into them. The last two rows carry the way to what they name: the figure (if any) at the column's left edge, an outline button at its right — the templates on GitHub, the published sample on Hugging Face. The `example dataset` row is button-only, so its cells keep an empty first flex item and its buttons line up under the row above. Labels are right-aligned, one line each, vertically centred. |
| `#explore`  | "Walk through either pipeline" — a walkthrough, not results, because roughly half of each report is the worked example and the pipeline that produced it. Two buttons carrying each dataset's name and nothing else, 40rem centred at rest so each sits under its own column, in a bar that pins to the top of the screen while a report is read and tightens as it goes. Both reports are _inside_ this section, in `.explore-body` — see "The chooser".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `#sdf`      | Synthetic documents, in full (`website/sdf.py`). Same skeleton as `#dad` and four pipeline stages instead of three. Hidden until chosen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `#dad`      | Difficult advice, in full (`website/dad.py`). Opens on the `what it is` overview — the vertical flow schematic and a trimmed specimen. Hidden until chosen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| footer      | Two rows: the byline and its affiliation key, then a split row — who made it and where to send feedback on the left, the two ways out (Datasets, Pipelines) on the right. The destinations are links, not buttons, and their marks are the only ones in the footer, because each names a place the reader has not been. No provenance: no run id, no commit, no dirty flag, no backend.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

Both reports take the same skeleton, so a reader learns it once: **the opening lede / the
pipeline / one example end to end / caveats / appendix**. The lede takes **no heading** on
either side — the `<h2>` is the heading, and one over a single sentence only names what a
reader can already see while costing a rail item and a hairline
(`test_neither_report_puts_a_heading_over_its_opening_line`). Every beat after it is an
`<h3>` with its own id (`#dad-weak`, `#sdf-example`), and each report's stages are `<h4 id>`s
under **the pipeline** and again under **one example** — one vocabulary per pipeline, used
twice. `test_both_reports_take_the_same_skeleton` pins the two lists against each other.

`what it is` is the overview, and it is two **named** halves, each a label, then a sentence,
then the thing itself. Under `The pipeline`, `render.flow()` draws the matrix, the three
stages and the record as a **vertical schematic**; under `The result`, `render.sidebyside()`
puts a **trimmed specimen** of one real record's question and answer beside each other. The
lede opens it, naming the pipeline and what it produces, for a reader who arrived on `#dad`
from a deep link and never saw the comparison.

The two labels are plain `<h4>`s with **no id**, so they stay out of the rail: an id is what
makes an `<h4>` a rail item (`render.substep`), and "The pipeline" listed there directly
above the beat "How it is built" puts back the ambiguity the labels exist to remove. They are
document subheads rather than `h4.pane-h`, because the specimen's own two panes are already
pane headings and a third above them reads as one list of three.

It was a bare lede, on the reasoning that a "What it is" heading over one sentence only names
what a reader can already see. That was right about one sentence and wrong about the beat: a
reader arriving cold could see neither the pipeline's shape nor a record until the worked
example, ~3,000px down. Three rules keep it from growing into a second report:

- **The sentences say what the visuals show**, rather than introducing them. A diagram's
  labels are not read aloud, and 30 words of a record is not a description. That redundancy
  is the beat's accessible reading, and `test_the_prose_says_what_the_diagram_shows` checks
  it against the shipped prose.
- **It links nowhere and names no record id.** It is the overview; the trail is two beats
  down and a reader reaches it by scrolling.
- **No figure, no tile, no chip, no score.** The flow is a schematic and the specimen is a
  quotation. A chart here would argue a result before the reader knows what the data is.

Four beats are open and the fifth is drawers, and the line between them is what a reader
has to read:

- **Open**: what the dataset is, the process, one record's whole trail through it, and
  caveats that hold for _any_ run of the pipeline.
- **Appendix**: everything specific to one run, in **five drawers, one per question a
  reader has**: what this run's audit flagged (the derived floor, first), the judged
  comparison against a plain model with its regression statement, every chart, every check,
  and the worked example's full stage-3 diff.

  **Which run that is, the appendix says**, in a muted line under its intro
  (`common.run_note()`, with the run directory's name and the audit's own count). The
  worked example carries the same line, because it is the other beat that is one batch
  rather than the pipeline — and it comes first, so the appendix's line repeats the id
  rather than referring back to a beat a reader arriving from the rail has not read. Both
  are derived and both vanish without a run id. The backend is in neither: see
  "provenance" below.

  It was eight, and the grouping contradicted itself — a drawer called "every chart from
  this run" beside two siblings that also held a figure and three stat tiles, so a reader
  who opened it had no way to know it was not every chart. Three rows now sit inside the
  drawer whose question they answer: the per-record retention chart is a chart
  (`_appendix_charts` appends it), and the diversity tiles (`_diversity_block`) and the
  rhetorical-move glossary (`_moves_drawer`) are the numbers and the gloss behind two rows
  of the checks table. Nothing was rewritten and nothing was dropped; the drawer meta still
  names each payload's size, so collapsing still costs the reader nothing.

The stages come _before_ the example that walks through them, because that is what the
chooser above promises. There is no "what we measured" beat: this is not a results report.

**The page has no contents rail; a report has one.** A column of page-wide links beside a
hero and a comparison is furniture, and it stays gone. But a report is ~2,700 visible words
of records with four beats and seven stages in it, and from inside one a reader could see
neither its shape nor a way past the worked example — so each report carries its own
contents, in the column to its left, sticky under the bar, with the stages nested under
their beat. See "The chooser" below. An earlier revision hung those links as a second row
under the bar instead; it read as clutter on the control and came out.

The type scale is the other thing that makes a report skimmable, and it had none: `h3` (a
beat) was `1.1rem` against a `1.0625rem` body and `h4` (a stage) was `.82rem`, _smaller_
than the prose under it. It steps 2 / 1.4 / 1.12rem now, each level clear of the body text,
and every beat is chunked off the one before it by a hairline above its `<h3>`.
`TestTypeScale` keeps it monotonic. `h4` doubles as a label over a block in exactly one
place — the two halves of a side-by-side — and `h4.pane-h` keeps the old small sans there.

## The chooser

Neither report is open on load — the choice is the point. Three things make that safe,
and all three are pinned by `TestChooser`:

- **`#dad` and `#sdf` in the URL open that report**, on load and on `hashchange`, so the
  dataset card's deep links land where they say they will. A hash naming anything _inside_
  a report (`#dad-weak`, from a quoted finding) opens the report it lives in and scrolls
  to it — that is what `closest('.panel')` in the inline JS is for.
- **The way across is on screen throughout**, because the bar the tabs live in is pinned.
  A report used to end with a filled button offering the other dataset, from when the
  chooser scrolled away behind the reader; a second way across at the foot of every report
  was then a button the page did not need.
- **Printing expands both**, so a PDF of the page is the whole thing.

The cost is real and was accepted deliberately: Cmd-F cannot see a closed report.
`.panel[hidden]{display:none}` is load-bearing — a panel is a `<section>`, and
`section{display:grid}` beats the browser's own `[hidden]` rule.

**The bar is pinned while you read**, and pressing a tab scrolls it to the top of the
screen. `TestStickyBar` pins the six things that make that work:

- **The panels live inside `#explore`**, wrapped with the bar and the rails in
  `.explore-body` (`render.explore_body`). A sticky box travels only inside its containing
  block, and the containing block of a _grid item_ is its own grid area — one row, as tall
  as the buttons — so a sticky bar left loose in `#explore`'s grid has nowhere to go. The
  wrapper is the travel: the bar pins for the length of the open report. It is two columns
  and two rows — the bar across the top, then `.railcol` beside `.panels` — and the panels
  are wrapped as **one grid item** on purpose: a grid item stretches to its row's height, so
  the rail's column is as tall as the open report. Left as loose siblings, each panel would
  start a row of its own and the rail would have one panel's worth of travel.
- **`.choicebar` carries the background, `.choices` the buttons.** The band is the full
  column in `var(--surface-0)`, the page's own paper, so the report scrolls under it and
  out of sight; the pair stays centred inside it. A sticky box the width of the buttons
  would let a figure scroll up either side.
- **The rail is the open report's own contents.** `.rail` is a column of jump links to that
  report's `<h3 id>`s with its `<h4 id>`s nested under them, hidden with the panel it
  belongs to and toggled by the same handler (`[data-rail]` in the inline JS), so what a
  reader sees is always the contents of what they are reading. It is **read back off the
  built panel** — `render.outline()` over the markup `website/page.py` just assembled, not a
  module's `BEATS` list — because the beats are conditional: the document report only earns
  `sdf-weak` when its run's audit flagged something, and
  `test_every_rail_link_lands_on_a_heading_that_rendered` builds a clean run to prove a link
  can't advertise a beat that isn't there. A stage becomes a rail item **by having an id**
  (`render.substep()`), which is why the appendix's `<h4>`s deliberately have none: they sit
  inside closed drawers, and a link to a collapsed heading goes nowhere.
- **The room for it came out of the shell, not the report.** `.shell` is 67rem rather than
  the 53rem the page was built at: 12rem of rail, so the reading column keeps its 38rem
  measure and the figure track has 812px. A rail taken out of the reading side would have
  shrunk the figure track, and every chart is drawn at 800px — an 11px label in a 600px
  track is no longer 11px. `test_the_room_for_the_rail_did_not_come_out_of_the_report`
  recomputes that from the tokens.
- **Nothing is drawn between the contents and the report.** The gutter was 2rem with a
  hairline down the middle of it; the line was a second separator, since a fixed column the
  links never leave, set in the sans with its stages indented under their beat, is already
  not the prose beside it. The one rule the contents do get is below 900px, _under_ them,
  where they wrap across the head of the document and need it.
- **With no line there, the gutter holds the columns apart on its own, and it is 3rem out of
  the shell's left margin.** `--pull` on `.explore-body` is a negative left margin of
  2.25rem — exactly what the gutter grew by — so the contents hang into the margin and the
  reading column does not move: 416px left edge, 812px figure track, charts at their drawn
  800px. It is clamped to `max(0px,(100vw - 67rem)/2)`, the room outside the shell, so below
  ~1088px it is 0 and the gutter narrows the reading column instead (as every width between
  900px and 67rem already does), and print gets 0 for free. `.choicebar` adds `--pull` back:
  the bar spans both columns, so without it the chooser's centred buttons sit 1.125rem left
  of the page's centre line.
- **The contents start level with the report's title.** `.panel` carries a 3.2rem top margin
  the rail did not, so the first beat sat 48px above the `<h2>` it is the contents of.
  `.railcol`'s `padding-top` plus the rail's own `.2rem` come to that margin —
  `test_the_contents_start_level_with_the_report_s_title` recomputes it, because a hardcoded
  3rem goes stale the moment the panel's margin is retuned. It applies at rest only; once the
  rail pins, its own `top` places it. Below 900px it is 0: the contents are above the report
  there, so there is no title to line up with.
- **Where the reader is: ink and a left edge, never a fill.** The current beat or stage takes
  `aria-current`, and the line for "arrived at" is the heading's **own
  `scroll-margin-top`**, read off the element: the CSS already states how far below the top
  of the screen a linked heading lands, so the same number decides whether it has been
  reached. Measured — with the bar's own bottom as the line instead, the marker sat one
  heading behind every jump. Marking runs inside the rAF-throttled scroll callback the bar's
  flag already uses; there is no second listener and no IntersectionObserver.
- **The script measures `.explore-body`, never the bar.** Once sticky takes hold, the
  bar's own `getBoundingClientRect()` and `offsetTop` report where it is _painted_, so
  scrolling to it means scrolling to wherever the reader already was. `.explore-body`'s
  top is the bar's flow top, and that is also the sticky threshold, so nothing jumps as
  the bar pins. Nothing else in the script queries the bar either.
- **The headroom is CSS, not arithmetic.** The bar measures 5.21rem, so `h3[id]`, `h4[id]`
  and `.panel` take `scroll-margin-top:7rem` and a linked beat or stage lands clear of it
  (60px, measured); a native fragment jump reads the same value, and so does the
  current-item pass. `_bar_rem()` in the test file recomputes the height from the six
  tokens, so retuning the bar without revisiting the headroom — or the rail's `top` —
  fails there rather than in a browser. `scrollIntoView()` carries no
  `behavior`, so `html{scroll-behavior}` — and therefore `prefers-reduced-motion` — still
  owns the smoothness.
- **Below 900px there is no beside.** The rail becomes a static wrapped block at the head of
  the report, held to the reading measure so it reads as part of the document, and **carries
  its beats only** (`.rail .r-s{display:none}`). Beside the report the two levels are a tree
  — an indented triplet under a bold parent — and the difficult-advice report names the same
  three stages twice on purpose, once to explain them and once to walk them. Flattened into
  a wrapped row the tree is gone and the duplication is all that is left: nine items in
  which "Stage 2 · the reasoning" appears twice, identically, with nothing to say which is
  which. Four beats on four lines say the same thing about the report's shape. Between 900px
  and the shell's own 67rem the reading column simply narrows, which needs no rule.
- **It has two sizes and crosses between them once.** Loose it is ~72px tall and 40rem wide,
  lined up under its heading with the two dataset columns; tight it is ~52px and 30rem, with
  the arrow faded out — `↓` means "the report is below", which is stale once the reader is
  in it. It tightens 96px past its own top and loosens again at 24px, animated by a 200ms
  transition. **Two thresholds, not one**, because a reader parked on a single boundary
  flips a layout change back and forth; and a trigger rather than a size that tracks the
  scroll, because tracking meant the bar moved whenever the page did, which reads as
  distraction beside prose. `--t` is a flag (0 loose, `.explore-body.tight` sets 1 — it
  lives on the wrapper because the rail's `top` reads it too, so a tightening bar does not
  leave a growing gap above the contents) and every
  dimension is one interpolation off it, so both states are one set of numbers and a
  breakpoint restates only the six tokens. The 30rem floor is measured, not chosen: below
  27.5rem "Synthetic documents" wraps and the tight bar is _taller_ than the loose one, and
  `test_the_bar_has_two_sizes_and_the_tight_one_is_smaller` recomputes it from `--w` and the
  factor. The transition sits on `padding`, `width`, `gap` and `font-size` rather than on
  `--t`, which is both necessary (a custom property is discrete unless registered) and what
  lets the page's own reduced-motion rule turn the animation off with the `transition:none`
  it already applies to everything — measured: 83 → 52 with no frames in between.
- **`overflow-anchor:none` on the wrapper.** Shrinking the pinned bar moves the report
  under it, so scroll anchoring corrects the scroll by the same amount — moving the element
  `--p` is computed from. Measured with anchoring on: the bar settled at 52px while sitting
  31px _below_ the top of the screen, or flipped between its two sizes depending on where
  the reader stopped. Resizing every frame costs nothing measurable otherwise (120 scroll
  frames: 16.6ms mean, no frame over 20ms, same as with the driver off).

`#explore>h2` is a child combinator on purpose: every panel opens with its own `<h2>`, so
a descendant selector centres and stretches both report titles too.

The bar stays pinned below 760px, so it is kept to **one row** there — two columns with
tighter tokens (57px at rest, 37px shrunk), and no arrow below 620px. Stacked, the two
buttons are ~10rem of permanent chrome, a quarter of a phone screen.

## Files

| File              | Role                                                                                                                                                                                                                                                                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `content_page.md` | **Page prose**: title, the intro's three blocks (`intro`, the two techniques `sdf_technique` / `dad_technique`, and `intro_close`), and the comparison's cells. A technique block's **leading bold run is its name** and the rest is its sentence; its index and the dataset it produced are supplied by `page.py` — the index off the enumeration, the dataset off `sdf.SECTION_TITLE` / `dad.SECTION_TITLE` — so neither is typed here. The comparison's three prose rows are `*_desc` (`output` — what each dataset _is_), `*_unit` (`output format` — what one record is) and `*_use` (what each is _for_); `prompt templates` and `example dataset` are counted and linked in code. A row's **label** lives in `page.section_datasets()`; only its cells are prose. The page's own prose interpolates nothing — a `{{placeholder}}` in it is a build error. |
| `content_dad.md`  | **Difficult-advice prose.** The file to iterate on for that report.                                                                                                                                                                                                                                                           |
| `content_sdf.md`  | **Synthetic-documents prose.** Same rules, same 800-word ceiling on the beats before the appendix. One placeholder is available to it, `{{matrix_clause}}`.                                                                                                                                                                     |
| `page.py`         | The page: hero, comparison, chooser, footer, and the one `document()` call.                                                                                                                                                                                                                                                   |
| `dad.py`          | The `#dad` beats: `facts()`, the block builders, `read_lineage()`, `judged_drawer()`, `derived_warnings()`.                                                                                                                                                                                                                   |
| `sdf.py`          | The `#sdf` beats: `facts()`, the block builders, `read_lineage()`, `read_matrix()`, `read_attrition()`, `judged_drawer()`, `derived_warnings()`.                                                                                                                                                                                |
| `common.py`       | Loading, prose parsing, `fill()`, the word diff (`diff_summary`/`diff_hunks`/`word_diff`, used by both reports), cost aggregation, the provenance warnings, the warnings table, `editorial_words()`, the CLI parser.                                                                                                                                                                                         |
| `render.py`       | CSS + inline-SVG chart primitives + the `document()` shell. No pipeline knowledge.                                                                                                                                                                                                                                            |
| `build_website.py` | The CLI.                                                                                                                                                                                                                                                                                                                      |

Each report module exposes `blocks()`, returning its body as one flat string; `page.py`
wraps that in `render.panel()`, which is the `<section>`, and both panels go inside
`#explore` via `render.explore_body()` so the chooser bar has something to stick to.
Blocks stay flat because a figure has to be a direct child of the section for the CSS grid
to bleed it past the text measure — nesting the panels does not change that: a panel is
still their parent, and `.explore-body` spans the full column, so a panel resolves to the
same width it had as a child of `<main>`.

## The rules

**1. No number is ever typed into a prose file.** Prose interpolates `{{placeholders}}`
resolved from the runs' own output, and an unknown one fails the build. Run-conditional
figures reach prose only with an explicit degraded string — `{{library_clause}}` and
`{{judge_arms_clause}}` — so a run missing the paid pass renders "not measured on this
run" where the figure would be and the sentence survives. The page's own prose has no
facts at all (`PAGE_FACTS = {}`), so any placeholder in `content_page.md` is a build
error. Do not add a bare conditional number to prose; add a clause to the owning module's
`facts()`.

Two blocks are stricter than that and carry **no figure at all**: the `caveats` list,
which is about the method rather than a run, and `dad_what`. Both are pinned by tests
against the shipped prose, because a fixture cannot prove it.

**2. The caveats a reader sees are general; the run's own findings are derived, and in the
appendix.** Two separate things, and the split is deliberate. `caveats` is authored, holds
for any run of this pipeline, and takes no `audit` argument at all, so a run number cannot
get into it. Everything the run's own audit flagged — every BAD/OK verdict, plus provenance
rules (a still-supported non-`api` backend, small n) and DAD-specific rules (a delivery
regression, per-measure arm asymmetry, length inflation, an unmeasured delivery pass) — is
still emitted by `derived_warnings()` whether or not anyone wrote it up, and renders in the
appendix's "What the audit flags" drawer.

The backend rule fires only for `common.UNFAITHFUL_BACKENDS` (`claude_code`, `auto`).
`bedrock` is not in the pipeline any more, so a row citing it sent a reader looking for a
backend that is not in the code — and a run's identity is a fact about provenance, said by
`run_note()`, not a finding an audit caught. The documents report's runs are `claude_code`
and still earn their BAD row.

Generalising the visible caveats must not lose that floor, and
`test_the_derived_floor_is_still_on_the_page` builds with the caveats prose _emptied_ and
asserts every derived row is still there. `evals/audit_sdf.py` only _prints_ its verdicts,
so `sdf.derived_warnings()` re-applies the eval's own thresholds instead.
`warnings_table()` may **collapse** rows into a drawer, and the drawer states how many it
holds — collapsing is a view, never a filter.

**3. The judged comparison does not lead, and the delivery regression is stated once.**
The whole comparison against the plain model — considerations, delivery, the scatter, the
scoreboard, retention — is one drawer in the appendix, headed with why it is there. It was
demoted because the delivery pass lost 19 of its 80 judgements on the pinned run, leaving
its two means over 33 pipeline against 26 control answers: different sets of records. A
page that led with that would rest on its least sound measurement.

Demoted is not deleted, and both halves are pinned by tests. The regression is written in
prose exactly once, by `dad._delivery_statement()`, _inside_ that drawer — next to the
comparison it is about, because the caveats beat is generalised and a figure from one run
cannot live there. The scoreboard row and the derived weakness carry the same number as
data. No figure of any kind appears outside the appendix —
`test_no_figure_appears_outside_the_appendix` is the restructure in one assertion.

The judged drawer reads **either** audit schema: the old
`valuable_welfare_considerations` + `delivery`, or the `delivery` + `welfare_impact` +
`composite` that PR #107 replaced it with upstream. A run with neither says so.

**4. Synthetic documents comes first**, in the comparison, the chooser and the panel
order, so the page reads in one order throughout.

**5. Both datasets are for midtraining.** "SFT" names the _format_ of the difficult-advice
data — chat transcripts, consumed as supervised fine-tuning — not a different training
phase; the documents are consumed as continued pretraining. The "what it is for" row says
both halves, because internal shorthand has the two sounding like different phases.

**6. Prose has a budget, and two ceilings.** The build prints `editorial_words()` for the
page it just wrote. `test_the_prose_has_a_ceiling` bounds the whole page;
`test_the_report_a_reader_reads_has_its_own_ceiling` bounds the difficult-advice beats
_before_ the appendix, which is the part that is open when a reader arrives — the
whole-page number is dominated by drawers nobody has to read. The second is the one that
matters: it came down from 1,199 words to under 800 over two rounds of cutting, by dropping
the results narrative, then the cost tiles, the commands and the run-specific caveats. Deks
— the aphoristic line under a heading — are rationed to two for the whole page.

`editorial_words()` counts what a person _wrote_, so it skips corpus text, chart internals,
every table and every `<nav>`. The rails are excluded because their labels are the
document's own headings, already counted where they are written; counting them twice would
spend the ceiling on navigation and let real prose in underneath it.

Section ids in each prose file must exactly match the owning module's `CONTENT_IDS`; a
missing or unknown id is a build error, and two files may not both define one, so moving
a block between prose files is a rename. `example_pick` holds the prompt_id of the DAD
worked example (or `auto`) and `example_extra` the ids in its carousel, so a rebuild
reproduces the same cases without a flag. A pinned id the run never shipped says so on the
page and falls back, rather than failing the build.

## The flow schematic

`render.flow()` draws the pipeline for the `what it is` beat: a source, a dot per stage down
one spine, an output box, and an optional dashed spur for a stage fed by something that is
not its predecessor (the control arm into stage 2, with a head — a dashed line with two bare
ends does not say which way it feeds). Its stage names are the ones "How it is built" and the
worked example use; three vocabularies for three views of one pipeline is how a reader stops
believing it is one pipeline.

**It is a schematic, not a chart, and the code enforces the difference.** Nothing in it is
proportional to a measurement, so it takes no series colour and no status colour — hairlines,
one ink, one muted grey, and `test_the_flow_is_a_schematic_so_it_carries_no_series_or_status_colour`
fails if that slips. Arrowheads are drawn paths, never typed glyphs, for the same reason the
outbound arrow is.

**Vertical, and that is what lets it live in the reading column.** Laid out left to right the
same five steps need 720px: too wide for the 38rem measure, so it had to bleed into the figure
track — which is for measurements — and on a 358px phone it needed a horizontal scroll box.
Turned down the page it needs 440, caps there, and scales to ~0.81 at 390px where a 12px label
still lands near 10px. It is one **flat** `<svg>`: `editorial_words()`'s `<svg>` strip is
non-greedy, so a nested one would start charging its labels against the prose ceiling.

Two things measured in a browser rather than asserted, because no HTML assertion catches
them: every label's box sits inside the viewBox at both widths (the branch label, hung off the
end of its spur and right-aligned, ran past `x=0` and was cut in half), and the arrow out of
the matrix carries no label — the gloss directly above it already says "dealt in code", and
the label only had somewhere to go by crowding the first dot.

## The worked example

`#dad-example` is one record's whole trail through the run, and every block in it is
verbatim from a file in the run directory — the dealt cards, the scenario the planner
wrote from them, the message that shipped, the scope and the library entries stage 2
pulled, the answer, and the three largest things stage 3 changed. Its `<h4>`s reuse the
stage headings from "How it is built" rather than inventing a second vocabulary.

Each stage opens on a muted line, because stage 2 is the one whose artefacts are all in
drawers: stages 1 and 3 show theirs inline (the dealt cards, the shipped answer) and stage 2
was a heading with nothing under it, which reads as a stage that did nothing. The scope
table stays collapsed — seven axes of dense prose, measured at 889px, and it would sit
between the message and the answer.

`dad.read_lineage()` assembles it at load time. Two things about the join: only step 1 is
keyed by `scenario_id` and everything downstream by `prompt_id`, so `step1/dilemmas.jsonl`
is the join table, with `audit.gid_map[pid]["scenario"]` as the fallback for a run that
kept no dilemmas file. And `step2/scopes.jsonl` is trimmed on the way in — 725 KB of it is
the reasoning library's prose repeated per case, and the page shows an entry's id,
category and claim.

A missing artefact **names the file it wanted** rather than disappearing, because a step
that silently vanishes reads as a step the pipeline does not have. A key that is not
available is left _absent_ rather than set to `None`, so renderers test membership; null
values in the dealt cards are dropped, since rendering an axis with "None" in it is a bug
that reads as data.

Below it, `render.tabs()` puts the ids in `example_extra` behind one set of buttons, using
the chooser's own mechanism — `data-pane`, `aria-selected`, the same inline JS. The first
pane renders _without_ `hidden`, so with JS off the carousel degrades to one example
rather than to none, and the print rule expands the rest.

**The carousel is inside a closed drawer.** That visible first pane is a second full
transcript — ~1,250 words on the pinned run — sitting under the pinned record's own trail,
which is what the beat is for; the drawer's summary counts what is behind it, and
`<details>` prints open, so nothing is lost on paper.

## The hero illustration

`website/assets/hero.png` is inlined as a `data:` URI at build time (`build_website.
data_uri()`), because the page must open offline and survive an artifact host's CSP: a
file reference, even a relative one, breaks the "one file" guarantee and
`test_is_self_contained` with it. `render.illustration()` raises on anything that is not
a `data:` URI, and renders a marked-TODO placeholder at the right proportions if the
asset is missing.

Its spacing is set in **one** place, `.illo.art{margin:3rem 0 .4rem}`. `.hero .illo` used to
declare `margin:0` and lose to that rule — same specificity, later in the file — so the hero
carried a third top margin nothing in its own block accounted for. With that and the two
6rem margins above the art and the title cut to 3rem, the title lands at ~330px instead of
~490px and the whole intro is above the fold on a 900px viewport; the comparison, which is
what does this page's work, starts at ~845px rather than 1,160px.

`website/assets/hero.png` is the artwork as supplied, unedited — an RGBA PNG, so the line
art sits straight on the cream with no background of its own. It is 2.1 MB, which
makes the built page ~3 MB — fine for a page you open or publish, worth knowing
before you email it.

## Constraints

- **Self-contained**: no external CSS, JS, fonts or images. Charts and the two link
  marks are inline `<svg>`, the hero is a data URI, and the only JS is a tooltip handler,
  the chooser and the example carousel. Enforced by `test_is_self_contained`, which allows
  a `data:` src and nothing else off-page — and which now looks for `url(` and `@import`
  _outside_ the run's own text, because the page quotes three records verbatim and a
  dilemma that happened to contain a CSS snippet would fail a test about the generator.
- **One accent, `--accent:#3b2fa0`.** The page's only interaction colour: the text
  selection and every link. Indigo because it cannot collide with anything the palette
  reserves — far from `--good`, `--warn` and `--bad`, so a selection can never read as a
  verdict, and deeper than `--series-7`, which only appears inside charts. There is no
  separate `--link` token; two names for one hex is how a palette drifts.
- **A link is marked, never re-faced**: it inherits its context's face and size — serif in
  prose, sans in the footer and the rail — and the mark is **weight 600**, accent ink and a
  2px accent underline at a `.2em` offset. The `a` rule sets no face and no size. Weight is
  part of the mark: inherited, a link in a long report read faint, and a bare link in the
  sans footer sat at 400 beside icon links that declared 600. It was `var(--mono)` at `.92em`/600,
  on the reasoning that mono carries identity; mono now means **a literal string** (an id, a
  path, code) and nothing else, because a work's title is language and setting it in mono
  changed x-height and letterfit mid-sentence in every paragraph of both reports. The offset
  is in em because a link sets at five sizes on this page and one px value cannot clear a
  descender at all of them.
- **A citation marker is authored as a name and drawn as a number.** `[^Teaching Claude
  Why](url)` in a prose file renders `<a class='cite-n'><sup>1</sup></a>` with the name in
  `aria-label` and `title`, numbered per prose block. The visible numeral keeps the sentence
  readable; the name is what the link announces, because "link, 1" tells a screen-reader user
  nothing (WCAG 2.4.4). It borrows the footnote convention without a footnote — the marker
  links straight out, and hovering is the only disclosure of which work it is.
- **A control declares the serif**, at its own size. Buttons are not links — `.lbtn`,
  `.choice` and `.ilink` each set their own `font:` shorthand, which beats the bare `a` rule,
  and that matters more now that the `a` rule sets no face: a control that forgets inherits
  whatever it sits in. `.tab` is the only mono control, and for its content — a record id —
  rather than for being a control.
- **Filled means selected, not important.** Every control is an outline button (`.lbtn`,
  `.choice`, `.tab`), a plain icon link (`.ilink`, in the footer), or prose; the accent
  ground with cream text appears only on the tab or pane that is currently open. There is
  no primary button — the end-of-report `.cta` went with the sticky bar, which offers the
  other dataset from anywhere in the one being read.
- **Four CSS traps, all hit and all commented in place.** `section` must use
  `minmax(0,1fr)`, never a bare `1fr`: a child with a definite width wider than the
  column grows the track past the page, and every percentage resolved against that grid
  area then points right of centre (measured: the comparison landed 116px off). And
  `.cmp th` sets the rule, alignment and padding for every cell, so a `.cmp-k` override
  has to out-specify it — `.cmp th.cmp-k` — not merely follow it. And a `position:sticky`
  grid item is confined to its own grid area, which is why the chooser bar needs
  `.explore-body` around it and both panels. And a _stuck_ sticky element's own
  `getBoundingClientRect()`/`offsetTop` report where it is painted, not where it sits, so
  anything scrolling to it has to measure a static element instead.
- **A link that leaves the page says so**, with an arrow that is _drawn_ — `EXT_ARROW`,
  an inline SVG at `stroke-width:2` in `currentColor`. As a glyph (U+2197) it is a
  hairline in most faces and a different shape in every one, and this page is printed and
  screenshotted. `inline_md()` adds it to any absolute link automatically, and
  `linkbutton()` carries it too.
- **One theme, aged paper.** `render.py` declares `color-scheme:only light` and emits a
  matching `<meta>`. `only` is load-bearing: it opts the page out of Chrome-Android and
  Samsung Internet's auto-darkening, which `prefers-color-scheme` does not cover.
- **The palette is contrast-verified, not eyeballed.** The page is `#f7f4ea` warm cream,
  panels `#f1ebdd`, code and table heads `#e9e1cd`, rules `#cec3a6`.
  `test_text_contrast_meets_wcag_aa` recomputes WCAG ratios from the CSS tokens
  themselves and fails if any text-on-surface pair drops below 4.5:1. Cream is much less
  forgiving than white: the pale chip washes only reach ~1.15:1 against the page, so
  every chip carries a tinted `--*-edge` hairline, and `segbar()` draws no text inside
  its bars.
- **Reserved status colours.** `--good`/`--warn`/`--bad` are not series hues.
  `test_status_colors_are_not_series_colors` pins the separation. Direction is carried by
  a labelled chip, so a status colour never travels alone.
- **Arm colours follow the arm.** `hbar(color=...)` takes a sequence; pass `R.ARM_PAIR`
  for any (control, pipeline) chart. Without it `hbar` colours bars by row order — that
  is how the considerations chart came to paint the pipeline in the control's own colour.
- **British English in prose, American in code.**
- **stdlib only**, and no imports from `viewer/` or `shared/` — the page has to build
  where the pipeline's dependencies are not installed, which is also what makes it
  portable. Cost: the row-building helpers in `viewer/rendering.py` are re-implemented
  here, so a schema change to `audit_report.json` can drift.
- Every DAD audit schema renders: `valuable_welfare_considerations`, the legacy
  reconstruction from `moral_patient_reasons` + `moves.alternatives` (exactly as
  `evals/audit_dad.py` did), and the `delivery` + `welfare_impact` + `composite` that
  PR #107 replaced both with. Only the appendix's judged drawer reads any of them, so a
  schema change cannot take the report down — the beats above it read the step files.

## Checking it renders

The generator's tests assert on the HTML it emits; they cannot tell you where anything
lands on screen. Two layout bugs got through that way — a `1fr` grid track silently
grown past the page by the comparison's wrapper, and a deep link scrolling before the
hero image had claimed its space — so if you are changing layout, measure it:

```bash
apt-get install -y chromium && npm install puppeteer   # chromium must match your arch
node -e "const p=require('puppeteer');(async()=>{
  const b=await p.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
  const pg=await b.newPage(); await pg.setViewport({width:1440,height:1000});
  await pg.goto('file://\$PWD/website/index.html',{waitUntil:'load'});
  console.log(await pg.evaluate(()=>{
    const th=[...document.querySelectorAll('.cmp thead th')].map(e=>e.getBoundingClientRect());
    return {centre:innerWidth/2, pairMid:(th[0].left+th[1].right)/2};}));
  // The bar: choosing a report puts it at the top, and it stays there while you read.
  await pg.evaluate(()=>document.getElementById('choose-dad').click());
  await new Promise(r=>setTimeout(r,1200));
  const probe=()=>{const b=document.querySelector('.choicebar');
    const r=b.getBoundingClientRect();
    const c=document.querySelector('.choices').getBoundingClientRect();
    const f=document.querySelector('.explore-body').getBoundingClientRect();
    return {tight:b.classList.contains('tight'), barTop:r.top, barH:Math.round(r.height),
            pairW:Math.round(c.width), past:Math.round(-f.top)};};
  const at=async past=>{await pg.evaluate(async past=>{
    const f=document.querySelector('.explore-body');
    window.scrollTo(0,f.getBoundingClientRect().top+scrollY+past);
    await new Promise(r=>setTimeout(r,400));},past);       // let the transition finish
    console.log(await pg.evaluate(probe));};
  await at(0);    // loose  ~72px x 640px
  await at(95);   // loose  — still, one pixel short of the trigger
  await at(97);   // TIGHT  ~52px x 480px
  await at(25);   // TIGHT  — still, coming back up: the second threshold
  await at(23);   // loose  again
  await pg.screenshot({path:'/tmp/page.png',fullPage:true}); await b.close();})()"
```

`pairMid` must equal `centre`: the two dataset columns straddle the page centre and the
field labels hang off their left, outside the pair. `barTop` must be `0` in every probe
after the click; the bar must be loose (~72px × 640px) at 95px past and tight (~52px × 480px)
at 97px, and coming back up it must stay tight to 25px and loosen at 23px — one threshold
each way means a reader stopped on the boundary flips it repeatedly. `past` must equal
exactly what you asked for; if it does not, scroll anchoring is back and the size change is
fighting itself. A deep link (`…/index.html#dad-weak`, `waitUntil:'load'`) must leave that
`<h3>`'s `top` greater than `barH` (measured 82 against the 52px bar), and on a `390x844`
viewport the two buttons must stay on one row (57px loose, 37px tight).

Sample the bar's height over consecutive `requestAnimationFrame`s just after the trigger:
it must pass through intermediate values (measured `83 83 81 76 70 …`). Under
`emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}])` it must jump
`83 → 52` with nothing in between. Note that `requestAnimationFrame` does not fire in a
backgrounded tab, so call `bringToFront()` on any page you sample this way.

The worked example put two wide tables and six new blocks inside that same grid, which is
the class of change the `1fr` trap bit last time, so measure the overflow too:

```js
console.log(
  await pg.evaluate(() => {
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = false));
    const s = document.querySelector("#dad");
    const over = [...s.querySelectorAll("*")]
      .filter((e) => e.getBoundingClientRect().right > innerWidth + 1)
      .map((e) => e.className || e.tagName);
    return {
      panel: s.getBoundingClientRect().width,
      vw: innerWidth,
      overflowing: over.slice(0, 5),
      beats: [...s.querySelectorAll("h3[id]")].map((h) => h.id),
    };
  }),
);
// then the carousel: clicking a tab swaps the pane, and only one is visible
await pg.evaluate(() => document.querySelectorAll(".tab")[1].click());
console.log(
  await pg.evaluate(() =>
    [...document.querySelectorAll(".pane-x")].map((p) => p.hidden),
  ),
);
```

`overflowing` must be empty, `panel <= vw`, and `beats` must read in skeleton order. The
one thing no assertion can check is whether the lineage scans as a walk or as a wall of
`<h4>`s — screenshot it with `#dad` open and read it.

## Tests

```bash
pytest tests/test_website_common.py tests/test_website_dad.py tests/test_website_sdf.py \
       tests/test_website_page.py
```

Offline. `test_website_common.py` covers the shared plumbing (prose ids, the placeholder
contract, the provenance floor, the warnings table, the word diff, the prose count);
`test_website_dad.py` and `test_website_sdf.py` cover the two reports along the same risk
axes — degradation, candour, not leading with the judge, the lineage naming what it could
not find, colour integrity; `test_website_page.py` covers the page itself, whose distinctive
risks are a report that cannot be reached, a column that shows nothing when a run is
missing, a chooser bar with nowhere to stick or a beat hidden under it, and prose growing
back.

Four of them are the boundary this page keeps being pulled across, and are worth knowing
by name: `test_the_page_does_not_explain_how_to_run_the_pipeline`,
`test_the_caveats_carry_no_run_figures`, `test_the_derived_floor_is_still_on_the_page` and
`test_each_report_a_reader_reads_has_its_own_ceiling`. Slice a beat with
`beat(html, anchor)` rather than by `index("id='dad-weak'")` — the naive slice keeps the
next beat's `<h3` and its stray `3` breaks any assertion about digits.

## The document report's own thresholds

`derived_warnings()` **cannot be shared between the two reports.** `evals/audit_dad.py`
records its verdicts into `sections[].rows[]`; `evals/audit_sdf.py` only prints them, so
`common.audit_verdict_warnings()` returns `[]` for an SDF audit and `sdf.derived_warnings()`
re-applies the eval's own thresholds instead. Every rule is pinned in
`test_website_sdf.py::TestDerivedThresholds` against the number the eval uses, so the two
cannot drift apart silently. Teaching `audit_sdf.py` to record rows the way `audit_dad.py`
does would give future runs the shared floor for free, and is the fix worth making.

Two of the rules are findings in their own right and worth knowing before reading the
appendix:

- **The gate and the judge are counted separately.** A layer-5 call whose JSON fails to
  parse is checkpointed at 5/5/5 rather than re-billed, and those records then fail the
  gate — so "documents the gate dropped" and "documents the judge rejected" are different
  numbers, and on the pinned run they are 12 and 2. `sdf.gate()` returns both.
- **Nothing in the pipeline measures plan-to-card fidelity.** Layer 5 is handed the *plan*
  as the spec, so a plan that quietly substituted one of its dealt cards is scored against
  its own substitution and passes. `audit/card_fidelity_report.json` is the only thing that
  looks at it, it is not produced by any committed script, and the caveats beat states the
  gap in general terms because it holds for every run.

`evals/report_sdf.py` builds the *other* document artefact, `audit/corpus_report.html` — an
internal audit page, not this one. Roughly half of its 866 lines is editorial prose welded
to a 477-document run ("across 477 documents", "nineteen drifted — 95%"), which is exactly
what rule 1 exists to prevent, so nothing here reads from it or from the
`audit/report_content.json` it consumes.
