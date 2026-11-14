# website/ — rules for agents working on the handoff page

What the page is, how to build and host it, and how to check it renders are in `website/README.md`.
This file holds what an agent needs before changing anything here: the invariants the tests pin,
the reasoning behind each decision, and the history of what was tried and undone. Do not undo
anything below without reading its reason.

## The page's shape and the skeleton both reports share

Both reports take the same skeleton, so a reader learns it once: **an opening lede → the
pipeline → one example end to end → appendix**, and both carry all of it. A **caveats** beat
sat between the example and the appendix and was cut from BOTH sides in the external-readiness
copy pass: its bullets restated what the pipeline openly does rather than conceding anything,
so they read as filler where self-criticism belonged. `test_both_reports_take_the_same_skeleton`
asserts its absence on both sides, so restoring it to one report alone fails rather than
quietly splitting the skeleton. `dad.blocks_weak()` and `sdf.blocks_weak()` are still defined
and now dead: nothing calls them, and the prose ids they read (`caveats`, `sdf_caveats`) are
gone from the prose files and from both `CONTENT_IDS`. **The lede takes no heading on either
side** — the `<h2>` is the heading, and one over a single sentence only names what a reader can
already see while costing a rail item and a hairline (`h3[id]` is what draws that rule).
`website/sdf.py` carried a `sdf-what` heading while that report was a stub whose whole content
was that one line and three stat tiles; it lost it when the beats below it landed, and
`test_neither_report_puts_a_heading_over_its_opening_line` keeps both sides that way. The
stages come before the example that walks through them, because the chooser promises a
walkthrough. There is still no "what we measured" beat: this is not a results report. Every
beat after the lede is an `<h3>` with its own id (`#dad-built`, `#sdf-example`, `#dad-appendix`),
and each pipeline's stages are `<h4 id>`s under **the pipeline** and again, under the same
names, in **one example** — one vocabulary per pipeline, used twice;
`test_both_reports_take_the_same_skeleton` pins the two beat lists against each other.

**There is no `what it is` beat, and the flow lives in the pipeline beat.** A `dad-what`
heading briefly carried a **vertical flow schematic** (`render.flow()`) and a **trimmed
specimen** of one record, each under its own label, on the reasoning that both were otherwise
invisible until the worked example ~3,000px down. Both moved: the diagram went down to **the
pipeline**, the beat whose prose reads it aloud, and the specimen was cut because the worked
example two beats below is the same record in full. So both reports now open on a bare lede
with no heading of its own, and nothing above the pipeline beat carries a figure, a tile, a
chip or a score. What survived the move is in `tests/test_website_dad.py::TestTheFlow`: the
schematic owes an accessible name, because SVG text is not read as prose, and it takes no
series or status colour, because it measures nothing. **Known drift:** the flow's stage names
still say `the constitution rewrite` while the copy pass rewrote `content_dad.md` to say "your
alignment documents" — one pipeline, two vocabularies, and a copy decision nobody has made yet.

The pipeline beat is cheap on purpose, and that is measured, not hoped: `<svg>`, `<table>`,
`<blockquote>` and `<div class='resp'>` are all uncounted by `editorial_words()`, so the
diagram costs nothing. `#dad` before the appendix sat at 786 of its 800 ceiling when this was
last measured — 14 words of headroom — so anything added to a report has to be paid for out of
its stage prose. (`TestBrevity` asserts the ceiling; the build prints the current count.)

Three beats are open and the last is drawers, and the line between them is what a reader has
to read:

- **Open**: the lede (what the dataset is), the process, and one record's whole trail through
  it. Nothing on the page explains how to install or run the pipeline, and nothing open holds
  a figure.
- **Appendix**: everything specific to one run, collapsed. The difficult-advice appendix is
  **two drawers**: "Comparison to the control" (the judged comparison against a plain model,
  with its regression statement; see rule 3) and "Composition and diversity" (rhetorical moves
  with their glossary nested inside, tracked phrases, and the semantic charts). The
  document report's appendix is **one**, "Composition and diversity" (the composition
  figures, principle coverage, the semantic charts and the Vendi curve). The earlier
  five-drawer layout (what the audit flagged, what the judge scored, every chart, every check,
  the worked example's full rewrite diff) was trimmed: the audit-flags drawer, the health-check
  triage tables and the full rewrite diff "belong to the review tool, not to a hand-off page"
  (`blocks_appendix` docstrings in `dad.py` and `sdf.py`).

  **Which run that is, the appendix says**, in a muted line under its intro
  (`common.run_note()`, with the run directory's name and the audit's own count). The
  worked example carries the same line, because it is the other beat that is one batch
  rather than the pipeline — and it comes first, so the appendix's line repeats the id
  rather than referring back to a beat a reader arriving from the rail has not read. Both
  are derived and both vanish without a run id. The report is about a pipeline; those two
  beats are one batch, and with nothing saying so a reader could not tell a property of the
  pipeline from a property of one run — the example carousel's "the same run" pointed at a run
  the page had never introduced. The id is derived from the run directory's name and the
  audit's own `n_prompts`, and stays out of the pipeline beat, which holds for any run
  (`TestWhichRun`). The backend is in neither, and `common.provenance_warnings()` flags only
  `UNFAITHFUL_BACKENDS` (`claude_code`, `auto`): `bedrock` is not in the code for a reader to
  go and look at.

The stages come _before_ the example that walks through them, because that is what the
chooser above promises.

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
  a report (`#dad-example`, from a quoted finding) opens the report it lives in and scrolls
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
  module's `BEATS` list — because the beats are conditional (a document report built with no run output renders only its lede and the two ways out), and
  `test_every_rail_link_lands_on_a_heading_that_rendered` builds a run without the optional beats to prove a link
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

## The rules

**1. No number is ever typed into a prose file.** Prose interpolates `{{placeholders}}`
resolved from the runs' own output, and an unknown one fails the build. Run-conditional
figures reach prose only with an explicit degraded string — `{{library_clause}}` and
`{{judge_arms_clause}}` — so a run missing the paid pass renders "not measured on this
run" where the figure would be and the sentence survives. The page's own prose has no
facts at all (`PAGE_FACTS = {}`), so any placeholder in `content_page.md` is a build
error. Do not add a bare conditional number to prose; add a clause to the owning module's
`facts()`.

One block is stricter than that and carries **no figure at all**: `dad_what`, the lede. It is
pinned by tests against the shipped prose, because a fixture cannot prove it.

**2. Caveats and the derived floor.** The `caveats` beat was authored, general, carried no
figure and no placeholder, and took no `audit` argument so a run number could not get in; it
was cut (see the skeleton above). The derived floor — every BAD/OK verdict the run's own audit
recorded, plus provenance rules (a still-supported non-`api` backend, small n) — was computed
rather than written, so that generalising the caveats could not quietly become softening
them, and `warnings_table()` could collapse rows into a drawer only with a visible count.
In the current build the difficult-advice floor is removed outright
(`TestCandour::test_the_audit_flags_drawer_is_gone_machinery_and_all`) and the document
report's (`sdf.derived_warnings()`, `sdf.audit_flags_drawer()`) is defined but not called.
Candour now lives with the measurements it qualifies: the delivery regression statement and
the arm-asymmetry note sit inside the judged drawer. `evals/audit_sdf.py` only _prints_ its
verdicts, so `sdf.derived_warnings()` re-applies the eval's own thresholds (see the last
section).

The backend rule fires only for `common.UNFAITHFUL_BACKENDS` (`claude_code`, `auto`).
`bedrock` is not in the pipeline any more, so a row citing it sent a reader looking for a
backend that is not in the code — and a run's identity is a fact about provenance, said by
`run_note()`, not a finding an audit caught. The documents report's pinned runs are
`claude_code`, which that rule marks BAD wherever `derived_warnings()` is rendered.

**3. The judged comparison does not lead, and the delivery regression is stated once.**
The whole comparison against the plain model — considerations, delivery, the scatter, the
scoreboard, retention — is one drawer in the appendix, headed with why it is there. It was
demoted because the delivery pass lost 19 of its 80 judgements on the pinned run, leaving
its two means over 33 pipeline against 26 control answers: different sets of records. A
page that led with that would rest on its least sound measurement. Judge and generator are also the same model family, and nothing checks whether the points the judge counted as added are correct. Do not restore the headline. Upstream agrees: PR #107 replaced that judge with two holistic ones.

Demoted is not deleted, and both halves are pinned by tests. The regression is written in
prose exactly once, by `dad._delivery_statement()`, _inside_ that drawer — next to the
comparison it is about, because a figure from one run cannot live anywhere but the appendix. The scoreboard row
carries the same number as data. No figure of any kind appears outside the appendix —
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
`test_each_report_a_reader_reads_has_its_own_ceiling` bounds each report's beats
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

`render.flow()` draws the pipeline for the pipeline beat (both reports; the document report's has four stages): a source, a dot per stage down
one spine, an output box, and an optional dashed spur for a stage fed by something that is
not its predecessor (the control arm into stage 2, with a head — a dashed line with two bare
ends does not say which way it feeds). Its stage names are the ones "The pipeline" and the
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
stage headings from "The pipeline" rather than inventing a second vocabulary.

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

## Decisions that must not be undone by accident

**Content style.** Sentence case for names and labels. Cut on sight: aphoristic two-beat deks
(at most two `> ` deks on the whole page, `test_at_most_two_deks`), negation-as-emphasis ("a
habit rather than a value"), portentous closers, and any sentence explaining why a section
exists. `test_the_prose_has_a_ceiling` only stops a third body of prose appearing somewhere the
per-report ceilings (rule 6) do not measure; both reports sat within a few words of 800 when
last measured, so anything added to one has to be paid for out of it.

**Neither report leads with its judge, and both for reasons in the data.** The difficult-
advice reasons are in rule 3. The document report's scoring judge (layer 5) graded every
document it could read 8 or 9 — no document it actually graded fell below the gate — so what
looks like a gate rejecting twelve documents is ten scoring calls whose JSON failed to parse
plus two the judge marked down; `sdf.gate()` returns both numbers. The only check on that
judge is `audit/realism_ablation.json`, a rerun of its own realism rubric by a judge that
cannot see the spec, and it scores the same documents 2.7 points lower. (The document
report's judged drawer, titled "What the judge scores, and why the report does not lead with
it", is defined in `sdf.py` but no longer built into the appendix.)

**The document report spends two chart hues and no more.** `R.PLAIN` and `R.PIPELINE` mean
"control" and "pipeline" in the difficult-advice report, and that pipeline has no control arm
— so `website/sdf.py` never borrows them. Its one pair is the matrix's dealt weight against
what shipped; every other chart is a single series in the palette's default, where a colour
carries no meaning to confuse. The greens are avoided outright: `--series-6` is `#008300`
against `--good`'s `#0ca30c`, so a magnitude drawn in it reads as a verdict.

**The document report's worked example.** `#sdf-example` is the same beat over the other
pipeline, assembled by `sdf.read_lineage()`: dealt cards, the planner's working notes, the
spec, the draft, the reviewer's own list of problems, the document as it ships, what the
rewrite changed, and the judge's three scores. Two things differ from the difficult-advice
one. The rewrite drawer **says which of two things the rewrite did** — the rewrite template
licenses a rewrite from the premise where the problems are structural, and past 60% of the
shipped words changed the drawer says "rewritten, not edited" instead of presenting three
windows as three edits (the pinned run's median is 81%). And the pinned document is
**English and dealt the central-subject slice** on purpose: 139 of 477 documents are English,
and a first worked example a reader cannot read, or one from a reserved slice, teaches the
wrong default. The two extras are the interesting slices — a skeptical author, and the arc
where there is no welfare stake and the AI correctly raises nothing.

**The tab icon is the one `<link>` on the page**, and the one hole in the self-contained
rule: a favicon has no other spelling — a browser will not read one out of a `<meta>`, and
the implicit `/favicon.ico` lookup only exists for a hosted copy, so it would leave the
emailed file without an icon. `test_is_self_contained` no longer bans `<link>` outright; it
requires every one to be `rel='icon'` with a `data:` href, so a stylesheet or a font cannot
follow it in. `render.icon_links()` emits them and, like `illustration()`, raises on a
non-`data:` value. It is deliberately **not** part of `head_meta()`, which is what a crawler
and a link preview see — an icon is neither. `make_preview.py` crops the butterfly without
its dashed trail by column alpha density (not a hardcoded box) and squares it on the page's
paper rather than transparency, because dark line work on dark browser chrome is an
invisible icon; after the `MinFilter` pass 149 of 256 pixels are inked at 16px. The two radii
in `ICONS` are eye-tuned constants, not a formula — a fit to two points would only dress the
eye up as arithmetic.

**Brand, addition.** Cream fills with a border are not a control style; the palette test
recomputes every contrast pair from the tokens.

**The chooser is a disclosure pair, not a tab set, and is marked up as one.** Two buttons
carrying `aria-expanded` + `aria-controls`; no `role='tablist'`, no `role='tab'`. It was a
tablist, which promises what this control cannot do: a tablist always has exactly one
selected tab and nothing here is selected on load — that is the point of the chooser — so a
screen reader announced "tab, 1 of 2, not selected" twice and the arrow keys the pattern owes
did nothing. The **example carousel is** a real tab set (one pane is always open), so it
keeps `role='tab'` and pays the rest of the pattern: `tabindex` roves with the selection,
Left/Right/Home/End move across the set, and each pane is named by the button that opens
it. `.choice[aria-expanded=true]` is what the accent fill hangs off
(`test_the_buttons_are_a_disclosure_pair_not_a_tab_set`).

**The comparison's heading is heard, not seen** (`<h2 class='vh'>`). With no heading at all,
pressing `H` went from the page title to the chooser, past both datasets. The documents column
used to carry a neutral `Report in preparation` chip, off `sdf.IS_PLACEHOLDER`, because that
column is first in the comparison, the chooser and the panels and opened ~200 words against the
other's ~10,000. **That report is written now**, so the flag, the chip and
`render.compare()`'s `status` argument all went with it — a state that is no longer true must not
survive as a string, and `test_the_comparison_no_longer_marks_this_column_as_a_stub` asserts all
three are gone.

**Each report ends with the two ways out, and the footer carries no provenance.** The
difficult-advice report carried no link at all through ten thousand words, so the reader most
likely to want the data had to scroll back past everything they had read; the pair (`Browse
the records` / `The pipeline`, `_ways_out()` in each report) now sits at the foot of the
worked example, which is the only place it appears. The footer has no run id, no commit, no
dirty flag, no backend. It has been added and removed twice, most recently restored on the
grounds that provenance had otherwise "appeared nowhere", and that is not true:
`common.run_note()` names the run twice inside the difficult-advice report, under
`#dad-example` and under the appendix intro, which is where a reader who wants it is. What the
footer added on top of that was a commit sha, a dirty flag and a backend name — none of which
a reader can act on, and one of which (`bedrock`) is not in the code to go and look at — plus
"+ uncommitted changes" on the last line of a handoff page, which reads as an unfinished
draft. `test_the_footer_carries_no_provenance` holds it out, and
`test_the_run_is_still_named_where_the_reader_needs_it` is the other half: the footer may
only stay empty for as long as the report names the run.

**The control edge answers to 3:1, not 4.5:1.** `--accent-edge` is a control boundary (WCAG
1.4.11), and at `#c9c3ea` it reached 1.53:1 on the paper — the page's only decision had a
border a low-vision reader could not see. `TestPalette.CONTROL_EDGES` recomputes it against
both `--surface-0` and the hover wash. `button` is in the `:focus-visible` rule for the same
reason: without it the only controls on the page fell back to the UA ring.

**The bar's resting height was narrowed.** The height range was 83px→52px and is now
~72px→52px: the pinned size is the one measured to sit beside prose, so a resting bar 61%
taller than it was oversized on arrival; the 40rem width is the comparison's two 20rem
columns and did not move.

**Layout cannot be tested by asserting on HTML.** Four real bugs shipped past the suite and
were only caught by measuring in a browser: a bare `1fr` grid track grown past the page by a
wide child (the comparison landed 116px off centre), a deep link scrolling before the
multi-megabyte hero had laid out, scroll anchoring fighting the bar's shrink, and every
section's named grid lines going undefined below 760px, which collapsed the prose to one word
per line. If you touch layout, measure it (see "Checking it renders" in `website/README.md`).

**The narrow layout keeps the named lines.** Below 760px `section` is a single track declared
`[text-start] minmax(0,1fr) [text-end full-end]`, not a bare `minmax(0,1fr)` with the children
re-placed. Re-placing them was tried and does not work: `section>*` is (0,0,1) and loses to
`section>figure` (0,0,2) and `section>.explore-body` (0,1,1), which then point at names the
same block deleted, so the figures, the comparison and the whole chooser landed in a 0px
implicit track. `TestNarrowLayout` pins both halves — the names stay declared, and nothing in
that block re-places a child or reaches for `!important`. Measured in Chromium at 390×844:
panel 358px in a 390px viewport, prose at ~45 characters a line, no horizontal page scroll,
bar one row at 57px.

## Pinned runs and open items

The datasets are CC0-1.0 — declared by hand in the Hugging Face card's frontmatter, beside the
files it governs, which is the only place it is stated — and the page deliberately says
nothing about it, the licence row having been removed. (Not to be confused with this repo's
own Apache-2.0 `LICENSE`, which covers the pipeline code.)

**The pinned DAD run is `2026-07-29_12-26_archetype200`** (191 examples), a matrix-dealt Opus-5
run that carries `step1/scenario_deals.jsonl` and `step1/scenarios.jsonl` and is on `main`. It
replaced `2026-07-20_20-51_bedrock-40`, which PR #108 pruned along with the other 35 pre-Opus-5
runs and which the merge of `main` therefore dropped; the report was rebuilt against the new
run rather than keeping the old directory, because the sweep that removed it also removed the
bedrock backend it was produced on. `2026-07-28_22-14_archetype10` is still not a candidate
(hand-seeded from a scratchpad file, so the matrix was bypassed, and its own
`step1/checklist.txt` fails four composition checks). The report still does not lead with the
judged comparison, and the reason still holds on this run: the delivery judge lost 24 of its
judgements, and its two arms cover **different sets of records** — 171 each side, but 16
records judged only on the pipeline side and 16 only on the control side — while the
welfare-impact judge is 179 against 187.

**Two things the repin broke are fixed, both in `website/dad.py`.** The pinned run is in the
two-holistic-judge schema (`delivery` / `welfare_impact` / `composite`), which dropped the
`moral_patient_reasons` metric `_pareto()` plots on its vertical axis — so "Substance against
manner" rendered its title, its axis note, a "not measured" placeholder and a typed caption
claiming "The pipeline arm sits up and to the left: it buys substance with manner", false
here, since the pipeline is higher on both axes (welfare impact 92.33 against 83.01, delivery
90.41 against 89.80). `_pareto_figure()` now needs **both** axes measured and renders nothing
otherwise, covered by `test_delivery_without_the_substance_measure_drops_the_pareto` (the
older `test_delivery_present_renders_the_pareto_in_the_appendix` fixture carries the old
metric, so it still pins the both-present case). The judged scale is no longer typed:
`_score_max()` reads `score_max` off the audit's `delivery`/`welfare_impact` block, falling
back to 10 for pre-rework runs, and every label that said `0–10` — `_JUDGED_AXES`, the
scoreboard row, the checks table, the dimension figure's note, both regression notes and
`_pareto()`'s own domain and tips — takes it from there. Rebuilding the DAD judge on held-out
labels is the separate, larger work that would let the report lead with a comparison again.

`page.MAKER_URL` is inferred from the team's domain.

**The pinned SDF run is `2026-07-25_15-57_fullscale-500-opus5`** (477 documents), which
replaced the 100-document `2026-07-11_20-06_matrix100-cli` when the documents report was
written: it is the only committed run carrying principle coverage, the LLM templating scan, a
10-mode compliance pass, card-fidelity drift, a blind realism ablation and a Vendi curve.
Both runs were generated on the `claude_code` backend, so both earn the same BAD provenance
row wherever the derived warnings are rendered. `matrix100-cli` still builds, and is the
cheapest way to exercise the degraded path.

## Tests: the boundary this page keeps being pulled across

Four of them are the boundary this page keeps being pulled across, and are worth knowing
by name: `test_the_page_does_not_explain_how_to_run_the_pipeline`,
`test_neither_report_puts_a_heading_over_its_opening_line`, `test_both_reports_take_the_same_skeleton` and
`test_each_report_a_reader_reads_has_its_own_ceiling`. Slice a beat with
`beat(html, anchor)` rather than by `index("id='dad-example'")` — the naive slice keeps the
next beat's `<h3` and its stray `3` breaks any assertion about digits.

## The document report's own thresholds

Status in the current build: `sdf.derived_warnings()`, `sdf.audit_flags_drawer()` and the
document report's `judged_drawer()` are still defined in `website/sdf.py`, but nothing in
`sdf.blocks()` calls them: that report's appendix is one "Composition and diversity" drawer.
What follows describes the thresholds as they stand in the code, ready to be rendered again;
`TestDerivedThresholds` still pins each one against the eval's own number.

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
  looks at it, it is not produced by any committed script, and no beat of the page now states the
  gap (the cut caveats beat did, in general terms, because it holds for every run).

`evals/report_sdf.py` builds the *other* document artefact, `audit/corpus_report.html` — an
internal audit page, not this one. Roughly half of its 866 lines is editorial prose welded
to a 477-document run ("across 477 documents", "nineteen drifted — 95%"), which is exactly
what rule 1 exists to prevent, so nothing here reads from it or from the
`audit/report_content.json` it consumes.
