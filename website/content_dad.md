<!--
Prose for the difficult-advice section of the handoff page (the #dad beats). Sections
are delimited by HTML comments of the form "id: <section>"; every id in website/dad.py's
CONTENT_IDS must appear exactly once here, and no others. Supported markup: paragraphs,
`- ` lists, **bold**, *italic*, `code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own audit JSON at build time, and an unknown placeholder fails
the build.

Run-conditional figures reach this file only with an explicit degraded string —
{{library_clause}} and {{judge_arms_clause}}. A run without the paid pass renders "not
measured on this run" in place of the figure, so the sentence survives and its claim does
not. Do not reach for a bare conditional number here; add a clause to facts() instead.

WHAT GOES WHERE. The beats a reader sees are the opening (`dad_what`, then the diagram and
a specimen record, both unnarrated), the process (`method_intro`, `stage1`-`3`, `control`),
one record's trail (`example_*`), and `caveats`. Everything specific to one run is in the
appendix. So:

  * `dad_what` is the whole of the opening's prose and it is BUDGETED: the beats before the
    appendix have to clear 800 counted words and sit within ~30 of it. The diagram and the
    specimen below it are free — `editorial_words` skips `<svg>`, `<blockquote>` and the
    answer — so any growth here is growth the ceiling feels. It is the lede: one line.
  * The opening does not narrate its own diagram or its own specimen. It shows them, under
    "The pipeline" and beside the panes' own labels, and the stages beat below explains
    them once.

  * `caveats` carries NO figures and NO placeholders. It is about the method, and it holds
    for any run of this pipeline. A number in it is a bug, not a tightening.
  * The delivery regression is written once, by dad.py, inside the appendix's judged
    drawer — next to the comparison it is about. This file must not restate it.
  * The comparison against a plain model does not lead. `judged_caveat` says why. Do not
    move it up: the judge's arms are not the same set of records, and the page would then
    rest on its least sound measurement.
  * Nothing here explains how to install or run the pipeline. That is the repository
    README's job, and it was cut from this page deliberately.

And no deks — the page allows two in total and both are spent elsewhere.
-->

<!-- id: dad_what -->

A fictional user brings a query that could help or harm animals. A wise AI assistant coaches the user through responsible conduct, reasoning skillfully about tradeoffs without being overbearing.

<!-- id: method_intro -->

Three stages, each a short chain of [model calls](https://github.com/sentfutures/animal-welfare-data-pipeline/tree/main/prompts/dad). Code deals a weighted mix of variables (you can adjust the weights); stage 1 turns it into a user message; stage 2 prompts a model to respond using generated and conditionally triggered supplemental reasoning; stage 3 revises the answer using your alignment documents.

<!-- id: stage1 -->

A dumb script draws from a weighted matrix of variables, feeding them to a model to generate a unique scenario. Rare values ensure a long tail of diverse situations when generating data in bulk.

Subsequent model calls draft a user message and filter out non-instructive situations before rewriting the message for human realism.

<!-- id: stage2 -->

The conversation is analysed according to the nature and severity of the ethical dilemma it contains. Based on that analysis, a model selects entries from our expert-generated reasoning library, taking these along with new context about the nature and purpose of this pipeline to generate a second draft.

<!-- id: stage3 -->

The second draft is rewritten against your alignment documents or a distilled set of excerpts concerning harm to third parties, limits on autonomous action, etc., ensuring compliance with the letter and spirit of your alignment goals.

<!-- id: control -->

The fictional input is given to a production model as a normal user message. This generates a control showing how well current models handle these dilemmas, and ensuring the pipeline only improves on the status quo.

<!-- id: example_pick -->

AW-0020

<!-- id: example_extra -->

AW-0031 AW-0011

<!-- id: appendix_intro -->

Corpus-wide evals from a {{n}}-example sample run using {{gen_models}}.

<!-- id: checks_intro -->

How varied the responses are across several dimensions: the **rhetorical moves** they make (classified by an LLM), the **wording and phrases** they repeat (detected automatically), and the **meanings or topics** they cover (measured using embedding similarity).
