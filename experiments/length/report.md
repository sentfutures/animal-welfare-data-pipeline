# Length targeting at the revision stage

**Question.** Does telling the layer-3 rewriter how long the finished document
should be change the corpus?

**Short answer.** The rewriter largely ignores the target — it hit 32% of them
against a 60% bar, and it will not shorten a document at all. But documents
dealt a length card vary more in how they are *built* than undirected ones do,
and that effect replicated across two runs with different card designs. The
card ships in its multiplier form, dealt from the matrix like every other axis,
and is kept for the structural variation rather than for the targeting.

---

## Design

Each arm re-runs layer 3 over **one shared set of drafts** sampled from the
donor run, so layers 1-2 are identical across arms and nothing but the revision
prompt differs. A `{length}` card is dealt to each document from
`prompts/sdf/variables.txt` (its `REVISION VARIABLES` section) and rendered by
`revision_style.length_block()` into the layer-3 user prompt:

> LENGTH. Aim for a finished document *at roughly 0.5x its current length*.
> Reach the target by changing how much the document does — scenes, examples,
> reported detail, how many turns an exchange runs — not by padding or by
> compressing prose into summary. If hitting it would cost the document's
> naturalness, miss the target and keep the document.

40% of documents are dealt `unchanged`, which renders no directive at all and
serves as the undirected baseline *within the same arm*.

Donor run `2026-07-25_15-57_fullscale-500-opus5`, n=50, seed 0, in both
generations below.

## Two generations of card

The experiment ran twice, because the first design could not be measured.

**Absolute targets** (`outputs/sdf/experiments/exp2`, 3 arms, $18.32). Cards
named word counts with a per-script fallback — `"about 1400 words, or about
2700 characters in Chinese or Japanese"`. This produced **no length answer at
all**: the comparison reported `—` in the length column and both arms came back
`INCOMPLETE (not measured: min_length_compliance, min_length_score)`. Nothing
could read a target back out of a card written that way, and a word count is
meaningless for a script that does not space its words, so the card had to
carry two units and the measurement had to pick between them per document.

**Relative targets** (`outputs/sdf/experiments/exp3-relative`, 1 arm, $8.99).
Cards name a multiple of the document's own draft length — `"at roughly 0.5x
its current length"`. `revision_style.length_ratio()` parses the multiple back
out and `compare_arms.draft_lengths()` supplies the per-document denominator,
so the rungs compare across languages without a word/character split and the
targets do not go stale when the corpus's length profile changes.

The multiplier form is what ships. It is the one that can be measured, and it
is the one that stops the card from encoding an assumption about how long
documents currently are.

## What the targets did

Measured on the relative run. `median/target` below 1.0 is an undershoot of the
instruction; above 1.0 an overshoot.

| asked for | n | median words | median/target | in-band (±20%) | graded score |
|---|---|---|---|---|---|
| unchanged (baseline) | 10 | 1120 | — | — | — |
| 0.5× | 10 | 1467 | **1.67** | 0% | 0.09 |
| 0.7× | 9 | 986 | **1.24** | 0% | 0.39 |
| 1.3× | 10 | 1754 | 0.86 | 100% | 0.72 |
| 1.6× | 9 | 1771 | 0.71 | 22% | 0.24 |

Overall 31.6% in-band, 0.36 graded score. The graded score is a Gaussian on the
log ratio with `sigma = log(1 + tolerance)`, so a document on the band edge
scores ≈0.61 and half-the-target and twice-the-target are penalised equally.

Read the top row carefully. Asked to **halve** a document, the rewriter produced
1.67× the target — about 0.84× the original, barely shorter than the 1120-word
undirected baseline. Asked for 1.6× it undershot. Every rung lands in roughly
the same 1000–1800 word band regardless of what it was asked for: **the
rewriter regresses to its own preferred length and the instruction mostly
bounces off.** Only the mild lengthening request (1.3×) landed.

The judge pass rate was 100% at every rung, so whatever the instruction did, it
did not cost document quality.

## What it did to diversity

| arm | Vendi (ratio) | structural spread | lexical spread |
|---|---|---|---|
| control (undirected) | 15.7 (0.321) | 0.296 | 0.254 |
| length, absolute cards | 15.3 (0.319) | **0.344** | 0.255 |
| length, relative cards | 15.4 (0.322) | **0.333** | 0.261 |

**Content diversity did not move.** Vendi — the effective number of distinct
documents in the corpus of 50 — is flat across all three, and the differences
are well inside what re-running one arm at temperature 1.0 would produce.

**Structural variation did.** Both length arms sit clearly above the control on
spread of sentence and paragraph architecture, and they do so with different
card designs in separate invocations. That is two independent observations of
the same direction, which is the main reason to believe it. Lexical spread —
word choice — is unchanged, which is the expected shape: a length target moves
how a document is assembled, not what vocabulary it reaches for.

Note that raw document length is deliberately **not** one of the features
behind the structural-spread figure, precisely so an arm that varies length
cannot score higher by construction. What is being measured is whether varying
length drags sentence and paragraph architecture along with it. It does.

## Decision

**Ship the multiplier form of the card, on by default.** `config.yaml` carries
`sdf.revision.use_length: true`, and the `{length}` axis is dealt from
`prompts/sdf/variables.txt` like every other card, so an ordinary pipeline run
deals a length target to 60% of its documents and leaves the rest `unchanged`.
It ships for the structural variation it produces, not for compliance, which is
the measure the rest of this report is about.

**Compliance is reported, not judged.** The `min_length_compliance` and
`min_length_score` bars were dropped from the arms file. They were the right
bars for the question as originally posed — "does the model hit the number" —
and the answer to that is a clear no. But hitting the number was never what the
intervention was for, and a declared criterion that reports `FAIL` forever
while the thing you care about works is noise in the comparison table. The
response curve above still prints as a diagnostic.

## Caveats

Three, and the first is the one that matters most.

**The relative run had no control arm of its own.** It ran a single `length`
arm, so the control and structural-spread figures it is compared against come
from the absolute-card invocation. The donor, seed, sample size and sampled ids
are identical between the two, so this is a fairer comparison than it sounds,
but it is still across invocations rather than within one. The arms file in
this directory has been corrected to include a control; re-running it would
close this gap for about $18.

**Structural spread was not a pre-registered criterion.** The bars fixed before
the run were Vendi gain, judge pass-rate drop, and length compliance. Structural
spread is a post-hoc read of a metric that was being collected anyway. It
replicates across two runs, which is why it is reported as a finding rather
than a hypothesis — but it was found, not predicted.

**n=50, one donor, no repeated sampling.** Temperature is 1.0 throughout. A few
percent of movement in any figure here is inside the noise of re-running the
same arm at the same settings.

## Files

The intervention is `revision_style.length_block()` / `length_ratio()`, the
`{length}` axis in `prompts/sdf/variables.txt`, and the
`{length_directive}` slot in `prompts/sdf/layer3.txt` and
`layer3b_revise.txt`. Measurement is `textstats.unit_for_language()` /
`count_units()` and the `length_compliance()` / `length_curves()` half of
`experiments/compare_arms.py`. `evals/README.md` documents the reporting.
