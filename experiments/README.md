# Experiments

Revision-stage interventions tried against the SDF pipeline, and what came of
them. One directory per experiment, each holding the arms file that defines it
and a report of what it measured.

An experiment here is always the same shape: several layer-3 variants ("arms")
run over **one shared set of drafts** sampled from a donor run, so layers 1-2
are literally identical across arms and any difference is attributable to the
revision stage. `experiments/experiment.py` is the driver,
`experiments/compare_arms.py` assembles the comparison.

```bash
python experiments/experiment.py --donor-run <run_id> \
    --arms experiments/length/arms.yaml --exp-id length --n 50 --dry-run
python experiments/compare_arms.py --exp outputs/sdf/experiments/length
```

Arms files live here because they define an experiment — a donor run, a sample
size, a seed, the arms. The revision axes themselves live in
`prompts/sdf/variables.txt`, under its `REVISION VARIABLES` section, because
they are ordinary matrix axes: dealt by the same deck sampler as every other
card, and rendered into the layer-3 prompt. `arms.example.yaml` in this
directory is the schema reference for every knob an arm can set.

## The index

| Experiment | Question | Verdict | Decision |
|---|---|---|---|
| [`length/`](length/report.md) | Does giving the rewriter a target length change the corpus? | Targets are largely ignored, but structural variation rises (0.296 → 0.333) | Shipped, dealt from the matrix; see report |
| voices _(report not yet written)_ | Does intervening on author voice at the revision stage raise corpus diversity? | No Vendi gain either way, but the comparative axes raise structural spread +8.4% where sampled dispositions move it −1.1% | Axes shipped; dispositions parked |
| [`dispositions/`](dispositions/README.md) | Can the rewriter reason from an author's temperament to prose itself? | Lands hard on some documents, does nothing at all on others; flat in aggregate | Not shipped; kept for reference |
| [`split/`](split/report.md) | Does running layer 3's critique and rewrite as two calls change the corpus? | Outputs flat on every measure; the critique doubles (495 → 1,124 words) and the rewriter changes 21.6% of the document against 12.6% | Shipped on, for the critique quality; see report |

The voices run is `outputs/sdf/experiments/exp1`. It predates the
structural-spread column, so the figures quoted for it were computed from its
run directories afterwards rather than read off its own comparison table.

`split` is the one intervention here that ships on. Every other knob in
`config.yaml`'s `sdf.revision` block is a per-arm choice, and an arm that leaves
them off sends the prompt layer 3 has always sent — which is what makes a
control arm a real control rather than a near-copy. Because the block no longer
ships uniformly off, **an arm must name every knob it wants held down**: the
driver's `merge_arm_config` only overrides what an arm mentions, so a silent
control inherits whatever the base config happens to have on.

## Reading the numbers

Two measures recur and they answer different questions.

**Vendi score** — the effective number of distinct documents in a corpus of N,
computed from an embedding similarity matrix. A corpus of 50 documents scoring
15.4 is worth about 15 genuinely distinct ones. The `ratio` beside it is that
score over N. It is a measure of *content* diversity: what the documents are
about.

**Structural spread** — mean relative spread (standard deviation over median)
of sentence length, sentence-length variability and paragraph length across the
corpus, from `shared/textstats.form_summary`. It measures how documents are
*built*, not what they say. Raw document length is deliberately excluded as a
feature, so an intervention that varies length cannot score higher merely by
having varied length.

An intervention can move one without the other, and the length experiment is
exactly that case.

## A caveat that applies to all of it

These runs are n=50 against a single donor. Temperature is 1.0 throughout, so a
few percent of movement in any figure is inside the noise of re-running the same
arm. The reports say which differences are large enough to read and which are
not.

The split experiment is the only one that **measured** that noise rather than
asserting it, by running two arms at identical settings: re-running one arm
moved the Vendi ratio by 3.4% and the median per-document rewrite extent by
4.5%. Treat those as the floor any future single-donor result here has to clear,
and prefer a duplicate control arm to a caveat — it costs one arm.
