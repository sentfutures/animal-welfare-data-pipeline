# exp3-relative

Donor run `/Users/markstanl/PycharmProjects/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`, n=50, seed 0.

| arm | final/scored | Vendi (ratio) | idea Vendi | form struct / lex | near-dup >0.90 | judge pass | align / real / spec | length in-band / score | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| length | 48/48 | 15.4 (0.322) [10-13] | — (—) | 0.333 / 0.261 | 0.0% | 100.0% | 8.79 / 8.54 / 8.98 | 31.6% / 0.36 (n=38) | $8.99 | no control arm |

Success criteria, fixed before the run: `min_vendi_gain` = 0.1, `max_pass_rate_drop` = 0.05, `length_tolerance` = 0.2, `min_length_compliance` = 0.6, `min_length_score` = 0.7

### Length response — `length`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| unchanged (baseline) | 10 | 1120 | — | — | — | 100.0% |
| 0.5x | 10 | 1467 | 1.67 | 0.0% | 0.09 | 100.0% |
| 0.7x | 9 | 986 | 1.24 | 0.0% | 0.39 | 100.0% |
| 1.3x | 10 | 1754 | 0.86 | 100.0% | 0.72 | 100.0% |
| 1.6x | 9 | 1771 | 0.71 | 22.2% | 0.24 | 100.0% |

Corpus spread over the 39 word-counted documents: p10 692 / median 1263 / p90 2123 words, p90/p10 = 3.07.
