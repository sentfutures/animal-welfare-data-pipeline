# split

Donor run `/Users/markstanl/PycharmProjects/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`, n=50, seed 0.

| arm | final/scored | Vendi (ratio) | idea Vendi | form struct / lex | near-dup >0.90 | judge pass | align / real / spec | length in-band / score | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 46/49 | 15.2 (0.330) [10-12] | — (—) | 0.278 / 0.260 | 0.0% | 93.9% | 8.55 / 8.22 / 8.76 | 3.5% / 0.14 (n=29) | $8.41 | control |
| control-b | 47/49 | 15.0 (0.319) [10-13] | — (—) | 0.290 / 0.241 | 0.0% | 95.9% | 8.73 / 8.31 / 8.73 | 3.5% / 0.14 (n=29) | $6.46 | PASS (Vendi ratio -3.4%; pass rate +2.0%) |
| split | 49/49 | 15.6 (0.318) [10-13] | — (—) | 0.287 / 0.257 | 0.0% | 100.0% | 8.82 / 8.35 / 8.92 | 3.5% / 0.12 (n=29) | $10.97 | PASS (Vendi ratio -3.6%; pass rate +6.1%) |
| as-shipped | 49/50 | 15.4 (0.314) [10-13] | — (—) | 0.252 / 0.251 | 0.0% | 98.0% | 8.76 / 8.52 / 8.92 | — | $1.64 | PASS (Vendi ratio -4.8%; pass rate +4.1%) |

Success criteria, fixed before the run: `min_vendi_gain` = -0.05, `max_pass_rate_drop` = 0.05

### Length response — `control`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| unchanged (baseline) | 20 | 1422 | — | — | — | 95.0% |
| 0.5x | 7 | 1117 | 1.97 | 0.0% | 0.00 | 100.0% |
| 0.7x | 8 | 1418 | 1.40 | 0.0% | 0.17 | 100.0% |
| 1.3x | 7 | 2271 | 0.75 | 14.3% | 0.36 | 71.4% |
| 1.6x | 7 | 1509 | 0.60 | 0.0% | 0.03 | 100.0% |

Corpus spread over the 40 word-counted documents: p10 876 / median 1174 / p90 2309 words, p90/p10 = 2.64.

### Length response — `control-b`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| unchanged (baseline) | 20 | 1564 | — | — | — | 100.0% |
| 0.5x | 7 | 1107 | 1.95 | 0.0% | 0.00 | 71.4% |
| 0.7x | 8 | 1444 | 1.42 | 0.0% | 0.17 | 100.0% |
| 1.3x | 7 | 2230 | 0.74 | 14.3% | 0.36 | 100.0% |
| 1.6x | 7 | 1539 | 0.61 | 0.0% | 0.03 | 100.0% |

Corpus spread over the 40 word-counted documents: p10 861 / median 1163 / p90 2230 words, p90/p10 = 2.59.

### Length response — `split`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| unchanged (baseline) | 20 | 1659 | — | — | — | 100.0% |
| 0.5x | 7 | 996 | 1.75 | 0.0% | 0.00 | 100.0% |
| 0.7x | 8 | 1472 | 1.45 | 0.0% | 0.09 | 100.0% |
| 1.3x | 7 | 2176 | 0.72 | 14.3% | 0.32 | 100.0% |
| 1.6x | 7 | 1597 | 0.64 | 0.0% | 0.05 | 100.0% |

Corpus spread over the 40 word-counted documents: p10 859 / median 1288 / p90 2114 words, p90/p10 = 2.46.
