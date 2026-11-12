# exp2-small

Donor run `/Users/markstanl/PycharmProjects/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`, n=5, seed 0.

| arm | final/scored | Vendi (ratio) | idea Vendi | near-dup >0.90 | judge pass | align / real / spec | length in-band / score | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|
| control | 5/5 | 3.9 (0.786) [2-4] | 3.7 (0.746) | 0.0% | 100.0% | 8.8 / 8.2 / 9.0 | 0.0% / 0.21 (n=3) | $0.64 | control |
| as-shipped | 5/5 | 3.9 (0.774) [2-4] | 3.7 (0.748) | 0.0% | 100.0% | 9.0 / 8.4 / 9.0 | — | $0.16 | INCOMPLETE (Vendi ratio -1.5%; pass rate -0.0%; not measured: min_length_compliance, min_length_score) |
| length | 5/5 | 3.9 (0.783) [2-4] | 3.8 (0.763) | 0.0% | 100.0% | 8.8 / 8.2 / 9.0 | 33.3% / 0.40 (n=3) | $0.63 | FAIL (Vendi ratio -0.3%; pass rate -0.0%; length in-band 33.3%; length score 0.40) |

Success criteria, fixed before the run: `min_vendi_gain` = 0.1, `max_pass_rate_drop` = 0.05, `length_tolerance` = 0.2, `min_length_compliance` = 0.6, `min_length_score` = 0.7

### Length response — `control`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| 2700 chars | 1 | 3266 | 1.21 | 0.0% | 0.58 | 100.0% |
| unchanged (baseline) | 2 | 2234 | — | — | — | 100.0% |
| 700 words | 1 | 1100 | 1.57 | 0.0% | 0.05 | 100.0% |
| 2400 words | 1 | 988 | 0.41 | 0.0% | 0.00 | 100.0% |

### Length response — `length`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| 2700 chars | 1 | 2863 | 1.06 | 100.0% | 0.95 | 100.0% |
| unchanged (baseline) | 2 | 2264 | — | — | — | 100.0% |
| 700 words | 1 | 973 | 1.39 | 0.0% | 0.20 | 100.0% |
| 2400 words | 1 | 1534 | 0.64 | 0.0% | 0.05 | 100.0% |
