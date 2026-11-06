# exp2-smoke

Donor run `/Users/markstanl/PycharmProjects/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`, n=2, seed 0.

| arm | final/scored | Vendi (ratio) | idea Vendi | near-dup >0.90 | judge pass | align / real / spec | length in-band / score | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|
| control | 2/2 | — (—) | — (—) | — | 100.0% | 8.5 / 8.5 / 9.0 | 0.0% / 0.01 (n=1) | $0.94 | control |
| as-shipped | 2/2 | — (—) | — (—) | — | 100.0% | 8.5 / 8.5 / 9.0 | — | $0.06 | INCOMPLETE (pass rate -0.0%; not measured: min_vendi_gain, min_length_compliance, min_length_score) |
| length | 1/2 | — (—) | — (—) | — | 50.0% | 7.0 / 6.5 / 7.0 | 100.0% / 0.87 (n=1) | $0.23 | INCOMPLETE (pass rate -50.0%; length in-band 100.0%; length score 0.87; not measured: min_vendi_gain) |

Success criteria, fixed before the run: `min_vendi_gain` = 0.1, `max_pass_rate_drop` = 0.05, `length_tolerance` = 0.2, `min_length_compliance` = 0.6, `min_length_score` = 0.7

### Length response — `control`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| 1400 chars | 1 | 2401 | 1.72 | 0.0% | 0.01 | 100.0% |
| unchanged (baseline) | 1 | 2283 | — | — | — | 100.0% |

### Length response — `length`

| target | n | median | median/target | in-band | score | judge pass |
|---|---|---|---|---|---|---|
| 1400 chars | 1 | 1274 | 0.91 | 100.0% | 0.87 | 0.0% |
| unchanged (baseline) | 1 | 2227 | — | — | — | 100.0% |
