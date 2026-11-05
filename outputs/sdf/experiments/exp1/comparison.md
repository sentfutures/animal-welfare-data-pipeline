# exp1

Donor run `/Users/markstanl/PycharmProjects/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`, n=50, seed 0.

| arm | final/scored | Vendi (ratio) | idea Vendi | near-dup >0.90 | judge pass | align / real / spec | length | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|
| control | 49/49 | 15.6 (0.319) [10-13] | 13.8 (0.281) | 0.0% | 100.0% | 8.8 / 8.35 / 9.0 | — | $9.06 | control |
| as-shipped | 50/50 | 15.7 (0.314) [10-13] | 14.3 (0.287) | 0.0% | 100.0% | 8.92 / 8.7 / 9.0 | — | $1.62 | FAIL (Vendi ratio -1.6%; pass rate -0.0%) |
| voices-a1 | 48/49 | 15.5 (0.322) [10-13] | 13.9 (0.290) | 0.0% | 98.0% | 8.69 / 8.41 / 8.92 | — | $6.31 | FAIL (Vendi ratio +1.1%; pass rate -2.0%) |
| personas-a3 | 47/48 | 15.4 (0.328) [10-13] | 13.7 (0.291) | 0.0% | 97.9% | 8.77 / 8.5 / 8.9 | — | $6.34 | FAIL (Vendi ratio +3.0%; pass rate -2.1%) |

Success criteria, fixed before the run: `min_vendi_gain` = 0.1, `max_pass_rate_drop` = 0.05, `length_tolerance` = 0.2
