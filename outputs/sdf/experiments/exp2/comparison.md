# exp2

Donor run `/Users/markstanl/PycharmProjects/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`, n=50, seed 0.

| arm | final/scored | Vendi (ratio) | idea Vendi | form struct / lex | near-dup >0.90 | judge pass | align / real / spec | length in-band / score | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 49/49 | 15.7 (0.321) [10-13] | — (—) | 0.296 / 0.254 | 0.0% | 100.0% | 8.88 / 8.49 / 9.02 | — | $9.10 | control |
| as-shipped | 48/50 | 15.2 (0.316) [10-13] | — (—) | 0.252 / 0.236 | 0.0% | 96.0% | 8.7 / 8.52 / 8.84 | — | $1.63 | INCOMPLETE (Vendi ratio -1.5%; pass rate -4.0%; not measured: min_length_compliance, min_length_score) |
| length | 48/48 | 15.3 (0.319) [10-13] | — (—) | 0.344 / 0.255 | 0.0% | 100.0% | 8.77 / 8.44 / 8.94 | — | $7.59 | INCOMPLETE (Vendi ratio -0.7%; pass rate -0.0%; not measured: min_length_compliance, min_length_score) |

Success criteria, fixed before the run: `min_vendi_gain` = 0.1, `max_pass_rate_drop` = 0.05, `length_tolerance` = 0.2, `min_length_compliance` = 0.6, `min_length_score` = 0.7
