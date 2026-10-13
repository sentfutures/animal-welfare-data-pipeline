# Model rerun — cross-judge results

30 examples from `2026-07-29_12-26_archetype200` (10 lowest / 10 middle / 10 highest by that run's welfare-impact score). Every arm's responses judged by every judge through the repo's own rubric prompts; scores are blended 0–100 (70% holistic, 30% dimension mean). The 'original ref' column is the source run's own claude-opus-5 judge.

## Pipeline arm (the responses under test)

**welfare impact** (mean blended, n per cell in parens)

| generation arm | judge: gpt61sol | judge: opus55 |
|---|---|---|
| original | 73.3 (30) | 79.3 (30) |
| opus55 | 86.7 (30) | 86.5 (30) |
| gpt61sol | 94.9 (30) | 80.5 (30) |

**delivery** (mean blended, n per cell in parens)

| generation arm | judge: gpt61sol | judge: opus55 |
|---|---|---|
| original | 78.8 (30) | 82.0 (30) |
| opus55 | 88.7 (30) | 88.2 (30) |
| gpt61sol | 94.8 (30) | 82.8 (30) |

## Plain baseline (identical text in every run — judge calibration)

**welfare impact** (mean blended, n per cell in parens)

| generation arm | judge: gpt61sol | judge: opus55 |
|---|---|---|
| original | 65.2 (30) | 66.1 (30) |
| opus55 | 65.7 (30) | 66.7 (30) |
| gpt61sol | 65.4 (30) | 65.8 (30) |

**delivery** (mean blended, n per cell in parens)

| generation arm | judge: gpt61sol | judge: opus55 |
|---|---|---|
| original | 78.1 (30) | 82.8 (30) |
| opus55 | 78.3 (30) | 82.5 (30) |
| gpt61sol | 78.9 (30) | 82.7 (30) |

## Same-family bias readout (welfare impact, pipeline arm)

Each judge's preference for the Claude-made over the GPT-made responses, on the same 30 prompts. If the two judges disagree mostly in their own family's favor, that gap is same-family bias.

- judge **gpt61sol** (GPT-family): Claude-made − GPT-made = **-14.9**
- judge **opus55** (Claude-family): Claude-made − GPT-made = **+2.4**
- **difference between the judges (the bias estimate): +17.2 points** — positive means each family favors its own side relative to the other judge

## Per-example — judge gpt61sol, welfare impact (pipeline arm)

| bucket | example | original ref (opus-5 judge) | original | opus55 | gpt61sol |
|---|---|---|---|---|---|
| low | AW-0183 | 42.8 | 65.6 | 81.0 | 95.5 |
| low | AW-0055 | 77.7 | 42.7 | 89.0 | 94.5 |
| low | AW-0187 | 80.1 | 54.0 | 69.8 | 89.5 |
| low | AW-0009 | 80.2 | 65.2 | 86.2 | 91.8 |
| low | AW-0001 | 85.2 | 76.2 | 90.8 | 89.5 |
| low | AW-0138 | 87.0 | 76.1 | 88.2 | 95.2 |
| low | AW-0008 | 87.5 | 81.8 | 78.0 | 88.3 |
| low | AW-0149 | 87.8 | 58.0 | 73.7 | 92.8 |
| low | AW-0172 | 88.0 | 70.0 | 89.8 | 93.4 |
| low | AW-0161 | 88.3 | 77.7 | 86.5 | 96.3 |
| **low mean** |  | **80.4** | **66.7** | **83.3** | **92.7** |
| mid | AW-0013 | 93.1 | 61.2 | 93.3 | 95.3 |
| mid | AW-0014 | 93.1 | 70.1 | 86.0 | 94.5 |
| mid | AW-0027 | 93.1 | 88.4 | 93.3 | 96.3 |
| mid | AW-0031 | 93.1 | 69.3 | 86.5 | 94.5 |
| mid | AW-0048 | 93.1 | 81.3 | 82.5 | 95.4 |
| mid | AW-0069 | 93.1 | 80.7 | 93.3 | 97.0 |
| mid | AW-0095 | 93.1 | 68.5 | 85.2 | 94.6 |
| mid | AW-0121 | 93.1 | 84.6 | 89.5 | 97.2 |
| mid | AW-0165 | 93.1 | 63.0 | 86.0 | 95.3 |
| mid | AW-0016 | 93.2 | 68.8 | 89.2 | 95.3 |
| **mid mean** |  | **93.1** | **73.6** | **88.5** | **95.6** |
| high | AW-0098 | 94.9 | 79.5 | 88.7 | 96.2 |
| high | AW-0103 | 94.9 | 85.6 | 91.3 | 97.1 |
| high | AW-0166 | 95.0 | 88.5 | 87.7 | 94.3 |
| high | AW-0181 | 95.0 | 89.5 | 92.3 | 98.0 |
| high | AW-0004 | 95.0 | 88.8 | 94.1 | 97.2 |
| high | AW-0062 | 95.0 | 69.3 | 87.7 | 96.1 |
| high | AW-0142 | 95.0 | 91.5 | 92.4 | 97.2 |
| high | AW-0047 | 95.1 | 69.0 | 81.5 | 95.4 |
| high | AW-0090 | 95.2 | 61.2 | 82.2 | 94.7 |
| high | AW-0141 | 95.8 | 73.2 | 85.0 | 97.2 |
| **high mean** |  | **95.1** | **79.6** | **88.3** | **96.4** |

## Per-example — judge gpt61sol, delivery (pipeline arm)

| bucket | example | original ref (opus-5 judge) | original | opus55 | gpt61sol |
|---|---|---|---|---|---|
| low | AW-0183 | 63.7 | 62.6 | 93.9 | 97.2 |
| low | AW-0055 | 92.2 | 80.5 | 93.2 | 94.3 |
| low | AW-0187 | 89.5 | 68.9 | 80.0 | 88.5 |
| low | AW-0009 | 80.5 | 88.4 | 89.7 | 96.2 |
| low | AW-0001 | 91.1 | 82.3 | 89.4 | 94.2 |
| low | AW-0138 | 90.3 | 72.8 | 88.5 | 96.2 |
| low | AW-0008 | 87.4 | 79.5 | 89.3 | 95.2 |
| low | AW-0149 | 94.1 | 87.5 | 85.8 | 97.2 |
| low | AW-0172 | 89.5 | 90.2 | 93.8 | 96.0 |
| low | AW-0161 | 92.1 | 88.1 | 88.3 | 96.2 |
| **low mean** |  | **87.0** | **80.1** | **89.2** | **95.1** |
| mid | AW-0013 | 85.4 | 62.6 | 77.2 | 93.2 |
| mid | AW-0014 | 91.4 | 77.7 | 87.5 | 88.5 |
| mid | AW-0027 | 92.2 | 91.2 | 90.1 | 92.1 |
| mid | AW-0031 | 92.9 | 79.2 | 86.3 | 95.1 |
| mid | AW-0048 | 94.3 | 81.5 | 82.5 | 95.0 |
| mid | AW-0069 | 93.0 | 76.5 | 89.8 | 96.0 |
| mid | AW-0095 | 88.1 | 56.8 | 83.6 | 95.2 |
| mid | AW-0121 | 94.2 | 84.7 | 92.1 | 97.2 |
| mid | AW-0165 | 75.0 | 56.7 | 84.2 | 86.6 |
| mid | AW-0016 | 86.3 | 81.3 | 93.0 | 95.1 |
| **mid mean** |  | **89.3** | **74.8** | **86.6** | **93.4** |
| high | AW-0098 | 94.8 | 71.4 | 86.0 | 94.1 |
| high | AW-0103 | 90.8 | 77.7 | 83.7 | 95.9 |
| high | AW-0166 | 94.1 | 90.2 | 91.3 | 96.2 |
| high | AW-0181 | 89.6 | 75.7 | 91.2 | 97.0 |
| high | AW-0004 | 91.2 | 86.4 | 94.0 | 96.2 |
| high | AW-0062 | 95.2 | 82.3 | 94.9 | 95.2 |
| high | AW-0142 | 93.9 | 82.4 | 87.6 | 96.9 |
| high | AW-0047 | 89.1 | 76.7 | 91.0 | 95.2 |
| high | AW-0090 | 93.0 | 87.1 | 88.2 | 96.0 |
| high | AW-0141 | 92.2 | 85.2 | 94.2 | 96.9 |
| **high mean** |  | **92.4** | **81.5** | **90.2** | **96.0** |

## Per-example — judge opus55, welfare impact (pipeline arm)

| bucket | example | original ref (opus-5 judge) | original | opus55 | gpt61sol |
|---|---|---|---|---|---|
| low | AW-0183 | 42.8 | 48.2 | 89.0 | 83.5 |
| low | AW-0055 | 77.7 | 49.9 | 83.2 | 77.3 |
| low | AW-0187 | 80.1 | 50.8 | 85.9 | 84.7 |
| low | AW-0009 | 80.2 | 45.6 | 86.4 | 67.5 |
| low | AW-0001 | 85.2 | 85.6 | 85.6 | 65.9 |
| low | AW-0138 | 87.0 | 72.7 | 87.5 | 72.7 |
| low | AW-0008 | 87.5 | 75.2 | 80.7 | 69.2 |
| low | AW-0149 | 87.8 | 74.4 | 81.7 | 74.7 |
| low | AW-0172 | 88.0 | 83.5 | 85.7 | 79.3 |
| low | AW-0161 | 88.3 | 83.0 | 86.3 | 77.5 |
| **low mean** |  | **80.4** | **66.9** | **85.2** | **75.2** |
| mid | AW-0013 | 93.1 | 82.6 | 87.7 | 83.6 |
| mid | AW-0014 | 93.1 | 85.3 | 87.7 | 80.4 |
| mid | AW-0027 | 93.1 | 89.2 | 87.5 | 85.3 |
| mid | AW-0031 | 93.1 | 87.4 | 85.4 | 79.2 |
| mid | AW-0048 | 93.1 | 87.2 | 86.5 | 76.8 |
| mid | AW-0069 | 93.1 | 89.3 | 89.3 | 85.8 |
| mid | AW-0095 | 93.1 | 83.0 | 86.3 | 81.5 |
| mid | AW-0121 | 93.1 | 80.4 | 85.3 | 88.5 |
| mid | AW-0165 | 93.1 | 78.5 | 85.3 | 81.1 |
| mid | AW-0016 | 93.2 | 80.5 | 85.6 | 81.8 |
| **mid mean** |  | **93.1** | **84.3** | **86.7** | **82.4** |
| high | AW-0098 | 94.9 | 84.4 | 87.3 | 80.5 |
| high | AW-0103 | 94.9 | 88.0 | 89.9 | 87.2 |
| high | AW-0166 | 95.0 | 86.8 | 86.4 | 77.9 |
| high | AW-0181 | 95.0 | 87.0 | 87.3 | 86.5 |
| high | AW-0004 | 95.0 | 87.4 | 87.5 | 85.0 |
| high | AW-0062 | 95.0 | 77.5 | 88.2 | 81.8 |
| high | AW-0142 | 95.0 | 90.2 | 88.3 | 87.7 |
| high | AW-0047 | 95.1 | 89.0 | 84.5 | 85.5 |
| high | AW-0090 | 95.2 | 87.5 | 87.3 | 79.8 |
| high | AW-0141 | 95.8 | 88.5 | 89.0 | 86.7 |
| **high mean** |  | **95.1** | **86.6** | **87.6** | **83.8** |

## Per-example — judge opus55, delivery (pipeline arm)

| bucket | example | original ref (opus-5 judge) | original | opus55 | gpt61sol |
|---|---|---|---|---|---|
| low | AW-0183 | 63.7 | 53.9 | 93.8 | 88.2 |
| low | AW-0055 | 92.2 | 92.9 | 91.1 | 86.6 |
| low | AW-0187 | 89.5 | 71.7 | 80.8 | 71.5 |
| low | AW-0009 | 80.5 | 88.2 | 87.6 | 91.3 |
| low | AW-0001 | 91.1 | 90.2 | 85.0 | 77.2 |
| low | AW-0138 | 90.3 | 79.5 | 92.1 | 80.8 |
| low | AW-0008 | 87.4 | 85.2 | 89.1 | 89.4 |
| low | AW-0149 | 94.1 | 88.5 | 85.0 | 84.5 |
| low | AW-0172 | 89.5 | 90.1 | 90.2 | 88.4 |
| low | AW-0161 | 92.1 | 88.8 | 88.2 | 88.5 |
| **low mean** |  | **87.0** | **82.9** | **88.3** | **84.6** |
| mid | AW-0013 | 85.4 | 68.5 | 82.0 | 77.5 |
| mid | AW-0014 | 91.4 | 75.0 | 85.6 | 69.3 |
| mid | AW-0027 | 92.2 | 89.1 | 91.1 | 82.8 |
| mid | AW-0031 | 92.9 | 86.2 | 87.9 | 85.4 |
| mid | AW-0048 | 94.3 | 91.4 | 89.8 | 76.8 |
| mid | AW-0069 | 93.0 | 86.0 | 89.9 | 92.0 |
| mid | AW-0095 | 88.1 | 69.0 | 91.9 | 82.3 |
| mid | AW-0121 | 94.2 | 92.1 | 93.2 | 93.0 |
| mid | AW-0165 | 75.0 | 55.4 | 74.5 | 42.5 |
| mid | AW-0016 | 86.3 | 78.2 | 84.5 | 73.9 |
| **mid mean** |  | **89.3** | **79.1** | **87.0** | **77.5** |
| high | AW-0098 | 94.8 | 80.1 | 89.2 | 82.2 |
| high | AW-0103 | 90.8 | 81.5 | 89.8 | 90.1 |
| high | AW-0166 | 94.1 | 88.1 | 88.0 | 88.2 |
| high | AW-0181 | 89.6 | 71.7 | 84.8 | 81.1 |
| high | AW-0004 | 91.2 | 84.2 | 89.0 | 88.3 |
| high | AW-0062 | 95.2 | 89.1 | 90.2 | 82.5 |
| high | AW-0142 | 93.9 | 86.9 | 89.1 | 90.1 |
| high | AW-0047 | 89.1 | 82.3 | 87.9 | 85.1 |
| high | AW-0090 | 93.0 | 87.2 | 91.8 | 88.3 |
| high | AW-0141 | 92.2 | 88.2 | 93.1 | 88.0 |
| **high mean** |  | **92.4** | **83.9** | **89.3** | **86.4** |

## Refusals, truncations, and drop-outs (generation arms)

### original (`2026-10-13_02-09_local_rerun-original30`)

- no generation in this run (judge-only arm: the source run's own responses)
- examples with a final record: **30** of 30
- final responses matching refusal phrasing (manual-review flags): 0
- cost: $0.00 (generation) + $1.11 (gpt61sol judge) + $5.27 (opus55 judge)

### opus55 (`2026-10-13_02-17_local_rerun-opus55`)

- generation calls: 125; hard refusals (stop_reason=refusal): **0** (0.0% of calls)
- truncations at max_tokens (each retried once bigger): 5
- examples with a final record: **30** of 30
- final responses matching refusal phrasing (manual-review flags): 0
- cost: $9.11 (generation) + $0.83 (gpt61sol judge) + $5.07 (opus55 judge)

### gpt61sol (`2026-10-13_02-09_local_rerun-gpt61sol`)

- generation calls: 123; hard refusals (stop_reason=refusal): **0** (0.0% of calls)
- truncations at max_tokens (each retried once bigger): 3
- examples with a final record: **30** of 30
- final responses matching refusal phrasing (manual-review flags): 0
- cost: $2.38 (generation) + $1.21 (gpt61sol judge) + $5.24 (opus55 judge)

