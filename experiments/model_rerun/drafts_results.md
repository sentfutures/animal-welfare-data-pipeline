# Draft vs final — what the step-3 rewrite adds

Step-2 drafts (pre-rewrite) and final responses for the same 30 examples, judged by the same three judges. Scores are mean blended (0-100), pipeline arm.

## welfare impact

| arm | judge | draft | final | rewrite delta |
|---|---|---|---|---|
| original | opus55 | 80.3 | 79.3 | **-1.0** |
| original | gpt61sol | 73.1 | 73.3 | **+0.2** |
| original | gemini31pro | 96.6 | 96.3 | **-0.3** |
| opus55 | opus55 | 86.5 | 86.5 | **-0.0** |
| opus55 | gpt61sol | 85.2 | 86.7 | **+1.5** |
| opus55 | gemini31pro | 98.1 | 97.1 | **-1.0** |
| gpt61sol | opus55 | 80.8 | 80.5 | **-0.3** |
| gpt61sol | gpt61sol | 94.6 | 94.9 | **+0.2** |
| gpt61sol | gemini31pro | 96.0 | 96.0 | **+0.0** |

## delivery

| arm | judge | draft | final | rewrite delta |
|---|---|---|---|---|
| original | opus55 | 77.7 | 82.0 | **+4.3** |
| original | gpt61sol | 72.5 | 78.8 | **+6.3** |
| original | gemini31pro | 98.2 | 99.3 | **+1.0** |
| opus55 | opus55 | 85.8 | 88.2 | **+2.4** |
| opus55 | gpt61sol | 85.1 | 88.7 | **+3.6** |
| opus55 | gemini31pro | 99.7 | 99.4 | **-0.3** |
| gpt61sol | opus55 | 81.6 | 82.8 | **+1.3** |
| gpt61sol | gpt61sol | 94.0 | 94.8 | **+0.8** |
| gpt61sol | gemini31pro | 98.3 | 96.7 | **-1.7** |

## Same-family bias on DRAFTS (welfare impact)

Claude-made minus GPT-made drafts, per judge:

- judge **opus55**: +2.6
- judge **gpt61sol**: -15.5
- judge **gemini31pro**: +1.4
