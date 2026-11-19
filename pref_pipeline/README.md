# pref_pipeline/

Preference pairs: one response from each of two arms per input prompt, then blind human A/B
rating. Use it to A/B test candidate response specs against each other or against the bare model.
(Moved from the repository's `CLAUDE.md`.)

`pref_pipeline/run.py` generates one pair per input prompt: a response from each of two arms
defined in `config.yaml` under `pref.arms` (`name` + inline `system_prompt` or
`system_prompt_file` relative to the repo root, optional per-arm `model`/`max_tokens`). Prompts
come from any JSONL with a `user_message`, `refined`, or `prompt` field (handwritten sets, DAD
step-1 `dilemmas.jsonl`). Runs live in `outputs/pref/runs/<run_id>/` with the same
manifest/checkpoint/resume/cost-log conventions as SDF/DAD; resolved arms are frozen into
`inputs/arm_prompts.yaml` at run creation so `--resume` replays them. Checkpointing is per **arm**
(`pairs/arm_responses.jsonl`), so one failed arm never discards or re-bills its sibling's paid
response.

```bash
python pref_pipeline/run.py --config config.yaml --prompts <prompts.jsonl> --label spec-v1-vs-plain
streamlit run pref_pipeline/rate.py
```

`streamlit run pref_pipeline/rate.py` is the blind rating UI: arm identities are hidden, side
order is fixed per pair (md5 of `pair_id` → `left_arm`, so it carries no signal but survives
reloads), choices are Response 1 / Response 2 / Tie / Both bad plus an optional note, keyed by
rater name. Ratings append to `ratings/ratings.jsonl` (both the blinded side and the deblinded
arm); after every rating `final/preferences.jsonl` is rebuilt with one
`{user_message, chosen, rejected, chosen_arm_name, rater}` record per decisive rating
(ties/both-bad excluded). Data logic lives in `pref_pipeline/prefdata.py` (no Streamlit imports).
