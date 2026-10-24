# Designing a persona set

A persona set is a folder `prompts/tools/persona_checks/<persona_set_name>/`. Copy `preachiness/` as a starting point, validate with `python evals/log_probability_persona_check/persona_sets.py --persona-set <name>`, and read `run_folder_layout_and_interfaces.md` ("Persona sets") for the exact formats.

## 1. The persona
- [ ] One sentence of observable behaviour, for example "moralizes about animal welfare in answers where nobody asked". Not a trait or an opinion.
- [ ] Written down: the behaviour it is not, and which documents should show it.
- [ ] A short name for condition ids and columns (`persona_name` in the configuration).

## 2. Rewordings (`target_persona_rewordings.txt`)
- [ ] About 30 lines, one per line. Each completes "You are an AI assistant who ..." and "an AI assistant who ...": lowercase start, a verb, one line, no final full stop.
- [ ] Equivalent in meaning. Vary wording, not intensity, scope or the domain named. Include a few that name the setting and a few that do not, so that no single phrase carries the score.
- [ ] Nothing that names the controls' content, or the corpus's own wording.
- [ ] After a scored pilot, check rewording agreement (mean pairwise Spearman, intraclass correlation, outlier flags). Replace outliers and rerun if the target's agreement is below the controls'.

## 3. Controls
- [ ] Topic controls (`topic_control_personas.txt`): about five personas that share the subject and take no stance on the behaviour. Cover different relationships to the subject (expert, aesthete, indifferent, typical assistant). Header `# persona_name`, then five rewordings.
- [ ] Same-values control (`same_values_control_rewordings.txt`, optional): shares the target's values, lacks the behaviour. About five rewordings. Without it a high score may only mean the document is about the topic. If no such persona exists, say so in the report.
- [ ] Counts in `persona_check_configuration.json` match the files.

## 4. Rewrite instructions
- [ ] `positive_control_rewrite_instructions.txt`: rewrite towards (and optionally away from) the persona, keeping facts, genre, language and length. Slots `{direction}` and `{document_text}`.
- [ ] `neutral_control_rewrite_instructions.txt`: an edit of similar size that changes register only, without adding or removing the behaviour.
- [ ] A person reads both and removes the `# DRAFT:` first line; paid rewriting is refused until then.
- [ ] Away direction: only if enough corpus documents already show the behaviour. Otherwise leave `rewrite_directions.away` out and run `--directions toward neutral`.
- [ ] Check the eligibility rules (documents where the AI barely appears cannot show a change) and the stratifying variable fit the question.

## 5. Pre-registered pass criteria (in the configuration, before scoring)
- [ ] Whole-document and changed-span: minimum number of rewrites moving the expected way, and the number tested per direction.
- [ ] Neutral control: maximum number of rewrites moving towards the persona, at changed-span level.
- [ ] The number of rewrites actually scored must equal the number the criteria were fixed for, or the verdict is "not evaluable".

## 6. Before any paid step
- [ ] Persona set validates; `build_scoring_inputs.py --dry-run` shows the expected condition count.
- [ ] Local rehearsal passes for the model type to be used.
- [ ] Model type chosen, with pinned revisions; the controls will be rerun for that model type.
- [ ] Contributor's explicit go-ahead for each paid step.

## 7. Reading results
- [ ] Controls first, with the level at which they passed.
- [ ] Per-persona changed-span breakdown: the signal should come from the target persona only.
- [ ] Extreme spans read across several documents and several spec variable values.
