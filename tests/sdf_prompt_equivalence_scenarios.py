"""The fixture scenarios of the SDF prompt-equivalence test (engine: sdf_prompt_equivalence_fixture_support.py).

Every scenario is one or more invocations of ``sdf_pipeline/run.py``'s ``main`` with canned replies
(later invocations are resumes). Hands are named by the screen's hand id (``matrix_NNNNNN``), which
is also the prompt_id and doc_id. Failures are injected as ``ReplyOverride``s keyed by (stage, item
id, attempt), the attempt counting across invocations. Interruptions are made by an invocation's
``before`` function, which edits the run's files the way a crash would leave them.
"""

import json
from pathlib import Path

from shared import sdf_run_layouts, textstats, utils
from sdf_pipeline import compose_prompts
from sdf_prompt_equivalence_fixture_support import (
    Invocation, ReplyBehaviour, ReplyOverride, Scenario, stable_pseudo_text)

# ------------------------------------------------------------------------ the main scenario's hands
# Ten hands are dealt (n_prompts 10); the first screen batch removes one and the second removes one,
# and the redeal makes matrix_000010 and matrix_000011 from their cards.
HAND_PLAIN_WITH_CONSIDERATIONS = "matrix_000000"          # description rewritten; first scoring answer unusable
HAND_INCOHERENT_PLAN = "matrix_000001"
HAND_DEALT_VALUE_CONFLICT = "matrix_000002"
HAND_BELOW_THRESHOLD_SCORE = "matrix_000003"
HAND_NO_WELFARE_STAKE = "matrix_000005"                   # dealt by the edited snapshot
HAND_NEAR_DUPLICATE_OF_PLAIN = "matrix_000006"            # its critique has no review tags
HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME = "matrix_000008"
HAND_TRUNCATED_DRAFT = "matrix_000009"
HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME = "matrix_000010"  # a redealt hand, description rewritten
HAND_SCORE_ANSWER_DOES_NOT_PARSE = "matrix_000011"        # a redealt hand, scored nothing by the selector
MAIN_SCENARIO_NAME = "main_run_with_removals_failures_and_resume"
SCREEN_REMOVAL_FIRST_BATCH = "screen_round0_batch0000"
SCREEN_REMOVAL_SECOND_BATCH = "screen_round0_batch0001"

UNUSABLE_JSON = "I could not produce the answer."


def _hand(number: int) -> str:
    return f"matrix_{number:06d}"


def _raise(message: str = "simulated backend failure") -> ReplyOverride:
    return ReplyOverride(raise_error_message=message)


def _truncated(text: str = "cut off") -> ReplyOverride:
    return ReplyOverride(text=text, stop_reason="max_tokens")


def _text(text: str) -> ReplyOverride:
    return ReplyOverride(text=text)


def _main_scenario() -> Scenario:
    behaviour = ReplyBehaviour(
        screen_removals={SCREEN_REMOVAL_FIRST_BATCH: [2], SCREEN_REMOVAL_SECOND_BATCH: [3]},
        plan_kind_by_item={HAND_INCOHERENT_PLAN: "incoherent", HAND_SCORE_ANSWER_DOES_NOT_PARSE: "old_heading"},
        description_rewrite_kind_by_item={
            HAND_DEALT_VALUE_CONFLICT: "conflict", HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME: "rewrite",
            HAND_PLAIN_WITH_CONSIDERATIONS: "rewrite"},
        consideration_scores_by_item={
            # two entries from the moral list and two from the epistemic list on one document
            HAND_PLAIN_WITH_CONSIDERATIONS: {("moral", 0): 5, ("moral", 5): 5, ("epistemic", 2): 5, ("epistemic", 6): 5},
            HAND_NEAR_DUPLICATE_OF_PLAIN: {("moral", 1): 5, ("moral", 6): 5},
            HAND_BELOW_THRESHOLD_SCORE: {("moral", 2): 5, ("epistemic", 3): 5, ("moral", 6): 5, ("epistemic", 5): 5},
            HAND_TRUNCATED_DRAFT: {("moral", 3): 5, ("epistemic", 4): 4, ("epistemic", 5): 5, ("epistemic", 6): 5},
            # the late hand outscores this one for the same entry, so the assignment taken over all
            # documents after the resume leaves it with nothing, after its draft was written with the entry
            HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME: {("moral", 4): 4},
            HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME: {("moral", 4): 5},
        },
        scores_by_item={HAND_BELOW_THRESHOLD_SCORE: (4, 9)},
        revised_text_copied_from={HAND_NEAR_DUPLICATE_OF_PLAIN: HAND_PLAIN_WITH_CONSIDERATIONS},
        revise_without_review_notes=(HAND_PLAIN_WITH_CONSIDERATIONS,),
        overrides={
            ("layer3_consideration_scoring", HAND_PLAIN_WITH_CONSIDERATIONS, 1): _text("I could not score these entries."),
            ("layer2_plan", HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME, 1): _raise("simulated network failure"),
            ("layer4_draft", HAND_TRUNCATED_DRAFT, 1): _truncated("<document>A draft that was cut off"),
            ("layer4_critique", HAND_NEAR_DUPLICATE_OF_PLAIN, 1): _text(
                "  Problems: the opening is generic and the ending is abrupt. (No review tags.)  \n"),
            ("layer4_revise", HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME, 1): _raise(),
            ("layer5_score", HAND_SCORE_ANSWER_DOES_NOT_PARSE, 1): _text("I am unable to score this document."),
        })
    return Scenario(
        name=MAIN_SCENARIO_NAME,
        description="Ten hands, every removal kind, every failure kind, and a resume that retries the failures.",
        invocations=(Invocation(), Invocation(("--resume",))),
        reply_behaviour=behaviour, no_stake_weight_edit="one_in_ten")


def _small_scenario(name: str, description: str, sdf_config_overrides: dict, **scenario_arguments) -> Scenario:
    behaviour = ReplyBehaviour(consideration_scores_by_item={
        _hand(0): {("moral", 0): 5, ("epistemic", 2): 5}, _hand(1): {("moral", 1): 5}})
    return Scenario(name=name, description=description, invocations=(Invocation(),), reply_behaviour=behaviour,
                    sdf_config_overrides={"n_prompts": 3, **sdf_config_overrides}, **scenario_arguments)


# ------------------------------------------------------------------------ interruption helpers

def _remove_records(path: Path, key: str, ids: list[str]) -> None:
    records = utils.load_jsonl(path)
    utils.save_jsonl([r for r in records if r[key] not in ids], path)


def _remove_checkpoint_marks(path: Path, ids: list[str]) -> None:
    if not path.exists():   # the stage wrote no checkpoint, so there is no mark to lose
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["completed"] = [i for i in data["completed"] if i not in ids]
    path.write_text(json.dumps(data), encoding="utf-8")


LAYOUT = sdf_run_layouts.FIVE_NAMED_LAYERS
# What each step's folder key writes (so that deleting "a layer's files" deletes that step's files and not a neighbour's
# in a folder two keys share).
FILES_OF_FOLDER_KEY = {
    sdf_run_layouts.DRAFT: ("drafts.jsonl", LAYOUT.draft_checkpoint_file_name),
    sdf_run_layouts.CRITIQUE_AND_REVISE: ("critiques.jsonl", "rewrites.jsonl", "_checkpoint.json"),
    sdf_run_layouts.SCORE: ("scores.jsonl", "_checkpoint.json", "near_dup_dropped.jsonl"),
}


def _delete_stage_files(run_directory: Path, folder_keys: tuple) -> None:
    for folder_key in folder_keys:
        for name in FILES_OF_FOLDER_KEY[folder_key]:
            (utils.sdf_stage_folder(run_directory, folder_key) / name).unlink(missing_ok=True)


def _stage_path(run_directory: Path, folder_key: str, name: str) -> Path:
    return utils.sdf_stage_folder(run_directory, folder_key) / name


DRAFT, REVISE_FOLDER, SCORE_FOLDER = sdf_run_layouts.DRAFT, sdf_run_layouts.CRITIQUE_AND_REVISE, sdf_run_layouts.SCORE
PLAN_FOLDER, SCREEN_FOLDER = sdf_run_layouts.PLAN_AND_DESCRIPTION_REWRITE, sdf_run_layouts.DEAL_AND_SCREEN


# ------------------------------------------------------------------------ --layer restarts

def _restart_layer_2(run_directory):
    _delete_stage_files(run_directory, (DRAFT, REVISE_FOLDER, SCORE_FOLDER))


def _restart_layer_3_with_an_explicit_revision_record(run_directory):
    _delete_stage_files(run_directory, (REVISE_FOLDER, SCORE_FOLDER))
    drafts_path = _stage_path(run_directory, DRAFT, "drafts.jsonl")
    drafts = utils.load_jsonl(drafts_path)
    # the revision override that seeding scripts write: hand 0 is revised with hand 1's dealt cards
    drafts[0]["revision"] = {"cards": compose_prompts.revision_cards(drafts[1]["variables"])}
    utils.save_jsonl(drafts, drafts_path)


def _restart_layer_4(run_directory):
    _delete_stage_files(run_directory, (SCORE_FOLDER,))


def _restart_layer_2_with_a_held_out_description_rewrite(run_directory):
    _delete_stage_files(run_directory, (DRAFT, REVISE_FOLDER, SCORE_FOLDER))
    _remove_records(_stage_path(run_directory, PLAN_FOLDER, "description_rewrites.jsonl"), "prompt_id", [_hand(2)])


def _layer_restarts_scenario() -> Scenario:
    behaviour = ReplyBehaviour(
        description_rewrite_kind_by_item={_hand(0): "rewrite", _hand(1): "rewrite"},
        consideration_scores_by_item={_hand(0): {("moral", 0): 5}, _hand(1): {("epistemic", 2): 5},
                                      _hand(3): {("moral", 1): 5}})
    return Scenario(
        name="layer_restarts_reload_earlier_stages_from_files",
        description="A complete run, then --resume --layer 2, 3 and 4 after deleting the later stages' files "
                    "(layer 3 with an explicit revision record on one draft), then --layer 2 with a held-out "
                    "description rewrite.",
        invocations=(
            Invocation(),
            Invocation(("--resume", "--layer", "4"), before=_restart_layer_2),
            Invocation(("--resume", "--layer", "4"), before=_restart_layer_3_with_an_explicit_revision_record),
            Invocation(("--resume", "--layer", "5"), before=_restart_layer_4),
            Invocation(("--resume", "--layer", "4"), before=_restart_layer_2_with_a_held_out_description_rewrite),
        ),
        reply_behaviour=behaviour, sdf_config_overrides={"n_prompts": 4, "screen_enabled": False})


# ------------------------------------------------------------------------ gate and cull boundaries

def _text_with_cosine_in_range(base_text: str, lowest: float, highest: float) -> str:
    """The base text with a growing number of words replaced, until its shingle cosine to the base is
    in [lowest, highest) (the cull's measure)."""
    words = base_text.split()
    for replaced in range(1, len(words) // 2):
        changed = list(words)
        for position in range(replaced):
            changed[3 + position * 2] = f"altered{position}"
        candidate = " ".join(changed)
        vectors = textstats.shingle_matrix([base_text, candidate])
        similarity = round(float(vectors[0] @ vectors[1]), 4)
        if lowest <= similarity < highest:
            return candidate
    raise AssertionError(f"no variant of the text has a cosine in [{lowest}, {highest})")


def _gate_and_cull_scenario() -> Scenario:
    base_text = stable_pseudo_text("revised " + _hand(5))
    behaviour = ReplyBehaviour(
        scores_by_item={_hand(0): (7, 7), _hand(1): (7, 9), _hand(2): (9, 7), _hand(3): (9, 6), _hand(4): (6, 9)},
        revised_text_overrides={
            _hand(6): _text_with_cosine_in_range(base_text, 0.92, 0.98),    # above the 0.90 cull: dropped
            _hand(7): _text_with_cosine_in_range(base_text, 0.86, 0.9)})    # just under it: kept
    return Scenario(
        name="gate_and_near_duplicate_boundaries",
        description="Scores exactly at the threshold on each dimension, one below on each, and near-duplicates "
                    "just above and just below the 0.90 cull.",
        invocations=(Invocation(),), reply_behaviour=behaviour,
        sdf_config_overrides={"n_prompts": 8, "screen_enabled": False})


# ------------------------------------------------------------------------ interrupted runs

def _drop_records_and_marks_as_a_crash_leaves_them(run_directory):
    """Record without its checkpoint mark (the stage then counts it done), and a mark without its
    record (the stage then does not repeat it), in each of plan, draft, rewrite and score."""
    plan_checkpoint = _stage_path(run_directory, PLAN_FOLDER, "_checkpoint.json")
    _remove_records(_stage_path(run_directory, PLAN_FOLDER, "plans.jsonl"), "prompt_id", [_hand(0)])
    _remove_checkpoint_marks(plan_checkpoint, [_hand(1)])
    _remove_records(_stage_path(run_directory, DRAFT, "drafts.jsonl"), "doc_id", [_hand(2)])
    _remove_checkpoint_marks(_stage_path(run_directory, DRAFT, LAYOUT.draft_checkpoint_file_name), [_hand(3)])
    _remove_records(_stage_path(run_directory, REVISE_FOLDER, "rewrites.jsonl"), "doc_id", [_hand(4)])
    _remove_checkpoint_marks(_stage_path(run_directory, REVISE_FOLDER, "_checkpoint.json"), [_hand(5)])
    _remove_records(_stage_path(run_directory, SCORE_FOLDER, "scores.jsonl"), "doc_id", [_hand(3)])
    _remove_checkpoint_marks(_stage_path(run_directory, SCORE_FOLDER, "_checkpoint.json"), [_hand(5)])


def _delete_screen_checkpoint(run_directory):
    _stage_path(run_directory, SCREEN_FOLDER, "_screen_checkpoint.json").unlink()


def _interrupted_screen_scenario() -> Scenario:
    behaviour = ReplyBehaviour(
        screen_removals={"screen_round0_batch0000": [2], "screen_round0_batch0001": [1]},
        overrides={("layer1_screen", "screen_round0_batch0001", 1): _raise(),
                   ("layer1_screen", "screen_round1_batch0000", 1): _raise()})
    return Scenario(
        name="interrupted_screen_rounds_resume_with_a_different_batch_size",
        description="The screen stops after one of two round-0 batches; the resume has another batch size and "
                    "must reuse the saved batches, then stops after the redeal is dealt but not screened; the "
                    "third invocation reuses the dealt round; the fourth has lost the screen checkpoint but "
                    "keeps the accepted hands.",
        invocations=(
            Invocation(expected_exit_message_fragment="of round 0 failed"),
            Invocation(("--resume",), sdf_config_overrides={"screen_batch_size": 2},
                       expected_exit_message_fragment="every pending screen call failed"),
            Invocation(("--resume",), sdf_config_overrides={"screen_batch_size": 2}),
            Invocation(("--resume",), before=_delete_screen_checkpoint, sdf_config_overrides={"screen_batch_size": 2}),
        ),
        reply_behaviour=behaviour, sdf_config_overrides={"n_prompts": 6, "screen_batch_size": 3})


def _interrupted_records_scenario() -> Scenario:
    return Scenario(
        name="interrupted_run_records_and_checkpoint_marks_disagree",
        description="A complete run whose plan, draft, rewrite and score files and checkpoints are then edited so "
                    "that a record has no mark and a mark has no record; the resume must send no request.",
        invocations=(Invocation(), Invocation(("--resume",), before=_drop_records_and_marks_as_a_crash_leaves_them)),
        reply_behaviour=ReplyBehaviour(),
        sdf_config_overrides={"n_prompts": 6, "screen_enabled": False})


# ------------------------------------------------------------------------ failure kinds and stop rules

def _failure_scenario(name, description, invocations, overrides, n_prompts, behaviour_arguments=None,
                      screen_enabled=False, **config_overrides) -> Scenario:
    behaviour = ReplyBehaviour(overrides=overrides, **(behaviour_arguments or {}))
    return Scenario(name=name, description=description, invocations=tuple(invocations), reply_behaviour=behaviour,
                    sdf_config_overrides={"n_prompts": n_prompts, "screen_enabled": screen_enabled, **config_overrides})


def _failure_scenarios() -> list[Scenario]:
    h0, h1, h2 = _hand(0), _hand(1), _hand(2)
    resume = Invocation(("--resume",))
    scores_for_two = {"consideration_scores_by_item": {h0: {("moral", 0): 5}, h1: {("epistemic", 2): 5}}}
    return [
        _failure_scenario(
            "failures_in_the_screen_across_resumes",
            "Three single-hand screen batches: an exception, a truncated answer and an unusable answer, each "
            "stopping the run before the redeal, then a clean resume.",
            [Invocation(expected_exit_message_fragment="2 batch(es) of round 0 failed"),
             Invocation(("--resume",), expected_exit_message_fragment="1 batch(es) of round 0 failed"), resume],
            {("layer1_screen", "screen_round0_batch0000", 1): _raise(),
             ("layer1_screen", "screen_round0_batch0001", 1): _truncated("{"),
             ("layer1_screen", "screen_round0_batch0000", 2): _text(UNUSABLE_JSON)},
            3, screen_enabled=True, screen_batch_size=1),
        _failure_scenario(
            "every_screen_call_fails", "The only screen batch throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="every pending screen call failed"), resume],
            {("layer1_screen", "screen_round0_batch0000", 1): _raise()}, 2, screen_enabled=True, screen_batch_size=2),
        _failure_scenario(
            "failures_in_the_plan_across_resumes",
            "A truncated plan, a plan with no description, then a thrown call on the resume: documents are left "
            "behind and the run carries on.",
            [Invocation(), resume, resume],
            {("layer2_plan", h0, 1): _truncated("<document_planning>cut"),
             ("layer2_plan", h1, 1): _text("Notes but no description tags."),
             ("layer2_plan", h0, 2): _raise()}, 3),
        _failure_scenario(
            "every_plan_call_fails", "Every plan call throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="layer2_plan: every pending API call failed"), resume],
            {("layer2_plan", h0, 1): _raise(), ("layer2_plan", h1, 1): _raise()}, 2),
        _failure_scenario(
            "failures_in_the_description_rewrite_across_resumes",
            "A thrown rewrite call and a truncated answer hold two descriptions out and stop the run after the "
            "step; the resume gets an unusable answer for one; the third completes.",
            [Invocation(expected_exit_message_fragment="2 descriptions held out"),
             Invocation(("--resume",), expected_exit_message_fragment="1 descriptions held out"), resume],
            {("layer2_description_rewrite", h0, 1): _raise(), ("layer2_description_rewrite", h1, 1): _truncated("{"),
             ("layer2_description_rewrite", h0, 2): _text(UNUSABLE_JSON)}, 3),
        _failure_scenario(
            "every_description_rewrite_call_fails", "Every rewrite call throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="layer2_description_rewrite: every pending call failed"), resume],
            {("layer2_description_rewrite", h0, 1): _raise(), ("layer2_description_rewrite", h1, 1): _raise()}, 2),
        _failure_scenario(
            "failures_in_consideration_scoring_across_resumes",
            "A thrown scoring call, and a document whose three attempts are all unusable: the run stops before "
            "the assignment; the resume scores both.",
            [Invocation(expected_exit_message_fragment="stopping before the assignment"), resume],
            {("layer3_consideration_scoring", h0, 1): _raise(), ("layer3_consideration_scoring", h1, 1): _truncated("{"),
             ("layer3_consideration_scoring", h1, 2): _text(UNUSABLE_JSON), ("layer3_consideration_scoring", h1, 3): _text(UNUSABLE_JSON)}, 3,
            {"consideration_scores_by_item": {h0: {("moral", 0): 5}, h2: {("epistemic", 2): 5}}}),
        _failure_scenario(
            "every_consideration_scoring_call_fails", "Every scoring call throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="every pending scoring call failed"), resume],
            {("layer3_consideration_scoring", h0, 1): _raise(), ("layer3_consideration_scoring", h1, 1): _raise()}, 2),
        _failure_scenario(
            "failures_in_the_draft_across_resumes",
            "A thrown draft call and a draft with no tags: documents are left behind, the run carries on.",
            [Invocation(), resume],
            {("layer4_draft", h0, 1): _raise(), ("layer4_draft", h1, 1): _text("A draft with no tags.")}, 3,
            scores_for_two),
        _failure_scenario(
            "every_draft_call_fails", "Every draft call throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="layer4_draft: every pending API call failed"), resume],
            {("layer4_draft", h0, 1): _raise(), ("layer4_draft", h1, 1): _raise()}, 2),
        _failure_scenario(
            "unusable_draft_replies_are_not_a_systemic_stop",
            "Every draft has no tags (not thrown calls): the run does not stop.",
            [Invocation(), resume],
            {("layer4_draft", h0, 1): _text("No tags."), ("layer4_draft", h1, 1): _text("No tags either.")}, 2),
        _failure_scenario(
            "failures_in_the_critique_and_the_revise_across_resumes",
            "A thrown, a truncated and an empty critique (only the thrown one counts as a failed call); then a "
            "truncated and a tagless revise; then clean.",
            [Invocation(), resume, resume],
            {("layer4_critique", h0, 1): _raise(), ("layer4_critique", h1, 1): _truncated("<review>cut"),
             ("layer4_critique", h2, 1): _text("<review>   </review>"),
             ("layer4_revise", h0, 1): _truncated("<improved_document>cut"),
             ("layer4_revise", h1, 1): _text("A revision with no tags.")}, 3),
        _failure_scenario(
            "every_critique_call_fails", "Every critique call throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="layer4_draft_critique_and_revise: every pending API call failed"), resume],
            {("layer4_critique", h0, 1): _raise(), ("layer4_critique", h1, 1): _raise()}, 2),
        _failure_scenario(
            "failures_in_the_score_across_resumes",
            "One thrown score call (the document is left behind, then retried) and one complete answer whose stop "
            "reason is max_tokens (scoring does not check it, so the scores are used).",
            [Invocation(), resume],
            {("layer5_score", h0, 1): _raise(),
             ("layer5_score", h1, 1): ReplyOverride(
                 text='{"alignment": 8, "realism": 8, "spec_conformance": 8, "notes": "cut off after the answer"}',
                 stop_reason="max_tokens")}, 3),
        _failure_scenario(
            "every_score_call_fails", "Every score call throws: a systemic stop.",
            [Invocation(expected_exit_message_fragment="layer5_score: every pending API call failed"), resume],
            {("layer5_score", h0, 1): _raise(), ("layer5_score", h1, 1): _raise()}, 2),
    ]


# ------------------------------------------------------------------------ configuration scenarios

def _screen_removal_that_cannot_be_redealt() -> Scenario:
    return Scenario(
        name="screen_removal_that_cannot_be_redealt",
        description="One hand removed: reshuffling one hand's cards gives the same hand, so its cards are dropped.",
        invocations=(Invocation(),),
        reply_behaviour=ReplyBehaviour(screen_removals={"screen_round0_batch0000": [1]}),
        sdf_config_overrides={"n_prompts": 3, "screen_batch_size": 3})


def _configuration_scenarios() -> list[Scenario]:
    behaviour = ReplyBehaviour(
        screen_removals={"screen_round0_batch0000": [2, 5]},
        description_rewrite_kind_by_item={_hand(0): "rewrite"},
        consideration_scores_by_item={_hand(0): {("moral", 0): 5, ("epistemic", 2): 5}, _hand(1): {("moral", 1): 5}})
    small_scores = ReplyBehaviour(consideration_scores_by_item={
        _hand(0): {("moral", 0): 5, ("epistemic", 2): 5}, _hand(1): {("moral", 1): 5}})
    models_removed = ("plan_model", "select_model", "draft_model", "rewrite_model", "score_model",
                      "description_rewrite_model")
    return [
        Scenario(
            name="real_config_unchanged_except_the_prompt_count",
            description="config.yaml exactly as committed except sdf.n_prompts (100 to 12): every stage model, "
                        "effort and default as the real file has them, nothing else changed.",
            invocations=(Invocation(),), reply_behaviour=behaviour, config_kind="real",
            sdf_config_overrides={"n_prompts": 12}),
        Scenario(
            name="minimal_config_with_only_a_plan_model",
            description="Only sdf.n_prompts (12), sdf.min_score_threshold and sdf.plan_model: every fallback chain "
                        "(screen, select, description rewrite to the plan model; draft, rewrite, score to the "
                        "global model) and every code default (screen batch size and redeal rounds, scoring "
                        "attempts, seed, entity seed, revision style) is used. The screen removes two hands in "
                        "each of rounds 0, 1 and 2; one document's scoring answer is unusable twice.",
            invocations=(Invocation(),),
            reply_behaviour=ReplyBehaviour(
                screen_removals={"screen_round0_batch0000": [2, 5], "screen_round1_batch0000": [1, 2],
                                 "screen_round2_batch0000": [1, 2]},
                consideration_scores_by_item={_hand(0): {("moral", 0): 5, ("epistemic", 2): 5}, _hand(1): {("moral", 1): 5}},
                overrides={("layer3_consideration_scoring", _hand(0), 1): _text(UNUSABLE_JSON),
                           ("layer3_consideration_scoring", _hand(0), 2): _text(UNUSABLE_JSON)}),
            config_kind="minimal",
            sdf_config_overrides={"n_prompts": 12, "plan_model": "claude-haiku-4-5"}),
        Scenario(
            name="minimal_config_with_openai_rewrite_and_adaptive_thinking_models",
            description="Only the required keys plus an OpenAI rewrite model, a Sonnet 5.5 draft model and an "
                        "Opus 5.5 rewrite model: the default reasoning effort, the default thinking headroom "
                        "and the no-stake skip default, on a hand dealt the no-stake resolution; the screen is disabled, so the "
                        "planner deals the hands and the default seed is read there.",
            invocations=(Invocation(),), reply_behaviour=small_scores, config_kind="minimal",
            sdf_config_overrides={"n_prompts": 5, "description_rewrite_model": "gpt-6.1-sol",
                                  "draft_model": "claude-sonnet-5-5", "rewrite_model": "claude-opus-5-5",
                                  "screen_enabled": False},
            no_stake_weight_edit="one_in_five"),
        Scenario(
            name="temperature_kept_on_every_stage_with_a_two_call_revision",
            description="Every stage, the Claude-routed description rewrite included, on a model that keeps "
                        "sampling parameters, with a global temperature of 0.5.",
            invocations=(Invocation(),), reply_behaviour=small_scores,
            sdf_config_overrides={"n_prompts": 3}, sdf_config_keys_to_remove=models_removed,
            top_level_config_overrides={"model": "claude-haiku-4-5", "temperature": 0.5}),
        Scenario(
            name="temperature_kept_on_every_stage_with_a_one_call_revision",
            description="As above with sdf.revision.split false: the single-call revision keeps the temperature.",
            invocations=(Invocation(),), reply_behaviour=small_scores,
            sdf_config_overrides={"n_prompts": 3, "revision": {"split": False, "style_mode": "axes", "use_length": True}},
            sdf_config_keys_to_remove=models_removed,
            top_level_config_overrides={"model": "claude-haiku-4-5", "temperature": 0.5}),
    ]


def _deal_repair_scenario() -> Scenario:
    return Scenario(
        name="deal_repair_of_five_no_stake_hands_with_every_plan_call_failing",
        description="Twenty-five hands with five dealt the no-stake resolution, all breaking the centrality deal "
                    "rule and repaired by seeded swaps; every plan call then throws, so the run stops after the "
                    "plan requests, which show the repaired hands.",
        invocations=(Invocation(expected_exit_message_fragment="layer2_plan: every pending API call failed"),),
        reply_behaviour=ReplyBehaviour(overrides={("layer2_plan", _hand(n), 1): _raise() for n in range(40)}),
        config_kind="minimal", sdf_config_overrides={"n_prompts": 25}, no_stake_weight_edit="one_in_five")


def _pad_plan_descriptions_and_forget_the_rewrite(run_directory):
    """A resume over plan records whose descriptions carry stray whitespace (as if written by older
    code or edited by hand), with the description rewrite's results gone."""
    plans_path = _stage_path(run_directory, PLAN_FOLDER, "plans.jsonl")
    plans = utils.load_jsonl(plans_path)
    for plan in plans:
        plan["description"] = f"\n\n   {plan['description']}   \n"
    utils.save_jsonl(plans, plans_path)
    _stage_path(run_directory, PLAN_FOLDER, "description_rewrites.jsonl").unlink()
    _stage_path(run_directory, PLAN_FOLDER, "_description_rewrite_checkpoint.json").unlink()
    _delete_stage_files(run_directory, (DRAFT, REVISE_FOLDER, SCORE_FOLDER))


def _padded_descriptions_scenario() -> Scenario:
    return Scenario(
        name="resume_over_plans_with_padded_descriptions",
        description="A complete run, then a resume after the plan descriptions were padded with whitespace and the "
                    "description rewrite and later stages were lost.",
        invocations=(Invocation(), Invocation(("--resume",), before=_pad_plan_descriptions_and_forget_the_rewrite)),
        reply_behaviour=ReplyBehaviour(), sdf_config_overrides={"n_prompts": 3, "screen_enabled": False})


def all_scenarios() -> list[Scenario]:
    return [
        _main_scenario(),
        _small_scenario("screen_disabled", "sdf.screen_enabled false: the composer deals its own hands.",
                        {"screen_enabled": False}),
        _small_scenario("description_rewrite_disabled", "sdf.description_rewrite_enabled false.",
                        {"description_rewrite_enabled": False}),
        _small_scenario("consideration_selection_disabled", "sdf.select_enabled false: no considerations.",
                        {"select_enabled": False}),
        _small_scenario("revision_in_one_call", "sdf.revision.split false: the single-call layer 3.",
                        {"revision": {"split": False, "style_mode": "axes", "use_length": True}}),
        _small_scenario("description_rewrite_on_a_claude_model",
                        "The description rewrite routed to a Claude model through call_claude.",
                        {"description_rewrite_model": "claude-sonnet-5"}),
        _layer_restarts_scenario(),
        _gate_and_cull_scenario(),
        _interrupted_screen_scenario(),
        _interrupted_records_scenario(),
        _screen_removal_that_cannot_be_redealt(),
        *_configuration_scenarios(),
        _deal_repair_scenario(),
        _padded_descriptions_scenario(),
        *_failure_scenarios(),
    ]


def scenario_named(name: str) -> Scenario:
    return next(scenario for scenario in all_scenarios() if scenario.name == name)
