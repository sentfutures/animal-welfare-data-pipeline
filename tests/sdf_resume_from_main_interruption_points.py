"""The interruption points of the test that runs started on origin/main are finished by the current pipeline.

An interruption point says where a run crashes (after how many answered calls of which stage) and how
it is resumed. Crashing "after N calls of stage S" leaves stage S partly written (its first N documents
in deal order, the run being serial) and every earlier stage complete; N of 0 is a crash between stages.
The fixture scenario is the main prompt-equivalence scenario without its injected failures (see
``sdf_resume_from_main_fixture_support.fixture_scenario``).
"""

from dataclasses import dataclass

FIXTURE_SCENARIO_NAME = "main_run_with_removals_failures_and_resume"
CRASH_MESSAGE = "simulated process crash"


@dataclass(frozen=True)
class InterruptionPoint:
    name: str                       # descriptive, used in file names
    description: str                # completes "a run interrupted ..."
    crash_stage: str                # the stage tag whose call number calls_allowed_before_crash + 1 crashes
    calls_allowed_before_crash: int
    extra_arguments: tuple = ()     # added to --resume, e.g. ("--layer", "2")

    @property
    def resume_command_description(self) -> str:
        return " ".join(("--resume", *self.extra_arguments))

    def as_recorded_description(self) -> dict:
        return {"name": self.name, "description": self.description, "crash_stage": self.crash_stage,
                "calls_allowed_before_crash": self.calls_allowed_before_crash,
                "resume_arguments": ["--resume", *self.extra_arguments]}


def all_interruption_points() -> list[InterruptionPoint]:
    return [
        # inside the screen (rounds 0 and 1)
        InterruptionPoint("in_the_screen_after_the_first_batch_of_round_0",
                          "in the screen after the first batch of round 0", "layer1_screen", 1),
        InterruptionPoint("in_the_screen_after_round_0_before_the_redealt_hands_are_screened",
                          "in the screen after round 0, before the redealt hands are screened", "layer1_screen", 2),
        # inside each stage that makes calls
        InterruptionPoint("in_the_middle_of_the_plan_stage",
                          "in the middle of the plan stage", "layer1_plan", 4),
        InterruptionPoint("in_the_middle_of_the_description_rewrite_stage",
                          "in the middle of the description rewrite stage", "layer1_description_rewrite", 4),
        InterruptionPoint("in_the_middle_of_the_consideration_scoring_stage",
                          "in the middle of the consideration scoring stage", "layer1_select", 3),
        InterruptionPoint("in_the_middle_of_the_draft_stage",
                          "in the middle of the draft stage", "layer2_draft", 3),
        InterruptionPoint("in_the_middle_of_the_critique_stage",
                          "in the middle of the critique stage", "layer3_critique", 3),
        InterruptionPoint("in_the_middle_of_the_revise_stage",
                          "in the middle of the revise stage", "layer3_revise", 3),
        InterruptionPoint("in_the_middle_of_the_score_stage",
                          "in the middle of the score stage", "layer4_score", 3),
        # between stages (the first call of the next stage crashes)
        InterruptionPoint("between_the_plan_and_the_description_rewrite_stages",
                          "between the plan and the description rewrite stages", "layer1_description_rewrite", 0),
        InterruptionPoint("between_layer_1_and_the_draft_stage",
                          "between layer 1 and the draft stage", "layer2_draft", 0),
        InterruptionPoint("between_the_draft_and_the_critique_stages",
                          "between the draft and critique stages", "layer3_critique", 0),
        InterruptionPoint("between_layer_3_and_the_score_stage",
                          "between layer 3 and the score stage", "layer4_score", 0),
        # resumed with --layer
        InterruptionPoint("in_the_draft_stage_resumed_from_layer_2",
                          "in the middle of the draft stage", "layer2_draft", 3, ("--layer", "2")),
        InterruptionPoint("in_the_revise_stage_resumed_from_layer_3",
                          "in the middle of the revise stage", "layer3_revise", 3, ("--layer", "3")),
        InterruptionPoint("in_the_score_stage_resumed_from_layer_4",
                          "in the middle of the score stage", "layer4_score", 3, ("--layer", "4")),
        InterruptionPoint("in_the_plan_stage_resumed_from_layer_1_explicitly",
                          "in the middle of the plan stage", "layer1_plan", 4, ("--layer", "1")),
        InterruptionPoint("in_the_draft_stage_resumed_from_layer_3_skipping_the_unfinished_drafts",
                          "in the middle of the draft stage", "layer2_draft", 3, ("--layer", "3")),
    ]
