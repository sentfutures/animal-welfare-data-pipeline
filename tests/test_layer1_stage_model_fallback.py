"""Each layer-1 model stage uses its own model knob, else sdf.plan_model, else the global model.

One parametrized test covers the three stages that share that fallback: the implausibility
screen, the description rewrite and the considerations selector. The stage's model is read
off the call that reached the stubbed shared.api.call_claude.
"""

import pytest

from sdf_pipeline import layer1_description_rewrite as rewrite
from sdf_pipeline import layer1_screen as screen
from sdf_pipeline import layer1_select_considerations as select
from test_layer1_description_rewrite import answer as rewrite_answer
from test_layer1_description_rewrite import make_plans as make_rewrite_plans
from test_layer1_screen import no_removals
from test_layer1_select_considerations import make_plans as make_select_plans
from test_layer1_select_considerations import score_everything_five


def stub_and_runner_for_screen(config, prompts, stub_claude):
    config["sdf"]["n_prompts"] = 3
    return stub_claude(no_removals), lambda directory: screen.run(config, prompts, directory)


def stub_and_runner_for_rewrite(config, prompts, stub_claude):
    plans = make_rewrite_plans(prompts)[:1]
    calls = stub_claude(lambda user_message, **kw: rewrite_answer())
    return calls, lambda directory: rewrite.run(config, prompts, directory, plans)


def stub_and_runner_for_select(config, prompts, stub_claude):
    plans = make_select_plans(2)
    calls = stub_claude(score_everything_five)
    return calls, lambda directory: select.run(config, prompts, directory, plans)


@pytest.mark.parametrize("stub_and_runner, stage_model_knob", [
    (stub_and_runner_for_screen, "screen_model"),
    (stub_and_runner_for_rewrite, "description_rewrite_model"),
    (stub_and_runner_for_select, "select_model"),
], ids=["screen", "rewrite", "select"])
def test_model_falls_back_from_stage_to_plan_to_global(stub_and_runner, stage_model_knob, tiny_config,
                                                       prompts_sdf, tmp_path, stub_claude):
    calls, run_stage = stub_and_runner(tiny_config, prompts_sdf, stub_claude)
    for name, knobs, expected in (
        ("first", {}, None),
        ("second", {"plan_model": "plan-model-x"}, "plan-model-x"),
        ("third", {"plan_model": "plan-model-x", stage_model_knob: "stage-model-x"}, "stage-model-x"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        tiny_config["sdf"].update(knobs)
        run_stage(directory)
        assert calls[-1]["model"] == expected
