"""Tests for shared/deal_constraints.py and its use in the SDF composer
(offline, no API). Real variables.txt, deal_constraints.txt and layer1.txt
stay in play; the no-stake resolution is turned back up through a tmp copy of
variables.txt, because at weight zero the no-stake rules have nothing to do."""

import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer1_plan
from shared import deal_constraints as dc
from shared import matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_SDF = REPO_ROOT / "prompts" / "sdf"
REAL_RULES_PATH = PROMPTS_SDF / "deal_constraints.txt"
NO_STAKE_MARKER = "NO real welfare stake"
SEEDS = range(50)
DECK_SIZES = (20, 100, 300)  # x 50 seeds; one larger size is covered below with fewer seeds


def load_real_matrix(variables_path=PROMPTS_SDF / "variables.txt"):
    values, weights = matrix.split_weights(cp.parse_variables(variables_path))
    axes = cp.matrix_axes((PROMPTS_SDF / "layer1.txt").read_text(encoding="utf-8"))
    axes = axes + [name for name in values if name not in axes]
    return values, weights, axes


def write_variables_with_no_stake_turned_up(tmp_path, no_stake_weight=0.05):
    """A tmp copy of variables.txt with the no-stake resolution at the given
    weight, taking it back from the four 0.14 and the 0.11 resolutions."""
    lines = (PROMPTS_SDF / "variables.txt").read_text(encoding="utf-8").splitlines()
    edited = []
    in_resolution_block = False
    for line in lines:
        if line.startswith("{"):
            in_resolution_block = line.startswith("{resolution}")
        if in_resolution_block and NO_STAKE_MARKER in line:
            line = line.replace("0.00 ::", f"{no_stake_weight:.2f} ::")
        elif in_resolution_block and re.match(r"\s+0\.14 ::", line):
            line = line.replace("0.14 ::", "0.13 ::")
        elif in_resolution_block and re.match(r"\s+0\.11 ::", line):
            line = line.replace("0.11 ::", "0.10 ::")
        edited.append(line)
    path = tmp_path / "variables_with_no_stake_turned_up.txt"
    path.write_text("\n".join(edited), encoding="utf-8")
    return path


def deal_and_repair(values, weights, axes, rules, n_prompts, seed):
    hands = matrix.deck_sample(values, weights, axes, n_prompts, random.Random(seed))
    repaired, report = dc.swap_repair(hands, rules, random.Random(f"{seed}:repair"))
    return hands, repaired, report


def column_counts(hands, axes):
    return {axis: Counter(hand[axis] for hand in hands) for axis in axes}


def all_violations(rules, hands):
    return [(rule.name, i) for i, hand in enumerate(hands) for rule in rules if rule.is_violated_by(hand)]


@pytest.fixture(scope="module")
def real_matrix():
    return load_real_matrix()


@pytest.fixture(scope="module")
def real_rules(real_matrix):
    values, _, _ = real_matrix
    return dc.load_deal_constraints(REAL_RULES_PATH, values)


# --- loading and validation ---

def test_real_rules_file_loads_both_kinds_of_rule(real_rules):
    kinds = Counter(rule.kind for rule in real_rules)
    assert kinds[dc.SWAP_KIND] == 4 and kinds[dc.RENDER_KIND] == 1
    render_rule = next(rule for rule in real_rules if rule.kind == dc.RENDER_KIND)
    assert set(render_rule.omitted_variables) == {"tradeoff", "decision_scale"}
    assert len({rule.name for rule in real_rules}) == len(real_rules)


def test_unmatched_substring_raises_naming_rule(real_matrix):
    values, _, _ = real_matrix
    text = "[broken_rule]\nwhen resolution is one of:\n    no such value anywhere\nrequire centrality is one of:\n    a minor detail\n"
    with pytest.raises(ValueError, match=r"broken_rule.*no such value anywhere.*matches 0"):
        dc.parse_deal_constraints(text, values)


def test_ambiguous_substring_raises(real_matrix):
    values, _, _ = real_matrix
    text = "[broken_rule]\nwhen resolution is one of:\n    The document features a specific scenario\nrequire centrality is one of:\n    a minor detail\n"
    with pytest.raises(ValueError, match=r"broken_rule.*exactly one.*matches \d+"):
        dc.parse_deal_constraints(text, values)


def test_reworded_value_breaks_loading_of_real_rules(real_matrix):
    """The drift guard: reword a value in variables.txt and the real rules refuse to load."""
    values, _, _ = real_matrix
    reworded = {name: list(vals) for name, vals in values.items()}
    reworded["naming"] = [v.replace("Claude, identified by name", "Claude (named)") for v in reworded["naming"]]
    with pytest.raises(ValueError, match="identity_documents_name_claude"):
        dc.load_deal_constraints(REAL_RULES_PATH, reworded)


@pytest.mark.parametrize("text, message", [
    ("when resolution is one of:\n    x\n", "expected a"),
    ("[r]\nwhen tone is one of:\n    neutral\n", "exactly one"),
    ("[r]\nwhen tone is one of:\n    neutral\nrequire tone is one of:\n    skeptical\n", "same variable"),
    ("[r]\nwhen nope is one of:\n    neutral\nrequire tone is one of:\n    skeptical\n", "unknown variable"),
    ("[r]\nwhen tone is one of:\n    neutral\nomit planning-prompt lines for:\n    nope\n", "unknown variable"),
    ("[r]\nwhen tone is one of:\n    neutral\nrequire culture is one of:\n    France\nrequire culture is one of:\n    Japan\n", "exactly one"),
    ("[r]\nwhen tone is one of:\n    neutral\nrequire culture is one of:\n    France\n[r]\n", "duplicate rule"),
    ("[r]\nwhen tone is one of:\n    neutral\nbogus line\n", "unrecognised directive"),
])
def test_malformed_rules_files_raise(text, message):
    values = {"tone": ["neutral", "skeptical"], "culture": ["France", "Japan"]}
    with pytest.raises(ValueError, match=message):
        dc.parse_deal_constraints(text, values)


# --- repair: zero violations, exact counts, determinism ---

@pytest.mark.parametrize("n_prompts", DECK_SIZES)
def test_repair_leaves_zero_violations_across_seeds(real_matrix, real_rules, n_prompts):
    values, weights, axes = real_matrix
    for seed in SEEDS:
        _, repaired, _ = deal_and_repair(values, weights, axes, real_rules, n_prompts, seed)
        assert all_violations(real_rules, repaired) == [], (seed, n_prompts)


@pytest.mark.parametrize("n_prompts", DECK_SIZES)
def test_repair_leaves_zero_violations_with_no_stake_turned_up(tmp_path, n_prompts):
    variables_path = write_variables_with_no_stake_turned_up(tmp_path, 0.05)
    values, weights, axes = load_real_matrix(variables_path)
    rules = dc.load_deal_constraints(REAL_RULES_PATH, values)
    saw_no_stake_hand = False
    for seed in SEEDS:
        hands, repaired, _ = deal_and_repair(values, weights, axes, rules, n_prompts, seed)
        assert all_violations(rules, repaired) == [], (seed, n_prompts)
        for hand in repaired:
            if NO_STAKE_MARKER in hand["resolution"]:
                saw_no_stake_hand = True
                assert "a minor detail" in hand["centrality"]
    assert saw_no_stake_hand


def test_repair_at_a_large_deal_size(real_matrix, real_rules):
    values, weights, axes = real_matrix
    for seed in range(4):
        hands, repaired, _ = deal_and_repair(values, weights, axes, real_rules, 1000, seed)
        assert all_violations(real_rules, repaired) == []
        assert column_counts(repaired, axes) == column_counts(hands, axes)


@pytest.mark.parametrize("n_prompts", DECK_SIZES)
def test_repair_actually_has_hands_to_fix(real_matrix, real_rules, n_prompts):
    """Guards the zero-violations tests against passing vacuously on an easy deal."""
    values, weights, axes = real_matrix
    broken_before = sum(
        len(all_violations(real_rules, hands))
        for hands in (matrix.deck_sample(values, weights, axes, n_prompts, random.Random(seed)) for seed in SEEDS)
    )
    assert broken_before > 0


@pytest.mark.parametrize("use_no_stake_copy", [False, True])
def test_value_counts_per_variable_are_identical_before_and_after_repair(
    tmp_path, real_matrix, use_no_stake_copy
):
    """The experiment-invalidating bug to guard against: repair must only move
    cards within a column, so every variable's value counts equal the quotas."""
    if use_no_stake_copy:
        values, weights, axes = load_real_matrix(write_variables_with_no_stake_turned_up(tmp_path))
    else:
        values, weights, axes = real_matrix
    rules = dc.load_deal_constraints(REAL_RULES_PATH, values)
    for n_prompts in DECK_SIZES:
        for seed in SEEDS:
            hands, repaired, report = deal_and_repair(values, weights, axes, rules, n_prompts, seed)
            assert column_counts(repaired, axes) == column_counts(hands, axes), (seed, n_prompts)
            assert len(repaired) == len(hands)
            for axis in axes:
                expected = matrix.largest_remainder(weights[axis], n_prompts)
                observed = column_counts(repaired, axes)[axis]
                assert [observed[value] for value in values[axis]] == expected
    assert sum(report.swaps_made.values()) > 0


def test_repair_does_not_modify_the_dealt_hands(real_matrix, real_rules):
    values, weights, axes = real_matrix
    hands = matrix.deck_sample(values, weights, axes, 200, random.Random(0))
    snapshot = [dict(hand) for hand in hands]
    dc.swap_repair(hands, real_rules, random.Random(0))
    assert hands == snapshot


def test_repair_is_deterministic_for_a_seed_and_varies_with_it(real_matrix, real_rules):
    values, weights, axes = real_matrix
    first = deal_and_repair(values, weights, axes, real_rules, 500, 7)[1]
    again = deal_and_repair(values, weights, axes, real_rules, 500, 7)[1]
    other = deal_and_repair(values, weights, axes, real_rules, 500, 8)[1]
    assert first == again
    assert first != other


def test_report_counts_violations_and_swaps_per_rule(real_matrix, real_rules):
    values, weights, axes = real_matrix
    hands, _, report = deal_and_repair(values, weights, axes, real_rules, 500, 0)
    for rule in dc.swap_rules(real_rules):
        assert report.violating_before[rule.name] == len(
            [1 for hand in hands if rule.is_violated_by(hand)]
        )
        assert report.violating_after[rule.name] == 0
    assert any(count > 0 for count in report.violating_before.values())
    assert any("violated before repair" in line for line in report.format_lines(real_rules))


# --- infeasible rules ---

SMALL_VALUES = {"mood": ["happy", "sad"], "weather": ["sun", "rain"]}
SMALL_RULE_TEXT = "[happy_needs_rain]\nwhen mood is one of:\n    happy\nrequire weather is one of:\n    rain\n"


def test_infeasible_rule_raises_naming_rule_and_counts():
    rules = dc.parse_deal_constraints(SMALL_RULE_TEXT, SMALL_VALUES)
    weights = {"mood": [0.8, 0.2], "weather": [0.9, 0.1]}  # 8 happy hands, 1 rain card
    hands = matrix.deck_sample(SMALL_VALUES, weights, ["mood", "weather"], 10, random.Random(0))
    with pytest.raises(dc.InfeasibleDealConstraintError, match=r"happy_needs_rain.*8 of 10.*only 1"):
        dc.swap_repair(hands, rules, random.Random(0))


def test_feasible_small_rule_is_repaired():
    rules = dc.parse_deal_constraints(SMALL_RULE_TEXT, SMALL_VALUES)
    weights = {"mood": [0.5, 0.5], "weather": [0.5, 0.5]}
    hands = matrix.deck_sample(SMALL_VALUES, weights, ["mood", "weather"], 10, random.Random(0))
    repaired, _ = dc.swap_repair(hands, rules, random.Random(0))
    assert all(h["weather"] == "rain" for h in repaired if h["mood"] == "happy")


def test_jointly_unsatisfiable_rules_raise_instead_of_looping():
    """Each rule is feasible alone (the single-rule check passes) but together
    they are not: hands that are happy need rain, and rain hands need sad."""
    text = (
        SMALL_RULE_TEXT
        + "[rain_needs_sad]\nwhen weather is one of:\n    rain\nrequire mood is one of:\n    sad\n"
    )
    rules = dc.parse_deal_constraints(text, SMALL_VALUES)
    weights = {"mood": [0.5, 0.5], "weather": [0.5, 0.5]}
    hands = matrix.deck_sample(SMALL_VALUES, weights, ["mood", "weather"], 10, random.Random(0))
    with pytest.raises(dc.InfeasibleDealConstraintError):
        dc.swap_repair(hands, rules, random.Random(0))


# --- weight zero ---

def test_zero_weight_value_parses_and_is_never_dealt(real_matrix):
    values, weights, axes = real_matrix
    no_stake_index = next(i for i, v in enumerate(values["resolution"]) if NO_STAKE_MARKER in v)
    assert weights["resolution"][no_stake_index] == 0.0
    assert sum(weights["resolution"]) == pytest.approx(1.0, abs=matrix.WEIGHT_TOLERANCE)
    for n_prompts in (1, 7, 100, 333, 1000, 5000):
        for seed in range(10):
            hands = matrix.deck_sample(values, weights, axes, n_prompts, random.Random(seed))
            assert not any(NO_STAKE_MARKER in hand["resolution"] for hand in hands)


def test_largest_remainder_gives_zero_weight_zero_quota():
    for n in range(1, 60):
        quotas = matrix.largest_remainder([0.5, 0.0, 0.3, 0.2], n)
        assert quotas[1] == 0 and sum(quotas) == n


# --- full-matrix enumeration filters instead of repairing ---

def test_full_matrix_enumeration_filters_rule_breaking_combinations():
    import itertools
    rules = dc.parse_deal_constraints(SMALL_RULE_TEXT, SMALL_VALUES)
    combos = (dict(zip(SMALL_VALUES, combo)) for combo in itertools.product(*SMALL_VALUES.values()))
    report = dc.RepairReport()
    kept = list(dc.without_rule_breaking_hands(combos, rules, report))
    assert {(h["mood"], h["weather"]) for h in kept} == {("happy", "rain"), ("sad", "sun"), ("sad", "rain")}
    assert report.combinations_filtered_out == {"happy_needs_rain": 1}


def test_compose_full_matrix_filters_with_rules(capsys):
    template = "Mood: {mood}. Weather: {weather}.\n"
    weights = {"mood": [0.5, 0.5], "weather": [0.5, 0.5]}
    rules = dc.parse_deal_constraints(SMALL_RULE_TEXT, SMALL_VALUES)
    records = list(cp.compose_records(template, SMALL_VALUES, weights, None, None, deal_rules=rules))
    assert len(records) == 3
    assert not any(r["variables"] == {"mood": "happy", "weather": "sun"} for r in records)


# --- rendering rules and composer wiring ---

def test_drop_template_lines_removes_only_the_named_paragraphs():
    template = "Head {a}.\n\nTradeoff is {b}.\n\nMiddle {c}.\n\nScale is {d}.\n\nTail.\n"
    assert dc.drop_template_lines(template, ["b", "d"]) == "Head {a}.\n\nMiddle {c}.\n\nTail.\n"
    assert dc.drop_template_lines(template, []) == template
    assert dc.drop_template_lines(template, ["missing"]) == template
    # a placeholder inside a multi-line paragraph drops only its own line
    assert dc.drop_template_lines("One {a}\nTwo {b}\nThree\n", ["b"]) == "One {a}\nThree\n"


def compose_real_records(variables_path, n_prompts, seed=0, with_rules=True):
    template = (PROMPTS_SDF / "layer1.txt").read_text(encoding="utf-8")
    preamble = (PROMPTS_SDF / "preamble.txt").read_text(encoding="utf-8")
    values, weights = matrix.split_weights(cp.parse_variables(variables_path))
    rules = cp.load_deal_rules(REAL_RULES_PATH, values) if with_rules else []
    records = list(cp.compose_records(template, values, weights, preamble, n_prompts, seed, deal_rules=rules))
    return template, records


def test_rendering_rule_drops_exactly_the_tradeoff_and_scale_lines_and_keeps_variables(tmp_path):
    variables_path = write_variables_with_no_stake_turned_up(tmp_path, 0.05)
    template, records = compose_real_records(variables_path, 200)
    no_stake_records = [r for r in records if NO_STAKE_MARKER in r["variables"]["resolution"]]
    other_records = [r for r in records if NO_STAKE_MARKER not in r["variables"]["resolution"]]
    assert len(no_stake_records) == 10 and other_records
    for record in no_stake_records:
        variables = record["variables"]
        # the dealt cards stay in the record, so shares are reported honestly
        assert variables["tradeoff"] and variables["decision_scale"]
        assert "The main tradeoff with animal welfare" not in record["prompt"]
        assert "The animal welfare stakes are on the scale of" not in record["prompt"]
        # every other planning line is still present
        for kept_line in ("Type: the document is", "Culture / locale:", "The animal welfare dilemma is",
                          "The AI assistant character", "The animal welfare issues discussed arise from",
                          "Regarding moral or constitutional questions"):
            assert kept_line in record["prompt"]
        assert "\n\n\n" not in record["prompt"]
    for record in other_records:
        assert "The main tradeoff with animal welfare" in record["prompt"]
        assert "The animal welfare stakes are on the scale of" in record["prompt"]


def test_composer_deals_rule_abiding_hands_with_rules_and_leaves_no_rules_unchanged(tmp_path):
    values, weights, axes = load_real_matrix()
    rules = dc.load_deal_constraints(REAL_RULES_PATH, values)
    _, with_rules = compose_real_records(PROMPTS_SDF / "variables.txt", 500)
    _, without_rules = compose_real_records(PROMPTS_SDF / "variables.txt", 500, with_rules=False)
    assert all_violations(rules, [r["variables"] for r in with_rules]) == []
    assert all_violations(rules, [r["variables"] for r in without_rules]) != []
    # same quotas either way
    assert column_counts([r["variables"] for r in with_rules], axes) == column_counts(
        [r["variables"] for r in without_rules], axes)


def test_load_deal_rules_without_a_file_means_no_rules(tmp_path, real_matrix):
    values, _, _ = real_matrix
    assert cp.load_deal_rules(tmp_path / "deal_constraints.txt", values) == []
    assert cp.load_deal_rules(None, values) == []


def test_compose_dry_run_prints_per_rule_violations_and_swaps(capsys):
    compose_real_records(PROMPTS_SDF / "variables.txt", 500)
    output = capsys.readouterr().out
    assert "deal constraints:" in output
    assert "rule identity_documents_name_claude:" in output
    assert re.search(r"hands violated before repair, \d+ swaps made, 0 violating after", output)


def test_layer1_plan_applies_the_rules_file_in_its_prompts_directory(
    tmp_path, tiny_config, stub_claude
):
    prompts_directory = tmp_path / "prompts_copy"
    shutil.copytree(PROMPTS_SDF, prompts_directory)
    variables_path = write_variables_with_no_stake_turned_up(tmp_path, 0.05)
    shutil.copy(variables_path, prompts_directory / "variables.txt")
    tiny_config["sdf"]["n_prompts"] = 60
    stub_claude(lambda user_message, **kw: (
        "<document_planning>n</document_planning>\n"
        "<document_description>\nA spec.\n</document_description>"
    ))
    stage_directory = tmp_path / "stage"
    stage_directory.mkdir()
    layer1_plan.run(tiny_config, prompts_directory, stage_directory)

    from shared import utils
    composed = utils.load_jsonl(stage_directory / "prompts.jsonl")
    values, _, _ = load_real_matrix(variables_path)
    rules = dc.load_deal_constraints(REAL_RULES_PATH, values)
    assert all_violations(rules, [r["variables"] for r in composed]) == []
    no_stake = [r for r in composed if NO_STAKE_MARKER in r["variables"]["resolution"]]
    assert no_stake and all("The main tradeoff with animal welfare" not in r["prompt"] for r in no_stake)
