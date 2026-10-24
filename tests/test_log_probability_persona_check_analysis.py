"""Offline tests for the analysis steps of the log-probability persona check: analyze_persona_scores.py,
analyze_changed_spans.py, summarize_pilot_for_go_or_no_go_decision.py and the loading and compatibility modules, on
small synthetic score files with planted effects. No torch or transformers is needed."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evals.log_probability_persona_check import analysis_run_loading as loading
from evals.log_probability_persona_check import analyze_changed_spans as changed_spans
from evals.log_probability_persona_check import analyze_persona_scores as analysis
from evals.log_probability_persona_check import summarize_pilot_for_go_or_no_go_decision as summarizer
from evals.log_probability_persona_check.legacy_run_format_compatibility_reader import (
    PERSONA_CHECKS_DIRECTORY, normalize_legacy_run_configuration, rename_legacy_condition_groups)
import synthetic_scored_run_folder_for_log_probability_persona_check_analysis as synthetic

TIMESTAMP = "2026-01-01_00-00"
NUMBER_OF_PERMUTATIONS = 200
EXPECTED_ELEVATED_SCORE = synthetic.PLANTED_SPAN_SHIFT * synthetic.PLANTED_SPAN_LENGTH / synthetic.TOKENS_PER_SOURCE_DOCUMENT


def analyze(run_folder, **options):
    options.setdefault("number_of_permutations", NUMBER_OF_PERMUTATIONS)
    return analysis.run_analysis(run_folder, timestamp=TIMESTAMP, **options)


def read_table(analysis_folder, stem):
    return pd.read_csv(next(analysis_folder.glob(f"{stem}_{TIMESTAMP}.csv")))


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    run_folder = tmp_path_factory.mktemp("synthetic_scored_run")
    planted = synthetic.build_synthetic_scored_run_folder(run_folder, seed=1)
    return planted, analyze(run_folder), run_folder


def documents_of_kind(planted, kind):
    return [document_id for document_id, info in planted["documents"].items() if info["kind"] == kind]


# ------------------------------------------------------------------------------------------ names and labels

def test_output_columns_use_the_persona_name_and_configured_groups(analyzed):
    _, folder, _ = analyzed
    scores = read_table(folder, "secondary_not_validated_per_document_scores")
    for column in ["sycophantic_mean_score", "topic_control_mean_score", "same_values_mean_score",
                   "sycophantic_minus_topic_control_score", "model_type", "model_name"]:
        assert column in scores.columns
    assert not any("preachy" in column for column in scores.columns)
    assert (scores["model_type"] == "instruct").all()
    tests = read_table(folder, "secondary_not_validated_per_document_significance_tests")
    assert "main_test_median_sycophantic_minus_topic_control" in tests.columns
    assert "second_test_median_sycophantic_minus_same_values" in tests.columns


def test_every_output_is_labelled_with_model_type_and_model(analyzed):
    _, folder, _ = analyzed
    for path in folder.glob("*.csv"):
        frame = pd.read_csv(path)
        assert {"model_type", "model_name"} <= set(frame.columns), path.name
    for record in map(json.loads, next(folder.glob("extreme_spans_*.jsonl")).read_text().splitlines()):
        assert record["model_type"] == "instruct" and record["model_name"] == "example-org/example-model"
    for stem in ("analysis_summary", "control_verdicts", "changed_span_summary"):
        assert json.loads(next(folder.glob(f"{stem}_*.json")).read_text())["model_type"] == "instruct"
    assert "instruct model example-org/example-model" in next(folder.glob("analysis_notes_*.md")).read_text()


def test_multi_persona_same_values_group_is_pooled_and_broken_down_by_persona(analyzed):
    _, folder, run_folder = analyzed
    run = loading.load_run(run_folder)
    assert run.setup.same_values_group == "same_values"
    assert run.persona_names_by_group["same_values"] == synthetic.SAME_VALUES_PERSONAS
    assert len(run.group_indices["same_values"]) == 4
    assert run.setup.pooled_same_values_persona_name == "pooled_same_values"
    scores = read_table(folder, "secondary_not_validated_per_document_scores")
    for persona_name in synthetic.SAME_VALUES_PERSONAS:
        assert f"{persona_name}_mean_score" in scores.columns
    pooled = scores["same_values_mean_score"]
    per_persona_mean = scores[[f"{name}_mean_score" for name in synthetic.SAME_VALUES_PERSONAS]].mean(axis=1)
    assert np.allclose(pooled, per_persona_mean)  # equal-sized personas: the pooled mean is the mean of persona means
    tests = read_table(folder, "secondary_not_validated_per_document_significance_tests")
    assert (tests["number_of_same_values_rewordings"] == 4).all()  # the second test pools every same-values persona


def test_second_test_compares_the_target_with_all_same_values_personas_together(analyzed):
    _, folder, run_folder = analyzed
    run = loading.load_run(run_folder)
    matrix = analysis.condition_score_matrix_of(run)
    tests = read_table(folder, "secondary_not_validated_per_document_significance_tests").set_index("document_id")
    document_id = run.documents[0]["document_id"]
    scores = matrix.loc[document_id].to_numpy()
    expected = stats.mannwhitneyu(scores[run.group_indices["sycophantic"]], scores[run.group_indices["same_values"]],
                                  alternative="two-sided").pvalue
    assert tests.loc[document_id, "second_test_p_value"] == pytest.approx(expected)


def test_run_recorded_with_a_one_persona_same_values_group_is_analysed_from_its_recorded_configuration(tmp_path):
    """A run built when the group held one persona records `persona_name` and `expected_rewording_count`; the
    analysis must use that record."""
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    configuration_path = next(tmp_path.glob("run_configuration_*.json"))
    configuration = json.loads(configuration_path.read_text())
    groups = configuration["persona_set"]["configuration"]["condition_groups"]
    groups["same_values_control"] = {"condition_group": "same_values", "persona_name": "restrained_advocate",
                                     "display_name": "same values", "expected_rewording_count": 2}
    configuration_path.write_text(json.dumps(configuration))
    conditions_path = tmp_path / "scoring_inputs" / "conditions.jsonl"
    kept = [json.loads(line) for line in conditions_path.read_text().splitlines()]
    assert [c["persona_name"] for c in kept].count("even_handed_advocate") == 2
    # Drop the first same-values persona from the scoring inputs and every score file, as an older run would lack it.
    removed = {c["condition_id"] for c in kept if c["persona_name"] == "even_handed_advocate"}
    conditions_path.write_text("".join(json.dumps(c) + "\n" for c in kept if c["condition_id"] not in removed))
    for path in (tmp_path / "scoring_outputs" / "per_token_log_probabilities").glob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            contents = {key: archive[key] for key in archive.files}
        keep = [i for i, condition_id in enumerate(contents["condition_ids"]) if str(condition_id) not in removed]
        contents["condition_ids"] = contents["condition_ids"][keep]
        contents["log_probabilities"] = contents["log_probabilities"][keep]
        np.savez(path, **contents)
    folder = analyze(tmp_path)
    agreement = read_table(folder, "rewording_agreement")
    assert set(agreement[agreement.condition_group == "same_values"]["persona_name"]) == {"restrained_advocate"}


def test_run_without_same_values_control(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path, include_same_values_control=False)
    folder = analyze(tmp_path)
    scores = read_table(folder, "secondary_not_validated_per_document_scores")
    assert "same_values_mean_score" not in scores.columns
    tests = read_table(folder, "secondary_not_validated_per_document_significance_tests")
    assert "second_test_p_value" not in tests.columns


# ------------------------------------------------------------------------------------------ primary: spans

def test_planted_span_is_the_top_extreme_span(analyzed):
    planted, folder, _ = analyzed
    spans = pd.read_json(next(folder.glob("extreme_spans_*.jsonl")), lines=True)
    for document_id in documents_of_kind(planted, "elevated"):
        top = spans[(spans.document_id == document_id) & (spans.direction == "highest") & (spans["rank"] == 1)].iloc[0]
        assert top["start_token_index"] == synthetic.PLANTED_SPAN_START
        assert top["end_token_index_exclusive"] == synthetic.PLANTED_SPAN_START + synthetic.PLANTED_SPAN_LENGTH
        assert top["window_mean_sycophantic_difference"] == pytest.approx(synthetic.PLANTED_SPAN_SHIFT, abs=0.03)
        assert top["text"].startswith(f"tok{int(document_id.split('_')[1]) * 1000 + synthetic.PLANTED_SPAN_START} ")


def test_at_most_three_non_overlapping_windows_each_direction(analyzed):
    _, folder, _ = analyzed
    spans = pd.read_json(next(folder.glob("extreme_spans_*.jsonl")), lines=True)
    for _, group in spans.groupby(["document_id", "direction"]):
        starts = sorted(group["start_token_index"])
        assert 1 <= len(group) <= 3
        assert all(later - earlier >= 32 for earlier, later in zip(starts, starts[1:]))


def test_short_document_uses_one_whole_document_window():
    records = analysis.extreme_spans_for_document("short", np.arange(10, dtype=float), [f"t{i}" for i in range(10)], 32, 3)
    assert len(records) == 2 and all(r["start_token_index"] == 0 and r["end_token_index_exclusive"] == 10 for r in records)


def test_window_selection_respects_room():
    starts = analysis.select_non_overlapping_windows(np.arange(39, dtype=float), 32, 3, True)
    assert len(starts) == 2 and abs(starts[0] - starts[1]) >= 32


def test_permutation_test_flags_planted_span_and_not_noise(analyzed):
    planted, folder, _ = analyzed
    table = read_table(folder, "span_significance_tests").set_index("document_id")
    minimum_p = 1 / (NUMBER_OF_PERMUTATIONS + 1)
    for document_id in documents_of_kind(planted, "elevated"):
        assert table.loc[document_id, "highest_window_p_value"] == pytest.approx(minimum_p)
        assert table.loc[document_id, "highest_window_q_value"] < 0.05
    for document_id in documents_of_kind(planted, "null"):
        assert table.loc[document_id, "highest_window_p_value"] > 0.01  # null documents: p is uniform, never planted-small
        assert table.loc[document_id, "lowest_window_p_value"] > 0.01
    assert table["highest_window_p_value"].between(minimum_p - 1e-12, 1.0).all()


def test_permutation_test_is_deterministic_and_whole_document_window_gives_p_one():
    values = np.random.default_rng(0).normal(size=80)
    first = analysis.permutation_test_of_extreme_windows(values, 32, 100, np.random.default_rng([0, 1]))
    second = analysis.permutation_test_of_extreme_windows(values, 32, 100, np.random.default_rng([0, 1]))
    assert first == second
    _, p_high, _, p_low = analysis.permutation_test_of_extreme_windows(values[:20], 32, 100, np.random.default_rng(0))
    assert p_high == 1.0 and p_low == 1.0


def test_permutation_test_detects_negative_span():
    values = np.zeros(100)
    values[10:42] = -1.0
    values += np.random.default_rng(0).normal(0, 0.01, 100)
    _, _, observed_low, p_low = analysis.permutation_test_of_extreme_windows(values, 32, 200, np.random.default_rng(0))
    assert observed_low == pytest.approx(-1.0, abs=0.02) and p_low == pytest.approx(1 / 201)


# ------------------------------------------------------------------------------------------ primary: changed spans

def test_changed_span_counts_follow_the_planted_inserted_tokens(analyzed):
    _, folder, _ = analyzed
    summary = json.loads(next(folder.glob("changed_span_summary_*.json")).read_text())["directions"]
    assert summary["more_sycophantic"]["changed_minus_unchanged_number_moving_expected_way"] == 2
    assert summary["less_sycophantic"]["changed_minus_unchanged_number_moving_expected_way"] == 2
    assert summary["tone_shift"]["changed_minus_unchanged_number_moving_expected_way"] == 0
    rows = read_table(folder, "changed_span_sycophantic_minus_topic_control")
    assert (rows["changed_tokens"] == synthetic.INSERTED_TOKEN_COUNT).all()
    toward = rows[rows.rewrite_direction == "more_sycophantic"]
    assert np.allclose(toward["sycophantic_minus_topic_control_per_token_changed"], synthetic.INSERTED_TOKEN_SHIFT, atol=0.03)


def test_changed_span_by_persona_has_every_persona_and_direction(analyzed):
    _, folder, _ = analyzed
    table = read_table(folder, "changed_span_by_persona")
    assert set(table["rewrite_direction"]) == set(synthetic.DIRECTION_NAMES.values())
    toward = table[table.rewrite_direction == "more_sycophantic"].set_index("persona")
    assert {"sycophantic", "bakery_expertise", "railway_expertise", "even_handed_advocate", "restrained_advocate",
            "pooled_topic_control", "pooled_same_values"} == set(toward.index)
    assert toward.loc["sycophantic", "mean_changed_minus_unchanged"] > 0.2
    assert abs(toward.loc["pooled_topic_control", "mean_changed_minus_unchanged"]) < 0.05


def test_changed_token_masks_mark_inserted_and_replaced_tokens():
    changed, replaced = changed_spans.changed_token_masks([1, 2, 3, 4], [1, 2, 9, 9, 3, 4])
    assert changed.tolist() == [False, False, True, True, False, False] and not replaced.any()
    changed, replaced = changed_spans.changed_token_masks([1, 2, 3], [1, 7, 3])
    assert changed.tolist() == [False, True, False] and replaced.tolist() == [False, True, False]


# ------------------------------------------------------------------------------------------ verdicts

def test_verdicts_pass_on_planted_effects(analyzed):
    _, folder, _ = analyzed
    verdicts = json.loads(next(folder.glob("control_verdicts_*.json")).read_text())
    assert verdicts["changed_span"]["positive_control"]["verdict"] == "pass"
    assert verdicts["changed_span"]["neutral_control"]["verdict"] == "pass"
    assert verdicts["changed_span"]["neutral_control"]["number_moving_toward_persona"] == 0
    whole_document = verdicts["whole_document"]["positive_control"]
    assert whole_document["verdict"] == "pass"
    assert {r["direction_name"] for r in whole_document["per_direction"].values()} == {"more_sycophantic", "less_sycophantic"}


def test_whole_document_check_matches_planted_shifts(analyzed):
    _, folder, _ = analyzed
    table = read_table(folder, "secondary_not_validated_whole_document_control_check")
    documents = table[table.row_type == "document"]
    assert len(documents) == 6 and documents["moved_in_expected_direction"].sum() == 4
    summary = table[table.row_type == "summary"].set_index("rewrite_direction")
    assert summary.loc["more_sycophantic", "number_moving_in_expected_direction"] == 2
    assert summary.loc["tone_shift", "number_moving_in_expected_direction"] == 0


def test_neutral_rewrites_that_move_toward_the_persona_fail_the_neutral_control(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path, neutral_rewrites_shift=synthetic.INSERTED_TOKEN_SHIFT)
    folder = analyze(tmp_path)
    neutral = json.loads(next(folder.glob("control_verdicts_*.json")).read_text())["changed_span"]["neutral_control"]
    assert neutral["number_moving_toward_persona"] == 2 and neutral["verdict"] == "fail"


def test_wrong_number_of_rewrites_is_not_evaluable_and_missing_criteria_are_reported(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path, rewrites_per_direction=1)
    folder = analyze(tmp_path)
    verdicts = json.loads(next(folder.glob("control_verdicts_*.json")).read_text())
    assert verdicts["changed_span"]["positive_control"]["verdict"] == "not evaluable"
    assert "fixed for 2" in verdicts["changed_span"]["positive_control"]["per_direction"]["toward"]["reason"]
    criteria_without_neutral = {k: v for k, v in synthetic.DEFAULT_PASS_CRITERIA.items() if k != "neutral_control"}
    other = tmp_path / "other"
    synthetic.build_synthetic_scored_run_folder(other, pass_criteria=criteria_without_neutral)
    verdict = json.loads(next(analyze(other).glob("control_verdicts_*.json")).read_text())
    assert verdict["changed_span"]["neutral_control"]["verdict"] == "no criterion pre-registered"


def test_failing_positive_control_when_rewrites_do_not_move_expected_way(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    # Overwrite the away rewrites' scores so they move toward the persona instead: flip the run to a mirrored persona.
    directory = tmp_path / "scoring_outputs" / "per_token_log_probabilities"
    for path in directory.glob("*__less_sycophantic.npz"):
        with np.load(path) as archive:
            contents = {key: archive[key] for key in archive.files}
        condition_ids = [str(c) for c in contents["condition_ids"]]
        generic = contents["log_probabilities"][condition_ids.index("generic")]
        for row, condition_id in enumerate(condition_ids):
            if condition_id.startswith("sycophantic__"):
                contents["log_probabilities"][row] = generic + np.abs(contents["log_probabilities"][row] - generic) + 0.2
        np.savez(path, **contents)
    verdicts = json.loads(next(analyze(tmp_path).glob("control_verdicts_*.json")).read_text())
    assert verdicts["changed_span"]["positive_control"]["per_direction"]["away"]["verdict"] == "fail"
    assert verdicts["changed_span"]["positive_control"]["verdict"] == "fail"


def test_judge_count_against_criterion():
    assert changed_spans.judge_count_against_criterion(8, 10, 8, 10)["verdict"] == "pass"
    assert changed_spans.judge_count_against_criterion(7, 10, 8, 10)["verdict"] == "fail"
    assert changed_spans.judge_count_against_criterion(0, 0, 8, 10)["verdict"] == "not evaluable"
    assert changed_spans.judge_count_against_criterion(9, 9, 8, 10)["verdict"] == "not evaluable"


# ------------------------------------------------------------------------------------------ rewording agreement

def test_rewording_agreement_covers_every_persona(analyzed):
    _, folder, _ = analyzed
    agreement = read_table(folder, "rewording_agreement")
    assert set(agreement["persona_name"]) == {"sycophantic", "bakery_expertise", "railway_expertise",
                                              *synthetic.SAME_VALUES_PERSONAS}
    assert len(agreement) == 3 + 4 + 4
    same_values_rows = agreement[agreement.condition_group == synthetic.SAME_VALUES_GROUP]
    assert same_values_rows.groupby("persona_name").size().to_dict() == {name: 2 for name in synthetic.SAME_VALUES_PERSONAS}
    sycophantic = agreement[agreement.persona_name == "sycophantic"].iloc[0]
    assert sycophantic["persona_intraclass_correlation_one_way_single_rewording"] > 0.5


def test_outlier_rewording_is_flagged(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path, seed=5, outlier_rewording_condition_id="sycophantic__rewording_03")
    agreement = read_table(analyze(tmp_path), "rewording_agreement")
    flagged = agreement[agreement["rewording_is_outlier"] & (agreement.persona_name == "sycophantic")]
    assert list(flagged["condition_id"]) == ["sycophantic__rewording_03"]


def test_intraclass_correlation_extremes():
    assert analysis.one_way_intraclass_correlation(np.column_stack([np.arange(10.0)] * 4)) == pytest.approx(1.0)
    assert abs(analysis.one_way_intraclass_correlation(np.random.default_rng(0).normal(size=(500, 4)))) < 0.1


# ------------------------------------------------------------------------------------------ secondary: document means

def test_document_means_equal_planted_shift_and_are_labelled_not_validated(analyzed):
    planted, folder, _ = analyzed
    assert not list(folder.glob("per_document_scores_*"))
    scores = read_table(folder, "secondary_not_validated_per_document_scores").set_index("document_id")
    for document_id in documents_of_kind(planted, "elevated"):
        assert scores.loc[document_id, "sycophantic_minus_topic_control_score"] == pytest.approx(EXPECTED_ELEVATED_SCORE, abs=0.01)
    for document_id in documents_of_kind(planted, "null"):
        assert abs(scores.loc[document_id, "sycophantic_minus_topic_control_score"]) < 0.01
    assert np.allclose(scores["score__generic"], 0)
    assert "not validated" in next(folder.glob("analysis_notes_*.md")).read_text()


def test_mann_whitney_and_benjamini_hochberg_across_originals_only(analyzed):
    _, folder, _ = analyzed
    scores = read_table(folder, "secondary_not_validated_per_document_scores")
    tests = read_table(folder, "secondary_not_validated_per_document_significance_tests")
    row = scores.iloc[0]
    target = [row[f"score__sycophantic__rewording_{i:02d}"] for i in (1, 2, 3)]
    controls = [row[c] for c in scores.columns if c.startswith("score__bakery") or c.startswith("score__railway")]
    expected = stats.mannwhitneyu(target, controls, alternative="two-sided")
    assert tests.iloc[0]["main_test_p_value"] == pytest.approx(expected.pvalue)
    merged = tests.merge(scores[["document_id", "rewrite_direction"]], on="document_id")
    originals = merged[merged.rewrite_direction.isna()]
    rewrites = merged[merged.rewrite_direction.notna()]
    assert originals["main_test_q_value"].to_numpy() == pytest.approx(analysis.benjamini_hochberg(originals["main_test_p_value"].to_numpy()))
    assert rewrites["main_test_q_value"].to_numpy() == pytest.approx(rewrites["main_test_p_value"].to_numpy())
    assert rewrites["q_value_is_uncorrected_rewrite"].all()


def test_benjamini_hochberg_matches_hand_computation_and_scipy():
    assert analysis.benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    result = analysis.benjamini_hochberg([0.9, np.nan, 0.8])
    assert result[0] == pytest.approx(0.9) and np.isnan(result[1]) and result[2] == pytest.approx(0.9)
    p_values = np.random.default_rng(3).uniform(size=50) ** 2
    assert analysis.benjamini_hochberg(p_values) == pytest.approx(stats.false_discovery_control(p_values, method="bh"))


def test_ordinary_least_squares_recovers_known_coefficients():
    generator = np.random.default_rng(0)
    design = np.column_stack([np.ones(200), generator.integers(0, 2, 200), generator.integers(0, 2, 200)]).astype(float)
    outcome = design @ np.array([1.0, 2.0, -3.0]) + generator.normal(0, 0.1, 200)
    coefficients, _, _, _ = analysis.ordinary_least_squares(design, outcome)
    assert coefficients == pytest.approx([1.0, 2.0, -3.0], abs=0.1)


def test_scores_by_spec_variable_uses_original_documents(analyzed):
    _, folder, _ = analyzed
    spec = read_table(folder, "secondary_not_validated_scores_by_spec_variable")
    summary = spec[(spec.section == "summary_by_level") & (spec.variable == "tone")]
    assert summary["count"].sum() == 6


# ------------------------------------------------------------------------------------------ refusals and checks

def test_mixed_model_types_across_run_configurations_are_refused(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    configuration = json.loads((tmp_path / "run_configuration_2026-01-01_00-00.json").read_text())
    configuration["model_type"] = "base"
    (tmp_path / "run_configuration_2026-01-02_00-00.json").write_text(json.dumps(configuration))
    with pytest.raises(loading.MixedModelTypesError, match="model_type"):
        analyze(tmp_path)
    assert not list(tmp_path.glob("analysis_*"))


def test_mixed_model_in_scoring_environment_or_verification_is_refused(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    (tmp_path / "scoring_outputs" / "scoring_environment_2026-01-01_00-10.json").write_text(
        json.dumps({"model_name": "example-org/another-model", "model_revision_requested": "a" * 40}))
    with pytest.raises(loading.MixedModelTypesError, match="model_name"):
        analyze(tmp_path)
    other = tmp_path / "other"
    synthetic.build_synthetic_scored_run_folder(other)
    (other / "scoring_outputs" / "full_logits_verification_2026-01-01_00-10.json").write_text(json.dumps({"model_type": "base"}))
    with pytest.raises(loading.MixedModelTypesError, match="model_type"):
        analyze(other)


def test_base_model_type_is_carried_into_the_labels(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path, model_type="base")
    folder = analyze(tmp_path)
    assert (read_table(folder, "rewording_agreement")["model_type"] == "base").all()


def test_mismatched_condition_sets_raise(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    path = tmp_path / "scoring_outputs" / "per_token_log_probabilities" / "source_03.npz"
    with np.load(path) as archive:
        contents = {key: archive[key] for key in archive.files}
    contents["condition_ids"] = contents["condition_ids"][:-1]
    contents["log_probabilities"] = contents["log_probabilities"][:-1]
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="does not match conditions.jsonl"):
        analyze(tmp_path)


def test_missing_document_output_raises(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    (tmp_path / "scoring_outputs" / "per_token_log_probabilities" / "source_01.npz").unlink()
    with pytest.raises(ValueError, match="Missing scoring output"):
        analyze(tmp_path)


def test_condition_group_sizes_must_match_the_configuration(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    path = tmp_path / "run_configuration_2026-01-01_00-00.json"
    configuration = json.loads(path.read_text())
    configuration["persona_set"]["configuration"]["condition_groups"]["target"]["expected_rewording_count"] = 30
    path.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="expects 30"):
        analyze(tmp_path)


def test_no_rewrites_still_analyses_and_verdicts_are_not_evaluable(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path, rewrites_per_direction=0)
    folder = analyze(tmp_path)
    table = read_table(folder, "secondary_not_validated_whole_document_control_check")
    assert "skipped" in table.iloc[0]["note"]
    verdicts = json.loads(next(folder.glob("control_verdicts_*.json")).read_text())
    assert verdicts["whole_document"]["positive_control"]["verdict"] == "not evaluable"


def test_analysis_folder_is_never_overwritten(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    analyze(tmp_path)
    with pytest.raises(FileExistsError):
        analyze(tmp_path)


def test_command_line_entry_points_run(tmp_path):
    synthetic.build_synthetic_scored_run_folder(tmp_path)
    analysis_folder = analysis.main(["--run-folder", str(tmp_path), "--number-of-permutations", "20", "--skip-changed-spans"])
    assert not list(analysis_folder.glob("changed_span_*"))
    other = tmp_path / "separate_changed_span_output"
    changed_spans.main(["--run-folder", str(tmp_path), "--analysis-folder", str(other)])
    assert list(other.glob("changed_span_summary_*.json"))


# ------------------------------------------------------------------------------------------ legacy run compatibility

def test_legacy_run_configuration_is_refused_without_the_explicit_flag(tmp_path):
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(
        json.dumps({"run_name": "legacy", "model_name": "example-org/example-model"}))
    with pytest.raises(ValueError, match="--legacy-run-persona-set"):
        loading.read_run_configurations(tmp_path)


def test_legacy_run_configuration_is_converted_with_the_persona_sets_criteria(tmp_path):
    (tmp_path / "scoring_outputs").mkdir()
    (tmp_path / "scoring_outputs" / "scoring_environment_2026-01-01_00-00.json").write_text(
        json.dumps({"model_name": "example-org/example-model", "model_revision_requested": "b" * 40}))
    converted = normalize_legacy_run_configuration(
        {"run_name": "legacy", "model_name": "example-org/example-model"}, "preachiness", tmp_path)
    assert converted["model_type"] == "instruct" and converted["model_revision"] == "b" * 40
    assert converted["pass_criteria"]["whole_document"]["minimum_number_moving_expected_way"] == 8
    setup = loading.build_setup([converted], tmp_path)
    assert setup.persona_name == "preachy" and setup.direction_names_by_role["toward"] == "preachier"
    assert setup.target_minus_control_score_column == "preachy_minus_control_score"


def write_legacy_format_conditions(run_folder, same_values_persona_names):
    (run_folder / "scoring_inputs").mkdir(parents=True)
    conditions = [{"condition_id": "generic", "condition_group": "generic", "persona_name": "generic"}]
    for persona_name in same_values_persona_names:
        for index in range(1, 6):
            conditions.append({"condition_id": f"{persona_name}__rewording_{index:02d}",
                               "condition_group": "proportionate_welfare", "persona_name": persona_name})
    (run_folder / "scoring_inputs" / "conditions.jsonl").write_text("".join(json.dumps(c) + "\n" for c in conditions))
    (run_folder / "scoring_outputs").mkdir()
    (run_folder / "scoring_outputs" / "scoring_environment_2026-01-01_00-00.json").write_text(
        json.dumps({"model_name": "example-org/example-model", "model_revision_requested": "b" * 40}))


def test_legacy_run_same_values_group_is_mapped_and_expected_counts_follow_the_run(tmp_path):
    write_legacy_format_conditions(tmp_path, ["proportionate_welfare"])
    converted = normalize_legacy_run_configuration(
        {"run_name": "legacy", "model_name": "example-org/example-model"}, "preachiness", tmp_path)
    same_values = converted["persona_set"]["configuration"]["condition_groups"]["same_values_control"]
    assert same_values["condition_group"] == "same_values_control"
    assert (same_values["expected_persona_count"], same_values["expected_rewordings_per_persona"]) == (1, 5)
    setup = loading.build_setup([converted], tmp_path)
    assert setup.expected_conditions_per_group["same_values_control"] == 5
    conditions = loading.utils.load_jsonl(tmp_path / "scoring_inputs" / "conditions.jsonl")
    renamed = rename_legacy_condition_groups(conditions, converted["persona_set"]["configuration"])
    assert {c["condition_group"] for c in renamed} == {"generic", "same_values_control"}
    assert {c["condition_group"] for c in conditions} == {"generic", "proportionate_welfare"}  # input left alone


def test_converting_a_legacy_run_does_not_change_the_current_persona_set_files(tmp_path):
    write_legacy_format_conditions(tmp_path, ["proportionate_welfare"])
    normalize_legacy_run_configuration({"run_name": "legacy", "model_name": "m"}, "preachiness", tmp_path)
    current = json.loads((PERSONA_CHECKS_DIRECTORY / "preachiness" / "persona_check_configuration.json").read_text())
    assert current["condition_groups"]["same_values_control"]["expected_persona_count"] == 3


# ------------------------------------------------------------------------------------------ pilot summary

PROGRESS_THROUGHPUT = {"documents_scored_this_session": 12, "input_tokens_processed": 1, "document_tokens_scored": 1,
                       "scoring_seconds": 1800.0, "input_tokens_per_second": 9000.0,
                       "document_tokens_scored_per_second": 8000.0, "documents_per_minute": 0.4}


def add_scoring_outputs_and_pod_cost_log(run_folder, verification_passed=True, write_verification=True, hourly_price=3.0):
    outputs = run_folder / "scoring_outputs"
    (outputs / "scoring_environment_2026-01-01_12-00.json").write_text(json.dumps({
        "model_name": "example-org/example-model", "model_revision_requested": "a" * 40, "model_type": "instruct",
        "dtype": "bfloat16", "gpu_name": "Example GPU", "torch_version": "2.8.0", "transformers_version": "5.0.0",
        "throughput": PROGRESS_THROUGHPUT}))
    documents = [json.loads(line) for line in (run_folder / "scoring_inputs" / "documents.jsonl").read_text().splitlines()]
    conditions = [json.loads(line) for line in (run_folder / "scoring_inputs" / "conditions.jsonl").read_text().splitlines()]
    with open(outputs / "scoring_completion_records.jsonl", "w") as handle:
        for document in documents:
            for condition in conditions:
                handle.write(json.dumps({"document_id": document["document_id"], "condition_id": condition["condition_id"],
                                         "number_of_document_tokens": 100, "sum_log_probability": -1.0,
                                         "mean_log_probability": -1.0}) + "\n")
    if write_verification:
        (outputs / "full_logits_verification_2026-01-01_12-00.json").write_text(json.dumps({
            "passed": verification_passed, "model_type": "instruct", "document_id": "source_00",
            "condition_id": "restrained_advocate__rewording_02", "number_of_document_tokens": 100,
            "maximum_absolute_difference": 0.02, "mean_absolute_difference": 0.004,
            "maximum_allowed_mean_difference": 0.05, "maximum_allowed_maximum_difference": 1.0}))
    (run_folder / "pod_cost_log_2026-01-01_11-00.json").write_text(json.dumps({
        "pod_id": "examplepod", "gpu_type": "Example GPU", "hourly_price_dollars": hourly_price,
        "start_time": "2026-01-01T11:00:00", "stop_time": "2026-01-01T12:00:00", "termination_reason": "scorer_finished",
        "total_cost_dollars": hourly_price}))


def build_pilot_run(tmp_path, analyse=True, **options):
    run_folder = tmp_path / "pilot_run"
    synthetic.build_synthetic_scored_run_folder(run_folder, **{k: options.pop(k) for k in list(options)
                                                               if k in ("pass_criteria", "rewrites_per_direction")})
    add_scoring_outputs_and_pod_cost_log(run_folder, **options)
    if analyse:
        analyze(run_folder)
    return run_folder


def test_pilot_facts_and_projection_range(tmp_path):
    facts = summarizer.collect_facts(build_pilot_run(tmp_path), full_run_number_of_documents=477, total_budget_dollars=30)
    assert facts["verification"]["passed"] is True and facts["model_type"] == "instruct"
    assert facts["scored_tokens_per_second"] == pytest.approx(8000) and facts["scoring_seconds"] == 1800
    assert facts["setup_seconds_per_pod"] == pytest.approx(1800)
    assert facts["number_of_documents_scored"] == 12 and facts["number_of_conditions"] == 13
    full_run = facts["full_run"]
    assert full_run["scored_tokens"] == pytest.approx(477 * 100 * 13)
    slowest, fastest = full_run["throughput_range_scored_tokens_per_second"]
    assert slowest == pytest.approx(6000) and fastest == pytest.approx(10000)
    assert full_run["low"]["scoring_seconds"] == pytest.approx(477 * 100 * 13 / 10000)
    assert full_run["high"]["cost_dollars"] == pytest.approx((477 * 100 * 13 / 6000 + 1800) / 3600 * 3.0)
    assert full_run["low"]["cost_dollars"] < full_run["high"]["cost_dollars"]
    assert facts["span_significance"]["number_of_original_documents_tested"] == 6


def write_pod_cost_log(run_folder, file_stamp, start_time, stop_time, total_cost_dollars, pod_id="examplepod", **flags):
    (run_folder / f"pod_cost_log_{file_stamp}.json").write_text(json.dumps({
        "pod_id": pod_id, "gpu_type": "Example GPU", "hourly_price_dollars": 3.0, "start_time": start_time,
        "stop_time": stop_time, "termination_reason": "scorer_finished", "total_cost_dollars": total_cost_dollars, **flags}))


def test_pilot_pod_time_and_cost_are_counted_once_when_a_launcher_attached_to_the_pod(tmp_path):
    run_folder = build_pilot_run(tmp_path)
    # The first launcher stopped at 11:40; a second one attached to the same pod and counts from the first one's
    # start, so its log covers the whole pod lifetime (11:00 to 12:00).
    write_pod_cost_log(run_folder, "2026-01-01_11-00", "2026-01-01T11:00:00", "2026-01-01T11:40:00", 2.0)
    write_pod_cost_log(run_folder, "2026-01-01_11-45", "2026-01-01T11:00:00", "2026-01-01T12:00:00", 3.0,
                       attached_to_existing_pod=True)
    facts = summarizer.collect_facts(run_folder, full_run_number_of_documents=477)
    assert facts["pod_cost_spent_dollars"] == pytest.approx(3.0)
    assert facts["pod_running_seconds"] == pytest.approx(3600)
    assert facts["setup_seconds_per_pod"] == pytest.approx(1800)  # one pod, 3600 seconds of which 1800 are scoring
    assert [log["file_name"] for log in facts["pod_cost_logs"]] == [
        "pod_cost_log_2026-01-01_11-00.json", "pod_cost_log_2026-01-01_11-45.json"]
    assert "included in the totals of the later log" in summarizer.render_markdown(facts)


def test_pilot_pod_time_and_cost_still_add_up_across_different_pods_and_separate_uses_of_one_pod(tmp_path):
    run_folder = build_pilot_run(tmp_path)
    write_pod_cost_log(run_folder, "2026-01-01_11-00", "2026-01-01T11:00:00", "2026-01-01T11:30:00", 1.5)
    # A reuse of the same pod later is a separate period counted from its own launch, not an attach.
    write_pod_cost_log(run_folder, "2026-01-01_13-00", "2026-01-01T13:00:00", "2026-01-01T13:30:00", 1.5,
                       reused_existing_pod=True)
    write_pod_cost_log(run_folder, "2026-01-01_15-00", "2026-01-01T15:00:00", "2026-01-01T15:30:00", 1.5, pod_id="otherpod")
    facts = summarizer.collect_facts(run_folder, full_run_number_of_documents=477)
    assert facts["pod_cost_spent_dollars"] == pytest.approx(4.5) and facts["pod_running_seconds"] == pytest.approx(5400)


def test_pilot_criteria_statuses_for_a_clean_pilot(tmp_path):
    facts = summarizer.collect_facts(build_pilot_run(tmp_path), full_run_number_of_documents=477, total_budget_dollars=30,
                                     minimum_mean_pairwise_spearman=0.0)
    statuses = {description: status for description, status, _ in facts["criteria"]}
    assert all(status == "MET" for status in statuses.values()), statuses
    assert len(statuses) == 1 + 2 + 2 + 1 + 1 + 1  # verification, whole document x2, changed span x2, neutral, agreement, budget


def test_pilot_facts_list_the_same_values_personas_by_group(tmp_path):
    facts = summarizer.collect_facts(build_pilot_run(tmp_path), full_run_number_of_documents=477)
    assert facts["persona_conditions_by_group"]["same_values"] == {name: 2 for name in synthetic.SAME_VALUES_PERSONAS}
    text = summarizer.render_markdown(facts)
    assert "Condition group `same_values` (pooled over its personas): 4 conditions" in text
    assert "restrained_advocate 2" in text


def test_pilot_budget_criterion_fails_when_projection_is_expensive(tmp_path):
    facts = summarizer.collect_facts(build_pilot_run(tmp_path, hourly_price=500.0), full_run_number_of_documents=477,
                                     total_budget_dollars=30)
    budget = [status for description, status, _ in facts["criteria"] if description.startswith("Projected")]
    assert budget == ["NOT MET"]


def test_pilot_failed_or_missing_verification(tmp_path):
    failed = summarizer.collect_facts(build_pilot_run(tmp_path / "failed", verification_passed=False))["criteria"]
    assert failed[0][1] == "NOT MET"
    missing = summarizer.collect_facts(build_pilot_run(tmp_path / "missing", write_verification=False))
    assert missing["criteria"][0][1] == "NOT EVALUABLE" and "No verification result" in summarizer.render_markdown(missing)


def test_pilot_without_analysis_or_corpus_still_summarises(tmp_path):
    facts = summarizer.collect_facts(build_pilot_run(tmp_path, analyse=False))
    markdown = summarizer.render_markdown(facts)
    assert "No control verdicts" in markdown and "Full-run projection not possible" in markdown
    assert any(status == "NOT EVALUABLE" for _, status, _ in facts["criteria"])


def test_pilot_without_neutral_criterion_is_not_evaluable(tmp_path):
    criteria = {k: v for k, v in synthetic.DEFAULT_PASS_CRITERIA.items() if k != "neutral_control"}
    facts = summarizer.collect_facts(build_pilot_run(tmp_path, pass_criteria=criteria), full_run_number_of_documents=477)
    neutral = [(status, detail) for description, status, detail in facts["criteria"] if description.startswith("Neutral")]
    assert neutral[0][0] == "NOT EVALUABLE"


def test_pilot_markdown_sections_and_no_cost_center_text(tmp_path):
    run_folder = build_pilot_run(tmp_path)
    output_path = summarizer.main(["--run-folder", str(run_folder), "--full-run-number-of-documents", "477"])
    text = output_path.read_text()
    for heading in ("(a) Full-logits verification", "(b) Throughput", "(c) Positive and neutral controls",
                    "(d) Rewording agreement", "(e) Span significance", "(f) Pods", "(g) Criteria"):
        assert heading in text
    assert "Projected full-run cost:" in text and " to " in text
    assert output_path.with_suffix(".json").exists() and "cost center" not in text.lower()
