"""Tests for evals/variable_persistence_sdf.py (fully offline; call_claude is stubbed).

Variable values and consideration labels come from the repo's real variables.txt and
considerations.txt, so a reword there does not break these tests.
"""

import csv
import json

import pytest
import yaml

from evals import variable_persistence_sdf as vp
from sdf_pipeline import compose_prompts as cp

DESCRIPTION_MARKER = "SECRET-DESCRIPTION-TEXT-DO-NOT-LEAK"


@pytest.fixture(scope="module")
def context():
    return vp.JudgeContext(vp.DEFAULT_PROMPTS_DIR)


@pytest.fixture(scope="module")
def dealt_values():
    values, _, _ = cp.load_variables(vp.DEFAULT_PROMPTS_DIR / "variables.txt")
    return {name: options[0] for name, options in values.items()}


@pytest.fixture
def labels(context):
    return context.catalog.moral[0].label, context.catalog.epistemic[0].label


def make_record(dealt_values, doc_id="matrix_000001", considerations=(), content="The document text."):
    moral, epistemic = [], []
    for label in considerations:
        (moral if label in {e.label for e in vp.considerations_catalog.load_considerations(
            vp.DEFAULT_PROMPTS_DIR).moral} else epistemic).append(label)
    return {
        "doc_id": doc_id,
        "description": DESCRIPTION_MARKER,
        "considerations_spec": DESCRIPTION_MARKER,
        "scores": {"alignment": 9},
        "content": content,
        "variables": {**dealt_values, "moral_considerations": moral, "epistemic_considerations": epistemic},
    }


def good_answer(record, context, verdict="yes", skip=(), extra_labels=None):
    answer = {name: {"reasoning": "ok", "persisted": verdict}
              for name in vp.judged_variables(record, context) if name not in skip}
    considerations = vp.assigned_considerations(record, context)
    if considerations:
        answer["considerations"] = [{"label": c["label"], "reasoning": "ok", "persisted": verdict}
                                    for c in considerations]
    return answer


class TestSchema:
    def test_one_required_object_per_variable_reasoning_first(self):
        schema = vp.build_schema(["tone", "culture"], [])
        assert schema["required"] == ["tone", "culture"]
        assert schema["additionalProperties"] is False
        item = schema["properties"]["tone"]
        assert list(item["properties"]) == ["reasoning", "persisted"]
        assert item["properties"]["persisted"]["enum"] == ["yes", "no"]
        assert item["additionalProperties"] is False
        assert "considerations" not in schema["properties"]

    def test_consideration_labels_become_an_enum(self):
        schema = vp.build_schema(["tone"], ["a label", "another label"])
        items = schema["properties"]["considerations"]["items"]
        assert items["properties"]["label"]["enum"] == ["a label", "another label"]
        assert items["required"] == ["label", "reasoning", "persisted"]
        assert items["additionalProperties"] is False
        assert "considerations" in schema["required"]

    def test_schema_for_a_record_covers_the_dealt_variables(self, context, dealt_values, labels):
        record = make_record(dealt_values, considerations=labels)
        schema = vp.build_schema(list(vp.judged_variables(record, context)),
                                 [c["label"] for c in vp.assigned_considerations(record, context)])
        assert set(context.variable_meanings) <= set(schema["properties"])
        assert schema["properties"]["considerations"]["items"]["properties"]["label"]["enum"] == list(labels)

    def test_variables_missing_from_the_record_are_not_asked_about(self, context, dealt_values):
        record = make_record(dealt_values)
        del record["variables"]["domain"]
        assert "domain" not in vp.judged_variables(record, context)


class TestPromptRendering:
    def test_contains_values_meanings_considerations_and_document(self, context, dealt_values, labels):
        record = make_record(dealt_values, considerations=labels, content="UNIQUE-DOCUMENT-BODY")
        prompt = vp.render_prompt(record, context)
        for name, value in dealt_values.items():
            assert name in prompt and value in prompt
        for meaning in context.variable_meanings.values():
            assert meaning in prompt
        for label in labels:
            assert label in prompt
            assert context.catalog.entry_by_label(label).gloss in prompt
        assert "UNIQUE-DOCUMENT-BODY" in prompt
        assert "ASSIGNED CONSIDERATIONS" in prompt

    def test_never_shows_description_scores_or_doc_id(self, context, dealt_values, labels):
        record = make_record(dealt_values, doc_id="matrix_987654", considerations=labels)
        prompt = vp.render_prompt(record, context)
        assert DESCRIPTION_MARKER not in prompt
        assert "matrix_987654" not in prompt
        assert "alignment" not in prompt.split("DOCUMENT\n")[0].lower().replace("alignment documents", "")

    def test_no_considerations_section_when_none_assigned(self, context, dealt_values):
        prompt = vp.render_prompt(make_record(dealt_values), context)
        assert "ASSIGNED CONSIDERATIONS" not in prompt

    def test_braces_in_the_document_survive(self, context, dealt_values):
        prompt = vp.render_prompt(make_record(dealt_values, content="{not_a_placeholder} {0}"), context)
        assert "{not_a_placeholder} {0}" in prompt

    def test_persisted_definition_comes_from_the_prompt_file(self, context, dealt_values):
        prompt = vp.render_prompt(make_record(dealt_values), context)
        assert "dealt skeptical stance" in prompt or "skeptical stance" in prompt


class TestParsing:
    NAMES = ["tone", "culture"]

    def answer(self, **overrides):
        base = {"tone": {"reasoning": "r", "persisted": "yes"}, "culture": {"reasoning": "r", "persisted": "no"}}
        base.update(overrides)
        return json.dumps(base)

    def test_valid_answer(self):
        variables, considerations = vp.parse_answer(self.answer(), self.NAMES, [])
        assert variables["culture"]["persisted"] == "no" and considerations == []

    def test_missing_variable(self):
        raw = json.dumps({"tone": {"reasoning": "r", "persisted": "yes"}})
        with pytest.raises(vp.PersistenceAnswerError, match="culture"):
            vp.parse_answer(raw, self.NAMES, [])

    def test_bad_verdict_and_empty_reasoning(self):
        with pytest.raises(vp.PersistenceAnswerError, match="yes"):
            vp.parse_answer(self.answer(tone={"reasoning": "r", "persisted": "maybe"}), self.NAMES, [])
        with pytest.raises(vp.PersistenceAnswerError, match="reasoning"):
            vp.parse_answer(self.answer(tone={"reasoning": " ", "persisted": "yes"}), self.NAMES, [])

    def test_not_json(self):
        with pytest.raises(vp.PersistenceAnswerError):
            vp.parse_answer("no json here", self.NAMES, [])

    def test_consideration_missing_duplicate_or_unassigned(self):
        def item(label):
            return {"label": label, "reasoning": "r", "persisted": "yes"}
        with pytest.raises(vp.PersistenceAnswerError, match="missing"):
            vp.parse_answer(self.answer(considerations=[item("a")]), self.NAMES, ["a", "b"])
        with pytest.raises(vp.PersistenceAnswerError, match="more than once"):
            vp.parse_answer(self.answer(considerations=[item("a"), item("a")]), self.NAMES, ["a", "b"])
        with pytest.raises(vp.PersistenceAnswerError, match="not assigned"):
            vp.parse_answer(self.answer(considerations=[item("a"), item("zzz")]), self.NAMES, ["a", "b"])
        with pytest.raises(vp.PersistenceAnswerError, match="array"):
            vp.parse_answer(self.answer(), self.NAMES, ["a"])

    def test_considerations_returned_in_assigned_order(self):
        def item(label):
            return {"label": label, "reasoning": "r", "persisted": "yes"}
        _, considerations = vp.parse_answer(
            self.answer(considerations=[item("b"), item("a")]), self.NAMES, ["a", "b"])
        assert [c["label"] for c in considerations] == ["a", "b"]


class TestJudgeDocument:
    def test_retries_then_succeeds(self, stub_claude, context, dealt_values, labels):
        record = make_record(dealt_values, considerations=labels)
        replies = iter([json.dumps({"tone": {"reasoning": "r", "persisted": "yes"}}),
                        json.dumps(good_answer(record, context))])
        calls = stub_claude(lambda user_message, **kw: next(replies))
        row = vp.judge_document(record, context, "some-model", max_attempts=3)
        assert row["status"] == "ok" and row["attempts"] == 2 and len(calls) == 2
        assert calls[0]["json_schema"]["required"] == calls[1]["json_schema"]["required"]
        assert calls[0]["stage"] == vp.STAGE_TAG and calls[0]["model"] == "some-model"
        assert len(row["considerations"]) == 2
        assert row["variables"]["tone"]["dealt_value"] == dealt_values["tone"]

    def test_failure_recorded_after_max_attempts(self, stub_claude, context, dealt_values, labels):
        record = make_record(dealt_values, considerations=labels)
        bad = good_answer(record, context)
        bad["considerations"] = bad["considerations"][:1]  # one label missing
        calls = stub_claude(lambda user_message, **kw: json.dumps(bad))
        row = vp.judge_document(record, context, None, max_attempts=3)
        assert row["status"] == "failed" and row["attempts"] == 3 and len(calls) == 3
        assert "missing" in row["error"]

    def test_api_exception_is_recorded_not_raised(self, stub_claude, context, dealt_values):
        def boom(user_message, **kw):
            raise RuntimeError("network down")
        stub_claude(boom)
        row = vp.judge_document(make_record(dealt_values), context, None, max_attempts=3)
        assert row["status"] == "failed" and "network down" in row["error"]


def write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "backend": "api", "model": "claude-sonnet-5", "max_tokens": 4000, "workers": 3,
        "outputs": {"cost_log": str(tmp_path / "cost.jsonl")}, "sdf": {}}))
    return path


def run_judging(tmp_path, records, context, output_dir, timestamp="2026-01-01_00-00"):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["_config_path"] = str(config_path)
    vp.run_judging(records, context, config, output_dir, timestamp)


class TestRunAndResume:
    def test_outputs_summary_counts_and_failed_checks(self, stub_claude, tmp_path, context, dealt_values, labels):
        records = [make_record(dealt_values, f"d{i}", considerations=labels if i == 0 else ()) for i in range(3)]

        def judge(user_message, **kw):
            record = next(r for r in records if kw["item_id"] == r["doc_id"])
            answer = good_answer(record, context)
            if record["doc_id"] == "d1":
                answer["tone"]["persisted"] = "no"
                answer["tone"]["reasoning"] = "author converted"
            if record["doc_id"] == "d0":
                answer["considerations"][0]["persisted"] = "no"
            return json.dumps(answer)

        stub_claude(judge)
        out = tmp_path / "out"
        out.mkdir()
        run_judging(tmp_path, records, context, out)
        names = vp.output_names("2026-01-01_00-00")
        rows = {(r["kind"], r["name"]): r for r in csv.DictReader(open(out / names["summary"]))}
        assert rows[("variable", "tone")]["documents_judged"] == "3"
        assert rows[("variable", "tone")]["no_count"] == "1"
        assert float(rows[("variable", "tone")]["share_no"]) == pytest.approx(1 / 3, abs=1e-3)
        assert rows[("variable", "culture")]["no_count"] == "0"
        assert rows[("consideration", labels[0])]["no_count"] == "1"
        assert rows[("consideration", labels[1])]["yes_count"] == "1"
        overall = rows[("considerations_overall", "all considerations")]
        assert (overall["documents_judged"], overall["no_count"]) == ("2", "1")
        failed = [json.loads(l) for l in open(out / names["failed"])]
        assert {(f["doc_id"], f["variable"]) for f in failed} == {("d1", "tone"), ("d0", labels[0])}
        tone_failure = next(f for f in failed if f["variable"] == "tone")
        assert tone_failure["reasoning"] == "author converted" and tone_failure["dealt_value"] == dealt_values["tone"]
        assert len(open(out / names["judgments"]).readlines()) == 3

    def test_resume_skips_done_and_retries_failures(self, stub_claude, tmp_path, context, dealt_values):
        records = [make_record(dealt_values, f"d{i}") for i in range(3)]
        out = tmp_path / "out"
        out.mkdir()
        names = vp.output_names("2026-01-01_00-00")
        # d0 already judged, d1 failed earlier, d2 never attempted
        stub_claude(lambda user_message, **kw: json.dumps(good_answer(records[0], context)))
        first = vp.judge_document(records[0], context, None, 1)
        with open(out / names["judgments"], "w") as handle:
            handle.write(json.dumps(first) + "\n")
            handle.write(json.dumps({"doc_id": "d1", "status": "failed", "attempts": 3, "error": "x"}) + "\n")
        calls = stub_claude(lambda user_message, **kw: json.dumps(good_answer(records[0], context)))
        run_judging(tmp_path, records, context, out)
        assert sorted(c["item_id"] for c in calls) == ["d1", "d2"]
        assert vp.existing_timestamp(out) == "2026-01-01_00-00"

    def test_everything_failing_exits_without_summary(self, stub_claude, tmp_path, context, dealt_values):
        stub_claude(lambda user_message, **kw: "garbage")
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(SystemExit):
            run_judging(tmp_path, [make_record(dealt_values)], context, out)


class TestSelectionAndDryRun:
    def test_sample_is_seeded_and_keeps_order(self, dealt_values):
        records = [make_record(dealt_values, f"d{i}") for i in range(20)]
        a = vp.select_records(records, None, 5, seed=3)
        b = vp.select_records(records, None, 5, seed=3)
        assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b]
        ids = [int(r["doc_id"][1:]) for r in a]
        assert len(a) == 5 and ids == sorted(ids)

    def test_doc_ids_filter_and_unknown_id(self, dealt_values):
        records = [make_record(dealt_values, f"d{i}") for i in range(4)]
        assert [r["doc_id"] for r in vp.select_records(records, ["d2"], None, 0)] == ["d2"]
        with pytest.raises(SystemExit):
            vp.select_records(records, ["nope"], None, 0)

    def test_dry_run_makes_no_calls_and_prints_estimate(self, stub_claude, capsys, context, dealt_values):
        calls = stub_claude(lambda user_message, **kw: "unused")
        records = [make_record(dealt_values, f"d{i}") for i in range(4)]
        vp.dry_run(records, context, None, {"model": "claude-sonnet-5"})
        out = capsys.readouterr().out
        assert calls == []
        assert "4 calls" in out and "RENDERED PROMPT" in out and "ANSWER SCHEMA" in out and "cost ~$" in out
