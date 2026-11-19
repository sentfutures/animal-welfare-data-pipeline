"""Routing of gpt-/gemini- models through call_claude, the Gemini wrapper's
request and response handling, per-stage effort and the thinking headroom."""


import pytest

from shared import api, gemini_api, utils

# The real network seam, captured before the autouse guard in conftest replaces it.
_REAL_GENERATE = gemini_api._generate


def _gemini_reply(text="ok", finish="STOP", prompt=1000, candidates=200, thoughts=800, thought_text=None):
    parts = [{"text": thought_text, "thought": True}] if thought_text else []
    parts.append({"text": text})
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": candidates,
                              "thoughtsTokenCount": thoughts}}


@pytest.fixture
def stub_gemini(monkeypatch):
    def install(reply):
        calls = []

        def fake(model, body):
            calls.append((model, body))
            return reply(body) if callable(reply) else reply

        monkeypatch.setattr(gemini_api, "_generate", fake)
        return calls

    return install


@pytest.fixture
def run_log(tmp_path, monkeypatch):
    path = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(api, "_cost_log_path", path)
    return path


class TestGeminiRouting:
    def test_gemini_model_routes_to_gemini_with_system_and_thinking_level(self, stub_gemini, run_log, monkeypatch):
        monkeypatch.setattr(api, "_config", {"max_tokens": 4000, "effort": {"response_draft": "high"}})
        calls = stub_gemini(_gemini_reply("the answer"))
        text, stop = api.call_claude("user turn", "system text", model="gemini-4-argon",
                                     stage="response_draft", item_id="p1", return_stop_reason=True)
        assert (text, stop) == ("the answer", "end_turn")
        model, body = calls[0]
        assert model == "gemini-4-argon"
        assert body["systemInstruction"] == {"parts": [{"text": "system text"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": "user turn"}]}]
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}
        assert body["generationConfig"]["maxOutputTokens"] == 4000 + api._DEFAULT_THINKING_HEADROOM

    def test_thought_parts_are_never_returned(self, stub_gemini, run_log):
        stub_gemini(_gemini_reply("answer only", thought_text="private reasoning"))
        assert api.call_claude("u", model="gemini-3.1-pro-preview") == "answer only"

    def test_thinking_tokens_are_billed_as_output(self, stub_gemini, run_log):
        stub_gemini(_gemini_reply(prompt=1_000_000, candidates=500_000, thoughts=500_000))
        api.call_claude("u", model="gemini-3.1-pro-preview", stage="response_draft")
        record = utils.load_jsonl(run_log)[0]
        assert record["backend"] == "gemini"
        assert record["output_tokens"] == 1_000_000
        assert record["reasoning_tokens"] == 500_000
        assert record["cost_usd"] == pytest.approx(2.00 + 12.00)

    def test_no_effort_configured_leaves_the_models_own_thinking_level(self, stub_gemini, run_log):
        calls = stub_gemini(_gemini_reply())
        api.call_claude("u", model="gemini-4-argon", stage="response_draft")
        assert "thinkingConfig" not in calls[0][1]["generationConfig"]

    @pytest.mark.parametrize("finish, expected", [
        ("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal"), ("PROHIBITED_CONTENT", "refusal"),
        ("OTHER", "other")])
    def test_finish_reasons_map_to_the_pipelines_stop_reasons(self, stub_gemini, run_log, finish, expected):
        stub_gemini(_gemini_reply("partial", finish=finish))
        _, stop = api.call_claude("u", model="gemini-4-argon", return_stop_reason=True)
        assert stop == expected

    def test_a_blocked_prompt_is_a_refusal(self, stub_gemini, run_log):
        stub_gemini({"promptFeedback": {"blockReason": "SAFETY"},
                     "usageMetadata": {"promptTokenCount": 10}})
        assert api.call_claude("u", model="gemini-4-argon", return_stop_reason=True) == ("", "refusal")

    def test_unpriced_gemini_model_is_refused_before_any_call(self, stub_gemini, run_log):
        calls = stub_gemini(_gemini_reply())
        with pytest.raises(ValueError, match="no price"):
            api.call_claude("u", model="gemini-0-unknown")
        assert calls == []

    def test_a_daily_quota_fails_at_once_without_retrying(self, monkeypatch):
        posts = []

        class _Response:
            status_code = 429
            text = '{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}}'

        def fake_post(*args, **kwargs):
            posts.append(args)
            return _Response()

        monkeypatch.setattr(gemini_api, "load_api_key", lambda name: "fake-key")
        monkeypatch.setattr(gemini_api.httpx, "post", fake_post)
        with pytest.raises(RuntimeError, match="daily quota"):
            _REAL_GENERATE("gemini-4-argon", {})
        assert len(posts) == 1

    def test_xhigh_and_max_are_sent_as_high(self):
        assert gemini_api.thinking_level("xhigh") == "high"
        assert gemini_api.thinking_level("max") == "high"
        assert gemini_api.thinking_level("low") == "low"


class TestOpenAIRouting:
    def test_gpt_model_routes_to_openai_with_stage_effort(self, stub_openai_responses, openai_response,
                                                         run_log, monkeypatch):
        monkeypatch.setattr(api, "_config", {"max_tokens": 4000, "effort": {"prompt_draft": "high"}})
        calls = stub_openai_responses([openai_response(text="drafted")])
        text, stop = api.call_claude("u", "sys", model="gpt-6.1-sol", stage="prompt_draft",
                                     return_stop_reason=True)
        assert (text, stop) == ("drafted", "end_turn")
        assert calls[0]["reasoning"] == {"effort": "high"}
        assert calls[0]["instructions"] == "sys"
        assert calls[0]["max_output_tokens"] == 4000 + api._DEFAULT_THINKING_HEADROOM

    def test_gpt_without_configured_effort_uses_the_openai_default(self, stub_openai_responses,
                                                                  openai_response, run_log):
        calls = stub_openai_responses([openai_response()])
        api.call_claude("u", model="gpt-6.1-sol", stage="prompt_draft")
        from shared import openai_api
        assert calls[0]["reasoning"] == {"effort": openai_api.DEFAULT_REASONING_EFFORT}


class TestClaudeEffortAndHeadroom:
    def _sent(self, monkeypatch, config, model, stage):
        monkeypatch.setattr(api, "_config", config)
        sent = {}

        def record(**kwargs):
            sent.update(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                                   content=[SimpleNamespace(type="text", text="ok")],
                                   stop_reason="end_turn")

        monkeypatch.setattr(api, "_call_with_retry", record)
        monkeypatch.setattr(api, "_get_client", lambda: None)
        api.call_claude("u", "s", model=model, stage=stage)
        return sent

    def test_stage_effort_reaches_claude(self, monkeypatch, run_log):
        sent = self._sent(monkeypatch, {"effort": {"constitution_rewrite": "high"}},
                          "claude-opus-5-5", "constitution_rewrite")
        assert sent["effort"] == "high"

    def test_unlisted_stage_sends_no_effort(self, monkeypatch, run_log):
        sent = self._sent(monkeypatch, {"effort": {"constitution_rewrite": "high"}},
                          "claude-opus-5-5", "response_draft")
        assert "effort" not in sent

    def test_headroom_only_for_models_that_must_think(self, monkeypatch, run_log):
        config = {"max_tokens": 4000, "thinking_headroom": 6000}
        assert self._sent(monkeypatch, config, "claude-sonnet-5-5", "x")["max_tokens"] == 10000
        assert self._sent(monkeypatch, config, "claude-opus-5", "x")["max_tokens"] == 4000

    def test_configured_effort_overrides_opus_5_5_default_low(self):
        import importlib
        real = importlib.reload(api)._call_with_retry

        class Messages:
            def create(self, **kwargs):
                self.kwargs = kwargs

        client = type("Client", (), {"messages": Messages()})()
        real(client=client, model="claude-opus-5-5", max_tokens=100, system="s", messages=[],
             temperature=1.0, effort="high")
        assert client.messages.kwargs["output_config"] == {"effort": "high"}

    def test_sonnet_5_5_requires_adaptive_thinking(self):
        assert api._requires_adaptive_thinking("claude-sonnet-5-5")
        assert not api._requires_adaptive_thinking("claude-sonnet-5")

    def test_a_bad_effort_level_fails_at_init(self, tmp_path):
        config = tmp_path / "c.yaml"
        config.write_text("backend: api\neffort: {response_draft: extreme}\n"
                          "outputs: {cost_log: " + str(tmp_path / "log.jsonl") + "}\n")
        with pytest.raises(ValueError, match="effort"):
            api.init(str(config))
