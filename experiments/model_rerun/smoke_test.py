#!/usr/bin/env python3
"""Offline end-to-end check of the model-rerun harness. No network, no keys.

Builds both mini runs under a temp PIPELINE_OUTPUT_ROOT, drives them with
canned stage replies (the anthropic arm through a stubbed call_claude, the
openrouter arm through a stubbed _post_chat so the whole shim code path runs),
and asserts every selected example reaches the final corpus with zero baseline
calls. Run before spending real money:

    .venv/bin/python experiments/model_rerun/smoke_test.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

os.environ.setdefault("ANTHROPIC_API_KEY", "smoke-test-fake-key")
os.environ.setdefault("OPENROUTER_API_KEY", "smoke-test-fake-key")
_tmp = tempfile.mkdtemp(prefix="model_rerun_smoke_")
os.environ["PIPELINE_OUTPUT_ROOT"] = _tmp

from shared import api, utils  # noqa: E402
from dad_pipeline import reasoning_library  # noqa: E402
import build_run  # noqa: E402
import drive  # noqa: E402
import or_backend  # noqa: E402

SELECTION = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
N = sum(len(v) for v in SELECTION["buckets"].values())
SOURCE = ROOT / build_run.SOURCE_RUN_DEFAULT

# Library ids come from the source run's own snapshot, so the canned 2a.5
# selection always names real entries.
_library = reasoning_library.load(SOURCE / "inputs" / "prompts")
_two_ids = ", ".join(reasoning_library.all_ids(_library)[:2])


def canned_reply(system_prompt: str, user_message: str) -> str:
    sys_lower = (system_prompt or "").lower()
    blob = (system_prompt or "") + "\n" + user_message
    if not (system_prompt or ""):
        raise AssertionError("baseline call reached the stub — copied baselines "
                             "should have made the stage a no-op")
    # Exact markers first (same ones tests/test_e2e_smoke.py keys on), loose
    # fallbacks after, so 2b and the step-3 rewrite can't shadow each other.
    if "build the full map of the case" in blob or "map of the case" in blob:
        return json.dumps({"patients": "p", "goal": "g", "levers": "l", "cost": "c",
                           "magnitude": "m", "upside": "u", "replaceability": "cf"})
    if "retrieving reasoning modules" in blob or "reasoning modules" in blob:
        return _two_ids
    if "advisor responding to a user's dilemma" in blob:
        return "Draft response."
    if "rewrite a draft assistant response" in blob:
        return "Rewritten careful answer."
    if "advisor" in sys_lower or "dilemma" in sys_lower:
        return "Draft response."
    if "rewrite" in sys_lower:
        return "Rewritten careful answer."
    raise AssertionError(f"Unrecognized prompt: system {(system_prompt or '')[:160]!r}")


def stub_call_claude(user_message, **kw):
    text = canned_reply(kw.get("system_prompt", ""), user_message)
    # Mirror the real backend's cost-log write so check_run can assert on it.
    api._log_usage(kw.get("model") or "stub-model", 10, 5, cost_usd=0.0,
                   stage=kw.get("stage"), item_id=kw.get("item_id"), backend="api")
    return (text, "end_turn") if kw.get("return_stop_reason") else text


def stub_post_chat(payload):
    msgs = {m["role"]: m["content"] for m in payload["messages"]}
    text = canned_reply(msgs.get("system", ""), msgs.get("user", ""))
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001}}


def check_run(run_dir: Path, expect_backend: str) -> None:
    finals = utils.load_jsonl(run_dir / "final" / "dad_corpus.jsonl")
    assert len(finals) == N, f"{run_dir.name}: {len(finals)} final records, wanted {N}"
    calls = utils.load_jsonl(run_dir / "call_log.jsonl")
    # scope + select + respond + rewrite per example
    assert len(calls) == 4 * N, f"{run_dir.name}: {len(calls)} calls, wanted {4 * N}"
    assert all(c["stop_reason"] == "end_turn" for c in calls)
    cost_backends = {r.get("backend") for r in utils.load_jsonl(run_dir / "cost_log.jsonl")}
    assert expect_backend in cost_backends, \
        f"{run_dir.name}: cost log backends {cost_backends}, wanted {expect_backend}"
    baselines = utils.load_jsonl(run_dir / "baseline" / "baseline_responses.jsonl")
    assert len(baselines) == N
    print(f"  OK {run_dir.name}: {len(finals)} finals, {len(calls)} calls, "
          f"backends {sorted(cost_backends)}")


def main() -> None:
    print(f"smoke outputs under {_tmp}")

    print("[1/2] anthropic-arm plumbing")
    run_a = build_run.main(["--label", "local_smoke-anthropic",
                            "--model", "claude-opus-5-5"])
    api.call_claude = stub_call_claude
    drive.main(["--run-dir", str(run_a), "--backend", "anthropic"])
    check_run(run_a, expect_backend="api")

    print("[2/2] openrouter-arm plumbing (stubbed HTTP seam)")
    run_b = build_run.main(["--label", "local_smoke-gpt",
                            "--model", "openai/fake-gpt-for-smoke"])
    or_backend._post_chat = stub_post_chat
    drive.main(["--run-dir", str(run_b), "--backend", "openrouter"])
    check_run(run_b, expect_backend="openrouter")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
