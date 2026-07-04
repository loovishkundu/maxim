"""The wrapper contract the Claude Code skill (and any script) depends on.

- stdout carries ONLY the result: the report path, or one line of RunResult
  JSON with --json; all progress and warnings go to stderr.
- Exit codes: 0 ok · 2 partial · 3 hard failure · 4 plan declined.
- The --json payload round-trips through the RunResult schema and carries
  schema_version so wrappers can detect contract drift.
"""

import json

from conftest import make_dossier, make_plan

import maxim.cli as cli
from maxim.llm import LLMError
from maxim.orchestrator import PlanRejected
from maxim.schemas import SCHEMA_VERSION, RunResult, RunUsage


def _result(partial=False, warnings=()) -> RunResult:
    return RunResult(
        schema_version=SCHEMA_VERSION,
        topic="topic",
        plan=make_plan(),
        dossiers=[make_dossier()],
        report_markdown="# The Report\n\nBody [F-ai1].\n",
        usage=RunUsage(stages=[], total_cost_usd=1.0, wall_seconds=10.0),
        partial=partial,
        warnings=list(warnings),
    )


def _install(monkeypatch, *, result=None, raises=None):
    async def fake_pipeline(topic, settings, progress, confirm, on_synthesis_text=None):
        if raises is not None:
            raise raises
        return result if result is not None else _result()

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)


def _main(monkeypatch, tmp_path, *argv) -> int:
    monkeypatch.chdir(tmp_path)  # reports land in tmp, not the repo
    return cli.main(["topic", "--quiet", *argv])


def test_json_mode_stdout_is_exactly_one_parseable_runresult(monkeypatch, tmp_path, capsys):
    _install(monkeypatch)
    code = _main(monkeypatch, tmp_path, "--json")
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1  # stdout purity: nothing but the payload
    payload = json.loads(lines[0])
    assert payload["schema_version"] == SCHEMA_VERSION
    # The payload round-trips through the schema wrappers pin against.
    RunResult.model_validate(payload)
    assert code == cli.EXIT_OK


def test_default_mode_stdout_is_exactly_the_report_path(monkeypatch, tmp_path, capsys):
    _install(monkeypatch)
    code = _main(monkeypatch, tmp_path)
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(out_lines) == 1
    assert out_lines[0].endswith(".md")
    assert code == cli.EXIT_OK


def test_partial_result_exits_2(monkeypatch, tmp_path):
    _install(monkeypatch, result=_result(partial=True, warnings=["ai_agentic: failed"]))
    assert _main(monkeypatch, tmp_path) == cli.EXIT_PARTIAL


def test_plan_rejected_exits_4(monkeypatch, tmp_path):
    _install(monkeypatch, raises=PlanRejected())
    assert _main(monkeypatch, tmp_path) == cli.EXIT_CANCELLED


def test_llm_failure_exits_3(monkeypatch, tmp_path):
    _install(monkeypatch, raises=LLMError("planner: API call failed"))
    assert _main(monkeypatch, tmp_path) == cli.EXIT_FAILURE


def test_unexpected_exception_exits_3_not_traceback(monkeypatch, tmp_path):
    _install(monkeypatch, raises=RuntimeError("surprise"))
    assert _main(monkeypatch, tmp_path) == cli.EXIT_FAILURE


def test_unknown_perspective_is_an_argparse_error(monkeypatch, tmp_path, capsys):
    import pytest

    _install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["topic", "--perspectives", "astrology"])
    assert "unknown perspective" in capsys.readouterr().err


def test_quiet_implies_yes(monkeypatch, tmp_path):
    # A suppressed confirmation prompt must never hang a wrapper.
    confirmed = {}

    async def fake_pipeline(topic, settings, progress, confirm, on_synthesis_text=None):
        confirmed["assume_yes"] = settings.assume_yes
        return _result()

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    _main(monkeypatch, tmp_path)
    assert confirmed["assume_yes"] is True
