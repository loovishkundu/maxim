"""The CLI's report-write salvage: a paid-for run must never be discarded."""

from pathlib import Path

from conftest import make_dossier, make_plan
from rich.console import Console

import maxim.cli as cli
from maxim.config import Settings
from maxim.schemas import RunResult, RunUsage


def _result() -> RunResult:
    return RunResult(
        schema_version="2",
        topic="topic",
        plan=make_plan(),
        dossiers=[make_dossier()],
        report_markdown="# The Report\n\nBody [F-ai1].\n",
        usage=RunUsage(stages=[], total_cost_usd=1.0, wall_seconds=10.0),
        partial=False,
        warnings=[],
    )


def _install(monkeypatch):
    async def fake_pipeline(topic, settings, progress, confirm, on_synthesis_text=None):
        return _result()

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)


async def test_write_failure_salvages_to_temp_file(monkeypatch, capsys):
    _install(monkeypatch)
    # /dev/null is not a directory: mkdir/write under it raises OSError.
    settings = Settings(out=Path("/dev/null/report.md"), assume_yes=True, quiet=True)
    code = await cli._run(settings, "topic", Console(stderr=True, quiet=True))

    out = capsys.readouterr().out.strip()
    assert code == cli.EXIT_OK  # salvage succeeded: full result exists on disk
    salvaged = Path(out)
    assert salvaged.exists()
    assert salvaged.read_text() == _result().report_markdown


async def test_double_write_failure_dumps_markdown_to_stdout(monkeypatch, capsys):
    _install(monkeypatch)
    monkeypatch.setattr(
        cli.tempfile, "mkdtemp", lambda **_: (_ for _ in ()).throw(OSError("disk full"))
    )
    settings = Settings(out=Path("/dev/null/report.md"), assume_yes=True, quiet=True)
    code = await cli._run(settings, "topic", Console(stderr=True, quiet=True))

    out = capsys.readouterr().out
    assert "# The Report" in out  # the paid-for content reached the user
    assert code == cli.EXIT_PARTIAL  # result delivered via stdout, not lost
