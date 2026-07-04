"""Golden-report regression test.

Runs the full pipeline on the canned two-wave FakeLLM topic and compares the
assembled markdown against a checked-in snapshot. Catches unintended drift
anywhere in the report chain — template, citation resolution, sources
section, pulse rendering, appendices, metadata footer.

Intentional changes: regenerate with

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden_report.py
"""

import os
import re
from pathlib import Path

from test_pipeline import _install, _settings

import maxim.orchestrator as orchestrator

GOLDEN = Path(__file__).parent / "golden" / "anomaly-detection-report.md"


def _normalize(markdown: str) -> str:
    """Strip the two genuinely volatile values (timestamp, wall time)."""
    markdown = re.sub(r"- Generated: \S+", "- Generated: <normalized>", markdown)
    markdown = re.sub(r"wall time: \d+s", "wall time: <normalized>", markdown)
    return markdown


async def test_report_matches_golden(monkeypatch):
    _install(monkeypatch, perspectives=("ai_agentic", "statistics", "community"))
    result = await orchestrator.run_pipeline(
        "anomaly detection for vehicle telemetry",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert not result.partial  # the golden run must be the healthy path
    got = _normalize(result.report_markdown)

    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(got, encoding="utf-8")

    assert GOLDEN.exists(), "golden file missing — run once with UPDATE_GOLDEN=1"
    expected = GOLDEN.read_text(encoding="utf-8")
    assert got == expected, (
        "report drifted from the golden snapshot; if the change is "
        "intentional, regenerate with UPDATE_GOLDEN=1"
    )
