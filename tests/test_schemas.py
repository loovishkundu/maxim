import pytest
from conftest import make_dossier, make_plan
from pydantic import ValidationError

from maxim.schemas import (
    SCHEMA_VERSION,
    ResearchBrief,
    RunResult,
    RunUsage,
)


def test_strict_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        ResearchBrief(
            perspective="ai_agentic",
            objective="x",
            sub_questions=[],
            seed_queries=[],
            must_cover_methods=[],
            avoid=[],
            surprise="nope",
        )


def test_run_result_round_trips():
    result = RunResult(
        schema_version=SCHEMA_VERSION,
        topic="t",
        plan=make_plan(),
        dossiers=[make_dossier()],
        report_markdown="# hi",
        usage=RunUsage(stages=[], total_cost_usd=0.0, wall_seconds=1.0),
        partial=False,
        warnings=[],
    )
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_perspective_literal_enforced():
    with pytest.raises(ValidationError):
        ResearchBrief(
            perspective="astrology",
            objective="x",
            sub_questions=[],
            seed_queries=[],
            must_cover_methods=[],
            avoid=[],
        )
