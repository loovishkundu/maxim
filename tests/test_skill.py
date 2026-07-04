"""The Claude Code skill must stay truthful about the CLI it wraps."""

import re
from pathlib import Path

from maxim.cli import _build_parser
from maxim.schemas import SCHEMA_VERSION

SKILL_PATH = Path(__file__).parent.parent / ".claude" / "skills" / "maxim" / "SKILL.md"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_exists_with_frontmatter():
    text = _skill_text()
    assert text.startswith("---\n")
    frontmatter = text.split("---", 2)[1]
    assert re.search(r"^name:\s*maxim\s*$", frontmatter, re.MULTILINE)
    assert re.search(r"^description:\s*\S", frontmatter, re.MULTILINE)


def test_skill_mentions_only_real_cli_flags():
    parser = _build_parser()
    real_flags = {opt for action in parser._actions for opt in action.option_strings}
    mentioned = set(re.findall(r"(--[a-z][a-z-]+)", _skill_text()))
    fake = mentioned - real_flags
    assert not fake, f"skill references CLI flags that do not exist: {sorted(fake)}"


def test_skill_pins_the_current_schema_version():
    # If the schema version bumps, the skill's contract note must move with it.
    assert f'`"{SCHEMA_VERSION}"`' in _skill_text()


def test_skill_documents_the_exit_codes():
    text = _skill_text()
    for code in ("0", "2", "3", "4"):
        assert re.search(rf"^- `{code}`", text, re.MULTILINE), f"exit code {code} undocumented"


def test_skill_names_real_runresult_fields():
    from maxim.schemas import RunResult

    text = _skill_text()
    for field in ("report_markdown", "dossiers", "pulse", "warnings", "schema_version"):
        assert field in RunResult.model_fields
        assert field in text
