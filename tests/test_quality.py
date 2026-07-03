"""The deterministic report-quality gate."""

from maxim.quality import repair_instruction, report_violations

KNOWN = {"F-ai1", "F-st1"}

CLEAN = """# Topic

## TL;DR
- STL wins [F-ai1]

## Method Landscape
| Method | Perspective | Maturity | Community Signal | Best When | Findings |
|---|---|---|---|---|---|
| STL | Statistics | mature | – | seasonal | [F-st1] |

## Decision Guide
Start with STL [F-st1].

## Caveats
Thin coverage on GPU baselines [F-ai1].
"""


def test_clean_report_passes():
    assert report_violations(CLEAN, KNOWN) == []


def test_missing_sections_flagged():
    violations = report_violations("# Topic\n\n## TL;DR\nok [F-ai1]\n", KNOWN)
    assert any("Method Landscape" in v for v in violations)
    assert any("Decision Guide" in v for v in violations)
    assert any("Caveats" in v for v in violations)


def test_unknown_citation_flagged():
    violations = report_violations(CLEAN + "\nAlso bogus [F-zz9].\n", KNOWN)
    assert any("F-zz9" in v for v in violations)


def test_uncited_landscape_row_flagged():
    body = CLEAN.replace(
        "| STL | Statistics | mature | – | seasonal | [F-st1] |",
        "| STL | Statistics | mature | – | seasonal | none |",
    )
    violations = report_violations(body, KNOWN)
    assert any("Landscape row 'STL'" in v for v in violations)


def test_long_uncited_paragraph_flagged():
    filler = (
        "STL decomposes the series into trend and seasonality and this "
        "sentence keeps going to be substantive prose. "
    ) * 5
    body = CLEAN + f"\n## Statistics Take\n\n{filler}\n"
    violations = report_violations(body, KNOWN)
    assert any("uncited factual paragraph" in v and "Statistics Take" in v for v in violations)


def test_short_transition_paragraph_not_flagged():
    body = CLEAN + "\n## Statistics Take\n\nIn short, the classics hold up.\n"
    assert report_violations(body, KNOWN) == []


def test_cited_long_paragraph_not_flagged():
    filler = (
        "STL decomposes the series into trend and seasonality and this "
        "sentence keeps going to be substantive prose. "
    ) * 5
    body = CLEAN + f"\n## Statistics Take\n\n{filler} [F-st1]\n"
    assert report_violations(body, KNOWN) == []


def test_repair_instruction_lists_violations():
    text = repair_instruction(["missing section", "row uncited"])
    assert "missing section" in text
    assert "row uncited" in text
    assert "never invent ids" in text


FILLER = (
    "STL decomposes the series into trend and seasonality and this "
    "sentence keeps going to be substantive prose. "
) * 5


def test_bold_led_paragraph_is_still_prose():
    body = CLEAN + f"\n## Statistics Take\n\n**STL wins.** {FILLER}\n"
    violations = report_violations(body, KNOWN)
    assert any("uncited factual paragraph" in v for v in violations)


def test_true_list_items_are_not_prose():
    bullets = "\n".join(f"- {FILLER}" for _ in range(2))
    body = CLEAN + f"\n## Statistics Take\n\n{bullets}\n"
    assert report_violations(body, KNOWN) == []


def test_fenced_code_blocks_are_opaque():
    # A '## Caveats' inside a fence must not satisfy the required-section
    # check, and fence bodies must not be flagged as uncited prose.
    body = CLEAN.replace("## Caveats\nThin coverage on GPU baselines [F-ai1].\n", "")
    body += f"\n## Statistics Take\n\n```\n## Caveats\n{FILLER}\n```\n"
    violations = report_violations(body, KNOWN)
    assert any("'## Caveats' is missing" in v for v in violations)
    assert not any("uncited factual paragraph" in v for v in violations)


def test_preamble_prose_is_checked():
    body = f"# Topic\n\n{FILLER}\n\n" + CLEAN.split("\n", 2)[2]
    violations = report_violations(body, KNOWN)
    assert any("report preamble" in v for v in violations)


def test_duplicate_landscape_sections_both_checked():
    extra = (
        "\n## Method Landscape (community)\n"
        "| Method | Findings |\n|---|---|\n| Prophet | none |\n"
    )
    violations = report_violations(CLEAN + extra, KNOWN)
    assert any("row 'Prophet'" in v for v in violations)
