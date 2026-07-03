import pytest

from maxim.cli import _build_parser


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "maxim" in capsys.readouterr().out.lower()


def test_parser_defaults():
    args = _build_parser().parse_args(["some topic"])
    assert args.topic == "some topic"
    assert args.depth == "standard"
    assert args.budget_usd is None  # resolved per-depth in main()
    assert not args.json
    assert not args.yes
