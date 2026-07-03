"""Maxim — Multi-Agent eXploration & Intelligence Machine."""


def main() -> None:
    import sys

    from .cli import main as cli_main

    sys.exit(cli_main())
