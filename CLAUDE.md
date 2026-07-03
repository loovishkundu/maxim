# Maxim — project instructions

## Linting (mandatory before every commit)

Always run all three, in this order, and fix anything they report **before**
committing:

```bash
uv run isort .
uv run black .
uv run ruff check .
```

Never commit with any of the three failing. All are configured in
`pyproject.toml` (line length 100, py312); isort uses the black profile so the
tools agree with each other.

## Commit messages

- Never include Claude's name in commit messages — no "Co-Authored-By: Claude"
  trailers, no "Generated with Claude" lines, no AI attribution of any kind.
- Keep them simple and direct: a short imperative subject line; a brief body
  only when the change genuinely needs explanation.

## Testing

`uv run pytest` must be green before committing. Tests use a FakeLLM — no
network or API key needed.

## Environment quirk

This repo lives on an iCloud-synced Desktop. If `import maxim` suddenly fails
with ModuleNotFoundError, iCloud has set macOS hidden flags inside `.venv`
(Python 3.12 skips hidden `.pth` files): run `chflags -R nohidden .venv`.
