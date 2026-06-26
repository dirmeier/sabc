.PHONY: tests lints format build example

build:
	uv pip install -e . --no-build-isolation

tests:
	uv run pytest

lints:
	uv run ruff check sabc examples

format:
	uv run ruff check --select I --fix sabc examples
	uv run ruff format sabc examples

example:
	uv run python examples/gaussian.py
