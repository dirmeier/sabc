.PHONY: build tests lints format check docs example

build:
	uv pip install -e . --no-build-isolation

tests:
	uv run pytest

lints:
	uv run ruff check src/sabc examples
	uvx cpplint --filter=-legal/copyright,-readability/casting,-whitespace/braces,-whitespace/indent,-build/include_subdir,-whitespace/line_length,-build/header_guard,-runtime/references,-build/include_order,-runtime/int src/csrc/*.cpp src/csrc/*.hpp

format:
	uv run ruff check --select I --fix src/sabc examples
	uv run ruff format src/sabc examples
	uvx clang-format@22.1.5 -i src/csrc/*.cpp src/csrc/*.hpp

check:
	cppcheck --enable=warning,style --inline-suppr --std=c++23 src/csrc

docs:
	doxygen

example:
	uv run python examples/gaussian.py
