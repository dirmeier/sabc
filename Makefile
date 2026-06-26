.PHONY: build tests lints format check docs example


tests:
	uv run pytest

lints:
	uv run ruff check src/sabc examples
	uvx cpplint --filter=-legal/copyright,-readability/casting,-whitespace/braces,-whitespace/indent,-build/include_subdir,-whitespace/line_length,-build/header_guard,-runtime/references,-build/include_order,-runtime/int src/cpp/*.cpp src/cpp/*.hpp

format:
	uv run ruff check --select I --fix src/sabc examples
	uv run ruff format src/sabc examples
	uvx clang-format@22.1.5 -i src/cpp/*.cpp src/cpp/*.hpp

check:
	cppcheck --enable=warning,style --inline-suppr --std=c++23 src/cpp
