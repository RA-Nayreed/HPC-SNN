.PHONY: test lint check

test:
	uv run pytest

lint:
	uv run ruff check src tests

check: lint test
