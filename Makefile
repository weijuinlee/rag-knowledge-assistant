.PHONY: install run test lint

install:
	pip install -e ".[dev]"

run:
	uvicorn rag_assistant.api:app --reload --app-dir src

test:
	pytest

lint:
	ruff check src tests
