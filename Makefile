.PHONY: install install-dev run test lint format clean

install:
	pip install -r requirements.txt && pip install -e .

install-dev:
	pip install -r requirements-dev.txt && pip install -e .

run:
	streamlit run ui/app.py

test:
	pytest

lint:
	ruff check src tests ui

format:
	ruff format src tests ui

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
