.PHONY: install dev run test lint format clean docker-up docker-down ingest

# --- Setup ---
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# --- Run ---
run:
	uvicorn atlas.main:app --reload --host 0.0.0.0 --port 8080

# --- Test ---
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=atlas --cov-report=term-missing

# --- Lint & Format ---
lint:
	ruff check atlas/ tests/

format:
	ruff format atlas/ tests/
	ruff check --fix atlas/ tests/

# --- Docker ---
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

# --- Data ---
ingest:
	python scripts/ingest_papers.py --dir data/eval/

# --- Clean ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
