# Shortcuts for the two supported workflows: uv locally, Docker for the demo.
# Run `make` with no target to see this list.

.DEFAULT_GOAL := help
.PHONY: help setup data pipeline pipeline-honest compare app app-honest test lint fmt clean \
        docker-build docker-app docker-pipeline docker-test docker-clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- local ----

setup:  ## Create the local venv from uv.lock (run this first)
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env — add your Kaggle credentials.")
	@echo "Ready. Next: make data"

data:  ## Download the raw dataset from Kaggle into data/raw/
	uv run python -m src.data

pipeline:  ## Full run: features, training, all four evaluation dimensions
	uv run python -m src.pipeline

pipeline-honest:  ## Same pipeline, leaking features excluded (writes artifacts-honest/)
	uv run python -m src.pipeline --config config.honest.yaml

compare:  ## Print both scorecards side by side
	uv run python -m src.compare

app:  ## Launch Streamlit on the leaky variant (artifacts/)
	uv run streamlit run app/Home.py

app-honest:  ## Launch Streamlit on the honest variant (artifacts-honest/)
	JOBAPP_CONFIG=config.honest.yaml uv run streamlit run app/Home.py

test:  ## Run the smoke suite (no Kaggle credentials needed)
	uv run pytest

lint:  ## Check formatting and lint rules
	uv run ruff check src app tests
	uv run ruff format --check src app tests

fmt:  ## Auto-fix formatting and lint rules
	uv run ruff check --fix src app tests
	uv run ruff format src app tests

clean:  ## Delete generated artifacts and processed data (keeps data/raw)
	rm -rf artifacts/models/* artifacts/results/* artifacts/figures/* data/processed/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned. data/raw was left alone."

# ---------------------------------------------------------------- docker ----

requirements:  ## Re-export requirements.txt from uv.lock
	uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
	@echo "Re-add the header comment at the top before committing."

docker-build:  ## Build the image
	docker compose build

docker-app:  ## Run the app in Docker on http://localhost:8501
	docker compose up app

docker-pipeline:  ## Run the full pipeline in Docker
	docker compose --profile pipeline run --rm pipeline

docker-test:  ## Run the test suite in Docker
	docker compose --profile test run --rm test

docker-clean:  ## Stop everything and drop volumes
	docker compose --profile pipeline --profile test down --volumes --remove-orphans
