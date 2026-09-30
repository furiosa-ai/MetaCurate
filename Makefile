.PHONY: setup lint format test

setup: ## Install dependencies and set up pre-commit hooks
	uv sync
	uv run pre-commit install

lint: ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

format: ## Run ruff formatter
	uv run ruff format .

test: ## Run the offline test suite
	uv run pytest tests/ -q
