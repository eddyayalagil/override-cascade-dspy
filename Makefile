.PHONY: help install install-dev test lint format check clean docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package and dependencies
	pip install -e .

install-dev: ## Install package in development mode with dev dependencies
	pip install -e .[dev]

test: ## Run tests
	pytest

test-fast: ## Run tests excluding slow tests
	pytest -m "not slow"

lint: ## Run linting (ruff)
	ruff check override_cascade tests

format: ## Format code (black + isort)
	black override_cascade tests examples
	isort override_cascade tests examples

check: ## Run all checks (lint + format + test)
	make lint
	make format
	make test

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Generate documentation
	cd docs && make html

demo: ## Run basic demo
	python -m override_cascade.main --demo

experiment: ## Run threshold dynamics experiment
	python -m override_cascade.experiments.threshold_dynamics

.DEFAULT_GOAL := help
