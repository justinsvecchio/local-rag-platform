.PHONY: help setup install dev start stop test test-unit test-int test-cov lint format clean bench eval docs

# Variables
PYTHON := python3.11
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTEST := $(BIN)/pytest
RUFF := $(BIN)/ruff
UVICORN := $(BIN)/uvicorn

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)RAG Platform - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make setup   # One-command setup"
	@echo "  make start   # Start API server"

# ============================================================================
# SETUP
# ============================================================================

setup: $(VENV) install-deps start-infra seed-data ## One-command setup (venv + deps + infra + seed)
	@echo "$(GREEN)Setup complete!$(NC)"
	@echo "Run '$(BLUE)make start$(NC)' to start the API server"
	@echo "API will be available at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

$(VENV):
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools

install-deps: $(VENV) ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -e ".[dev,local]"

install: $(VENV) ## Install production dependencies only
	$(PIP) install -e .

dev: $(VENV) ## Install development dependencies
	$(PIP) install -e ".[dev]"

# ============================================================================
# INFRASTRUCTURE
# ============================================================================

start-infra: ## Start Qdrant via Docker
	@echo "$(BLUE)Starting Qdrant...$(NC)"
	docker compose -f infra/docker-compose.yml up -d
	@echo "$(YELLOW)Waiting for Qdrant to be ready...$(NC)"
	@until curl -s http://localhost:6333/health > /dev/null 2>&1; do \
		sleep 1; \
		echo "  Waiting..."; \
	done
	@echo "$(GREEN)Qdrant is ready at http://localhost:6333$(NC)"

stop-infra: ## Stop Qdrant
	@echo "$(BLUE)Stopping Qdrant...$(NC)"
	docker compose -f infra/docker-compose.yml down

start: start-infra ## Start Qdrant and API server
	@echo "$(BLUE)Starting API server...$(NC)"
	$(UVICORN) apps.api.main:app --reload --host 0.0.0.0 --port 8000

start-api: ## Start only API server (assumes infra is running)
	$(UVICORN) apps.api.main:app --reload --host 0.0.0.0 --port 8000

stop: stop-infra ## Stop all services
	@echo "$(GREEN)All services stopped$(NC)"

# ============================================================================
# TESTING
# ============================================================================

test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-int: start-infra ## Run integration tests (starts Qdrant)
	$(PYTEST) tests/integration/ -v --tb=short -m integration

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=rag --cov=apps --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated at htmlcov/index.html$(NC)"

test-watch: ## Run tests in watch mode
	$(BIN)/ptw tests/ -- -v --tb=short

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: ## Run linters (ruff + mypy)
	@echo "$(BLUE)Running Ruff check...$(NC)"
	$(RUFF) check .
	@echo "$(BLUE)Running Ruff format check...$(NC)"
	$(RUFF) format --check .
	@echo "$(BLUE)Running MyPy...$(NC)"
	$(BIN)/mypy rag apps --ignore-missing-imports
	@echo "$(GREEN)All checks passed!$(NC)"

format: ## Format code with Ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	$(RUFF) check --fix .
	$(RUFF) format .
	@echo "$(GREEN)Code formatted!$(NC)"

pre-commit: ## Run pre-commit hooks
	$(BIN)/pre-commit run --all-files

# ============================================================================
# BENCHMARKS & EVALUATION
# ============================================================================

bench: start-infra ## Run benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTEST) benchmarks/ -v --benchmark-only --benchmark-json=benchmark-results.json
	@echo "$(GREEN)Benchmark results saved to benchmark-results.json$(NC)"

eval: start-infra ## Run evaluation suite
	@echo "$(BLUE)Running evaluation suite...$(NC)"
	$(BIN)/python -m evals.runner --dataset evals/datasets/sample.json --output evals/reports/
	@echo "$(GREEN)Evaluation complete! Check evals/reports/$(NC)"

# ============================================================================
# DATA & MAINTENANCE
# ============================================================================

seed-data: ## Seed sample data for development
	@echo "$(BLUE)Seeding sample data...$(NC)"
	@if [ -f scripts/seed_data.py ]; then \
		$(BIN)/python scripts/seed_data.py; \
	else \
		echo "$(YELLOW)No seed script found, skipping...$(NC)"; \
	fi

migrate: ## Run index migrations
	$(BIN)/python scripts/migrate_index.py

clean: ## Remove build artifacts and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf benchmark-results.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleaned!$(NC)"

clean-docker: ## Remove Docker volumes (WARNING: deletes all indexed data)
	@echo "$(YELLOW)WARNING: This will delete all indexed data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose -f infra/docker-compose.yml down -v
	@echo "$(GREEN)Docker volumes removed$(NC)"

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs: ## Generate API documentation
	@echo "$(BLUE)API documentation available at http://localhost:8000/docs when server is running$(NC)"

docs-serve: start-api ## Serve documentation (starts API for OpenAPI docs)

# ============================================================================
# DEVELOPMENT UTILITIES
# ============================================================================

shell: ## Open Python shell with project context
	$(BIN)/python -i -c "from rag import *; print('RAG Platform shell ready')"

repl: ## Open IPython REPL
	$(BIN)/ipython

logs: ## Tail API logs
	tail -f logs/api.log 2>/dev/null || echo "No log file found"

health: ## Check service health
	@echo "$(BLUE)Checking Qdrant...$(NC)"
	@curl -s http://localhost:6333/health && echo " $(GREEN)OK$(NC)" || echo " $(YELLOW)NOT RUNNING$(NC)"
	@echo "$(BLUE)Checking API...$(NC)"
	@curl -s http://localhost:8000/health && echo " $(GREEN)OK$(NC)" || echo " $(YELLOW)NOT RUNNING$(NC)"
