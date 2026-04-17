.DEFAULT_GOAL := help

PACKAGES := eager-tools-core eager-tools-anthropic eager-tools-openai
CORE     := packages/eager-tools-core
ANTHRO   := packages/eager-tools-anthropic
OPENAI   := packages/eager-tools-openai

# Auto-load .env if present (export every var while sourcing).
ENV_LOAD := set -a; [ -f .env ] && . ./.env; set +a;

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: sync
sync: ## uv sync all packages with dev extras
	@for pkg in $(PACKAGES); do \
		echo "→ syncing $$pkg"; \
		(cd packages/$$pkg && uv sync --extra dev) || exit 1; \
	done

.PHONY: install
install: sync ## Alias for sync

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test: test-core test-anthropic test-openai ## Run all package test suites

.PHONY: test-core
test-core: ## Run eager-tools-core tests
	cd $(CORE) && uv run pytest -xvs

.PHONY: test-anthropic
test-anthropic: ## Run eager-tools-anthropic tests
	cd $(ANTHRO) && uv run pytest -xvs

.PHONY: test-openai
test-openai: ## Run eager-tools-openai tests
	cd $(OPENAI) && uv run pytest -xvs

# ---------------------------------------------------------------------------
# Lint / format / typecheck
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Ruff check all packages
	@for pkg in $(PACKAGES); do \
		echo "→ ruff check $$pkg"; \
		(cd packages/$$pkg && uv run ruff check .) || exit 1; \
	done

.PHONY: fmt
fmt: ## Ruff format (write) all packages
	@for pkg in $(PACKAGES); do \
		echo "→ ruff format $$pkg"; \
		(cd packages/$$pkg && uv run ruff format .) || exit 1; \
	done

.PHONY: fmt-check
fmt-check: ## Ruff format --check (no write)
	@for pkg in $(PACKAGES); do \
		echo "→ ruff format --check $$pkg"; \
		(cd packages/$$pkg && uv run ruff format --check .) || exit 1; \
	done

.PHONY: typecheck
typecheck: ## Pyright across the workspace
	cd $(CORE) && uv run pyright

.PHONY: check
check: lint fmt-check typecheck test ## Lint + format-check + typecheck + tests

# ---------------------------------------------------------------------------
# Examples (auto-loads .env)
# ---------------------------------------------------------------------------

EX_RUN_OPENAI = uv run --project $(OPENAI) --with-editable $(CORE) python
EX_RUN_ANTHRO = uv run --project $(ANTHRO) --with-editable $(CORE) python

.PHONY: example-1
example-1: ## examples/01_minimal — no key needed
	$(EX_RUN_ANTHRO) examples/01_minimal.py

.PHONY: example-2
example-2: ## examples/02_anthropic_live — needs ANTHROPIC_API_KEY
	$(ENV_LOAD) $(EX_RUN_ANTHRO) examples/02_anthropic_live.py

.PHONY: example-3
example-3: ## examples/03_openai_live — needs OPENAI_API_KEY
	$(ENV_LOAD) $(EX_RUN_OPENAI) examples/03_openai_live.py

.PHONY: example-4
example-4: ## examples/04_cancellation — no key needed
	$(EX_RUN_ANTHRO) examples/04_cancellation.py

.PHONY: example-5
example-5: ## examples/05_openrouter_live — needs OPENROUTER_API_KEY
	$(ENV_LOAD) $(EX_RUN_OPENAI) examples/05_openrouter_live.py

.PHONY: examples
examples: example-1 example-4 ## Run offline examples (1, 4)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps .venv)
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .pyrefly_cache -o -name .mypy_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +
	find . -type d \( -name build -o -name dist \) -not -path "*/.venv/*" -prune -exec rm -rf {} +

.PHONY: clean-venv
clean-venv: ## Remove all .venv directories (forces re-sync)
	find packages -type d -name .venv -prune -exec rm -rf {} +
