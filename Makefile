.PHONY: help install test lint typecheck format clean run

# =============================================================================
# ScholAI development tasks
# =============================================================================

help:
	@echo "ScholAI development tasks:"
	@echo ""
	@echo "  make install       Install dependencies (including dev dependencies)"
	@echo "  make test          Run unit tests (skips heavy integration tests)"
	@echo "  make test-all     Run all tests including heavy integration tests"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run ruff formatter + import sorter"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make clean         Remove build artifacts and caches"
	@echo "  make run TEXT=...  Run the CLI with TEXT as input"

install:
	pip install -e ".[dev]"

test:
	# Fast unit tests only — skips tests that require BGE-M3 or DeepSeek API
	python -m pytest tests/ \
		--ignore=tests/test_pipeline.py \
		--ignore=tests/test_coherence_fixer.py \
		--ignore=tests/test_rewriter.py \
		--ignore=tests/test_semantic_validator.py \
		--ignore=tests/test_quality_controller.py \
		--ignore=tests/test_semantic_anchor.py \
		-q

test-all:
	# All tests — requires DEEPSEEK_API_KEY and sentence-transformers
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

typecheck:
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run:
	@if [ -z "$(TEXT)" ]; then \
		echo "Usage: make run TEXT='your AI text here'"; \
	else \
		python main.py "$(TEXT)"; \
	fi
