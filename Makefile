.PHONY: run dev test benchmark lint build clean security audit

run:
	docker compose up --build

dev:
	docker compose -f docker-compose.dev.yml up --build

test:
	uv run pytest tests/ -v --cov=agent --cov=mcp_servers --cov-report=term || [ "$$?" = "5" ]

benchmark:
	uv run python evaluation/run_benchmark.py --profile deep --fail-below 0.75

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy agent/ mcp_servers/ api/ config/ utils/ --ignore-missing-imports

security:
	uv run uv-secure uv.lock
	uv export --format requirements-txt --no-emit-project > /tmp/req-audit.txt && uv run pip-audit -r /tmp/req-audit.txt
	uv run bandit -r agent/ mcp_servers/ api/ -ll -ii

build:
	docker compose build

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
