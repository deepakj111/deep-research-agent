.PHONY: run dev test benchmark lint build clean security audit docker-test

run:
	docker compose up --build

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

test:
	uv run pytest tests/ -v --cov=agent --cov=mcp_servers --cov-report=term || [ "$$?" = "5" ]

benchmark:
	uv run python evaluation/run_benchmark.py --profile fast --queries 3 --fail-below 0.75

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

docker-test:
	docker compose build
	docker compose up -d redis web-search-mcp arxiv-mcp github-mcp
	@echo "Waiting for MCP services..."
	@timeout 60 bash -c 'until curl -sf http://localhost:8001/health; do sleep 2; done'
	@timeout 60 bash -c 'until curl -sf http://localhost:8002/health; do sleep 2; done'
	@timeout 60 bash -c 'until curl -sf http://localhost:8003/health; do sleep 2; done'
	@echo "All MCP services healthy ✓"
	docker compose down -v

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
