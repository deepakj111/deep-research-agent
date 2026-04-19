.PHONY: run dev test benchmark lint build clean

run:
	docker compose up --build

dev:
	docker compose -f docker-compose.dev.yml up --build

test:
	pytest tests/ -v --cov=agent --cov=mcp_servers

benchmark:
	python evaluation/run_benchmark.py --profile deep --fail-below 0.75

lint:
	ruff check .
	mypy agent/ mcp_servers/

build:
	docker compose build

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
