.PHONY: up down migrate api worker reaper test unit integration chaos demo demo-idempotency eval load-test lint fmt

up:            ## start postgres + redis
	docker compose up -d

down:
	docker compose down

migrate:
	alembic upgrade head

api:
	python -m app.api --reload

worker:
	python -m app.worker

reaper:
	python -m app.reaper

unit:
	pytest tests/unit -q

integration:
	pytest -m integration -q

chaos:
	pytest -m chaos -q

test:
	pytest -q

demo:          ## kill a real worker mid-step and watch the system recover
	python scripts/demo_crash_recovery.py

demo-idempotency:  ## prove a side effect happens once despite a mid-flight crash
	python scripts/demo_idempotency.py

eval:          ## run the 17-task evaluation suite; writes docs/EVAL_REPORT.md
	python -m eval

load-test:     ## concurrent load over HTTP; needs `make api` + workers running first
	python scripts/load_test.py --runs 200 --concurrency 40

lint:
	ruff check app tests scripts eval
	mypy app eval

fmt:
	ruff format app tests scripts eval
	ruff check --fix app tests scripts eval
