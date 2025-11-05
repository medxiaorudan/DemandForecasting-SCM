.PHONY: format test run eval

format:
	python -m pip install ruff black
	ruff check --fix . || true
	black .

test:
	pytest -q

eval:
	python -m src.evals.run_forecast_eval

run:
	uvicorn src.app.main:app --reload --port 8000
