# DemandForecasting-SCM

Time-series **demand forecasting** + a tiny **RAG docs assistant for planners** + **inventory optimization** (OR-Tools).
Exposes a **FastAPI** service with a **p95 latency target mindset** and a **CI evaluation gate** (WAPE threshold).

## Features
- **Forecasting**: simple baselines (moving average, exponential smoothing) with `pandas`/`statsmodels`.
- **Metrics**: WAPE / MAPE.
- **RAG (docs assistant)**: TF‑IDF retrieval over local `docs/`, returns top matches with snippets (no external calls).
- **Optimization**: single-item inventory **reorder planning** using **OR-Tools** with service-level and capacity constraints.
- **API**: `/forecast`, `/plan`, `/ask` via FastAPI.
- **CI gate**: fails if offline eval WAPE exceeds threshold (see `configs/config.yaml`).

## Quickstart

```bash
# 1) Python 3.11+ recommended
python -m venv .venv && source .venv/bin/activate

# 2) Install
pip install -r requirements.txt

# 3) (Optional) regenerate synthetic sample data
python scripts/seed_data.py

# 4) Run tests + eval gate
pytest -q
python -m src.evals.run_forecast_eval

# 5) Launch API
uvicorn src.app.main:app --reload --port 8000
# Open http://127.0.0.1:8000/docs
```

## Endpoints
- `GET /forecast?item_id=SKU1&horizon=14&model=exp_smoothing` → returns forecast for next `horizon` days.
- `POST /plan` (JSON body) → returns reorder plan with costs and service-level satisfaction.
- `POST /ask` (JSON body) → returns top relevant docs/snippets from `docs/`.

### Example requests
```bash
curl "http://127.0.0.1:8000/forecast?item_id=SKU1&horizon=14&model=exp_smoothing"

curl -X POST "http://127.0.0.1:8000/plan" -H "Content-Type: application/json" -d '{
  "item_id":"SKU1",
  "horizon":14,
  "initial_inventory":120,
  "demand_forecast":[20,22,19,23,25,18,20,22,19,23,25,18,21,22],
  "holding_cost":0.2,
  "stockout_cost":1.0,
  "service_level":0.95,
  "max_order":80,
  "lead_time":2
}'

curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{
  "question":"How do I interpret WAPE and MAPE for weekly forecasts?",
  "top_k": 3
}'
```

## CI Evaluation Gate
- Offline evaluation reads `data/sample_sales.csv` and computes **WAPE/MAPE** for a simple baseline.
- The gate **exits non-zero** if WAPE exceeds the `max_wape` set in `configs/config.yaml`.
- Wire this as a required GitHub check (see `.github/workflows/ci.yml`).

## Repo Layout
```
src/
  app/                      # FastAPI app
  forecasting/              # models, metrics, data loader
  optimization/             # OR-Tools reorder planning
  evals/                    # offline eval & CI gate
tests/                      # pytest unit tests & smoke tests
data/                       # synthetic sample sales
docs/                       # planner docs for RAG
configs/                    # thresholds
.github/workflows/ci.yml    # tests + eval gate
Dockerfile
requirements.txt
Makefile
```
