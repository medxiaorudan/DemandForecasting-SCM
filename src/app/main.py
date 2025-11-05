from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time

from src.forecasting.data import load_sales
from src.forecasting.model import forecast_series
from src.forecasting.metrics import mape, wape
from src.optimization.inventory_optimization import plan_reorder
from src.forecasting.rag_assistant import RAGAssistant

app = FastAPI(title="DemandForecasting-SCM", version="0.1.0")

rag = RAGAssistant(docs_path="docs")

class PlanRequest(BaseModel):
    item_id: str
    horizon: int
    initial_inventory: int
    demand_forecast: List[float]
    holding_cost: float = 0.1
    stockout_cost: float = 1.0
    service_level: float = 0.95
    max_order: Optional[int] = None
    lead_time: int = 0

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/forecast")
def get_forecast(
    item_id: str = Query(..., description="SKU ID"),
    horizon: int = Query(14, ge=1, le=90),
    model: str = Query("exp_smoothing", description="moving_avg|exp_smoothing|naive")
) -> Dict[str, Any]:
    t0 = time.time()
    df = load_sales("data/sample_sales.csv")
    series = df[df["item_id"]==item_id].set_index("date")["qty"]
    fc = forecast_series(series, horizon=horizon, method=model)
    elapsed_ms = (time.time()-t0)*1000.0
    return {
        "item_id": item_id,
        "model": model,
        "horizon": horizon,
        "forecast": list(map(float, fc)),
        "latency_ms": round(elapsed_ms,2)
    }

@app.post("/plan")
def post_plan(req: PlanRequest) -> Dict[str, Any]:
    plan = plan_reorder(
        demand=req.demand_forecast,
        initial_inventory=req.initial_inventory,
        holding_cost=req.holding_cost,
        stockout_cost=req.stockout_cost,
        service_level=req.service_level,
        max_order=req.max_order,
        lead_time=req.lead_time
    )
    return {"item_id": req.item_id, **plan}

@app.post("/ask")
def ask_docs(req: AskRequest) -> Dict[str, Any]:
    results = rag.search(req.question, top_k=req.top_k)
    return {"query": req.question, "results": results}
