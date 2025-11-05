from src.forecasting.data import load_sales
from src.forecasting.model import forecast_series
from src.optimization.inventory_optimization import plan_reorder

def test_forecast_smoke():
    s = load_sales("data/sample_sales.csv")
    series = s[s["item_id"]=="SKU1"]["qty"].values
    fc = forecast_series(series, horizon=7, method="exp_smoothing")
    assert len(fc) == 7

def test_opt_smoke():
    res = plan_reorder(
        demand=[20]*7,
        initial_inventory=100,
        holding_cost=0.1,
        stockout_cost=1.0,
        service_level=0.9
    )
    assert len(res["order_plan"]) == 7
