import yaml
import numpy as np
from src.forecasting.data import load_sales
from src.forecasting.model import forecast_series
from src.forecasting.metrics import wape, mape

def _train_test_split(series, test_horizon):
    # Simple holdout: last H days as test
    return series[:-test_horizon], series[-test_horizon:]

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    item_id = cfg["eval"]["item_id"]
    horizon = int(cfg["eval"]["horizon"])
    method = cfg["eval"]["model"]
    max_wape = float(cfg["eval"]["max_wape"])

    df = load_sales("data/sample_sales.csv")
    s = df[df["item_id"]==item_id].set_index("date")["qty"].astype(float).values

    train, test = _train_test_split(s, horizon)
    fc = forecast_series(train, horizon=horizon, method=method)

    w = wape(test, fc)
    m = mape(test, fc)
    print(f"WAPE: {w:.4f}  MAPE: {m:.4f}  (threshold WAPE <= {max_wape})")

    if w > max_wape:
        print("::error ::WAPE exceeds threshold — failing CI gate.")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
