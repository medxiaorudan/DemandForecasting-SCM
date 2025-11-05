import numpy as np
import pandas as pd
from typing import Iterable
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def _moving_average(series: Iterable[float], horizon: int, window: int = 7):
    s = np.asarray(series, dtype=float)
    if len(s) < window:
        window = max(1, len(s))
    ma = pd.Series(s).rolling(window=window).mean().iloc[-1]
    return np.repeat(ma, horizon)

def _naive(series: Iterable[float], horizon: int):
    s = np.asarray(series, dtype=float)
    last = s[-1] if len(s)>0 else 0.0
    return np.repeat(last, horizon)

def _exp_smoothing(series: Iterable[float], horizon: int):
    s = np.asarray(series, dtype=float)
    if len(s) < 10:
        return _moving_average(series, horizon)
    model = ExponentialSmoothing(s, trend=None, seasonal=None, damped_trend=False, initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(horizon)
    return np.clip(fc, a_min=0, a_max=None)

def forecast_series(series, horizon: int = 14, method: str = "exp_smoothing"):
    method = method.lower()
    if method == "moving_avg":
        return _moving_average(series, horizon).tolist()
    if method == "naive":
        return _naive(series, horizon).tolist()
    if method == "exp_smoothing":
        return _exp_smoothing(series, horizon).tolist()
    raise ValueError(f"Unknown method: {method}")
