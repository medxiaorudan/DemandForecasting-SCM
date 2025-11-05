from src.forecasting.metrics import mape, wape

def test_metrics_basic():
    y_true = [10, 20, 30]
    y_pred = [10, 22, 27]
    assert 0 <= mape(y_true, y_pred) < 0.2
    assert 0 <= wape(y_true, y_pred) < 0.2
