import pandas as pd
import numpy as np
from pathlib import Path

def main():
    rng = pd.date_range("2024-01-01", periods=200, freq="D")
    skus = ["SKU1","SKU2","SKU3"]
    rows = []
    np.random.seed(7)
    for i, sku in enumerate(skus):
        level = 20 + 10*i
        season = np.sin(np.arange(len(rng))/7*2*np.pi)*3
        noise = np.random.normal(0,2,len(rng))
        base = level + season + noise
        demand = np.maximum(np.round(base).astype(int), 0)
        rows += [{"date": d.strftime("%Y-%m-%d"), "item_id": sku, "qty": int(q)} for d,q in zip(rng, demand)]
    df = pd.DataFrame(rows)
    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/sample_sales.csv", index=False)
    print("Wrote data/sample_sales.csv")

if __name__ == "__main__":
    main()
