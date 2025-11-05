import pandas as pd

def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["item_id","date"]).reset_index(drop=True)
    return df
