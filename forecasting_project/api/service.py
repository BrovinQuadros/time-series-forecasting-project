import pandas as pd
from pathlib import Path
from fastapi import HTTPException

try:
    from src.config import FORECASTS_FILE
except ModuleNotFoundError:
    from forecasting_project.src.config import FORECASTS_FILE


def load_forecasts():
    if not Path(FORECASTS_FILE).exists():
        raise FileNotFoundError("Forecast file not found. Run training pipeline first.")
    df = pd.read_csv(FORECASTS_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_state_forecast(state: str):
    df = load_forecasts()
    ds = df[df["State"].str.lower() == state.lower()].sort_values("Date")
    if ds.empty:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found in forecast output")
    points = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(y), "best_model": m} for d, y, m in zip(ds["Date"], ds["Forecast"], ds["BestModel"])]
    return {"state": state, "horizon_days": len(points), "points": points}
