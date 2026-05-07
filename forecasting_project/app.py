from fastapi import FastAPI
from api.service import get_state_forecast
from api.schemas import ForecastResponse

app = FastAPI(title="State Sales Forecasting API", version="1.0.0")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Forecasting API is running"}


@app.get("/forecast/{state}", response_model=ForecastResponse)
def forecast_state(state: str):
    return get_state_forecast(state)
