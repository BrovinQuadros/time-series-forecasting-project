from pydantic import BaseModel
from typing import List


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    best_model: str


class ForecastResponse(BaseModel):
    state: str
    horizon_days: int
    points: List[ForecastPoint]
