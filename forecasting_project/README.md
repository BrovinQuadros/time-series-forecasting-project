# End-to-End Time Series Forecasting System

This project forecasts next **8 weeks (56 days)** of sales for each state and serves results via FastAPI.

## Structure

```
forecasting_project/
├── data/
├── models/
├── notebooks/
├── src/
├── api/
├── reports/
├── logs/
├── requirements.txt
├── README.md
└── app.py
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r forecasting_project/requirements.txt
cd forecasting_project
python -m src.train_pipeline
uvicorn app:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`

## Endpoints

- `GET /` health check
- `GET /forecast/{state}` returns 56-day forecast for a state
