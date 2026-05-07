from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA_FILE = BASE_DIR.parent / "Forecasting Case- Study.xlsx"
CLEAN_DATA_FILE = DATA_DIR / "cleaned_data.csv"
FEATURE_DATA_FILE = DATA_DIR / "featured_data.csv"
METRICS_FILE = REPORTS_DIR / "model_performance.csv"
FORECASTS_FILE = REPORTS_DIR / "state_forecasts_8_weeks.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model_registry.pkl"
