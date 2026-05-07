import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RAW_DATA_FILE, CLEAN_DATA_FILE, FEATURE_DATA_FILE, METRICS_FILE, FORECASTS_FILE, BEST_MODEL_FILE
from src.logger import get_logger
from src.data_processing import load_and_clean_data, expand_state_dates, add_features
from src.evaluation import calculate_metrics
from src.models import time_split, run_sarima, run_prophet, run_xgboost, run_lstm, forecast_next_56_with_best

logger = get_logger("train_pipeline")

FEATURE_COLS = ["lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_std_7", "day_of_week", "month", "week_of_year", "holiday_flag"]


def ensure_dirs():
    for p in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def run_training():
    ensure_dirs()
    logger.info("Loading and preprocessing data...")
    df = load_and_clean_data(str(RAW_DATA_FILE))
    df = expand_state_dates(df)
    df.to_csv(CLEAN_DATA_FILE, index=False)

    df_feat = add_features(df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    df_feat.to_csv(FEATURE_DATA_FILE, index=False)

    states = sorted(df_feat["State"].unique())
    metrics_rows, forecast_rows, best_registry = [], [], {}

    for state in states:
        logger.info(f"Training models for state: {state}")
        ds = df_feat[df_feat["State"] == state].sort_values("Date").reset_index(drop=True)
        if len(ds) < 150:
            logger.warning(f"Skipping state {state}: insufficient data")
            continue

        train, val = time_split(ds, val_days=56)
        y_val = val["Total"].values
        model_store, result_store = {}, {}

        for name, fn in [
            ("SARIMA", lambda: run_sarima(train, val)),
            ("Prophet", lambda: run_prophet(train, val)),
            ("XGBoost", lambda: run_xgboost(train, val, FEATURE_COLS)),
            ("LSTM", lambda: run_lstm(train, val, lookback=30)),
        ]:
            try:
                m, p = fn()
                result_store[name] = calculate_metrics(y_val, p)
                model_store[name] = m
            except Exception as e:
                logger.exception(f"{name} failed for {state}: {e}")

        if not result_store:
            continue

        best_model_name = min(result_store.keys(), key=lambda k: result_store[k]["RMSE"])
        best_model_obj = model_store[best_model_name]
        best_registry[state] = {"model_name": best_model_name, "model_object": best_model_obj, "feature_cols": FEATURE_COLS}

        for model_name, met in result_store.items():
            metrics_rows.append({"State": state, "Model": model_name, **met, "is_best": model_name == best_model_name})

        preds = forecast_next_56_with_best(best_model_name, best_model_obj, ds, FEATURE_COLS, horizon=56)
        future_dates = pd.date_range(ds["Date"].max() + pd.Timedelta(days=1), periods=56, freq="D")
        for d, yhat in zip(future_dates, preds):
            forecast_rows.append({"State": state, "Date": d, "Forecast": float(yhat), "BestModel": best_model_name})

        plt.figure(figsize=(10, 4))
        plt.plot(ds["Date"].tail(120), ds["Total"].tail(120), label="History")
        plt.plot(future_dates, preds, label="Forecast (8 weeks)")
        plt.title(f"{state} - Best: {best_model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"forecast_plot_{state.replace(' ', '_')}.png")
        plt.close()

    metrics_df = pd.DataFrame(metrics_rows)
    forecasts_df = pd.DataFrame(forecast_rows)
    metrics_df.to_csv(METRICS_FILE, index=False)
    forecasts_df.to_csv(FORECASTS_FILE, index=False)

    with open(BEST_MODEL_FILE, "wb") as f:
        pickle.dump(best_registry, f)

    joblib.dump({k: v["model_name"] for k, v in best_registry.items()}, MODELS_DIR / "best_model_index.joblib")
    logger.info("Training pipeline completed successfully")
    return metrics_df, forecasts_df


if __name__ == "__main__":
    run_training()
