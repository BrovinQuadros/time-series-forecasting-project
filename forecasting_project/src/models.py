import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
except Exception:
    Prophet = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    Sequential = None
    LSTM = None
    Dense = None
    Input = None
    EarlyStopping = None


def time_split(df_state: pd.DataFrame, val_days: int = 56):
    df_state = df_state.sort_values("Date").reset_index(drop=True)
    train = df_state.iloc[:-val_days].copy()
    val = df_state.iloc[-val_days:].copy()
    return train, val


def run_sarima(train: pd.DataFrame, val: pd.DataFrame):
    model = SARIMAX(train["Total"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    preds = fit.forecast(steps=len(val))
    return fit, np.array(preds)


def run_prophet(train: pd.DataFrame, val: pd.DataFrame):
    if Prophet is None:
        raise ImportError("prophet is not installed")
    tr = train[["Date", "Total"]].rename(columns={"Date": "ds", "Total": "y"})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(tr)
    future = pd.DataFrame({"ds": val["Date"].values})
    fc = m.predict(future)
    return m, fc["yhat"].values


def run_xgboost(train: pd.DataFrame, val: pd.DataFrame, feature_cols):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    x_train = train[feature_cols]
    y_train = train["Total"]
    x_val = val[feature_cols]
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(x_train, y_train)
    return model, model.predict(x_val)


def _to_sequences(series, lookback=30):
    X, y = [], []
    vals = series.values
    for i in range(lookback, len(vals)):
        X.append(vals[i - lookback:i])
        y.append(vals[i])
    X, y = np.array(X), np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y


def run_lstm(train: pd.DataFrame, val: pd.DataFrame, lookback=30):
    if Sequential is None:
        raise ImportError("tensorflow is not installed")
    train_X, train_y = _to_sequences(train["Total"], lookback=lookback)
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(train_X, train_y, epochs=10, batch_size=32, verbose=0, callbacks=[es])

    history = train["Total"].tolist()
    preds = []
    for _ in range(len(val)):
        x = np.array(history[-lookback:]).reshape(1, lookback, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(float(p))
        history.append(float(p))
    return model, np.array(preds)


def forecast_next_56_with_best(model_name, model_obj, df_state, feature_cols, horizon=56):
    df_state = df_state.sort_values("Date").copy()

    if model_name == "SARIMA":
        return np.array(model_obj.forecast(steps=horizon))

    if model_name == "Prophet":
        start = df_state["Date"].max() + pd.Timedelta(days=1)
        fut = pd.DataFrame({"ds": pd.date_range(start, periods=horizon, freq="D")})
        return model_obj.predict(fut)["yhat"].values

    if model_name == "XGBoost":
        temp = df_state.copy()
        out = []
        for _ in range(horizon):
            next_date = temp["Date"].max() + pd.Timedelta(days=1)
            row = {"Date": next_date, "State": temp["State"].iloc[-1], "Category": temp["Category"].iloc[-1]}
            row["lag_1"] = temp["Total"].iloc[-1]
            row["lag_7"] = temp["Total"].iloc[-7] if len(temp) >= 7 else temp["Total"].iloc[-1]
            row["lag_30"] = temp["Total"].iloc[-30] if len(temp) >= 30 else temp["Total"].iloc[-1]
            row["rolling_mean_7"] = temp["Total"].iloc[-7:].mean()
            row["rolling_std_7"] = temp["Total"].iloc[-7:].std() if len(temp) >= 2 else 0.0
            row["day_of_week"] = next_date.dayofweek
            row["month"] = next_date.month
            row["week_of_year"] = int(next_date.isocalendar().week)
            row["holiday_flag"] = 1 if next_date.dayofweek >= 5 else 0
            x = pd.DataFrame([row])[feature_cols]
            p = float(model_obj.predict(x)[0])
            row["Total"] = p
            temp = pd.concat([temp, pd.DataFrame([row])], ignore_index=True)
            out.append(p)
        return np.array(out)

    if model_name == "LSTM":
        lookback = 30
        history = df_state["Total"].tolist()
        out = []
        for _ in range(horizon):
            x = np.array(history[-lookback:]).reshape(1, lookback, 1)
            p = float(model_obj.predict(x, verbose=0)[0, 0])
            history.append(p)
            out.append(p)
        return np.array(out)

    raise ValueError(f"Unsupported model_name: {model_name}")
