import pandas as pd


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def expand_state_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for state, g in df.groupby("State"):
        g = g.sort_values("Date")
        full_dates = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        expanded = pd.DataFrame({"Date": full_dates})
        expanded = expanded.merge(g[["Date", "Total", "Category"]], on="Date", how="left")
        expanded["State"] = state
        expanded["Category"] = expanded["Category"].ffill().bfill()
        expanded["Total"] = expanded["Total"].interpolate(method="linear").ffill().bfill()
        out.append(expanded)
    return pd.concat(out, ignore_index=True).sort_values(["State", "Date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 7, 30]:
        df[f"lag_{lag}"] = df.groupby("State")["Total"].shift(lag)

    df["rolling_mean_7"] = df.groupby("State")["Total"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_std_7"] = df.groupby("State")["Total"].transform(lambda x: x.shift(1).rolling(7).std())

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["holiday_flag"] = (df["day_of_week"] >= 5).astype(int)
    return df
