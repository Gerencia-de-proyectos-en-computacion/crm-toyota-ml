"""Contrato de variables compartido entre entrenamiento e inferencia."""
import pandas as pd

CATEGORICAL = ["part_id", "vehicle_model", "part_category"]
NUMERIC = ["avg_mileage", "vehicles_served", "month", "week_of_year",
           "lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4"]
FEATURES = CATEGORICAL + NUMERIC


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["week_date"] = pd.to_datetime(df["week_date"])
    df = df.sort_values(["part_id", "week_date"]).reset_index(drop=True)
    groups = df.groupby("part_id", sort=False)
    # Las variables operativas describen la semana anterior, no la futura.
    for col in ["avg_mileage", "vehicles_served"]:
        df[col] = groups[col].shift(1)
    for lag in [1, 2, 4]:
        df[f"lag_{lag}"] = groups["demand"].shift(lag)
    df["rolling_mean_4"] = groups["demand"].transform(lambda x: x.shift(1).rolling(4).mean())
    df["rolling_std_4"] = groups["demand"].transform(lambda x: x.shift(1).rolling(4).std(ddof=0))
    df["month"] = df["week_date"].dt.month
    df["week_of_year"] = df["week_date"].dt.isocalendar().week.astype(int)
    return df.dropna(subset=FEATURES).sort_values(["week_date", "part_id"]).reset_index(drop=True)
