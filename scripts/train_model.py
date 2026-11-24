"""Entrena y evalúa pronósticos de una semana con particiones estrictas por fecha."""
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from app.features import CATEGORICAL, NUMERIC, FEATURES, build_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "0.2.0"


def wape(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    denominator = np.abs(y).sum()
    return float(np.abs(y - pred).sum() / denominator) if denominator else float("nan")


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    positive = y > 0
    return {
        "r2": round(float(r2_score(y, pred)), 4),
        "wape_pct": round(100.0 * wape(y, pred), 2),
        "mape_positive_pct": round(float(100.0 * np.mean(np.abs(y[positive] - pred[positive]) / y[positive])), 2)
        if positive.any()
        else None,
        "rmse": round(float(root_mean_squared_error(y, pred)), 4),
        "mae": round(float(mean_absolute_error(y, pred)), 4),
    }


def main():
    raw = pd.read_csv(ROOT / "data" / "toyota_demand.csv")
    adi_series = raw.groupby("part_id")["demand"].agg(lambda x: len(x) / max(1, int((x > 0).sum())))
    adi_by_part = {k: round(float(v), 2) for k, v in adi_series.items()}

    df = build_features(raw)
    dates = np.sort(df.week_date.unique())
    cut = dates[int(len(dates) * 0.8)]
    train = df[df.week_date < cut].reset_index(drop=True)
    test = df[df.week_date >= cut].reset_index(drop=True)

    train_dates = np.sort(train.week_date.unique())
    cv = [
        (
            np.flatnonzero(train.week_date.isin(train_dates[a])),
            np.flatnonzero(train.week_date.isin(train_dates[b])),
        )
        for a, b in TimeSeriesSplit(n_splits=5).split(train_dates)
    ]

    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("rf", RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=1)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "rf__max_depth": [12, 18],
            "rf__min_samples_leaf": [1, 2],
        },
        cv=cv,
        scoring=make_scorer(wape, greater_is_better=False),
        n_jobs=2,
    )
    search.fit(train[FEATURES], train.demand)
    raw_pred = np.clip(search.predict(test[FEATURES]), 0.0, None)
    pred = np.where(raw_pred < 0.45, 0.0, raw_pred)

    rf_metrics = metrics(test.demand, pred)
    baseline_metrics = metrics(test.demand, test.rolling_mean_4)

    report = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic_calibrated_toyota_lima",
        "evaluation": "rolling one-week forecast, observed history through prior week",
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": str(pd.Timestamp(cut).date()),
        "intermittency_adi": {
            "min": round(float(adi_series.min()), 2),
            "mean": round(float(adi_series.mean()), 2),
            "max": round(float(adi_series.max()), 2),
            "parts_meeting_syntetos_boylan_1_32": int((adi_series >= 1.32).sum()),
            "total_parts": int(len(adi_series)),
        },
        "best_params": search.best_params_,
        "random_forest": rf_metrics,
        "baseline_rolling_mean_4": baseline_metrics,
        "targets_met": {
            "r2": rf_metrics["r2"] >= 0.80,
            "wape": rf_metrics["wape_pct"] <= 15.0,
            "mape_positive": rf_metrics["mape_positive_pct"] is not None and rf_metrics["mape_positive_pct"] <= 15.0,
        },
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": search.best_estimator_,
            "version": MODEL_VERSION,
            "known_parts": sorted(train.part_id.unique().tolist()),
            "adi_by_part": adi_by_part,
        },
        ROOT / "models" / "random_forest.joblib",
    )
    (ROOT / "models" / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
