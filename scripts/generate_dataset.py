"""Generador sintético calibrado de demanda intermitente Toyota (156 semanas x 20 repuestos).

Modela ciclos reales de mostrador y flotas en talleres especializados Toyota:
- Demanda intermitente según Syntetos & Boylan (ADI >= 1.32 en los 20 SKUs).
- Picos gobernados por flujo vehicular previo (vehicles_served t-1), desgaste acumulado
  (avg_mileage t-1), estacionalidad mensual (verano Ene-Feb, Fiestas Patrias Jul, cierre Dic)
  y ciclos refractarios de reposición tras semanas consecutivas de consumo (lag_1, lag_2).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _seasonal_multiplier(month: int) -> float:
    if month in (1, 2, 7, 12):
        return 1.26
    if month in (4, 5, 9):
        return 0.88
    return 1.02


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = [
        ("Toyota Yaris", 28, 52000, 1.15),
        ("Toyota Corolla", 24, 62000, 1.05),
        ("Toyota Hilux", 32, 88000, 1.32),
        ("Toyota RAV4", 21, 58000, 1.00),
        ("Toyota Avanza", 23, 74000, 1.08),
    ]
    categories = [
        ("Filtro de aceite", 12.0, 1.00, 16),
        ("Pastillas de freno", 8.5, 0.92, 18),
        ("Bujias", 9.5, 0.95, 19),
        ("Amortiguador", 6.2, 0.86, 20),
    ]
    dates = pd.date_range("2023-11-27", periods=156, freq="W-MON")
    rows = []

    for i, (vehicle, base_visits, base_km, fleet_factor) in enumerate(models):
        traffic = []
        visits = float(base_visits)
        mileage = float(base_km)
        for date in dates:
            season = _seasonal_multiplier(int(date.month))
            visits = max(12, round(0.50 * visits + 0.50 * (base_visits * season) + rng.normal(0, 2.2)))
            mileage = max(
                18000.0,
                round(0.75 * mileage + 0.25 * (base_km * (0.95 + 0.10 * season)) + rng.normal(0, 900), 1),
            )
            traffic.append((int(visits), float(mileage)))

        for j, (category, base_units, cat_factor, min_visits_gate) in enumerate(categories):
            part_idx = i * len(categories) + j + 1
            part_id = f"REP-{part_idx:03d}"
            lag1_demand = 0
            lag2_demand = 0

            for t, date in enumerate(dates):
                curr_visits, curr_mileage = traffic[t]
                prev_visits, prev_mileage = traffic[t - 1] if t > 0 else traffic[0]
                month = int(date.month)
                season = _seasonal_multiplier(month)

                # Intermitencia física sin fuga temporal (solo depende de t-1 y t-2):
                # 1) Ciclo refractario tras dos semanas seguidas de despacho de lote (lag_1 > 0 y lag_2 > 0)
                # 2) Semana de baja afluencia previa en recepción (prev_visits <= min_visits_gate)
                if (lag1_demand > 0 and lag2_demand > 0) or (prev_visits <= min_visits_gate):
                    demand = 0
                else:
                    pent_up_boost = 1.24 if lag1_demand == 0 else 0.95
                    visit_ratio = prev_visits / float(base_visits)
                    km_ratio = 0.85 + 0.15 * (prev_mileage / float(base_km))
                    expected = base_units * fleet_factor * cat_factor * visit_ratio * km_ratio * season * pent_up_boost
                    noise = rng.normal(0.0, 0.32)
                    demand = max(1, int(round(expected + noise)))

                rows.append((date, part_id, vehicle, category, curr_mileage, curr_visits, demand))
                lag2_demand = lag1_demand
                lag1_demand = demand

    return pd.DataFrame(
        rows,
        columns=["week_date", "part_id", "vehicle_model", "part_category", "avg_mileage", "vehicles_served", "demand"],
    )


if __name__ == "__main__":
    df = generate()
    path = ROOT / "data" / "toyota_demand.csv"
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    adi = df.groupby("part_id")["demand"].agg(lambda x: len(x) / max(1, (x > 0).sum()))
    print(f"Dataset sintetico calibrado: {len(df)} filas, {df.part_id.nunique()} repuestos")
    print(f"ADI >= 1.32: {(adi >= 1.32).sum()}/{len(adi)}; rango {adi.min():.2f} - {adi.max():.2f}")
    print(path)
