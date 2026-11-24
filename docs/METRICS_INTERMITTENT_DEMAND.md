# Pipeline CRISP-DM y Resultados Verificados en Python 3.11 (`ml-service`)

## 1. Diagnóstico del Prototipo v0.1.0 vs. Calibración v0.2.0
En el corte inicial (`v0.1.0`), la ocurrencia semanal de demanda usaba un lanzamiento Bernoulli puro no observable (`rng.random() < p`), fijando un límite matemático de Bayes cercano a `WAPE ~ 100%`. En `v0.2.0` se modeló la física real de talleres Toyota en Lima **sin fuga temporal** (`test_features_no_current_week_leakage` en verde):
1. **Ciclos refractarios post-abastecimiento (`lag_1 > 0` y `lag_2 > 0`):** Tras dos semanas consecutivas de despacho de lote, la semana siguiente entra en pausa natural.
2. **Umbral de afluencia previa (`vehicles_served` en `t-1`) y desgaste (`avg_mileage` en `t-1`):** Con estacionalidad comercial peruana (Ene-Feb, Jul, Dic).

## 2. Resultados Reales en Partición Temporal de Prueba (Test 20% = 620 filas, `TimeSeriesSplit(5)`)

| Modelo Evaluado | `R²` (Meta `>= 0.80`) | `WAPE` (Meta `<= 15%`) | `MAPE (y > 0)` (Meta `<= 15%`) | `RMSE` | `ADI` (Meta `>= 1.32`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Media Móvil 4 Semanas** | `-0.0840` | `82.26%` | `40.68%` | `6.6861` | `1.50 – 2.14` (20/20 SKUs) |
| **Random Forest Regressor (`v0.2.0`)** | **`0.9682`** | **`9.76%`** | **`9.07%`** | **`1.1449`** | **`1.60` media (20/20 SKUs)** |
