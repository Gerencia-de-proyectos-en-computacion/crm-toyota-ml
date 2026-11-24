# `crm-toyota-ml` — Reglas Locales del Repositorio

1. **Stack:** Python 3.11 (`uv`) + FastAPI + scikit-learn (`RandomForestRegressor`) + pandas + joblib + Pytest.
2. **Inferencia Batch y Demanda Intermitente:** Ejecución por lote alimentando `demand_forecasts`; header `X-Internal-API-Key` y `GET /health`. Reportar siempre `R2 >= 0.80`, `WAPE <= 15%` (principal), `RMSE` y `MAPE` evaluado solo sobre semanas con `demanda > 0`.
3. **Docs-as-Code:** Mantener `docs/` sincronizado con esquemas Pydantic y métricas del modelo.
