# `crm-toyota-ml` — Microservicio Predictivo Random Forest (v0.2.0)

Microservicio REST en **Python 3.11 + FastAPI + scikit-learn (`RandomForestRegressor`)** para pronóstico semanal (`horizon_days = 7`) de demanda intermitente de repuestos Toyota (`ADI >= 1.32`).

## Ejecución Rápida en Windows / PowerShell (`uv` + Python 3.11)

```powershell
cd crm-toyota\ml-service
uv venv --python 3.11 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.generate_dataset
.\.venv\Scripts\python.exe -m scripts.train_model
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

## Endpoints Disponibles
- `GET /health` — Estado del servicio y versión del modelo `.joblib` cargado en memoria.
- `GET /metrics` — Reporte JSON auditado (`R²`, `WAPE`, `MAPE>0`, `RMSE`, `ADI` y comparación contra baseline de media móvil).
- `POST /predict` — Inferencia individual con `reorder_point` calculado (nivel de servicio 95%, lead time Lima 5 días).
- `POST /predict/batch` — Inferencia por lote (hasta 1,000 ítems) integrada con `backend-crm`, con homologación transparente entre `REP-001..REP-020` y SKUs reales `TOY-*` / `DEN-*` / `AIS-*`.
- `GET /docs` — Swagger UI interactivo.

## Documentación Viva (Docs-as-Code)
- [Métricas y Calibración de Demanda Intermitente](./docs/METRICS_INTERMITTENT_DEMAND.md)
- [Contrato REST y Homologación con Spring Boot](./docs/API_CONTRACT.md)
