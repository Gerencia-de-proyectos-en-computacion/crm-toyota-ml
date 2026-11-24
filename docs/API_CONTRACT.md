# Contrato Oficial de Integración REST (`ml-service` v0.2.0)

## 1. Endpoints Expuestos (Puerto local `8000` / Docker `7860`)
- `GET /health` *(Público)* — Healthcheck de contenedor (`status`, `model_version`, `data_source`).
- `GET /metrics` *(Protegido por `X-Internal-API-Key` cuando `ML_INTERNAL_API_KEY` está definido)* — Retorna el reporte completo de evaluación temporal (`models/metrics.json`): `r2`, `wape_pct`, `mape_positive_pct`, `rmse`, `mae`, `intermittency_adi` y `targets_met`.
- `POST /predict` — Predicción individual semanal (`horizon_days = 7`) incluyendo `reorder_point`.
- `POST /predict/batch` — Predicción por lote de hasta 1,000 repuestos consumida por `RoutingMlForecastClient` de Spring Boot.

## 2. Homologación Transparente de SKUs
`app/main.py` soporta tanto los identificadores sintéticos de entrenamiento (`REP-001`..`REP-020`) como los SKUs reales del catálogo `backend-crm` (`TOY-FIL-041`, `AIS-PAD-0D1`, `DEN-AIR-178`, `TOY-FIL-0L0`, etc.) mediante `CRM_SKU_HOMOLOGATION`, rechazando con `HTTP 422` cualquier identificador desconocido.
