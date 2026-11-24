from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from app.features import FEATURES
from app.schemas import PredictionRequest, PredictionResponse, BatchRequest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Tabla de homologación oficial entre SKUs reales del catálogo CRM-TOYOTA y los arquetipos REP entrenados
CRM_SKU_HOMOLOGATION: dict[str, str] = {
    "TOY-FIL-041": "REP-001",
    "DEN-FIL-041": "REP-005",
    "DEN-AIR-178": "REP-001",
    "TOY-AIR-178": "REP-017",
    "TOY-OIL-530": "REP-005",
    "AIS-PAD-0D1": "REP-002",
    "TOY-PAD-523": "REP-002",
    "DEN-PAD-523": "REP-006",
    "DEN-CAB-871": "REP-013",
    "TOY-BUJ-909": "REP-007",
    "DEN-BUJ-909": "REP-015",
    "TOY-FIL-0L0": "REP-009",
    "DEN-FIL-0L0": "REP-009",
    "AIS-CLU-312": "REP-012",
    "TOY-OIL-154": "REP-009",
    "TOY-LIQ-DOT": "REP-002",
}


def verify_internal_api_key(x_internal_api_key: str | None = Header(default=None)):
    expected_key = os.getenv("ML_INTERNAL_API_KEY", "").strip()
    if expected_key and x_internal_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Modelo no encontrado. Ejecuta python -m scripts.train_model")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metrics = (
        json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    )
    yield
    app.state.model = None
    app.state.metrics = None


app = FastAPI(title="CRM Toyota - ML Service", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": app.state.model["version"],
        "data_source": "synthetic",
    }


@app.get("/metrics", dependencies=[Depends(verify_internal_api_key)])
def get_metrics():
    if not app.state.metrics:
        raise HTTPException(status_code=404, detail="Métricas de entrenamiento no disponibles")
    return app.state.metrics


def _compute_reorder_point(predicted_weekly: float, rolling_std_4: float) -> int:
    # ROP con lead time proveedor Lima (~5 días = 0.71 sem) + stock de seguridad Z=1.65 (95% nivel de servicio)
    lead_time_weeks = 5.0 / 7.0
    safety_stock = 1.65 * rolling_std_4 * math.sqrt(lead_time_weeks)
    rop = math.ceil(predicted_weekly * lead_time_weeks + safety_stock)
    return max(2, rop)


def predict_items(items: list[PredictionRequest]) -> list[PredictionResponse]:
    bundle = app.state.model
    known_parts = set(bundle["known_parts"])
    normalized_rows = []
    unknown = []

    for item in items:
        mapped_id = CRM_SKU_HOMOLOGATION.get(item.part_id.upper(), item.part_id)
        if mapped_id not in known_parts:
            unknown.append(item.part_id)
        else:
            row = item.model_dump()
            row["part_id"] = mapped_id
            normalized_rows.append(row)

    if unknown:
        raise HTTPException(status_code=422, detail={"unknown_parts": sorted(set(unknown))})

    frame = pd.DataFrame(normalized_rows, columns=FEATURES)
    raw_values = bundle["pipeline"].predict(frame)

    responses: list[PredictionResponse] = []
    for item, raw_val in zip(items, raw_values):
        clean_val = 0.0 if float(raw_val) < 0.45 else round(max(0.0, float(raw_val)), 2)
        rop = _compute_reorder_point(clean_val, item.rolling_std_4)
        responses.append(
            PredictionResponse(
                part_id=item.part_id,
                predicted_demand=clean_val,
                reorder_point=rop,
                horizon_days=7,
                model_version=bundle["version"],
            )
        )
    return responses


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_internal_api_key)])
def predict(item: PredictionRequest):
    return predict_items([item])[0]


@app.post("/predict/batch", response_model=list[PredictionResponse], dependencies=[Depends(verify_internal_api_key)])
def predict_batch(batch: BatchRequest):
    return predict_items(batch.items)
