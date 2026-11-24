import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.features import build_features
from scripts.generate_dataset import generate

SAMPLE = {
    "part_id": "REP-001",
    "vehicle_model": "Toyota Yaris",
    "part_category": "Filtro de aceite",
    "avg_mileage": 65000,
    "vehicles_served": 32,
    "month": 11,
    "week_of_year": 48,
    "lag_1": 8,
    "lag_2": 10,
    "lag_4": 7,
    "rolling_mean_4": 8.5,
    "rolling_std_4": 1.11803398875,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_metrics_endpoint_meets_pmbok_targets(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["targets_met"]["r2"] is True
    assert body["targets_met"]["wape"] is True
    assert body["targets_met"]["mape_positive"] is True
    assert body["intermittency_adi"]["min"] >= 1.32


def test_predict_and_sku_homologation(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json()["predicted_demand"] >= 0
    assert r.json()["reorder_point"] >= 2
    assert r.json()["horizon_days"] == 7

    crm_sku_sample = {**SAMPLE, "part_id": "TOY-FIL-041"}
    r_sku = client.post("/predict", json=crm_sku_sample)
    assert r_sku.status_code == 200
    assert r_sku.json()["part_id"] == "TOY-FIL-041"


def test_batch(client):
    single = client.post("/predict", json=SAMPLE).json()
    r = client.post("/predict/batch", json={"items": [SAMPLE, SAMPLE]})
    assert r.status_code == 200
    assert r.json() == [single, single]


def test_invalid_and_api_key_guard(client, monkeypatch):
    assert client.post("/predict", json={**SAMPLE, "lag_1": -1}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "part_id": "UNKNOWN"}).status_code == 422
    assert client.post("/predict/batch", json={"items": []}).status_code == 422

    monkeypatch.setenv("ML_INTERNAL_API_KEY", "secret-test-key")
    assert client.post("/predict", json=SAMPLE).status_code == 401
    ok = client.post("/predict", json=SAMPLE, headers={"X-Internal-API-Key": "secret-test-key"})
    assert ok.status_code == 200


def test_features_no_current_week_leakage():
    raw = generate()
    features = build_features(raw)
    target = features.iloc[0]
    original = raw[(raw.part_id == target.part_id)].sort_values("week_date")
    prior = original[original.week_date < target.week_date].tail(4)
    assert target.lag_1 == prior.demand.iloc[-1]
    assert target.rolling_mean_4 == prior.demand.mean()
    assert target.vehicles_served == prior.vehicles_served.iloc[-1]
    raw.loc[(raw.part_id == target.part_id) & (raw.week_date == target.week_date), "demand"] = 999
    modified = build_features(raw).iloc[0]
    assert modified.rolling_mean_4 == target.rolling_mean_4
