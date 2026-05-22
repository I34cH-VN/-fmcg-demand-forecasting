from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.models import inference
from src.api.app import create_app
from src.api.routes import get_session
from src.api import services
from src.storage.models import Base
from src.storage.repository import create_run, finish_run, get_latest_metrics_run, get_run


def make_test_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def make_client(session_factory):
    app = create_app(create_tables=False)

    def override_get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_get_session
    return TestClient(app)


def test_storage_create_read_run():
    session_factory = make_test_session_factory()
    with session_factory() as session:
        run = create_run(session, run_type="train", config_path="configs/default.yaml")
        finish_run(session, run, status="success", metrics_json=[{"target": "total_qty", "wmape": 0.1}])

        fetched = get_run(session, run.run_id)
        latest = get_latest_metrics_run(session)

    assert fetched is not None
    assert fetched.status == "success"
    assert latest is not None
    assert latest.run_id == run.run_id


def test_api_health_endpoint():
    client = make_client(make_test_session_factory())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_analyze_train_report_with_mocked_workflows(monkeypatch):
    session_factory = make_test_session_factory()
    client = make_client(session_factory)

    monkeypatch.setattr(
        services,
        "run_analyze_workflow",
        lambda config_path: {"data_quality_json": {"quality": {"summary": {"warning_checks": 0}}}, "report_path": "outputs/data_quality_report.json"},
    )
    monkeypatch.setattr(
        services,
        "run_train_workflow",
        lambda config_path: {"metrics_json": [{"target": "total_qty", "wmape": 0.1}], "report_path": "reports/metrics.json"},
    )
    monkeypatch.setattr(
        services,
        "run_report_workflow",
        lambda config_path: {"metrics_json": [{"target": "total_qty", "wmape": 0.1}], "data_quality_json": {"quality": {}}, "report_path": "outputs/reports/forecast_report.md"},
    )

    analyze_response = client.post("/runs/analyze", json={"config_path": "configs/default.yaml"})
    train_response = client.post("/runs/train", json={"config_path": "configs/default.yaml"})
    report_response = client.post("/runs/report", json={"config_path": "configs/default.yaml"})
    runs_response = client.get("/runs")
    latest_response = client.get("/metrics/latest")

    assert analyze_response.status_code == 200
    assert analyze_response.json()["status"] == "success"
    assert train_response.status_code == 200
    assert train_response.json()["metrics_json"][0]["target"] == "total_qty"
    assert report_response.status_code == 200
    assert report_response.json()["report_path"] == "outputs/reports/forecast_report.md"
    assert runs_response.status_code == 200
    assert len(runs_response.json()) == 3
    assert latest_response.status_code == 200
    assert latest_response.json()["metrics_json"][0]["wmape"] == 0.1


class DummyModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame):
        return [self.value] * len(frame)


def _predict_config(models_path="models"):
    return {
        "paths": {"models": models_path},
        "columns": {"target_columns": ["total_qty", "total_cbm"]},
        "validation": {"forecast_horizon_weeks": 1},
        "features": {"lag_weeks": [1, 2], "rolling_windows": [2]},
    }


def _predict_payload(config_path):
    return {
        "config_path": str(config_path),
        "history": [
            {
                "week_start": f"2025-01-{day:02d}",
                "category": "bev",
                "brand": "alpha",
                "whseid": "hcm",
                "total_qty": 100 + index,
                "total_cbm": 10 + index,
            }
            for index, day in enumerate([6, 13, 20, 27])
        ],
        "forecasts": [
            {
                "week_start": "2025-02-03",
                "category": "bev",
                "brand": "alpha",
                "whseid": "hcm",
            }
        ],
    }


def test_api_predict_loads_models_and_returns_forecast(monkeypatch):
    client = make_client(make_test_session_factory())

    def fake_load_model(config, target_col):
        if target_col == "total_qty":
            return DummyModel(123.4)
        return DummyModel(12.3)

    monkeypatch.setattr(services, "load_config", lambda config_path: _predict_config())
    monkeypatch.setattr(inference, "_load_model", fake_load_model)

    response = client.post("/predict", json=_predict_payload("configs/default.yaml"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_horizon_weeks"] == 1
    assert payload["forecasts"][0]["week_start"] == "2025-02-03"
    assert payload["forecasts"][0]["category"] == "BEV"
    assert payload["forecasts"][0]["predicted_total_qty"] == 123.4
    assert payload["forecasts"][0]["predicted_total_cbm"] == 12.3


def test_api_predict_validates_input_schema():
    client = make_client(make_test_session_factory())
    payload = _predict_payload("configs/default.yaml")
    payload["history"][0]["total_qty"] = -1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    payload = _predict_payload("configs/default.yaml")
    payload["forecasts"][0]["week_start"] = "not-a-date"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_api_predict_returns_404_when_model_is_missing(monkeypatch):
    client = make_client(make_test_session_factory())
    config = _predict_config(models_path="missing-models")
    config["columns"]["target_columns"] = ["total_qty"]
    monkeypatch.setattr(services, "load_config", lambda config_path: config)

    response = client.post("/predict", json=_predict_payload("configs/default.yaml"))

    assert response.status_code == 404
    assert "Model file not found" in response.json()["detail"]
