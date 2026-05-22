from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

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
