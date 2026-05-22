from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api import services
from src.storage.database import get_session
from src.storage.repository import create_run, finish_run, get_latest_metrics_run, get_run, list_runs
from src.storage.serialization import run_to_dict


DEFAULT_CONFIG_PATH = "configs/default.yaml"


class RunRequest(BaseModel):
    config_path: str = DEFAULT_CONFIG_PATH


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _execute_run(session: Session, run_type: str, config_path: str) -> dict:
    run = create_run(session, run_type=run_type, config_path=config_path)
    try:
        if run_type == "analyze":
            result = services.run_analyze_workflow(config_path)
        elif run_type == "train":
            result = services.run_train_workflow(config_path)
        elif run_type == "report":
            result = services.run_report_workflow(config_path)
        else:
            raise ValueError(f"Unsupported run type: {run_type}")
        run = finish_run(
            session,
            run,
            status="success",
            metrics_json=result.get("metrics_json"),
            data_quality_json=result.get("data_quality_json"),
            report_path=result.get("report_path"),
        )
    except Exception as exc:
        run = finish_run(session, run, status="failed", error_message=str(exc))
        raise HTTPException(status_code=500, detail=run_to_dict(run)) from exc
    return run_to_dict(run)


@router.post("/runs/analyze")
def analyze_run(request: RunRequest, session: Annotated[Session, Depends(get_session)]) -> dict:
    return _execute_run(session, "analyze", request.config_path)


@router.post("/runs/train")
def train_run(request: RunRequest, session: Annotated[Session, Depends(get_session)]) -> dict:
    return _execute_run(session, "train", request.config_path)


@router.post("/runs/report")
def report_run(request: RunRequest, session: Annotated[Session, Depends(get_session)]) -> dict:
    return _execute_run(session, "report", request.config_path)


@router.get("/runs")
def runs(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict]:
    return [run_to_dict(run) for run in list_runs(session, limit=limit)]


@router.get("/runs/{run_id}")
def run_detail(run_id: str, session: Annotated[Session, Depends(get_session)]) -> dict:
    run = get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_to_dict(run)


@router.get("/metrics/latest")
def latest_metrics(session: Annotated[Session, Depends(get_session)]) -> dict:
    run = get_latest_metrics_run(session)
    if run is None:
        raise HTTPException(status_code=404, detail="No successful training metrics found")
    return {"run_id": run.run_id, "metrics_json": run.metrics_json, "finished_at": run.finished_at.isoformat() if run.finished_at else None}
