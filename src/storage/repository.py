from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.storage.models import RunRecord


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_run(session: Session, run_type: str, config_path: str, status: str = "running") -> RunRecord:
    run = RunRecord(
        run_id=uuid4().hex,
        run_type=run_type,
        status=status,
        started_at=utc_now(),
        config_path=config_path,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def finish_run(
    session: Session,
    run: RunRecord,
    status: str,
    metrics_json: dict | list | None = None,
    data_quality_json: dict | list | None = None,
    report_path: str | None = None,
    error_message: str | None = None,
) -> RunRecord:
    run.status = status
    run.finished_at = utc_now()
    run.metrics_json = metrics_json
    run.data_quality_json = data_quality_json
    run.report_path = report_path
    run.error_message = error_message
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, run_id: str) -> RunRecord | None:
    return session.execute(select(RunRecord).where(RunRecord.run_id == run_id)).scalar_one_or_none()


def list_runs(session: Session, limit: int = 50) -> list[RunRecord]:
    statement = select(RunRecord).order_by(RunRecord.started_at.desc()).limit(limit)
    return list(session.execute(statement).scalars())


def get_latest_metrics_run(session: Session) -> RunRecord | None:
    statement = (
        select(RunRecord)
        .where(RunRecord.run_type == "train", RunRecord.status == "success", RunRecord.metrics_json.is_not(None))
        .order_by(RunRecord.finished_at.desc())
        .limit(1)
    )
    return session.execute(statement).scalar_one_or_none()
