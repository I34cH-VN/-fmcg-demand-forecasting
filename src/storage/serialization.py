from __future__ import annotations

from src.storage.models import RunRecord


def run_to_dict(run: RunRecord) -> dict:
    return {
        "run_id": run.run_id,
        "run_type": run.run_type,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "config_path": run.config_path,
        "metrics_json": run.metrics_json,
        "data_quality_json": run.data_quality_json,
        "report_path": run.report_path,
        "error_message": run.error_message,
    }
