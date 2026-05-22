from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    run_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    config_path: Mapped[str] = mapped_column(String(512), nullable=False)
    metrics_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    data_quality_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
