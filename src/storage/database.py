from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.storage.models import Base


DEFAULT_DATABASE_URL = "sqlite:///./outputs/runs.sqlite3"


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_db_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_database_url()
    if url.startswith("sqlite:///"):
        sqlite_path = url.removeprefix("sqlite:///")
        if sqlite_path not in {":memory:", ""}:
            Path(sqlite_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or create_db_engine(), autoflush=False, autocommit=False, future=True)


def init_db(engine: Engine | None = None) -> None:
    Base.metadata.create_all(bind=engine or create_db_engine())


ENGINE = create_db_engine()
SessionLocal = create_session_factory(ENGINE)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
