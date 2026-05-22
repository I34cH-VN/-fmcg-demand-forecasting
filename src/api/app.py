from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.storage.database import init_db


def create_app(create_tables: bool = False) -> FastAPI:
    app = FastAPI(
        title="Demand Forecasting Agent API",
        version="1.0.0",
        description="FastAPI service for data quality analysis, model training, and forecasting reports.",
    )
    app.include_router(router)
    if create_tables:
        init_db()
    return app


app = create_app()
