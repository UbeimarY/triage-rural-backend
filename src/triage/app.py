"""FastAPI application factory."""

import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from triage.api import accounts, auth, dashboard, sync
from triage.api.deps import SessionDep
from triage.api.errors import install_error_handlers
from triage.config import Settings, get_settings
from triage.db.session import create_engine


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = create_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await engine.dispose()

    app = FastAPI(title="Triage Rural API", version=_app_version(), lifespan=lifespan)
    app.state.settings = settings
    app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=3600,
    )
    install_error_handlers(app)
    app.include_router(auth.router)
    app.include_router(sync.router)
    app.include_router(dashboard.router)
    app.include_router(accounts.router)

    @app.get("/healthz", response_class=PlainTextResponse, include_in_schema=False)
    async def healthz() -> str:
        return "ok"

    @app.get("/status", tags=["system"])
    async def status(session: SessionDep) -> dict[str, object]:
        try:
            await session.execute(text("SELECT 1"))
            database_ok = True
        except Exception:
            database_ok = False
        return {
            "version": app.version,
            "databaseOk": database_ok,
            "serverTime": datetime.now(UTC).isoformat(),
        }

    return app


_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _app_version() -> str:
    """Installed package version; from a source checkout, read from pyproject.toml."""
    try:
        return version("triage")
    except PackageNotFoundError:
        with _PYPROJECT.open("rb") as file:
            return str(tomllib.load(file)["project"]["version"])
