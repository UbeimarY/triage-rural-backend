from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from triage.app import create_app
from triage.config import Settings
from triage.db import models  # noqa: F401  (registers every table)
from triage.db.base import Base
from triage.db.models import User, UserRole
from triage.security import hash_password

TEST_JWT_SECRET = "test-secret-with-at-least-32-characters"
PASSWORD = "very-secure-password"


class _TestEnvironment(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    test_database_url: str


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def settings() -> AsyncIterator[Settings]:
    """Settings pointing at a test database with a fresh, empty schema."""
    settings = Settings(
        database_url=_TestEnvironment().test_database_url, jwt_secret=TEST_JWT_SECRET
    )
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()
    yield settings


@pytest.fixture
async def api(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


UserFactory = Callable[[str, UserRole], Awaitable[str]]


@pytest.fixture
def make_user(settings: Settings, api: httpx.AsyncClient) -> UserFactory:
    """Creates an account directly in the database and returns its access token."""

    async def create(email: str, role: UserRole = UserRole.PROMOTER) -> str:
        engine = create_async_engine(settings.database_url)
        async with async_sessionmaker(engine)() as session:
            session.add(
                User(
                    email=email,
                    full_name=email.split("@")[0].title(),
                    role=role,
                    password_hash=await hash_password(PASSWORD),
                )
            )
            await session.commit()
        await engine.dispose()
        response = await api.post("/auth/login", json={"email": email, "password": PASSWORD})
        return str(response.json()["accessToken"])

    return create
