import httpx
import pytest

from tests.conftest import PASSWORD, UserFactory
from triage.db.models import UserRole

pytestmark = pytest.mark.anyio


async def test_login_returns_a_token_that_identifies_the_user(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)

    response = await api.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["role"] == "promoter"


async def test_wrong_password_and_unknown_email_get_the_same_error(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    await make_user("ana@test.co", UserRole.PROMOTER)

    wrong = await api.post("/auth/login", json={"email": "ana@test.co", "password": "nope-nope"})
    unknown = await api.post("/auth/login", json={"email": "who@test.co", "password": PASSWORD})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


async def test_a_tampered_token_is_rejected(api: httpx.AsyncClient) -> None:
    response = await api.get("/auth/me", headers={"Authorization": "Bearer not.a.token"})

    assert response.status_code == 401
    assert response.json()["code"] == "SESSION_EXPIRED"
