import httpx
import pytest

from tests.conftest import UserFactory
from triage.db.models import UserRole

pytestmark = pytest.mark.anyio

NEW = {"email": " Rosa@Vereda.co ", "fullName": "Rosa  Pantoja", "password": "clave-larga-2026"}


async def test_a_coordinator_opens_an_account_the_promoter_can_sign_in_with(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    headers = {"Authorization": f"Bearer {coordinator}"}

    created = await api.post("/accounts", json=NEW, headers=headers)

    assert created.status_code == 201
    assert created.json() | {"id": None, "createdAt": None} == {
        "id": None,
        "email": "rosa@vereda.co",
        "fullName": "Rosa Pantoja",
        "role": "promoter",
        "createdAt": None,
        "visits": 0,
    }
    login = await api.post(
        "/auth/login", json={"email": "rosa@vereda.co", "password": NEW["password"]}
    )
    assert login.status_code == 200
    listed = await api.get("/accounts", headers=headers)
    assert [account["email"] for account in listed.json()] == ["rosa@vereda.co"]


async def test_an_email_can_only_have_one_account(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    headers = {"Authorization": f"Bearer {coordinator}"}
    await api.post("/accounts", json=NEW, headers=headers)

    again = await api.post("/accounts", json=NEW, headers=headers)

    assert again.status_code == 409
    assert again.json()["code"] == "EMAIL_TAKEN"


async def test_short_passwords_and_bad_emails_are_refused(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    headers = {"Authorization": f"Bearer {coordinator}"}

    response = await api.post(
        "/accounts", json={**NEW, "email": "sin-arroba", "password": "corta"}, headers=headers
    )

    assert response.status_code == 422
    assert response.json()["fields"] == ["email", "password"]


async def test_promoters_cannot_open_accounts(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)

    response = await api.post(
        "/accounts", json=NEW, headers={"Authorization": f"Bearer {promoter}"}
    )

    assert response.status_code == 403
