from typing import Any

import httpx
import pytest

from tests import factories as f
from tests.conftest import UserFactory
from triage.db.models import UserRole

pytestmark = pytest.mark.anyio


async def push(
    api: httpx.AsyncClient, token: str, operations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    response = await api.post(
        "/sync/push", json={"operations": operations}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200, response.text
    return response.json()["results"]


def statuses(results: list[dict[str, Any]]) -> list[str]:
    return [result["status"] for result in results]


async def test_a_new_visit_is_applied(api: httpx.AsyncClient, make_user: UserFactory) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    _, _, ops = f.new_visit()

    results = await push(api, token, ops)

    assert statuses(results) == ["applied", "applied", "applied"]
    assert [r["operationId"] for r in results] == [op["id"] for op in ops]


async def test_retrying_a_batch_never_duplicates_records(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    _, _, ops = f.new_visit()

    await push(api, promoter, ops)
    # The answer was lost on a bad network: the phone sends the same operations again.
    retry = await push(api, promoter, ops)

    assert statuses(retry) == ["duplicate", "duplicate", "duplicate"]
    summary = await api.get(
        "/dashboard/summary", headers={"Authorization": f"Bearer {coordinator}"}
    )
    assert summary.json()["totalVisits"] == 1


async def test_an_edit_on_the_current_version_is_applied(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    patient_id, encounter_id, ops = f.new_visit()
    await push(api, token, ops)

    edited = f.encounter(
        patient_id, encounter_id, symptoms="Fiebre alta y dolor de cabeza", version=2
    )
    results = await push(api, token, [f.operation("encounter", "update", encounter_id, edited, 1)])

    assert results[0]["status"] == "applied"
    assert results[0]["serverVersion"] == 2


async def test_an_edit_based_on_an_old_version_is_a_conflict(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    patient_id, encounter_id, ops = f.new_visit()
    await push(api, token, ops)
    first = f.encounter(patient_id, encounter_id, symptoms="Primera corrección", version=2)
    await push(api, token, [f.operation("encounter", "update", encounter_id, first, 1)])

    # A second edit also made from version 1 (for example on another device).
    stale = f.encounter(patient_id, encounter_id, symptoms="Edición desactualizada", version=2)
    results = await push(api, token, [f.operation("encounter", "update", encounter_id, stale, 1)])

    conflict = results[0]
    assert conflict["status"] == "conflict"
    assert conflict["code"] == "VERSION_CONFLICT"
    assert conflict["serverVersion"] == 2
    # The server never overwrites health data silently: it returns its copy.
    assert conflict["serverRecord"]["symptoms"] == "Primera corrección"


async def test_a_visit_whose_patient_has_not_arrived_is_retried_later(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    _, _, ops = f.new_visit()

    results = await push(api, token, ops[1:2])

    assert results[0] == {
        "operationId": ops[1]["id"],
        "status": "retry",
        "code": "PATIENT_NOT_FOUND",
        "serverVersion": None,
        "serverRecord": None,
    }


async def test_one_invalid_operation_does_not_block_the_batch(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    patient_id, _, ops = f.new_visit()
    without_consent = f.encounter(patient_id, consentGiven=False)
    invalid = f.operation("encounter", "create", without_consent["id"], without_consent)

    results = await push(api, token, [ops[0], invalid, ops[1]])

    assert statuses(results) == ["applied", "rejected", "applied"]
    assert results[1]["code"] == "VALIDATION_FAILED"


async def test_minors_are_rejected(api: httpx.AsyncClient, make_user: UserFactory) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    minor = f.patient(birthYear=2020)

    results = await push(api, token, [f.operation("patient", "create", minor["id"], minor)])

    assert results[0]["code"] == "VALIDATION_FAILED"


async def test_a_promoter_cannot_touch_another_promoters_records(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    ana = await make_user("ana@test.co", UserRole.PROMOTER)
    luis = await make_user("luis@test.co", UserRole.PROMOTER)
    patient_id, encounter_id, ops = f.new_visit()
    await push(api, ana, ops)

    hijack = f.encounter(patient_id, encounter_id, symptoms="Cambiado por otro", version=2)
    results = await push(api, luis, [f.operation("encounter", "update", encounter_id, hijack, 1)])

    assert results[0]["status"] == "rejected"
    assert results[0]["code"] == "FORBIDDEN"


async def test_a_result_for_an_older_version_never_replaces_a_newer_one(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    token = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    patient_id, encounter_id, ops = f.new_visit()
    await push(api, token, ops)
    edited = f.encounter(patient_id, encounter_id, version=2)
    newer = f.assessment(encounter_id, version=2, priority="high")
    older = f.assessment(encounter_id, version=1, priority="low")

    await push(
        api,
        token,
        [
            f.operation("encounter", "update", encounter_id, edited, 1),
            f.operation("assessment", "update", encounter_id, newer),
            f.operation("assessment", "update", encounter_id, older),
        ],
    )

    visits = await api.get("/dashboard/visits", headers={"Authorization": f"Bearer {coordinator}"})
    assert visits.json()[0]["suggestedPriority"] == "high"


async def test_only_promoters_upload_records(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    _, _, ops = f.new_visit()

    response = await api.post(
        "/sync/push", json={"operations": ops}, headers={"Authorization": f"Bearer {coordinator}"}
    )

    assert response.status_code == 403
