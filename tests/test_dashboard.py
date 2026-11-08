import httpx
import pytest

from tests import factories as f
from tests.conftest import UserFactory
from tests.test_sync import push
from triage.db.models import UserRole

pytestmark = pytest.mark.anyio


async def test_visits_are_listed_most_urgent_first(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    for priority in ("low", "high", "medium"):
        p = f.patient()
        e = f.encounter(p["id"])
        await push(
            api,
            promoter,
            [
                f.operation("patient", "create", p["id"], p),
                f.operation("encounter", "create", e["id"], e),
                f.operation(
                    "assessment", "create", e["id"], f.assessment(e["id"], priority=priority)
                ),
            ],
        )

    response = await api.get(
        "/dashboard/visits", headers={"Authorization": f"Bearer {coordinator}"}
    )

    assert [visit["suggestedPriority"] for visit in response.json()] == ["high", "medium", "low"]


async def test_the_promoters_decision_overrides_the_suggestion(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    _, encounter_id, ops = f.new_visit()
    reviewed = f.assessment(
        encounter_id,
        priority="high",
        finalPriority="medium",
        reviewNote="Signos vitales estables al revisar",
        reviewedAt=f.now(),
    )
    await push(api, promoter, [*ops, f.operation("assessment", "update", encounter_id, reviewed)])

    summary = await api.get(
        "/dashboard/summary", headers={"Authorization": f"Bearer {coordinator}"}
    )

    body = summary.json()
    assert body["byPriority"]["medium"] == 1
    assert body["loweredByPromoter"] == 1
    assert body["byCommunity"] == [{"community": "Vereda El Encano", "visits": 1}]


async def test_promoters_cannot_open_the_dashboard(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)

    response = await api.get("/dashboard/summary", headers={"Authorization": f"Bearer {promoter}"})

    assert response.status_code == 403
