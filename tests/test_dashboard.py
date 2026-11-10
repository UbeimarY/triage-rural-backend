from datetime import UTC, datetime, timedelta

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


async def _reviewed_visit(
    api: httpx.AsyncClient, promoter: str, suggested: str, final: str, **encounter: object
) -> None:
    p = f.patient()
    e = f.encounter(p["id"], **encounter)
    reviewed = f.assessment(
        e["id"], priority=suggested, finalPriority=final, reviewNote="Revisado", reviewedAt=f.now()
    )
    await push(
        api,
        promoter,
        [
            f.operation("patient", "create", p["id"], p),
            f.operation("encounter", "create", e["id"], e),
            f.operation("assessment", "create", e["id"], reviewed),
        ],
    )


async def test_insights_count_each_day_and_how_often_promoters_kept_the_ai(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    await _reviewed_visit(api, promoter, "high", "high")
    await _reviewed_visit(api, promoter, "medium", "high")
    await _reviewed_visit(api, promoter, "high", "low")

    response = await api.get(
        "/dashboard/insights?days=7", headers={"Authorization": f"Bearer {coordinator}"}
    )

    body = response.json()
    assert len(body["daily"]) == 7
    today = body["daily"][-1]
    assert (today["high"], today["low"]) == (2, 1)
    assert body["agreement"] | {"cells": None} == {
        "reviewed": 3,
        "kept": 1,
        "raised": 1,
        "lowered": 1,
        "cells": None,
    }
    assert body["syncDelay"]["visits"] == 3
    assert body["syncDelay"]["medianMinutes"] >= 0
    assert body["byPromoter"] == [{"promoterName": "Ana", "visits": 3, "high": 2}]


async def test_old_visits_stay_out_of_the_period(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    await _reviewed_visit(api, promoter, "high", "high", createdAt=old, updatedAt=old)

    body = (
        await api.get(
            "/dashboard/insights?days=7", headers={"Authorization": f"Bearer {coordinator}"}
        )
    ).json()

    assert sum(day["high"] for day in body["daily"]) == 0
    assert body["agreement"]["reviewed"] == 0


async def test_the_export_is_a_csv_without_notes_and_safe_for_spreadsheets(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)
    p = f.patient(community="=HYPERLINK(1)")
    e = f.encounter(p["id"], symptoms="Dolor de cabeza muy fuerte")
    await push(
        api,
        promoter,
        [
            f.operation("patient", "create", p["id"], p),
            f.operation("encounter", "create", e["id"], e),
        ],
    )

    response = await api.get(
        "/dashboard/visits.csv", headers={"Authorization": f"Bearer {coordinator}"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    lines = response.text.lstrip("﻿").splitlines()
    assert lines[0].startswith("patient_code,community,age")
    assert "'=HYPERLINK(1)" in lines[1]
    assert "Dolor de cabeza" not in response.text


async def test_promoters_cannot_read_insights_or_export(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    promoter = await make_user("ana@test.co", UserRole.PROMOTER)
    headers = {"Authorization": f"Bearer {promoter}"}

    assert (await api.get("/dashboard/insights", headers=headers)).status_code == 403
    assert (await api.get("/dashboard/visits.csv", headers=headers)).status_code == 403


async def test_insights_refuse_periods_longer_than_ninety_days(
    api: httpx.AsyncClient, make_user: UserFactory
) -> None:
    coordinator = await make_user("jefe@test.co", UserRole.COORDINATOR)

    response = await api.get(
        "/dashboard/insights?days=10000", headers={"Authorization": f"Bearer {coordinator}"}
    )

    assert response.status_code == 422
