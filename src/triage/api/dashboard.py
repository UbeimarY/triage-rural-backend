"""Follow-up dashboard for coordinators: synchronized visits, most urgent first."""

import csv
import io
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query
from fastapi.responses import Response
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.deps import CoordinatorUser, SessionDep
from triage.api.schemas import (
    Agreement,
    AgreementCell,
    CommunityCount,
    DailyCount,
    DashboardInsights,
    DashboardSummary,
    DashboardVisit,
    PriorityCounts,
    PromoterCount,
    SyncDelay,
)
from triage.db.models import Assessment, Encounter, Patient, User

# Days are counted as the promoters live them, not in UTC.
LOCAL_ZONE = "America/Bogota"

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# The promoter's decision wins over the AI suggestion.
_effective = func.coalesce(Assessment.final_priority, Assessment.priority)
_attention_rank = case(
    (_effective == "high", 0),
    (_effective.is_(None), 1),
    (_effective == "medium", 2),
    else_=3,
)
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@router.get("/summary")
async def summary(_user: CoordinatorUser, session: SessionDep) -> DashboardSummary:
    rows = (
        await session.execute(
            select(_effective, func.count())
            .select_from(Encounter)
            .outerjoin(Assessment, Assessment.encounter_id == Encounter.id)
            .group_by(_effective)
        )
    ).all()
    counts = PriorityCounts()
    for priority, total in rows:
        setattr(counts, priority or "unclassified", total)

    awaiting_review = await session.scalar(
        select(func.count())
        .select_from(Assessment)
        .where(Assessment.priority == "high", Assessment.reviewed_at.is_(None))
    )
    lowered_rows = (
        await session.execute(
            select(Assessment.priority, Assessment.final_priority).where(
                Assessment.final_priority.is_not(None)
            )
        )
    ).all()
    lowered_by_promoter = sum(
        1
        for suggested, final in lowered_rows
        if _PRIORITY_ORDER[final] > _PRIORITY_ORDER[suggested]
    )

    communities = (
        await session.execute(
            select(Patient.community, func.count(Encounter.id))
            .join(Encounter, Encounter.patient_id == Patient.id)
            .group_by(Patient.community)
            .order_by(func.count(Encounter.id).desc())
            .limit(8)
        )
    ).all()
    last_received = await session.scalar(select(func.max(Encounter.received_at)))

    return DashboardSummary(
        total_visits=sum(total for _, total in rows),
        by_priority=counts,
        awaiting_review=awaiting_review or 0,
        lowered_by_promoter=lowered_by_promoter,
        by_community=[CommunityCount(community=name, visits=total) for name, total in communities],
        last_received_at=last_received,
    )


VisitFilter = Literal["high", "medium", "low", "unclassified"] | None


@router.get("/visits")
async def visits(
    _user: CoordinatorUser,
    session: SessionDep,
    priority: VisitFilter = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[DashboardVisit]:
    return await _visits(session, priority, limit)


# Columns of the export. The symptoms note stays out: a spreadsheet travels by
# email and USB sticks, so it only carries what the follow-up needs.
_CSV_COLUMNS = (
    "patient_code",
    "community",
    "age",
    "sex",
    "alarm_count",
    "suggested_priority",
    "final_priority",
    "source",
    "model_version",
    "reviewed",
    "promoter_name",
    "created_at",
    "received_at",
)


@router.get("/visits.csv", response_class=Response)
async def visits_csv(
    _user: CoordinatorUser, session: SessionDep, priority: VisitFilter = None
) -> Response:
    rows = await _visits(session, priority, limit=5000)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)
    for visit in rows:
        data = visit.model_dump()
        writer.writerow([_cell(data[column]) for column in _CSV_COLUMNS])
    today = datetime.now(ZoneInfo(LOCAL_ZONE)).date().isoformat()
    return Response(
        # A byte order mark lets Excel open the accents correctly.
        content="\ufeff" + buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="visitas-{today}.csv"'},
    )


def _cell(value: object) -> str:
    """Text a spreadsheet will not run as a formula (CSV injection)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


@router.get("/insights")
async def insights(
    _user: CoordinatorUser, session: SessionDep, days: int = Query(default=14, ge=1, le=90)
) -> DashboardInsights:
    today = datetime.now(ZoneInfo(LOCAL_ZONE)).date()
    first_day = today - timedelta(days=days - 1)
    since = datetime.combine(first_day, datetime.min.time(), tzinfo=ZoneInfo(LOCAL_ZONE))
    return DashboardInsights(
        days=days,
        daily=await _daily(session, first_day, days, since),
        agreement=await _agreement(session, since),
        sync_delay=await _sync_delay(session, since),
        by_promoter=await _by_promoter(session, since),
    )


def _current_assessment():
    """Only the assessment of the visit's current version counts."""
    return (Assessment.encounter_id == Encounter.id) & (
        Assessment.encounter_version == Encounter.version
    )


async def _daily(
    session: AsyncSession, first_day: date, days: int, since: datetime
) -> list[DailyCount]:
    local_day = func.date(func.timezone(LOCAL_ZONE, Encounter.created_at))
    rows = (
        await session.execute(
            select(local_day, _effective, func.count())
            .select_from(Encounter)
            .outerjoin(Assessment, _current_assessment())
            .where(Encounter.created_at >= since)
            .group_by(local_day, _effective)
        )
    ).all()
    by_day = {first_day + timedelta(days=i): DailyCount(day="") for i in range(days)}
    for day, priority, total in rows:
        if day in by_day:
            setattr(by_day[day], priority or "unclassified", total)
    return [count.model_copy(update={"day": day.isoformat()}) for day, count in by_day.items()]


async def _agreement(session: AsyncSession, since: datetime) -> Agreement:
    rows = (
        await session.execute(
            select(Assessment.priority, Assessment.final_priority, func.count())
            .join(Encounter, _current_assessment())
            .where(
                Assessment.source == "model",
                Assessment.final_priority.is_not(None),
                Encounter.created_at >= since,
            )
            .group_by(Assessment.priority, Assessment.final_priority)
        )
    ).all()
    cells = [AgreementCell(suggested=s, final=f, visits=n) for s, f, n in rows]
    rank = _PRIORITY_ORDER
    return Agreement(
        reviewed=sum(c.visits for c in cells),
        kept=sum(c.visits for c in cells if c.suggested == c.final),
        raised=sum(c.visits for c in cells if rank[c.final] < rank[c.suggested]),
        lowered=sum(c.visits for c in cells if rank[c.final] > rank[c.suggested]),
        cells=sorted(cells, key=lambda c: (rank[c.suggested], rank[c.final])),
    )


async def _sync_delay(session: AsyncSession, since: datetime) -> SyncDelay:
    minutes = func.extract("epoch", Encounter.received_at - Encounter.created_at) / 60
    total, median, p90 = (
        await session.execute(
            select(
                func.count(),
                func.percentile_cont(0.5).within_group(minutes),
                func.percentile_cont(0.9).within_group(minutes),
            ).where(Encounter.created_at >= since)
        )
    ).one()
    return SyncDelay(
        visits=total,
        median_minutes=None if median is None else round(max(0.0, float(median)), 1),
        p90_minutes=None if p90 is None else round(max(0.0, float(p90)), 1),
    )


async def _by_promoter(session: AsyncSession, since: datetime) -> list[PromoterCount]:
    rows = (
        await session.execute(
            select(User.full_name, func.count(), func.count().filter(_effective == "high"))
            .select_from(Encounter)
            .join(User, User.id == Encounter.promoter_id)
            .outerjoin(Assessment, _current_assessment())
            .where(Encounter.created_at >= since)
            .group_by(User.full_name)
            .order_by(func.count().desc())
        )
    ).all()
    return [PromoterCount(promoter_name=n, visits=v, high=h) for n, v, h in rows]


async def _visits(session: AsyncSession, priority: VisitFilter, limit: int) -> list[DashboardVisit]:
    query = (
        select(Encounter, Patient, Assessment, User.full_name)
        .join(Patient, Patient.id == Encounter.patient_id)
        .join(User, User.id == Encounter.promoter_id)
        .outerjoin(Assessment, Assessment.encounter_id == Encounter.id)
        .order_by(_attention_rank, Encounter.created_at.asc())
        .limit(limit)
    )
    if priority == "unclassified":
        query = query.where(_effective.is_(None))
    elif priority is not None:
        query = query.where(_effective == priority)

    year = datetime.now(UTC).year
    result = []
    for encounter, patient, assessment, promoter_name in (await session.execute(query)).all():
        current = (
            assessment if assessment and assessment.encounter_version == encounter.version else None
        )
        result.append(
            DashboardVisit(
                encounter_id=encounter.id,
                patient_code=patient.code,
                community=patient.community,
                age=year - patient.birth_year,
                sex=patient.sex,
                symptoms=encounter.symptoms,
                alarm_count=sum(1 for value in encounter.alarms.values() if value),
                suggested_priority=current.priority if current else None,
                final_priority=current.final_priority if current else None,
                source=current.source if current else None,
                model_version=current.model_version if current else None,
                reviewed=bool(current and current.reviewed_at),
                promoter_name=promoter_name,
                version=encounter.version,
                created_at=encounter.created_at,
                received_at=encounter.received_at,
            )
        )
    return result
