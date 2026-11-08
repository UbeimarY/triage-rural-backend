"""Follow-up dashboard for coordinators: synchronized visits, most urgent first."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Query
from sqlalchemy import case, func, select

from triage.api.deps import CoordinatorUser, SessionDep
from triage.api.schemas import CommunityCount, DashboardSummary, DashboardVisit, PriorityCounts
from triage.db.models import Assessment, Encounter, Patient, User

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


@router.get("/visits")
async def visits(
    _user: CoordinatorUser,
    session: SessionDep,
    priority: Literal["high", "medium", "low", "unclassified"] | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[DashboardVisit]:
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
