"""Promoter accounts, created by a coordinator from the dashboard.

There is still no public sign-up: only a coordinator can open an account, and only
for a promoter. Coordinator accounts are created on the server (see triage.cli).
"""

from fastapi import APIRouter
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from triage.api.deps import CoordinatorUser, SessionDep
from triage.api.errors import ApiError
from triage.api.schemas import AccountOut, NewPromoter
from triage.db.models import Encounter, User, UserRole
from triage.security import hash_password

router = APIRouter(prefix="/accounts", tags=["accounts"])


@router.get("")
async def list_accounts(_user: CoordinatorUser, session: SessionDep) -> list[AccountOut]:
    visits = (
        select(func.count())
        .where(Encounter.promoter_id == User.id)
        .correlate(User)
        .scalar_subquery()
    )
    rows = (
        await session.execute(
            select(User, visits).where(User.role == UserRole.PROMOTER).order_by(User.full_name)
        )
    ).all()
    return [
        AccountOut.model_validate(user, from_attributes=True).model_copy(update={"visits": count})
        for user, count in rows
    ]


@router.post("", status_code=201)
async def create_promoter(
    body: NewPromoter, _user: CoordinatorUser, session: SessionDep
) -> AccountOut:
    if await session.scalar(select(User.id).where(User.email == body.email)):
        raise ApiError(409, "EMAIL_TAKEN", "An account with this email already exists")
    user = User(
        email=body.email,
        full_name=body.full_name,
        role=UserRole.PROMOTER,
        password_hash=await hash_password(body.password),
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as error:  # two coordinators at the same time
        await session.rollback()
        raise ApiError(409, "EMAIL_TAKEN", "An account with this email already exists") from error
    await session.refresh(user)
    return AccountOut.model_validate(user, from_attributes=True)
