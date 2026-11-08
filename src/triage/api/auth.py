"""Sign-in. Accounts are provisioned by the health organization (see triage.cli):
there is no public sign-up for a system that handles health data."""

from datetime import timedelta

from fastapi import APIRouter
from sqlalchemy import select

from triage.api.deps import CurrentUser, SessionDep, SettingsDep
from triage.api.errors import ApiError
from triage.api.schemas import LoginRequest, LoginResponse, UserOut
from triage.db.models import User
from triage.security import issue_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login(body: LoginRequest, session: SessionDep, settings: SettingsDep) -> LoginResponse:
    user = await session.scalar(select(User).where(User.email == body.email.strip().lower()))
    # Same work and same answer for unknown emails and wrong passwords.
    valid = await verify_password(body.password, user.password_hash if user else None)
    if user is None or not valid:
        raise ApiError(401, "INVALID_CREDENTIALS", "Wrong email or password")
    token, expires_at = issue_token(
        settings.jwt_secret, user.id, user.role, timedelta(days=settings.jwt_expire_days)
    )
    return LoginResponse(
        access_token=token,
        expires_at=expires_at,
        user=UserOut.model_validate(user, from_attributes=True),
    )


@router.get("/me")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user, from_attributes=True)
