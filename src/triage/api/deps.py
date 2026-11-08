"""Authentication and authorization dependencies."""

import uuid
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.errors import ApiError
from triage.config import Settings
from triage.db.models import User, UserRole
from triage.db.session import get_session
from triage.security import read_token

_bearer = HTTPBearer(auto_error=False)

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


async def current_user(
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None:
        raise ApiError(401, "AUTHENTICATION_REQUIRED", "Missing bearer token")
    try:
        claims = read_token(settings.jwt_secret, credentials.credentials)
        user_id = uuid.UUID(claims["sub"])
    except (jwt.InvalidTokenError, ValueError) as error:
        raise ApiError(401, "SESSION_EXPIRED", "Invalid or expired token") from error
    user = await session.get(User, user_id)
    if user is None:
        raise ApiError(401, "SESSION_EXPIRED", "Unknown user")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require_role(role: UserRole):
    async def check(user: CurrentUser) -> User:
        if user.role != role:
            raise ApiError(403, "FORBIDDEN", f"Only {role.value} accounts can do this")
        return user

    return check


PromoterUser = Annotated[User, Depends(require_role(UserRole.PROMOTER))]
CoordinatorUser = Annotated[User, Depends(require_role(UserRole.COORDINATOR))]
