"""Endpoint where phones upload their outbox."""

from datetime import UTC, datetime

from fastapi import APIRouter

from triage.api.deps import PromoterUser, SessionDep, SettingsDep
from triage.api.errors import ApiError
from triage.api.schemas import PushRequest, PushResponse
from triage.sync import apply_operations

router = APIRouter(prefix="/sync", tags=["sync"])


@router.post("/push")
async def push(
    body: PushRequest, user: PromoterUser, session: SessionDep, settings: SettingsDep
) -> PushResponse:
    """Applies a batch of outbox operations in order. Always answers per operation,
    so the phone knows exactly which ones to delete, retry or show as a conflict."""
    if len(body.operations) > settings.max_sync_batch:
        raise ApiError(413, "BATCH_TOO_LARGE", "Too many operations in one request")
    results = await apply_operations(session, user, body.operations)
    return PushResponse(results=results, server_time=datetime.now(UTC))
