"""Request and response bodies. JSON uses camelCase, like the Angular client."""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

MIN_AGE = 18
MAX_AGE = 120


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


# ---------- Auth ----------


class LoginRequest(ApiModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class UserOut(ApiModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: str


class LoginResponse(ApiModel):
    access_token: str
    expires_at: datetime
    user: UserOut


# ---------- Records sent by the phone ----------


class AlarmAnswers(ApiModel):
    breathing_difficulty: bool
    loss_of_consciousness: bool
    chest_pain: bool
    severe_bleeding: bool
    seizures: bool


class PatientPayload(ApiModel):
    id: uuid.UUID
    code: str = Field(min_length=3, max_length=20)
    birth_year: int
    sex: Literal["female", "male", "other"]
    community: str = Field(min_length=2, max_length=100)
    created_at: datetime
    updated_at: datetime

    @field_validator("birth_year")
    @classmethod
    def adults_only(cls, value: int) -> int:
        year = datetime.now(UTC).year
        if not year - MAX_AGE <= value <= year - MIN_AGE:
            raise ValueError("The prototype only registers adults")
        return value


class EncounterPayload(ApiModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    symptoms: str = Field(min_length=3, max_length=2000)
    alarms: AlarmAnswers
    consent_given: bool
    created_at: datetime
    updated_at: datetime
    version: int = Field(ge=1)

    @field_validator("consent_given")
    @classmethod
    def consent_required(cls, value: bool) -> bool:
        # Health data is sensitive (Ley 1581 de 2012): nothing is stored without consent.
        if not value:
            raise ValueError("Consent is required")
        return value


Priority = Literal["high", "medium", "low"]


class AssessmentPayload(ApiModel):
    encounter_id: uuid.UUID
    encounter_version: int = Field(ge=1)
    priority: Priority
    source: Literal["rules", "model"]
    triggered_alarms: list[str] = Field(default_factory=list, max_length=10)
    reason_codes: list[str] = Field(default_factory=list, max_length=20)
    model_version: str | None = Field(default=None, max_length=50)
    created_at: datetime
    final_priority: Priority | None = None
    review_note: str | None = Field(default=None, max_length=500)
    reviewed_at: datetime | None = None


# ---------- Sync ----------

Entity = Literal["patient", "encounter", "assessment"]


class Operation(ApiModel):
    """One entry of the phone's outbox."""

    id: uuid.UUID
    entity: Entity
    entity_id: uuid.UUID
    type: Literal["create", "update"]
    payload: dict[str, Any]
    base_version: int | None = None
    created_at: datetime


class PushRequest(ApiModel):
    operations: list[Operation] = Field(max_length=100)


OperationStatus = Literal["applied", "duplicate", "conflict", "rejected", "retry"]


class OperationResult(ApiModel):
    operation_id: uuid.UUID
    status: OperationStatus
    # Stable code for the client to translate (for example "VERSION_CONFLICT").
    code: str | None = None
    server_version: int | None = None
    # On a conflict: the server's current copy, so the promoter can compare.
    server_record: dict[str, Any] | None = None


class PushResponse(ApiModel):
    results: list[OperationResult]
    server_time: datetime


# ---------- Dashboard ----------


class PriorityCounts(ApiModel):
    high: int = 0
    medium: int = 0
    low: int = 0
    unclassified: int = 0


class CommunityCount(ApiModel):
    community: str
    visits: int


class DashboardSummary(ApiModel):
    total_visits: int
    by_priority: PriorityCounts
    awaiting_review: int
    lowered_by_promoter: int
    by_community: list[CommunityCount]
    last_received_at: datetime | None


class DashboardVisit(ApiModel):
    encounter_id: uuid.UUID
    patient_code: str
    community: str
    age: int
    sex: str
    symptoms: str
    alarm_count: int
    suggested_priority: Priority | None
    final_priority: Priority | None
    source: str | None
    model_version: str | None
    reviewed: bool
    promoter_name: str
    version: int
    created_at: datetime
    received_at: datetime
