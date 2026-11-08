"""Server-side copy of the records promoters create on their phones.

Ids are UUIDs generated on the device: the same record keeps the same id
offline and online, so a retried upload can never create a duplicate.
"""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from triage.db.base import Base


class UserRole(enum.StrEnum):
    PROMOTER = "promoter"
    COORDINATOR = "coordinator"


class Priority(enum.StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('promoter', 'coordinator')", name="valid_role"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    full_name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Patient(Base):
    """Minimal data only (Ley 1581 de 2012): no names, documents or phone numbers."""

    __tablename__ = "patients"
    __table_args__ = (
        CheckConstraint("sex IN ('female', 'male', 'other')", name="valid_sex"),
        CheckConstraint("birth_year BETWEEN 1900 AND 2100", name="plausible_birth_year"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20))
    birth_year: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(10))
    community: Mapped[str] = mapped_column(String(100))
    promoter_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Encounter(Base):
    __tablename__ = "encounters"
    __table_args__ = (
        CheckConstraint("version >= 1", name="positive_version"),
        CheckConstraint("consent_given", name="consent_required"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("patients.id"), index=True)
    promoter_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    symptoms: Mapped[str] = mapped_column(Text)
    alarms: Mapped[dict[str, Any]] = mapped_column(JSONB)
    consent_given: Mapped[bool]
    # Optimistic concurrency: every edit must say which version it was based on.
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Assessment(Base):
    """AI suggestion for one visit version, plus the promoter's final decision."""

    __tablename__ = "assessments"
    __table_args__ = (
        CheckConstraint("priority IN ('high', 'medium', 'low')", name="valid_priority"),
        CheckConstraint(
            "final_priority IS NULL OR final_priority IN ('high', 'medium', 'low')",
            name="valid_final_priority",
        ),
    )

    encounter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE"), primary_key=True
    )
    encounter_version: Mapped[int] = mapped_column(Integer)
    priority: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(10))
    triggered_alarms: Mapped[list[str]] = mapped_column(ARRAY(String))
    reason_codes: Mapped[list[str]] = mapped_column(ARRAY(String))
    model_version: Mapped[str | None] = mapped_column(String(50))
    final_priority: Mapped[str | None] = mapped_column(String(10))
    review_note: Mapped[str | None] = mapped_column(String(500))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ProcessedOperation(Base):
    """Every outbox operation the server has seen: the key to idempotent sync."""

    __tablename__ = "processed_operations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    entity: Mapped[str] = mapped_column(String(20))
    entity_id: Mapped[uuid.UUID]
    status: Mapped[str] = mapped_column(String(20))
    server_version: Mapped[int | None] = mapped_column(Integer)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
