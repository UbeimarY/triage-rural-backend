"""Applies outbox operations sent by phones.

Guarantees:
- Idempotent: an operation id the server already applied is acknowledged
  again ("duplicate"), never applied twice. Retries on bad networks are safe.
- No silent overwrites of health data: an edit based on an old version is a
  conflict, returned with the server's copy so a person decides.
- Isolation: each operation runs in its own savepoint, so one bad operation
  never blocks the rest of the batch.
- Ownership: a promoter can only write their own patients and visits.
"""

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.schemas import (
    AssessmentPayload,
    EncounterPayload,
    Operation,
    OperationResult,
    PatientPayload,
)
from triage.db.models import Assessment, Encounter, Patient, ProcessedOperation, User

T = TypeVar("T", bound=BaseModel)


class _Rejected(Exception):
    def __init__(self, code: str, status: str = "rejected") -> None:
        self.code = code
        self.status = status


async def apply_operations(
    session: AsyncSession, user: User, operations: list[Operation]
) -> list[OperationResult]:
    results = []
    for operation in operations:
        results.append(await _apply_one(session, user, operation))
    await session.commit()
    return results


async def _apply_one(session: AsyncSession, user: User, operation: Operation) -> OperationResult:
    seen = await session.get(ProcessedOperation, operation.id)
    if seen is not None:
        if seen.user_id != user.id:
            return _result(operation, "rejected", "FORBIDDEN")
        return _result(operation, "duplicate", server_version=seen.server_version)

    try:
        async with session.begin_nested():
            outcome = await _dispatch(session, user, operation)
            if outcome.status == "applied":
                session.add(
                    ProcessedOperation(
                        id=operation.id,
                        user_id=user.id,
                        entity=operation.entity,
                        entity_id=operation.entity_id,
                        status=outcome.status,
                        server_version=outcome.server_version,
                    )
                )
                await session.flush()
            return outcome
    except _Rejected as rejection:
        return _result(operation, rejection.status, rejection.code)  # type: ignore[arg-type]
    except ValidationError:
        return _result(operation, "rejected", "VALIDATION_FAILED")
    except IntegrityError:
        # The same operation arrived twice at the same time; the other request won.
        return _result(operation, "duplicate")


async def _dispatch(session: AsyncSession, user: User, operation: Operation) -> OperationResult:
    match operation.entity:
        case "patient":
            return await _apply_patient(session, user, operation)
        case "encounter":
            return await _apply_encounter(session, user, operation)
        case "assessment":
            return await _apply_assessment(session, user, operation)


def _parse(model: type[T], operation: Operation) -> T:
    payload = model.model_validate(operation.payload)
    entity_id = getattr(payload, "id", None) or getattr(payload, "encounter_id", None)
    if entity_id != operation.entity_id:
        raise _Rejected("ENTITY_ID_MISMATCH")
    return payload


async def _apply_patient(
    session: AsyncSession, user: User, operation: Operation
) -> OperationResult:
    data = _parse(PatientPayload, operation)
    patient = await session.get(Patient, data.id, with_for_update=True)
    if patient is not None and patient.promoter_id != user.id:
        raise _Rejected("FORBIDDEN")

    if patient is None:
        session.add(
            Patient(
                id=data.id,
                code=data.code,
                birth_year=data.birth_year,
                sex=data.sex,
                community=data.community.strip(),
                promoter_id=user.id,
                created_at=data.created_at,
                updated_at=data.updated_at,
            )
        )
    elif operation.type == "update":
        patient.birth_year = data.birth_year
        patient.sex = data.sex
        patient.community = data.community.strip()
        patient.updated_at = data.updated_at
    return _result(operation, "applied")


async def _apply_encounter(
    session: AsyncSession, user: User, operation: Operation
) -> OperationResult:
    data = _parse(EncounterPayload, operation)
    # Row lock: two devices editing the same visit are processed one at a time.
    encounter = await session.get(Encounter, data.id, with_for_update=True)
    if encounter is not None and encounter.promoter_id != user.id:
        raise _Rejected("FORBIDDEN")

    if encounter is None:
        if operation.type == "update":
            # The create is still on its way (or failed): ask the phone to retry later.
            raise _Rejected("ENCOUNTER_NOT_FOUND", status="retry")
        patient = await session.get(Patient, data.patient_id)
        if patient is None:
            raise _Rejected("PATIENT_NOT_FOUND", status="retry")
        if patient.promoter_id != user.id:
            raise _Rejected("FORBIDDEN")
        session.add(
            Encounter(
                id=data.id,
                patient_id=data.patient_id,
                promoter_id=user.id,
                symptoms=data.symptoms.strip(),
                alarms=data.alarms.model_dump(by_alias=True),
                consent_given=True,
                version=data.version,
                created_at=data.created_at,
                updated_at=data.updated_at,
            )
        )
        return _result(operation, "applied", server_version=data.version)

    if operation.type == "create":
        # Already created by an earlier attempt that lost its acknowledgement.
        return _result(operation, "applied", server_version=encounter.version)

    if operation.base_version is None:
        raise _Rejected("BASE_VERSION_REQUIRED")
    if encounter.version != operation.base_version or data.version <= encounter.version:
        # Someone changed the visit since this edit started: a person must decide.
        return _result(
            operation,
            "conflict",
            "VERSION_CONFLICT",
            server_version=encounter.version,
            server_record=encounter_to_record(encounter),
        )

    encounter.symptoms = data.symptoms.strip()
    encounter.alarms = data.alarms.model_dump(by_alias=True)
    encounter.version = data.version
    encounter.updated_at = data.updated_at
    return _result(operation, "applied", server_version=encounter.version)


async def _apply_assessment(
    session: AsyncSession, user: User, operation: Operation
) -> OperationResult:
    data = _parse(AssessmentPayload, operation)
    encounter = await session.get(Encounter, data.encounter_id)
    if encounter is None:
        raise _Rejected("ENCOUNTER_NOT_FOUND", status="retry")
    if encounter.promoter_id != user.id:
        raise _Rejected("FORBIDDEN")

    assessment = await session.scalar(
        select(Assessment).where(Assessment.encounter_id == data.encounter_id).with_for_update()
    )
    if assessment is not None and assessment.encounter_version > data.encounter_version:
        # Result for an older version of the visit: keep the newer one.
        return _result(operation, "applied", "STALE_IGNORED")

    fields: dict[str, Any] = {
        "encounter_version": data.encounter_version,
        "priority": data.priority,
        "source": data.source,
        "triggered_alarms": data.triggered_alarms,
        "reason_codes": data.reason_codes,
        "model_version": data.model_version,
        "final_priority": data.final_priority,
        "review_note": data.review_note,
        "reviewed_at": data.reviewed_at,
        "created_at": data.created_at,
    }
    if assessment is None:
        session.add(Assessment(encounter_id=data.encounter_id, **fields))
    else:
        for name, value in fields.items():
            setattr(assessment, name, value)
    return _result(operation, "applied")


def encounter_to_record(encounter: Encounter) -> dict[str, Any]:
    """The server's copy in the same shape the phone stores."""
    return {
        "id": str(encounter.id),
        "patientId": str(encounter.patient_id),
        "symptoms": encounter.symptoms,
        "alarms": encounter.alarms,
        "consentGiven": encounter.consent_given,
        "createdAt": encounter.created_at.isoformat(),
        "updatedAt": encounter.updated_at.isoformat(),
        "version": encounter.version,
    }


def _result(
    operation: Operation,
    status: str,
    code: str | None = None,
    *,
    server_version: int | None = None,
    server_record: dict[str, Any] | None = None,
) -> OperationResult:
    return OperationResult(
        operation_id=operation.id,
        status=status,  # type: ignore[arg-type]
        code=code,
        server_version=server_version,
        server_record=server_record,
    )
