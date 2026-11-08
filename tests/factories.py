"""Outbox operations exactly as the Angular app builds them."""

import uuid
from datetime import UTC, datetime
from typing import Any

NO_ALARMS = {
    "breathingDifficulty": False,
    "lossOfConsciousness": False,
    "chestPain": False,
    "severeBleeding": False,
    "seizures": False,
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def operation(
    entity: str, kind: str, entity_id: str, payload: dict[str, Any], base_version: int | None = None
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "entity": entity,
        "entityId": entity_id,
        "type": kind,
        "payload": payload,
        "baseVersion": base_version,
        "createdAt": now(),
        "attempts": 0,
    }


def patient(patient_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    patient_id = patient_id or str(uuid.uuid4())
    return {
        "id": patient_id,
        "code": f"P-{patient_id[:6].upper()}",
        "birthYear": 1980,
        "sex": "female",
        "community": "Vereda El Encano",
        "createdAt": now(),
        "updatedAt": now(),
        **overrides,
    }


def encounter(patient_id: str, encounter_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    return {
        "id": encounter_id or str(uuid.uuid4()),
        "patientId": patient_id,
        "symptoms": "Fiebre de tres días y tos con flema",
        "alarms": dict(NO_ALARMS),
        "consentGiven": True,
        "createdAt": now(),
        "updatedAt": now(),
        "version": 1,
        **overrides,
    }


def assessment(
    encounter_id: str, version: int = 1, priority: str = "medium", **overrides: Any
) -> dict[str, Any]:
    return {
        "encounterId": encounter_id,
        "encounterVersion": version,
        "priority": priority,
        "source": "model",
        "triggeredAlarms": [],
        "reasonCodes": ["MODEL_PREDICTION"],
        "modelVersion": "triage-text-v1",
        "elapsedMs": 4,
        "createdAt": now(),
        **overrides,
    }


def new_visit() -> tuple[str, str, list[dict[str, Any]]]:
    """A new patient with their first visit and its assessment, as three operations."""
    p = patient()
    e = encounter(p["id"])
    ops = [
        operation("patient", "create", p["id"], p),
        operation("encounter", "create", e["id"], e),
        operation("assessment", "create", e["id"], assessment(e["id"])),
    ]
    return p["id"], e["id"], ops
