"""Guards the exported model the Angular app ships: it must keep its safety bar."""

import json
from pathlib import Path

ARTIFACTS = Path(__file__).parents[1] / "ml" / "artifacts"
CARD = json.loads((ARTIFACTS / "model_card.json").read_text(encoding="utf-8"))
V1, V2 = list(CARD["comparison_same_data"].values())


def test_the_model_keeps_a_high_recall_for_urgent_cases() -> None:
    assert V2["held_out_scenarios"]["recall"]["high"] >= 0.95
    assert V2["typo_stress"]["recall"]["high"] >= 0.95


def test_v2_handles_denied_findings_better_than_v1() -> None:
    assert V2["negation_stress_false_high_rate"] <= 0.1
    assert V2["negation_stress_false_high_rate"] < V1["negation_stress_false_high_rate"]


def test_the_model_is_calibrated() -> None:
    calibration = CARD["calibration"]
    assert calibration["expected_calibration_error_after"] < 0.03
    assert (
        calibration["expected_calibration_error_after"]
        <= calibration["expected_calibration_error_before"]
    )


def test_the_exported_model_is_consistent() -> None:
    model = json.loads((ARTIFACTS / "triage-text-v2.json").read_text(encoding="utf-8"))

    size = len(model["vocabulary"])
    assert sorted(model["classes"]) == ["high", "low", "medium"]
    assert len(model["idf"]) == size
    assert all(len(row) == size for row in model["coef"])
    assert 0 < model["highThreshold"] < 1
    assert model["temperature"] > 0
    assert model["unknownFallback"] == "medium"
