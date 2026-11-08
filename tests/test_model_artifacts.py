"""Guards the exported model the Angular app ships: it must keep its safety bar."""

import json
from pathlib import Path

ARTIFACTS = Path(__file__).parents[1] / "ml" / "artifacts"


def test_the_model_keeps_a_high_recall_for_urgent_cases() -> None:
    card = json.loads((ARTIFACTS / "model_card.json").read_text(encoding="utf-8"))

    metrics = card["metrics_on_held_out_scenarios"]
    assert metrics["model"]["recall"]["high"] >= 0.9
    assert metrics["model"]["macro_f1"] > metrics["baseline_rules_only"]["macro_f1"]


def test_the_exported_model_is_consistent() -> None:
    model = json.loads((ARTIFACTS / "triage-text-v1.json").read_text(encoding="utf-8"))

    size = len(model["vocabulary"])
    assert sorted(model["classes"]) == ["high", "low", "medium"]
    assert len(model["idf"]) == size
    assert all(len(row) == size for row in model["coef"])
    assert 0 < model["highThreshold"] < 1
