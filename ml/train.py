"""Trains the on-device triage text model (v2) and exports it for the browser.

Run from the repository root:  uv run python -m ml.train
Outputs (ml/artifacts/):
  triage-text-v2.json        weights, vocabulary and temperature, copied to the Angular app
  model_card.json            data, metrics (v2 against the v1 method), intended use, limits
  equivalence-fixtures.json  texts with the exact features and probabilities Python
                             computes; the TypeScript tests must reproduce them

v2 against v1, trained on the same data and judged on the same held-out scenarios:
  * negation-aware features and four-letter pieces of words (ml/features.py);
  * temperature scaling, so "certainty" means what it says (expected calibration error);
  * two stress tests with phrasings never seen in training: negated urgent findings
    and typos.
"""

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from ml.dataset import (
    LABELS,
    Example,
    generate,
    negation_stress,
    split_by_scenario,
    typo_stress,
)
from ml.features import (
    CHAR_N,
    CLAUSE_BREAKS,
    NEGATION_SCOPE,
    NEGATORS,
    PRESENCE_VERBS,
    SCOPE_FREE,
    analyze,
)

ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_VERSION = "triage-text-v2"
# Missing an urgent case is far worse than a false alarm.
TARGET_HIGH_RECALL = 0.95
MAX_FEATURES = 6000
UNKNOWN_FALLBACK = "medium"
DECIMALS = 6


def v2_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyze,
        sublinear_tf=True,
        min_df=2,
        max_features=MAX_FEATURES,
        norm="l2",
        smooth_idf=True,
    )


def v1_vectorizer() -> TfidfVectorizer:
    """The previous method, kept only as a baseline."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"[a-z]{2,}",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        norm="l2",
        smooth_idf=True,
    )


def classifier() -> LogisticRegression:
    return LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def decide(probabilities: np.ndarray, classes: list[str], high_threshold: float) -> list[str]:
    """High when its probability reaches the threshold; otherwise the likelier of the rest."""
    high = classes.index("high")
    others = [i for i in range(len(classes)) if i != high]
    return [
        "high" if row[high] >= high_threshold else classes[max(others, key=lambda i: row[i])]
        for row in probabilities
    ]


def out_of_fold_logits(make_vectorizer, train: list[Example]) -> tuple[np.ndarray, list[str]]:
    """Logits for every training example from a model that never saw its scenario."""
    texts = [e.text for e in train]
    labels = np.array([e.label for e in train])
    groups = [e.scenario for e in train]
    logits = np.zeros((len(train), len(LABELS)))
    classes: list[str] = []
    for fit_index, eval_index in GroupKFold(n_splits=5).split(texts, labels, groups):
        vectorizer = make_vectorizer()
        model = classifier().fit(
            vectorizer.fit_transform([texts[i] for i in fit_index]), labels[fit_index]
        )
        classes = list(model.classes_)
        logits[eval_index] = model.decision_function(
            vectorizer.transform([texts[i] for i in eval_index])
        )
    return logits, classes


def fit_temperature(logits: np.ndarray, labels: list[str], classes: list[str]) -> float:
    """Single scalar that minimises the negative log-likelihood (temperature scaling)."""
    index = np.array([classes.index(label) for label in labels])
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.3, 6.01, 0.05):
        p = softmax(logits, float(t))[np.arange(len(index)), index]
        nll = float(-np.mean(np.log(np.clip(p, 1e-12, 1))))
        if nll < best_nll:
            best_t, best_nll = round(float(t), 2), nll
    return best_t


def choose_high_threshold(
    probabilities: np.ndarray, labels: list[str], classes: list[str]
) -> float:
    """Highest threshold reaching the target recall of "high" on out-of-fold data."""
    truth = np.array(labels)
    for threshold in np.arange(0.70, 0.04, -0.01):
        predicted = np.array(decide(probabilities, classes, float(threshold)))
        if recall_score(truth, predicted, labels=["high"], average="macro") >= TARGET_HIGH_RECALL:
            return round(float(threshold), 2)
    return 0.05


def expected_calibration_error(
    probabilities: np.ndarray, labels: list[str], classes: list[str], bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """How far the stated confidence is from the observed accuracy (0 is perfect)."""
    confidence = probabilities.max(axis=1)
    correct = np.array([classes[i] for i in probabilities.argmax(axis=1)]) == np.array(labels)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            ece += gap * mask.mean()
            table.append(
                {
                    "from": round(float(low), 1),
                    "to": round(float(high), 1),
                    "count": int(mask.sum()),
                    "confidence": round(float(confidence[mask].mean()), 3),
                    "accuracy": round(float(correct[mask].mean()), 3),
                }
            )
    return round(float(ece), 4), table


def evaluate(truth: list[str], predicted: list[str]) -> dict[str, object]:
    labels = list(LABELS)
    rounded = lambda values: [round(float(v), 3) for v in values]  # noqa: E731
    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "recall": dict(
            zip(
                labels,
                rounded(recall_score(truth, predicted, labels=labels, average=None)),
                strict=True,
            )
        ),
        "precision": dict(
            zip(
                labels,
                rounded(
                    precision_score(truth, predicted, labels=labels, average=None, zero_division=0)
                ),
                strict=True,
            )
        ),
        "macro_f1": round(float(f1_score(truth, predicted, labels=labels, average="macro")), 3),
    }


class Method:
    """A complete method: features, weights, temperature and decision threshold."""

    def __init__(self, name: str, make_vectorizer, calibrate: bool, train: list[Example]):
        self.name = name
        labels = [e.label for e in train]
        logits, classes = out_of_fold_logits(make_vectorizer, train)
        self.temperature = fit_temperature(logits, labels, classes) if calibrate else 1.0
        oof = softmax(logits, self.temperature)
        self.threshold = choose_high_threshold(oof, labels, classes)
        self.vectorizer = make_vectorizer()
        self.model = classifier().fit(
            self.vectorizer.fit_transform([e.text for e in train]), labels
        )
        self.classes = list(self.model.classes_)

    def probabilities(self, texts: list[str]) -> np.ndarray:
        return softmax(
            self.model.decision_function(self.vectorizer.transform(texts)), self.temperature
        )

    def predict(self, texts: list[str]) -> list[str]:
        decisions = decide(self.probabilities(texts), self.classes, self.threshold)
        # No known feature at all: the model has nothing to say (the app shows a
        # provisional "medium" and asks the promoter to describe more).
        known = self.vectorizer.transform(texts).getnnz(axis=1)
        return [d if k > 0 else UNKNOWN_FALLBACK for d, k in zip(decisions, known, strict=True)]


def export_model(method: Method) -> dict[str, object]:
    vocabulary = method.vectorizer.vocabulary_
    terms = sorted(vocabulary, key=vocabulary.__getitem__)
    return {
        "version": MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": method.classes,
        "highThreshold": method.threshold,
        "temperature": method.temperature,
        "unknownFallback": UNKNOWN_FALLBACK,
        "features": {
            "negators": sorted(NEGATORS),
            "clauseBreaks": sorted(CLAUSE_BREAKS),
            "presenceVerbs": sorted(PRESENCE_VERBS),
            "negationScope": NEGATION_SCOPE,
            "scopeFree": sorted(SCOPE_FREE),
            "charN": CHAR_N,
            "sublinearTf": True,
            "norm": "l2",
        },
        "vocabulary": terms,
        "idf": [round(float(v), DECIMALS) for v in method.vectorizer.idf_],
        "coef": [[round(float(v), DECIMALS) for v in row] for row in method.model.coef_],
        "intercept": [round(float(v), DECIMALS) for v in method.model.intercept_],
    }


def exported_probabilities(exported: dict, texts: list[str]) -> np.ndarray:
    """Probabilities from the ROUNDED exported weights: exactly what the browser computes."""
    coef = np.array(exported["coef"])
    intercept = np.array(exported["intercept"])
    idf = np.array(exported["idf"])
    index = {term: i for i, term in enumerate(exported["vocabulary"])}
    rows = []
    for text in texts:
        tf = np.zeros(len(index))
        for feature in analyze(text):
            if feature in index:
                tf[index[feature]] += 1
        weights = np.where(tf > 0, 1 + np.log(np.where(tf > 0, tf, 1)), 0) * idf
        norm = np.linalg.norm(weights)
        x = weights / norm if norm > 0 else weights
        rows.append(coef @ x + intercept)
    return softmax(np.array(rows), exported["temperature"])


def false_high_rate(predicted: list[str]) -> float:
    return round(sum(p == "high" for p in predicted) / max(1, len(predicted)), 3)


def main() -> None:
    examples = generate()
    train, test = split_by_scenario(examples)
    negated = negation_stress()
    typos = typo_stress(test)

    v1 = Method("v1: palabras y pares de palabras", v1_vectorizer, calibrate=False, train=train)
    v2 = Method(
        "v2: negación + fragmentos de letras + temperatura",
        v2_vectorizer,
        calibrate=True,
        train=train,
    )
    v2_raw = Method("v2 sin calibrar", v2_vectorizer, calibrate=False, train=train)

    truth = [e.label for e in test]
    typo_truth = [e.label for e in typos]
    test_texts = [e.text for e in test]
    report = {}
    for method in (v1, v2):
        report[method.name] = {
            "held_out_scenarios": evaluate(truth, method.predict(test_texts)),
            "negation_stress_false_high_rate": false_high_rate(
                method.predict([e.text for e in negated])
            ),
            "typo_stress": evaluate(typo_truth, method.predict([e.text for e in typos])),
        }
    ece_raw, _ = expected_calibration_error(v2_raw.probabilities(test_texts), truth, v2_raw.classes)
    ece, reliability = expected_calibration_error(v2.probabilities(test_texts), truth, v2.classes)

    exported = export_model(v2)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for old in ARTIFACTS.glob("triage-text-v*.json"):
        old.unlink()
    (ARTIFACTS / f"{MODEL_VERSION}.json").write_text(
        json.dumps(exported, separators=(",", ":")) + "\n"
    )

    majority = max(LABELS, key=[e.label for e in train].count)
    card = {
        "model": "On-device text classifier for rural triage priority",
        "version": MODEL_VERSION,
        "algorithm": (
            "TF-IDF over negation-aware words, word pairs and four-letter pieces "
            "+ multinomial logistic regression + temperature scaling"
        ),
        "intended_use": (
            "Suggest a priority for visits WITHOUT alarm signs, inside a Web Worker, offline. "
            "Alarm-sign rules always take precedence. A health promoter confirms every suggestion."
        ),
        "not_for": "Diagnosis or any real clinical decision. Academic prototype.",
        "training_data": {
            "source": "Synthetic, fictitious Spanish descriptions generated by ml/dataset.py",
            "examples": len(examples),
            "train_examples": len(train),
            "test_examples": len(test),
            "split": "By scenario: test scenarios never appear in training",
            "negation_stress_examples": len(negated),
            "typo_stress_examples": len(typos),
        },
        "decision_rule": f"high if P(high) >= {v2.threshold}, else the likelier of medium and low",
        "temperature": v2.temperature,
        "calibration": {
            "expected_calibration_error_before": ece_raw,
            "expected_calibration_error_after": ece,
            "reliability": reliability,
        },
        "vocabulary_size": len(exported["vocabulary"]),
        "comparison_same_data": report,
        "baseline_majority_class": evaluate(truth, [majority] * len(truth)),
        "limitations": [
            "Synthetic data written by the developer: real promoter notes will differ.",
            "Not validated by health professionals; the scale is not an official triage scale.",
            "Negation is handled by rules (a word list and a four-word scope), not by "
            "understanding: unusual phrasings can still fool it.",
            "Unknown words are ignored: very unusual descriptions fall back to the priors.",
        ],
        "sklearn_version": sklearn.__version__,
    }
    (ARTIFACTS / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n"
    )

    sample = (
        random.Random(1).sample(test, 30)
        + random.Random(2).sample(negated, 8)
        + random.Random(3).sample(typos, 8)
        + [
            Example("", "medium", "edge", "edge"),
            Example("¡¡ÑOÑO!! Árbol 123 ñandú", "medium", "edge", "edge"),
            Example("No tiene fiebre, PERO vomita y no come", "medium", "edge", "edge"),
        ]
    )
    texts = [e.text for e in sample]
    probabilities = exported_probabilities(exported, texts)
    fixtures = [
        {
            "text": text,
            "features": analyze(text),
            "probabilities": [round(float(p), 8) for p in row],
        }
        for text, row in zip(texts, probabilities, strict=True)
    ]
    (ARTIFACTS / "equivalence-fixtures.json").write_text(
        json.dumps(
            {"version": MODEL_VERSION, "classes": v2.classes, "cases": fixtures},
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )

    size = len(exported["vocabulary"])
    print(f"Model {MODEL_VERSION}: {size} features, T={v2.temperature}, high>={v2.threshold}")
    for name, metrics in report.items():
        held = metrics["held_out_scenarios"]
        typo = metrics["typo_stress"]
        print(f"  {name}")
        print(f"    held-out   recall {held['recall']}  macro F1 {held['macro_f1']}")
        print(f"    negations  false 'high' rate {metrics['negation_stress_false_high_rate']}")
        print(f"    typos      recall high {typo['recall']['high']}  macro F1 {typo['macro_f1']}")
    print(f"  calibration ECE {ece_raw} -> {ece}")


if __name__ == "__main__":
    main()
