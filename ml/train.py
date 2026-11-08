"""Trains the on-device triage text model and exports it for the browser.

Run from the repository root:  uv run python -m ml.train
Outputs (ml/artifacts/):
  triage-text-v1.json        weights and vocabulary, copied to the Angular app
  model_card.json            data, metrics, intended use and limitations
  equivalence-fixtures.json  texts with the exact probabilities Python computes;
                             the TypeScript tests must reproduce them
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

from ml.dataset import LABELS, Example, generate, split_by_scenario

ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_VERSION = "triage-text-v1"
# Missing an urgent case is far worse than a false alarm.
TARGET_HIGH_RECALL = 0.95
DECIMALS = 6

# Deliberately simple preprocessing, so TypeScript can reproduce it exactly:
# lowercase, strip accents (NFKD without combining marks), words of 2+ ASCII letters.
TOKEN_PATTERN = r"[a-z]{2,}"


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        norm="l2",
        smooth_idf=True,
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000)


def decide(probabilities: np.ndarray, classes: list[str], high_threshold: float) -> list[str]:
    """High when its probability reaches the threshold; otherwise the likelier of the rest."""
    high = classes.index("high")
    others = [i for i in range(len(classes)) if i != high]
    decisions = []
    for row in probabilities:
        if row[high] >= high_threshold:
            decisions.append("high")
        else:
            decisions.append(classes[max(others, key=lambda i: row[i])])
    return decisions


def choose_high_threshold(train: list[Example]) -> float:
    """Highest threshold reaching the target recall in cross-validation (by scenario)."""
    texts = [e.text for e in train]
    labels = np.array([e.label for e in train])
    groups = [e.scenario for e in train]
    probabilities = np.zeros((len(train), len(LABELS)))
    classes: list[str] = []
    for fit_index, eval_index in GroupKFold(n_splits=5).split(texts, labels, groups):
        vectorizer = build_vectorizer()
        model = build_classifier().fit(
            vectorizer.fit_transform([texts[i] for i in fit_index]), labels[fit_index]
        )
        classes = list(model.classes_)
        probabilities[eval_index] = model.predict_proba(
            vectorizer.transform([texts[i] for i in eval_index])
        )
    for threshold in np.arange(0.60, 0.04, -0.01):
        predicted = np.array(decide(probabilities, classes, float(threshold)))
        if recall_score(labels, predicted, labels=["high"], average="macro") >= TARGET_HIGH_RECALL:
            return round(float(threshold), 2)
    return 0.05


def evaluate(truth: list[str], predicted: list[str]) -> dict[str, object]:
    labels = list(LABELS)
    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "recall": dict(
            zip(
                labels,
                _rounded(recall_score(truth, predicted, labels=labels, average=None)),
                strict=True,
            )
        ),
        "precision": dict(
            zip(
                labels,
                _rounded(
                    precision_score(truth, predicted, labels=labels, average=None, zero_division=0)
                ),
                strict=True,
            )
        ),
        "macro_f1": round(float(f1_score(truth, predicted, labels=labels, average="macro")), 3),
    }


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in values]


def export_model(
    vectorizer: TfidfVectorizer, model: LogisticRegression, threshold: float
) -> dict[str, object]:
    terms = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.__getitem__)
    return {
        "version": MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": list(model.classes_),
        "highThreshold": threshold,
        "preprocessing": {
            "lowercase": True,
            "stripAccents": "unicode",
            "tokenPattern": TOKEN_PATTERN,
            "ngramRange": [1, 2],
            "sublinearTf": True,
            "norm": "l2",
        },
        "vocabulary": terms,
        "idf": [round(float(v), DECIMALS) for v in vectorizer.idf_],
        "coef": [[round(float(v), DECIMALS) for v in row] for row in model.coef_],
        "intercept": [round(float(v), DECIMALS) for v in model.intercept_],
    }


def exported_probabilities(
    exported: dict, texts: list[str], vectorizer: TfidfVectorizer
) -> np.ndarray:
    """Probabilities computed from the ROUNDED exported weights: what the browser will see."""
    coef = np.array(exported["coef"])
    intercept = np.array(exported["intercept"])
    idf = np.array(exported["idf"])
    analyzer = vectorizer.build_analyzer()  # same tokenization as training
    index = {term: i for i, term in enumerate(exported["vocabulary"])}
    rows = []
    for text in texts:
        tf = np.zeros(len(index))
        for token in analyzer(text):
            if token in index:
                tf[index[token]] += 1
        weights = np.where(tf > 0, 1 + np.log(np.where(tf > 0, tf, 1)), 0) * idf
        norm = np.linalg.norm(weights)
        x = weights / norm if norm > 0 else weights
        logits = coef @ x + intercept
        exp = np.exp(logits - logits.max())
        rows.append(exp / exp.sum())
    return np.array(rows)


def main() -> None:
    examples = generate()
    train, test = split_by_scenario(examples)
    threshold = choose_high_threshold(train)

    vectorizer = build_vectorizer()
    model = build_classifier().fit(
        vectorizer.fit_transform([e.text for e in train]), [e.label for e in train]
    )
    classes = list(model.classes_)
    exported = export_model(vectorizer, model, threshold)

    test_texts = [e.text for e in test]
    truth = [e.label for e in test]
    probabilities = exported_probabilities(exported, test_texts, vectorizer)
    predicted = decide(probabilities, classes, threshold)

    majority = max(LABELS, key=[e.label for e in train].count)
    metrics = {
        "model": evaluate(truth, predicted),
        "baseline_majority_class": evaluate(truth, [majority] * len(truth)),
        # What the app did before the model: without alarm signs, always "medium".
        "baseline_rules_only": evaluate(truth, ["medium"] * len(truth)),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{MODEL_VERSION}.json").write_text(
        json.dumps(exported, separators=(",", ":")) + "\n"
    )

    card = {
        "model": "On-device text classifier for rural triage priority",
        "version": MODEL_VERSION,
        "algorithm": "TF-IDF (unigrams + bigrams) + multinomial logistic regression",
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
            "train_scenarios": len({e.scenario for e in train}),
            "test_scenarios": len({e.scenario for e in test}),
        },
        "decision_rule": f"high if P(high) >= {threshold}, else the likelier of medium and low",
        "high_threshold_selection": (
            f"Highest threshold reaching recall(high) >= {TARGET_HIGH_RECALL} in 5-fold "
            "cross-validation grouped by scenario, on training data only"
        ),
        "vocabulary_size": len(exported["vocabulary"]),
        "metrics_on_held_out_scenarios": metrics,
        "limitations": [
            "Synthetic data written by the developer: real promoter notes will differ.",
            "Not validated by health professionals; the scale is not an official triage scale.",
            "Bag-of-words model: it can miss negations beyond bigrams ('no tiene ... fiebre').",
            "Unknown words are ignored: very unusual descriptions fall back to the priors.",
        ],
        "sklearn_version": sklearn.__version__,
    }
    (ARTIFACTS / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n"
    )

    sample = random.Random(1).sample(test, 40) + [
        Example("", "medium", "edge", "edge"),
        Example("¡¡ÑOÑO!! Árbol 123 ñandú", "medium", "edge", "edge"),
    ]
    fixture_probabilities = exported_probabilities(exported, [e.text for e in sample], vectorizer)
    fixtures = [
        {"text": e.text, "probabilities": [round(float(p), 8) for p in row]}
        for e, row in zip(sample, fixture_probabilities, strict=True)
    ]
    (ARTIFACTS / "equivalence-fixtures.json").write_text(
        json.dumps(
            {"version": MODEL_VERSION, "classes": classes, "cases": fixtures},
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )

    model_metrics = metrics["model"]
    print(f"Model {MODEL_VERSION}: {len(exported['vocabulary'])} terms, high threshold {threshold}")
    print(f"  recall    {model_metrics['recall']}")
    print(f"  precision {model_metrics['precision']}")
    print(f"  macro F1  {model_metrics['macro_f1']}")
    print(f"  confusion {model_metrics['confusion_matrix']}  (rows truth, cols {LABELS})")
    print(f"  baseline rules-only macro F1 {metrics['baseline_rules_only']['macro_f1']}")


if __name__ == "__main__":
    main()
