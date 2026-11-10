"""The text features the browser reproduces: denial, inability and positions."""

from ml.features import analyze, tokenize


def negated(text: str) -> dict[str, bool]:
    return {token.text: token.negated for token in tokenize(text)}


def test_sin_and_niega_always_deny() -> None:
    assert negated("sin rigidez en el cuello")["cuello"] is True
    assert negated("niega dolor de pecho")["pecho"] is True


def test_a_plain_no_denies_only_before_a_verb_of_presence() -> None:
    assert negated("no tiene fiebre")["fiebre"] is True
    assert negated("no se ha desmayado")["desmayado"] is True
    # "no puede" describes the symptom itself.
    assert negated("no puede mover el brazo")["brazo"] is False


def test_a_denial_ends_with_the_clause() -> None:
    marks = negated("no tiene fiebre, pero vomita")
    assert marks["fiebre"] is True
    assert marks["vomita"] is False


def test_positions_point_into_the_original_text() -> None:
    text = "Niega DOLOR en el pecho"
    token = tokenize(text)[0]
    assert text[token.start : token.end] == "DOLOR"


def test_denied_function_words_and_pieces_weigh_nothing() -> None:
    features = analyze("sin rigidez en el cuello")
    assert "no_el" not in features
    assert not any(feature.startswith("c:") for feature in features)


def test_typos_share_letter_pieces_with_the_right_word() -> None:
    right = {f for f in analyze("fiebre") if f.startswith("c:")}
    typo = {f for f in analyze("fievre") if f.startswith("c:")}
    assert right & typo
