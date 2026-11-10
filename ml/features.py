"""Text features of the triage model, version 2.

The browser runs the same code in TypeScript (frontend/src/app/core/triage/text-model.ts),
so it is deliberately small and explicit; an equivalence test compares both.

What changed from v1 (word unigrams and bigrams only):

* Negation. "no tiene fiebre" and "sin dificultad para respirar" used to look like
  fever and breathing difficulty. Now the words after a denial, up to the end of the
  clause and at most four content words, are marked with "no_".
  "sin", "niega", "nunca", "ni", "tampoco" always deny. A plain "no" denies only before
  a verb of presence ("no tiene", "no presenta", "no hay", "no se ha..."): in
  "no puede hablar" or "no orina" the "no" describes the symptom itself.
* Typos. Each affirmed word also contributes its four-letter pieces
  ("<fiebre>" -> "<fie", "fieb", ...), so "fievre" or "dolr" still share most pieces.
* Positions. Every token keeps where it is in the original text, so the app can
  highlight the words that pushed the decision.
"""

import re
import unicodedata
from dataclasses import dataclass

NEGATORS = frozenset({"sin", "niega", "nunca", "ni", "tampoco", "ningun", "ninguna"})
# After a plain "no", these words mean "absent": "no tiene", "no presenta", "no hay".
PRESENCE_VERBS = frozenset(
    {
        "tiene",
        "tuvo",
        "presenta",
        "hay",
        "ha",
        "han",
        "refiere",
        "reporta",
        "siente",
        "sufre",
        "muestra",
        "esta",
        "estuvo",
        "es",
    }
)
# A negation stops at punctuation or at these words: "no tiene fiebre pero vomita".
CLAUSE_BREAKS = frozenset({"y", "pero", "aunque", "mas", "luego", "ahora"})
NEGATION_SCOPE = 4
# Short function words stay inside a negation but do not use it up:
# in "niega que se le torcio la boca" the denial still reaches "boca".
SCOPE_FREE = frozenset(
    {
        "que",
        "se",
        "le",
        "lo",
        "la",
        "el",
        "los",
        "las",
        "un",
        "una",
        "de",
        "del",
        "al",
        "en",
        "su",
        "sus",
        "mi",
        "muy",
        "ha",
        "han",
        "haya",
        "tiene",
        "tenido",
        "presenta",
    }
)
CHAR_N = 4

_TOKEN = re.compile(r"[a-z]+|[.,;:!?\n]")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    negated: bool
    start: int  # index of the first character in the original text (code points)
    end: int


def normalize(text: str) -> str:
    """Lowercase without accents, one output character per input character.

    Keeping the length lets token positions point into the text the promoter typed.
    """
    out = []
    for char in text:
        base = unicodedata.normalize("NFKD", char.lower())
        base = "".join(c for c in base if not unicodedata.combining(c))
        out.append(base[:1] if base else " ")
    return "".join(out)


def _denies(words: list[str], i: int) -> bool:
    """Whether the word at i opens a denial."""
    if words[i] in NEGATORS:
        return True
    if words[i] != "no" or i + 1 >= len(words):
        return False
    following = words[i + 1]
    if following in PRESENCE_VERBS:
        return True
    # "no se ha desmayado", "no le ha dado fiebre"
    return following in {"se", "le", "lo"} and i + 2 < len(words) and words[i + 2] in PRESENCE_VERBS


def tokenize(text: str) -> list[Token]:
    matches = list(_TOKEN.finditer(normalize(text)))
    words = [m.group() for m in matches]
    tokens: list[Token] = []
    scope = 0
    for i, match in enumerate(matches):
        word = words[i]
        if not word.isalpha() or word in CLAUSE_BREAKS:
            scope = 0
            continue
        if _denies(words, i):
            scope = NEGATION_SCOPE
            continue
        if len(word) < 2:
            continue
        tokens.append(Token(word, scope > 0, match.start(), match.end()))
        if word not in SCOPE_FREE:
            scope = max(0, scope - 1)
    return tokens


def features(tokens: list[Token]) -> list[tuple[str, tuple[int, ...]]]:
    """(feature, indexes of the tokens it comes from), in a fixed order.

    A denied finding should weigh nothing rather than something: negated function
    words ("no_el") and letter pieces of negated words are left out, so a denial only
    counts through the content words the model has actually learned in negated form.
    """
    kept = [
        (("no_" if t.negated else "") + t.text, i)
        for i, t in enumerate(tokens)
        if not (t.negated and t.text in SCOPE_FREE)
    ]
    out: list[tuple[str, tuple[int, ...]]] = [(word, (i,)) for word, i in kept]
    for (w1, i1), (w2, i2) in zip(kept, kept[1:], strict=False):
        out.append((f"{w1} {w2}", (i1, i2)))
    for i, token in enumerate(tokens):
        if token.negated:
            continue
        padded = f"<{token.text}>"
        for k in range(len(padded) - CHAR_N + 1):
            out.append(("c:" + padded[k : k + CHAR_N], (i,)))
    return out


def analyze(text: str) -> list[str]:
    """Analyzer for scikit-learn's TfidfVectorizer."""
    return [feature for feature, _ in features(tokenize(text))]
