"""Synthetic, documented triage dataset (fictitious, Spanish, rural Nariño phrasing).

Honesty for the defense: there is no real clinical dataset here.

Each class has scenario *families* (for example "possible stroke"). A family is
described by COMPONENTS: groups of interchangeable phrases for one clinical clue
("the mouth is twisted", "loses strength on one side"...). A *scenario* is one
specific combination of components; variants of a scenario only change the words
around it (who reports, duration, fillers, accents).

The split is made by SCENARIO, never by sentence: test sentences are combinations
of clues the model never saw together, written in ways it never saw. Every family
keeps scenarios in training, so the model can learn its clues. The test therefore
measures generalization to new descriptions of known situations, NOT to clinical
situations absent from the data (see the model card's limitations).

The text model only decides cases WITHOUT alarm signs; the five alarm questions
are rules that always win. Labels are a simplified prototype scale, not an official
triage scale, and were not validated by health professionals.
"""

import itertools
import random
from dataclasses import dataclass

LABELS = ("high", "medium", "low")

Components = list[list[str]]

FAMILIES: dict[str, dict[str, Components]] = {
    "high": {
        "stroke": [
            ["se le torcio la boca", "tiene la cara caida de un lado", "la boca se le ve chueca"],
            [
                "no puede mover el brazo derecho",
                "perdio la fuerza en un lado del cuerpo",
                "se le duerme medio cuerpo",
                "no puede levantar la pierna izquierda",
            ],
            [
                "habla enredado de repente",
                "no le entienden cuando habla",
                "no puede hablar bien desde hace una hora",
            ],
        ],
        "meningitis": [
            ["fiebre muy alta", "calentura muy fuerte que no baja"],
            [
                "rigidez en el cuello",
                "no puede doblar el cuello",
                "le duele mucho el cuello al moverlo",
            ],
            ["no soporta la luz", "manchas moradas en la piel", "dolor de cabeza terrible"],
        ],
        "snakebite": [
            ["lo mordio una culebra", "le pico una serpiente", "mordedura de culebra"],
            ["en el pie", "en la mano", "trabajando en la finca", "en el cañaduzal"],
            [
                "se le esta hinchando mucho",
                "sangra por las encias",
                "tiene mucho dolor y se pone morado",
            ],
        ],
        "anaphylaxis": [
            [
                "se le hincharon la cara y los labios",
                "se le hincho la lengua",
                "siente la garganta cerrada",
            ],
            [
                "despues de comer mariscos",
                "lo picaron muchas abejas",
                "despues de una inyeccion",
                "despues de tomar una pastilla nueva",
            ],
            ["le broto todo el cuerpo", "le cuesta tragar"],
        ],
        "acute_abdomen": [
            [
                "dolor de barriga muy fuerte",
                "dolor de estomago insoportable",
                "dolor intenso al lado derecho abajo",
            ],
            [
                "el abdomen esta duro como una tabla",
                "no se calma con nada",
                "no aguanta caminar del dolor",
            ],
            ["vomita varias veces", "desde la madrugada"],
        ],
        "pregnancy_danger": [
            ["esta embarazada", "embarazada de siete meses", "en embarazo"],
            [
                "dolor de cabeza muy fuerte",
                "ve lucecitas",
                "vision borrosa",
                "dejo de sentir al bebe",
                "los pies y la cara muy hinchados",
            ],
        ],
        "dehydration": [
            [
                "vomito y diarrea todo el dia",
                "diarrea liquida muchas veces",
                "no retiene ni el agua",
            ],
            [
                "no orina desde ayer",
                "boca muy seca y ojos hundidos",
                "esta muy debil y casi no responde",
                "se marea al pararse",
            ],
        ],
        "confusion": [
            ["esta confundido", "esta muy desorientado", "muy somnoliento y cuesta despertarlo"],
            [
                "no reconoce a la familia",
                "dice cosas sin sentido",
                "con fiebre",
                "desde esta mañana",
            ],
        ],
        "burns": [
            [
                "se quemo con agua hirviendo",
                "quemadura grande con ampollas",
                "se quemo con una explosion de gas",
            ],
            ["en el pecho y los brazos", "en toda la espalda", "en la cara"],
        ],
        "poisoning": [
            [
                "se tomo veneno para las plagas",
                "se intoxico fumigando",
                "trago un liquido de limpieza",
                "tomo matamaleza por equivocacion",
            ],
            ["tiene vomito", "le sale mucha saliva", "le arde la garganta", "esta temblando"],
        ],
    },
    "medium": {
        "fever": [
            ["fiebre", "calentura", "fiebre y escalofrios"],
            ["hace tres dias", "desde el lunes", "desde anteayer"],
            [
                "con dolor de cuerpo",
                "que baja con acetaminofen y vuelve",
                "pero come y toma liquidos",
            ],
        ],
        "respiratory_infection": [
            ["tos con flema verde", "tos que no se le quita", "gripa fuerte"],
            ["con fiebre", "desde hace una semana", "hace diez dias"],
            ["dolor de pecho al toser", "le cuesta dormir por la tos"],
        ],
        "urinary": [
            ["ardor al orinar", "dolor al orinar", "orina muchas veces"],
            [
                "orina con mal olor",
                "con fiebre",
                "con dolor en la espalda baja",
                "desde hace tres dias",
            ],
        ],
        "gastroenteritis": [
            ["diarrea desde ayer", "vomito dos veces esta mañana", "diarrea con colicos"],
            [
                "pero toma liquidos",
                "y orina bien",
                "despues de comer en la feria",
                "dolor de estomago leve",
            ],
        ],
        "infected_wound": [
            ["herida en la mano", "cortada con machete hace dos dias", "uña enterrada"],
            ["roja y caliente", "con pus", "se ve inflamada", "le duele al tocarla"],
        ],
        "hypertension": [
            ["presion alta", "se le olvido la pastilla de la presion", "hipertenso"],
            ["con dolor de cabeza en la nuca", "y esta mareado", "con zumbido en los oidos"],
        ],
        "animal_bite": [
            ["lo mordio un perro", "rasguño de gato", "lo mordio el perro del vecino"],
            ["en la pierna", "en la mano", "la herida es pequeña", "se puso rojo"],
        ],
        "ear_infection": [
            ["dolor de oido fuerte", "le sale liquido del oido", "oido tapado y adolorido"],
            ["con fiebre", "desde ayer", "no oye bien de ese lado"],
        ],
        "abdominal_pain": [
            ["dolor de estomago", "dolor en la boca del estomago", "colicos fuertes"],
            ["que va y viene", "despues de comer grasa", "con la menstruacion", "desde ayer"],
        ],
        "diabetes": [
            ["diabetico", "diabetica", "con diabetes"],
            ["con el azucar alta", "con mucha sed", "orina mucho", "se siente muy cansado"],
        ],
    },
    "low": {
        "common_cold": [
            ["gripa con moco", "nariz tapada", "resfriado leve", "estornudos"],
            ["sin fiebre", "un poco de tos seca", "se siente bien", "desde ayer"],
        ],
        "mild_headache": [
            ["dolor de cabeza leve", "un poco de dolor de cabeza"],
            ["que se quita con acetaminofen", "por no dormir bien", "despues de trabajar al sol"],
        ],
        "follow_up": [
            ["viene a control de la presion", "control de rutina", "control del azucar"],
            ["se siente bien", "sin molestias", "todo normal", "trae los examenes"],
        ],
        "prescription": [
            [
                "necesita renovar la formula",
                "se le acabaron las pastillas",
                "pide la formula de siempre",
            ],
            ["de la tiroides", "de la presion", "de los medicamentos de siempre"],
        ],
        "minor_injury": [
            ["raspon en la rodilla", "golpe leve en el dedo", "pequeña cortada en el dedo"],
            ["ya no sangra", "puede moverlo bien", "ya cerro", "solo un poco de dolor"],
        ],
        "chronic_pain": [
            ["dolor de espalda", "dolor en las rodillas", "dolor en los hombros"],
            ["de hace meses", "por cargar bultos", "al subir lomas", "de toda la vida"],
        ],
        "skin": [
            ["picazon en la piel", "manchas blancas en la cara", "granitos en los brazos"],
            ["sin fiebre", "que no pican", "desde hace semanas", "piel reseca"],
        ],
        "information": [
            [
                "pregunta por las vacunas",
                "quiere informacion sobre planificacion familiar",
                "solicita certificado medico",
                "pregunta por la cita de odontologia",
            ],
            ["para el trabajo", "que le faltan", "para la familia"],
        ],
        "sore_throat": [
            ["dolor de garganta leve", "garganta irritada", "un poco de carraspera"],
            ["sin fiebre", "por gritar en el partido", "desde ayer"],
        ],
        "digestive_mild": [
            ["acidez", "llenura y gases", "agrieras"],
            ["despues de comer", "por las tardes", "cuando come frito"],
        ],
    },
}

PREFIXES = [
    "",
    "",
    "paciente refiere ",
    "la familia cuenta que ",
    "dice que ",
    "consulta porque ",
    "la señora comenta que ",
    "el señor manifiesta ",
    "refiere ",
    "acude porque ",
]
SUFFIXES = [
    "",
    "",
    "",
    ", vive lejos del puesto de salud",
    ", esta preocupado",
    ", no ha tomado nada",
    " y pide ayuda",
    ", vino caminando",
]
CONNECTORS = [", ", " y ", ", ademas ", ". "]
ACCENTS = str.maketrans({"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"})


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    label: str
    family: str
    scenario: str


def _scenarios(components: Components, rng: random.Random, limit: int) -> list[tuple[str, ...]]:
    """Distinct clue combinations: one phrase from each of two or more components."""
    combos: list[tuple[str, ...]] = []
    for size in range(2, len(components) + 1):
        for chosen in itertools.combinations(components, size):
            combos.extend(itertools.product(*chosen))
    rng.shuffle(combos)
    return combos[:limit]


def _render(clues: tuple[str, ...], rng: random.Random) -> str:
    text = clues[0]
    for clue in clues[1:]:
        text += rng.choice(CONNECTORS) + clue
    text = f"{rng.choice(PREFIXES)}{text}{rng.choice(SUFFIXES)}"
    if rng.random() < 0.3:
        # Some promoters type accents, others do not: the model must handle both.
        words = text.split()
        index = rng.randrange(len(words))
        words[index] = words[index].translate(ACCENTS)
        text = " ".join(words)
    if rng.random() < 0.25:
        text = text.capitalize()
    return text


def generate(scenarios_per_family: int = 12, variants: int = 10, seed: int = 42) -> list[Example]:
    """Same seed, same dataset: training is fully reproducible."""
    rng = random.Random(seed)
    examples = []
    for label, families in FAMILIES.items():
        for family, components in families.items():
            for index, clues in enumerate(_scenarios(components, rng, scenarios_per_family)):
                scenario = f"{label}/{family}/{index}"
                seen: set[str] = set()
                for _ in range(variants * 4):
                    text = _render(clues, rng)
                    if text not in seen:
                        seen.add(text)
                        examples.append(Example(text, label, family, scenario))
                    if len(seen) == variants:
                        break
    return examples


def split_by_scenario(
    examples: list[Example], test_share: float = 0.25, seed: int = 7
) -> tuple[list[Example], list[Example]]:
    """Holds out whole scenarios of every family; each family keeps most for training."""
    rng = random.Random(seed)
    test_scenarios: set[str] = set()
    for families in FAMILIES.values():
        for family in families:
            scenarios = sorted({e.scenario for e in examples if e.family == family})
            count = min(len(scenarios) - 1, max(1, round(len(scenarios) * test_share)))
            test_scenarios.update(rng.sample(scenarios, count))
    train = [e for e in examples if e.scenario not in test_scenarios]
    test = [e for e in examples if e.scenario in test_scenarios]
    return train, test
