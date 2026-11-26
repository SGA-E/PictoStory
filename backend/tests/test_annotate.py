import pytest

from pictostory.annotate import annotate_story
from pictostory.models import Pictogram
from pictostory.pictograms import InMemoryProvider
from pictostory.wsd import BaselineDisambiguator

STORY = (
    "Max se promène dans la forêt. Il rencontre le loup. "
    "Le loup a faim. Le loup mange un avocat. Il est content."
)

PROVIDER = InMemoryProvider(
    {
        "avocat": [Pictogram(1, ["avocat", "fruit"]), Pictogram(2, ["avocat", "métier"])],
        "forêt": [Pictogram(3, ["forêt"])],
    }
)


def tokens(story, text):
    return [t for s in story.sentences for t in s.tokens if t.text == text]


@pytest.mark.needs_model
def test_personnages_detectes():
    story = annotate_story(STORY, PROVIDER, known_characters=["Max"])
    assert {c.id for c in story.characters} == {"max", "loup"}


@pytest.mark.needs_model
def test_pronoms_relies_au_bon_personnage():
    story = annotate_story(STORY, PROVIDER, known_characters=["Max"])
    il_1, il_2 = tokens(story, "Il")
    assert il_1.character_id == "max"  # « Il rencontre le loup »
    assert il_2.character_id == "loup"  # « Il est content », après « Le loup mange »


@pytest.mark.needs_model
def test_mot_ambigu_marque_a_valider():
    story = annotate_story(STORY, PROVIDER, known_characters=["Max"])
    (avocat,) = tokens(story, "avocat")
    assert len(avocat.candidates) == 2
    assert avocat.to_validate
    (foret,) = tokens(story, "forêt")
    assert foret.pictogram.id == 3 and not foret.to_validate


def test_baseline_sans_candidat():
    assert BaselineDisambiguator().choose("chat", "Le chat dort.", []) == (None, 0.0)
