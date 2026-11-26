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


@pytest.mark.needs_model
def test_autres_noms_du_personnage():
    story = annotate_story(
        "Blanquette va dans la montagne. La petite chèvre est heureuse.",
        PROVIDER,
        known_characters=["Blanquette: chèvre"],
    )
    (chevre,) = tokens(story, "chèvre")
    assert chevre.character_id == "blanquette"


@pytest.mark.needs_model
def test_pluriel_et_pronom_complement():
    story = annotate_story(
        "Ruset sort du terrier. Les renardeaux le suivent. Ils courent autour du terrier.",
        PROVIDER,
        known_characters=["Ruset: renard", "renardeaux: renardeau"],
    )
    (le,) = tokens(story, "le")
    (ils,) = tokens(story, "Ils")
    assert le.character_id == "ruset"
    assert ils.character_id == "renardeaux"


@pytest.mark.needs_model
def test_il_impersonnel_ignore():
    story = annotate_story(
        "Monsieur Seguin aime sa chèvre. Elle a tout ce qu'il faut.",
        PROVIDER,
        known_characters=["Monsieur Seguin"],
    )
    (il,) = tokens(story, "il")
    assert il.character_id is None


def test_nettoyage_ocr():
    from pictostory.ocr import clean_text

    assert clean_text("Les renar-\ndeaux le\nsuivent.\n\nIls courent.") == (
        "Les renardeaux le suivent.\n\nIls courent."
    )
