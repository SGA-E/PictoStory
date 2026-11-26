import re

import pytest

from pictostory.annotate import annotate_story
from pictostory.coref_llm import OllamaLLM
from pictostory.pictograms import InMemoryProvider


class FakeLLM:
    """Répond comme le ferait un bon LLM, en cherchant le choix qui contient `answers[mot clé]`."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.prompts: list[str] = []

    def choose(self, prompt: str, n_choices: int) -> int | None:
        self.prompts.append(prompt)
        for clue, wanted in self.answers.items():
            if clue in prompt:
                for line in prompt.splitlines():
                    m = re.match(r"(\d+)\. (.*)", line)
                    if m and wanted in m.group(2):
                        return int(m.group(1))
        return None  # pas d'avis : on garde les règles


def tokens(story, text):
    return [t for s in story.sentences for t in s.tokens if t.text == text]


TEXT = "Blanquette mange. Et quelle herbe ! Elle est savoureuse."


@pytest.mark.needs_model
def test_llm_relie_un_pronom_a_une_chose():
    llm = FakeLLM({"[[Elle]] est savoureuse": "herbe"})
    story = annotate_story(TEXT, InMemoryProvider(), known_characters=["Blanquette"], llm=llm)
    (elle,) = tokens(story, "Elle")
    assert elle.character_id is None
    assert elle.referent == "herbe"
    assert "1. Blanquette (personnage)" in llm.prompts[0]


@pytest.mark.needs_model
def test_sans_reponse_du_llm_on_garde_les_regles():
    story = annotate_story(TEXT, InMemoryProvider(), known_characters=["Blanquette"], llm=FakeLLM({}))
    (elle,) = tokens(story, "Elle")
    assert elle.character_id == "blanquette"


@pytest.mark.needs_model
def test_je_dans_un_dialogue_est_celui_qui_parle():
    text = "Un soir, Blanquette dit à monsieur Seguin :\n- Je veux aller dans la montagne."
    story = annotate_story(
        text, InMemoryProvider(), known_characters=["Blanquette", "monsieur Seguin"]
    )
    (je,) = tokens(story, "Je")
    assert je.character_id == "blanquette"


def test_ollama_absent_ne_bloque_pas():
    llm = OllamaLLM(url="http://127.0.0.1:9", timeout=0.5)
    assert llm.choose("question", 3) is None
