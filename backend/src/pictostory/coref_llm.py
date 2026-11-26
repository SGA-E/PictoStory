"""Résolution des reprises difficiles par un LLM local (Ollama).

Les règles de `nlp.py` gèrent les cas simples. Quand plusieurs référents sont
possibles (« Et quelle herbe ! Elle est savoureuse » : la chèvre ou l'herbe ?),
on pose la question au LLM avec une liste fermée de choix numérotés : il ne
peut pas inventer de réponse, seulement choisir.

Le LLM tourne sur l'ordinateur de l'enseignant avec Ollama
(https://ollama.com). Exemple : `ollama pull qwen2.5:7b`.
Sans Ollama, le pipeline garde simplement le résultat des règles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
from spacy.tokens import Doc

from .models import Character
from .nlp import FIRST_PERSON_PRONOUNS, THIRD_PERSON_PRONOUNS, _is_impersonal, _morph, dialogue_tokens

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
CONTEXT_SENTENCES = 2  # phrases précédentes montrées au LLM

SYSTEM_PROMPT = (
    "Tu aides un enfant à comprendre une histoire. On te montre un extrait où un mot "
    "est entouré de [[ ]]. Tu dois dire à qui ou à quoi ce mot renvoie, en choisissant "
    "un numéro dans la liste. Réponds uniquement en JSON."
)


class ChoiceLLM(Protocol):
    def choose(self, prompt: str, n_choices: int) -> int | None:
        """Renvoie le numéro choisi (1 à n_choices), ou None si pas de réponse."""
        ...


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, url: str = "http://localhost:11434", timeout: float = 120.0):
        self.model = model
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def choose(self, prompt: str, n_choices: int) -> int | None:
        schema = {
            "type": "object",
            "properties": {"choix": {"type": "integer", "minimum": 1, "maximum": n_choices}},
            "required": ["choix"],
        }
        try:
            response = self.client.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "format": schema,
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            import json

            value = json.loads(response.json()["message"]["content"]).get("choix")
        except (httpx.HTTPError, ValueError, KeyError):
            return None
        return value if isinstance(value, int) and 1 <= value <= n_choices else None


@dataclass
class Option:
    label: str
    character_id: str | None = None
    referent: str | None = None  # lemme d'un nom qui n'est pas un personnage


GENERIC = Option(label="personne ou rien en particulier (sens général)")


def _compatible(tok, gender: str | None, number: str | None) -> bool:
    g, n = _morph(tok, "Gender"), _morph(tok, "Number")
    return (gender is None or g is None or g == gender) and (number is None or n is None or n == number)


def build_options(doc: Doc, pronoun, characters: list[Character], seen: list[str], token_to_char: dict[int, str]) -> list[Option]:
    """Personnages déjà cités et noms compatibles des phrases proches, puis « sens général »."""
    by_id = {c.id: c for c in characters}
    first_person = pronoun.text.lower() in FIRST_PERSON_PRONOUNS
    gender, number = (None, None) if first_person else (_morph(pronoun, "Gender"), _morph(pronoun, "Number"))
    options: list[Option] = []

    for cid in seen:
        c = by_id[cid]
        if first_person or (
            (gender is None or c.gender is None or c.gender == gender)
            and (number is None or c.number is None or c.number == number)
        ):
            options.append(Option(label=f"{c.name} (personnage)", character_id=cid))

    if not first_person:
        sents = list(doc.sents)
        idx = next(i for i, s in enumerate(sents) if s.start <= pronoun.i < s.end)
        window_start = sents[max(0, idx - CONTEXT_SENTENCES)].start
        lemmas = set()
        for tok in doc[window_start : pronoun.i]:
            if tok.pos_ == "NOUN" and tok.i not in token_to_char and _compatible(tok, gender, number):
                lemma = tok.lemma_.lower()
                if lemma not in lemmas:
                    lemmas.add(lemma)
                    options.append(Option(label=f"« {tok.text} »", referent=lemma))

    options.append(GENERIC)
    return options


def build_prompt(doc: Doc, pronoun, options: list[Option]) -> str:
    sents = list(doc.sents)
    idx = next(i for i, s in enumerate(sents) if s.start <= pronoun.i < s.end)
    start = sents[max(0, idx - CONTEXT_SENTENCES)].start
    end = sents[idx].end
    words = []
    for tok in doc[start:end]:
        text = f"[[{tok.text}]]" if tok.i == pronoun.i else tok.text
        words.append(text + tok.whitespace_)
    extract = " ".join("".join(words).split())
    choices = "\n".join(f"{n}. {o.label}" for n, o in enumerate(options, start=1))
    return (
        f"Extrait :\n{extract}\n\n"
        f"À qui ou à quoi renvoie [[{pronoun.text}]] ?\n{choices}\n\n"
        'Réponds en JSON : {"choix": <numéro>}'
    )


def refine_with_llm(
    doc: Doc,
    characters: list[Character],
    token_to_char: dict[int, str],
    references: dict[int, str],
    llm: ChoiceLLM,
) -> tuple[dict[int, str], dict[int, str]]:
    """Repose au LLM les pronoms qui ont plusieurs référents possibles.

    Renvoie (personnage par token, référent non-personnage par token).
    Si le LLM ne répond pas, le résultat des règles est gardé.
    """
    references = dict(references)
    referents: dict[int, str] = {}
    in_dialogue = dialogue_tokens(doc)
    seen: list[str] = []

    for tok in doc:
        if tok.i in token_to_char:
            cid = token_to_char[tok.i]
            if cid in seen:
                seen.remove(cid)
            seen.insert(0, cid)
            continue
        word = tok.text.lower()
        third = tok.pos_ == "PRON" and word in THIRD_PERSON_PRONOUNS and not _is_impersonal(tok)
        first = word in FIRST_PERSON_PRONOUNS and tok.i in in_dialogue and tok.i not in references
        generic_on = word == "on"
        if not (third or first or generic_on):
            continue

        options = build_options(doc, tok, characters, seen, token_to_char)
        if len(options) <= 2 and not generic_on and tok.i in references:
            continue  # un seul référent possible : les règles suffisent

        choice = llm.choose(build_prompt(doc, tok, options), len(options))
        if choice is None:
            continue
        picked = options[choice - 1]
        references.pop(tok.i, None)
        if picked.character_id:
            references[tok.i] = picked.character_id
        elif picked.referent:
            referents[tok.i] = picked.referent

    return references, referents
