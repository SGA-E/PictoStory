"""Analyse du texte avec spaCy : personnages et reprises (anaphores).

La résolution des reprises est ici une version simple par règles :
un pronom « il » ou « elle » renvoie au dernier personnage cité de même
genre. Elle servira de secours hors ligne et de point de comparaison
pour la version par LLM local.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .models import Character

DEFAULT_MODEL = "fr_core_news_md"

# Catégories de mots qui reçoivent un pictogramme.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}

SUBJECT_PRONOUNS = {"il": "Masc", "elle": "Fem", "ils": "Masc", "elles": "Fem"}


@lru_cache(maxsize=2)
def load_nlp(model: str = DEFAULT_MODEL) -> Language:
    return spacy.load(model)


def add_known_characters(nlp: Language, names: list[str]) -> None:
    """Ajoute des noms de personnages fournis par l'enseignant ou l'ontologie."""
    if not names:
        return
    if "entity_ruler" in nlp.pipe_names:
        ruler = nlp.get_pipe("entity_ruler")
    else:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    existing = {p["pattern"] for p in ruler.patterns}
    ruler.add_patterns(
        [{"label": "PER", "pattern": name} for name in names if name not in existing]
    )


def _gender(token) -> str | None:
    values = token.morph.get("Gender")
    return values[0] if values else None


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def find_characters(doc: Doc, min_subject_count: int = 2) -> tuple[list[Character], dict[int, str]]:
    """Repère les personnages et renvoie aussi, pour chaque token, le personnage qu'il désigne.

    Un personnage est soit une entité de type personne (PER), soit un nom
    commun qui est sujet d'un verbe au moins `min_subject_count` fois
    (« le loup », « la grand-mère »).
    """
    characters: dict[str, Character] = {}
    token_to_char: dict[int, str] = {}

    for ent in doc.ents:
        if ent.label_ != "PER":
            continue
        cid = _slug(ent.text)
        characters.setdefault(cid, Character(id=cid, name=ent.text, gender=_gender(ent.root)))
        for tok in ent:
            token_to_char[tok.i] = cid

    subject_nouns = Counter(
        t.lemma_.lower() for t in doc if t.pos_ == "NOUN" and t.dep_.startswith("nsubj")
    )
    recurring = {lemma for lemma, n in subject_nouns.items() if n >= min_subject_count}
    for tok in doc:
        if tok.pos_ == "NOUN" and tok.lemma_.lower() in recurring and tok.i not in token_to_char:
            cid = _slug(tok.lemma_)
            if cid not in characters:
                det = next((c.text.lower() for c in tok.children if c.dep_ == "det"), "")
                name = f"{det} {tok.text}".strip() if det else tok.text
                characters[cid] = Character(id=cid, name=name, gender=_gender(tok))
            token_to_char[tok.i] = cid

    return list(characters.values()), token_to_char


def resolve_pronouns(doc: Doc, characters: list[Character], token_to_char: dict[int, str]) -> dict[int, str]:
    """Relie les pronoms sujets au dernier personnage cité de même genre."""
    by_id = {c.id: c for c in characters}
    resolved = dict(token_to_char)
    last_by_gender: dict[str | None, str] = {}
    last_any: str | None = None

    for tok in doc:
        if tok.i in token_to_char:
            cid = token_to_char[tok.i]
            last_by_gender[by_id[cid].gender] = cid
            last_any = cid
            continue
        pronoun = tok.text.lower()
        if tok.pos_ == "PRON" and pronoun in SUBJECT_PRONOUNS:
            target = last_by_gender.get(SUBJECT_PRONOUNS[pronoun]) or last_by_gender.get(None) or last_any
            if target:
                resolved[tok.i] = target
    return resolved
