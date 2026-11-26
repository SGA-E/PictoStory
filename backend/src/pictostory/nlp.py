"""Analyse du texte avec spaCy : personnages et reprises (anaphores).

La résolution des reprises est ici une version simple par règles :
un pronom (« il », « elle », « le », « les »...) renvoie au dernier
personnage cité de même genre et de même nombre. Elle sert de secours
hors ligne et de point de comparaison pour la version par LLM local.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .models import Character

DEFAULT_MODEL = "fr_core_news_md"

# Catégories de mots qui reçoivent un pictogramme.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}

# Pronoms de 3e personne que l'on cherche à relier à un personnage.
THIRD_PERSON_PRONOUNS = {"il", "elle", "ils", "elles", "le", "la", "l'", "les", "lui", "leur"}

# Pronoms de 1re personne : dans un dialogue, ils désignent celui qui parle.
FIRST_PERSON_PRONOUNS = {"je", "j'", "me", "m'", "moi"}

# Verbes qui introduisent une parole : « elle dit à monsieur Seguin : »
SPEECH_VERBS = {"dire", "demander", "répondre", "crier", "écrier", "murmurer", "ajouter", "penser", "chuchoter", "hurler"}

DIALOGUE_DASHES = {"-", "–", "—", "«"}

# Verbes impersonnels : dans « il faut », « il pleut », « il y a », « il » ne désigne personne.
IMPERSONAL_VERBS = {"falloir", "pleuvoir", "neiger", "venter", "agir", "avoir"}


@dataclass
class KnownCharacter:
    """Personnage déclaré par l'enseignant ou tiré de l'ontologie.

    Syntaxe texte : « Blanquette: chèvre, coureuse » (nom, puis autres noms
    communs qui la désignent dans l'histoire).
    """

    name: str
    aliases: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, spec: str) -> "KnownCharacter":
        name, _, rest = spec.partition(":")
        aliases = [a.strip().lower() for a in rest.split(",") if a.strip()]
        return cls(name=name.strip(), aliases=aliases)


@lru_cache(maxsize=2)
def load_nlp(model: str = DEFAULT_MODEL) -> Language:
    return spacy.load(model)


def add_known_characters(nlp: Language, known: list[KnownCharacter]) -> None:
    """Ajoute les noms propres des personnages connus au repérage d'entités."""
    names = [k.name for k in known if any(w[:1].isupper() for w in k.name.split())]
    if not names:
        return
    if "entity_ruler" in nlp.pipe_names:
        ruler = nlp.get_pipe("entity_ruler")
    else:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    existing = {p["pattern"] for p in ruler.patterns}
    ruler.add_patterns([{"label": "PER", "pattern": n} for n in names if n not in existing])


def _morph(token, key: str) -> str | None:
    values = token.morph.get(key)
    return values[0] if values else None


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def find_characters(
    doc: Doc, known: list[KnownCharacter] | None = None, min_subject_count: int = 2
) -> tuple[list[Character], dict[int, str]]:
    """Repère les personnages et renvoie aussi, pour chaque token, le personnage qu'il désigne.

    Un personnage est :
    - un personnage connu (nom propre ou nom commun, avec ses autres noms) ;
    - une entité de type personne (PER) ;
    - un nom commun sujet d'un verbe au moins `min_subject_count` fois (« le loup »).
    """
    characters: dict[str, Character] = {}
    token_to_char: dict[int, str] = {}
    alias_to_char: dict[str, str] = {}

    for k in known or []:
        cid = _slug(k.name)
        characters[cid] = Character(id=cid, name=k.name, aliases=list(k.aliases))
        is_common_noun = not any(w[:1].isupper() for w in k.name.split())
        for alias in k.aliases + ([k.name.lower()] if is_common_noun else []):
            alias_to_char[alias] = cid

    for ent in doc.ents:
        if ent.label_ != "PER":
            continue
        if ent.start > 0 and doc[ent.start - 1].text in {"«", '"'}:
            continue  # un mot entre guillemets (« Mê… ») n'est pas un personnage
        cid = _slug(ent.text)
        char = characters.setdefault(cid, Character(id=cid, name=ent.text))
        char.gender = char.gender or _morph(ent.root, "Gender")
        char.number = char.number or "Sing"
        for tok in ent:
            token_to_char[tok.i] = cid

    subject_nouns = Counter(
        t.lemma_.lower() for t in doc if t.pos_ == "NOUN" and t.dep_.startswith("nsubj")
    )
    recurring = {lemma for lemma, n in subject_nouns.items() if n >= min_subject_count}

    for tok in doc:
        if tok.pos_ != "NOUN" or tok.i in token_to_char:
            continue
        lemma = tok.lemma_.lower()
        cid = alias_to_char.get(lemma) or alias_to_char.get(tok.text.lower())
        if cid is None and lemma in recurring:
            cid = _slug(lemma)
            if cid not in characters:
                det = next((c.text.lower() for c in tok.children if c.dep_ == "det"), "")
                characters[cid] = Character(id=cid, name=f"{det} {tok.text}".strip())
        if cid is None:
            continue
        char = characters[cid]
        # « les jeunes renards » (pluriel) ne désigne pas Ruset le renard (singulier).
        if char.number and _morph(tok, "Number") and char.number != _morph(tok, "Number"):
            continue
        char.gender = char.gender or _morph(tok, "Gender")
        char.number = char.number or _morph(tok, "Number")
        token_to_char[tok.i] = cid

    return list(characters.values()), token_to_char


def _is_impersonal(tok) -> bool:
    """« il faut », « il pleut », « il y a ». spaCy marque aussi « il s'endort » comme
    explétif, donc on ne se fie pas à son étiquette seule mais au verbe."""
    if tok.text.lower() != "il":
        return False
    verb = tok.head
    if verb.lemma_ == "avoir":  # seulement « il y a »
        return any(c.text.lower() == "y" for c in verb.children)
    return verb.lemma_ in IMPERSONAL_VERBS


def resolve_pronouns(doc: Doc, characters: list[Character], token_to_char: dict[int, str]) -> dict[int, str]:
    """Relie chaque pronom de 3e personne au dernier personnage cité de même genre et nombre."""
    by_id = {c.id: c for c in characters}
    resolved = dict(token_to_char)
    recent: list[str] = []  # personnages, du plus récent au plus ancien

    for tok in doc:
        if tok.i in token_to_char:
            cid = token_to_char[tok.i]
            if cid in recent:
                recent.remove(cid)
            recent.insert(0, cid)
            continue
        if tok.pos_ != "PRON" or tok.text.lower() not in THIRD_PERSON_PRONOUNS:
            continue
        if _is_impersonal(tok):
            continue
        gender, number = _morph(tok, "Gender"), _morph(tok, "Number")

        def compatible(cid: str) -> bool:
            c = by_id[cid]
            same_number = number is None or c.number is None or c.number == number
            same_gender = gender is None or c.gender is None or c.gender == gender
            return same_number and same_gender

        target = next((cid for cid in recent if compatible(cid)), None)
        if target:
            resolved[tok.i] = target
    return resolved


def dialogue_tokens(doc: Doc) -> set[int]:
    """Indices des tokens qui sont dans une réplique : de « - » en début de ligne jusqu'à la fin de la ligne."""
    inside, in_dialogue = set(), False
    for tok in doc:
        at_line_start = tok.i == 0 or "\n" in doc[tok.i - 1].text or doc[tok.i - 1].text_with_ws.endswith("\n")
        if at_line_start:
            in_dialogue = tok.text in DIALOGUE_DASHES
        if "\n" in tok.text:
            in_dialogue = False
            continue
        if in_dialogue:
            inside.add(tok.i)
    return inside


def resolve_speakers(doc: Doc, references: dict[int, str]) -> dict[int, str]:
    """Relie « je », « me », « moi » d'une réplique au personnage qui vient de prendre la parole."""
    resolved = dict(references)
    in_dialogue = dialogue_tokens(doc)
    speaker: str | None = None
    for tok in doc:
        if tok.i not in in_dialogue and tok.lemma_ in SPEECH_VERBS:
            subject = next((c for c in tok.children if c.dep_.startswith("nsubj")), None)
            if subject is not None and subject.i in resolved:
                speaker = resolved[subject.i]
        if tok.i in in_dialogue and tok.text.lower() in FIRST_PERSON_PRONOUNS and speaker:
            resolved[tok.i] = speaker
    return resolved
