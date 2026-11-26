"""Structures de données échangées entre le pipeline et l'application.

Elles reprennent les classes de l'ontologie : Histoire, Phrase, Token,
Personnage, Mention et Pictogramme.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Pictogram:
    """Un pictogramme candidat pour un mot."""

    id: int
    keywords: list[str]
    source: str = "arasaac"

    @property
    def image_url(self) -> str:
        if self.source == "arasaac":
            return f"https://static.arasaac.org/pictograms/{self.id}/{self.id}_500.png"
        return ""


@dataclass
class Character:
    """Un personnage de l'histoire, avec un identifiant stable."""

    id: str
    name: str
    gender: str | None = None  # "Masc", "Fem" ou None
    number: str | None = None  # "Sing", "Plur" ou None
    aliases: list[str] = field(default_factory=list)  # autres noms : « chèvre », « coureuse »


@dataclass
class Token:
    """Un mot de l'histoire tel qu'il sera affiché à l'enfant."""

    text: str
    lemma: str
    pos: str
    pictogram: Pictogram | None = None
    candidates: list[Pictogram] = field(default_factory=list)
    confidence: float = 0.0
    to_validate: bool = False
    character_id: str | None = None  # renseigné si le mot désigne un personnage


@dataclass
class Sentence:
    text: str
    tokens: list[Token]


@dataclass
class Story:
    sentences: list[Sentence]
    characters: list[Character]

    def to_dict(self) -> dict:
        return asdict(self)
