"""Désambiguïsation lexicale : choisir le bon pictogramme selon le contexte.

Pour l'instant seule la version de base existe (`BaselineDisambiguator`) :
elle sert de point de comparaison pour mesurer l'apport de l'IA.
Les versions par embeddings puis par LLM local (Ollama) viendront ensuite,
en respectant la même interface.
"""

from __future__ import annotations

from typing import Protocol

from .models import Pictogram

# En dessous de ce seuil, le mot est marqué « à valider » par l'enseignant.
CONFIDENCE_THRESHOLD = 0.7


class Disambiguator(Protocol):
    def choose(
        self, lemma: str, sentence: str, candidates: list[Pictogram]
    ) -> tuple[Pictogram | None, float]:
        """Renvoie le pictogramme choisi et un score de confiance entre 0 et 1."""
        ...


class BaselineDisambiguator:
    """Sans IA : prend le premier candidat et signale l'ambiguïté."""

    def choose(self, lemma, sentence, candidates):
        if not candidates:
            return None, 0.0
        if len(candidates) == 1:
            return candidates[0], 1.0
        # Plusieurs sens possibles : on ne sait pas trancher sans contexte.
        return candidates[0], 1.0 / len(candidates)
