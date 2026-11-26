"""Accès aux pictogrammes.

`ArasaacProvider` interroge l'API publique d'ARASAAC (https://arasaac.org,
licence CC BY-NC-SA) et met les réponses en cache sur disque pour pouvoir
travailler ensuite sans connexion. `InMemoryProvider` sert aux tests et aux
pictogrammes ajoutés à la main.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import httpx

from .models import Pictogram

ARASAAC_API = "https://api.arasaac.org/v1"


class PictogramProvider(Protocol):
    def search(self, lemma: str) -> list[Pictogram]:
        """Renvoie les pictogrammes dont un mot-clé correspond au lemme."""
        ...


class InMemoryProvider:
    def __init__(self, index: dict[str, list[Pictogram]] | None = None):
        self.index = {k.lower(): v for k, v in (index or {}).items()}

    def search(self, lemma: str) -> list[Pictogram]:
        return list(self.index.get(lemma.lower(), []))


class ArasaacProvider:
    def __init__(self, lang: str = "fr", cache_dir: Path | None = None, timeout: float = 10.0):
        self.lang = lang
        self.cache_dir = cache_dir or Path.home() / ".cache" / "pictostory" / "arasaac"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=timeout)

    def _cache_path(self, lemma: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in lemma.lower())
        return self.cache_dir / f"{self.lang}_{safe}.json"

    def _fetch(self, lemma: str) -> list[dict]:
        path = self._cache_path(lemma)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = f"{ARASAAC_API}/pictograms/{self.lang}/bestsearch/{lemma}"
        try:
            response = self.client.get(url)
        except httpx.HTTPError:
            return []  # hors ligne : pas de picto, et rien en cache pour réessayer plus tard
        if response.status_code == 404:
            data = []  # aucun picto pour ce mot : on le retient
        elif response.status_code == 200:
            data = response.json()
        else:
            return []
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def search(self, lemma: str) -> list[Pictogram]:
        return [
            Pictogram(
                id=item["_id"],
                keywords=[k.get("keyword", "") for k in item.get("keywords", [])],
            )
            for item in self._fetch(lemma)
        ]
