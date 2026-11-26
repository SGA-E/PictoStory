"""API locale utilisée par l'atelier de l'enseignant et l'app tablette.

Lancement : `uvicorn pictostory.api:app --reload`
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .annotate import annotate_story
from .pictograms import ArasaacProvider

app = FastAPI(title="PictoStory")
_provider: ArasaacProvider | None = None


def get_provider() -> ArasaacProvider:
    global _provider
    if _provider is None:
        _provider = ArasaacProvider()
    return _provider


class AnnotateRequest(BaseModel):
    text: str
    characters: list[str] = []


@app.post("/histoires/annoter")
def annoter(request: AnnotateRequest) -> dict:
    story = annotate_story(request.text, get_provider(), known_characters=request.characters)
    return story.to_dict()
