"""Lecture du texte d'une photo ou d'un PDF image (OCR).

Version de base avec Tesseract (langue française). Il faut installer
Tesseract sur l'ordinateur et les dépendances optionnelles :
`pip install -e ".[ocr]"`. Le texte extrait doit toujours être relu
par l'enseignant avant l'analyse.
"""

from __future__ import annotations

import re
from pathlib import Path


def clean_text(raw: str) -> str:
    """Recolle les lignes d'un même paragraphe et les mots coupés en fin de ligne."""
    paragraphs = re.split(r"\n\s*\n", raw.strip())
    cleaned = []
    for para in paragraphs:
        para = re.sub(r"(\w)-\n(\w)", r"\1\2", para)  # « renar-\ndeaux » → « renardeaux »
        cleaned.append(" ".join(para.split()))
    return "\n\n".join(p for p in cleaned if p)


def image_to_text(path: str | Path, lang: str = "fra") -> str:
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OCR non installé : pip install -e ".[ocr]"') from exc

    image = ImageOps.exif_transpose(Image.open(path))  # remet la photo dans le bon sens
    image = ImageOps.grayscale(image)
    return clean_text(pytesseract.image_to_string(image, lang=lang))
