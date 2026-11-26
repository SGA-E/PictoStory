"""Ligne de commande : `pictostory histoire.txt` affiche l'histoire annotée."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotate import annotate_story
from .pictograms import ArasaacProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Annoter une histoire avec des pictogrammes.")
    parser.add_argument("fichier", type=Path, help="fichier texte (.txt) ou image (.jpg, .png)")
    parser.add_argument("--personnage", action="append", default=[], help='personnage connu, ex. "Blanquette: chèvre, coureuse"')
    parser.add_argument("--json", action="store_true", help="sortie JSON complète")
    parser.add_argument("--llm", metavar="MODELE", help="modèle Ollama local pour les reprises difficiles, ex. qwen2.5:7b")
    args = parser.parse_args()

    if args.fichier.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        from .ocr import image_to_text

        text = image_to_text(args.fichier)
    else:
        text = args.fichier.read_text(encoding="utf-8")

    llm = None
    if args.llm:
        from .coref_llm import OllamaLLM

        llm = OllamaLLM(args.llm)
    story = annotate_story(text, ArasaacProvider(), known_characters=args.personnage, llm=llm)

    if args.json:
        print(json.dumps(story.to_dict(), ensure_ascii=False, indent=2))
        return

    names = {c.id: c.name for c in story.characters}
    print("Personnages :", ", ".join(names.values()) or "aucun")
    for sentence in story.sentences:
        parts = []
        for t in sentence.tokens:
            mark = ""
            if t.character_id:
                mark = f"[{names[t.character_id]}]"
            elif t.referent:
                mark = f"[→ {t.referent}]"
            elif t.pictogram:
                mark = f"[#{t.pictogram.id}{' ?' if t.to_validate else ''}]"
            parts.append(t.text + mark)
        print(" ".join(parts))


if __name__ == "__main__":
    main()
