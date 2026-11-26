"""Évaluation de la résolution des reprises sur un corpus annoté à la main.

Un corpus est un dossier contenant, pour chaque histoire, `nom.txt` (le texte)
et `nom.json` (les réponses attendues) :

    {
      "personnages": ["Blanquette: chèvre, coureuse", "monsieur Seguin"],
      "reprises": [
        {"phrase": "Elle broute l'herbe", "mot": "Elle", "attendu": "blanquette"},
        {"phrase": "Elle est savoureuse", "mot": "Elle", "attendu": "herbe"},
        {"phrase": "Comme on doit", "mot": "on", "attendu": null}
      ]
    }

`attendu` est l'identifiant du personnage, un mot qui n'est pas un personnage
(le pipeline doit alors ne relier le pronom à aucun personnage), ou null pour
un pronom générique.

Lancement : `python -m pictostory.evaluate chemin/du/corpus`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotate import annotate_story
from .pictograms import InMemoryProvider


def evaluate_story(text: str, gold: dict) -> list[dict]:
    story = annotate_story(text, InMemoryProvider(), known_characters=gold.get("personnages", []))
    character_ids = {c.id for c in story.characters}
    results = []
    for item in gold["reprises"]:
        sentence = next(
            (s for s in story.sentences if item["phrase"] in " ".join(s.text.split())), None
        )
        token = None
        if sentence:
            token = next((t for t in sentence.tokens if t.text == item["mot"]), None)
        got = token.character_id if token else None
        expected = item["attendu"]
        # Un référent qui n'est pas un personnage (« herbe ») attend « aucun personnage ».
        expected_char = expected if expected in character_ids else None
        results.append(
            {
                **item,
                "obtenu": got,
                "trouve": token is not None,
                "correct": token is not None and got == expected_char,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluer la résolution des reprises.")
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()

    total = correct = 0
    for gold_path in sorted(args.corpus.glob("*.json")):
        text = gold_path.with_suffix(".txt").read_text(encoding="utf-8")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        print(f"\n== {gold_path.stem}")
        for r in evaluate_story(text, gold):
            total += 1
            correct += r["correct"]
            mark = "OK " if r["correct"] else "NON"
            found = "" if r["trouve"] else " (mot introuvable)"
            print(f"  {mark} « {r['mot']} » dans « {r['phrase']}… » : attendu {r['attendu']}, obtenu {r['obtenu']}{found}")
    if total:
        print(f"\nScore : {correct}/{total} ({100 * correct / total:.0f} %)")


if __name__ == "__main__":
    main()
