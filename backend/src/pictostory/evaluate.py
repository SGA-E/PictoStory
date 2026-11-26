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

`attendu` est l'identifiant du personnage, le lemme (forme au singulier) d'un
nom qui n'est pas un personnage (« herbe », « fleur »), ou null pour un
pronom générique.

Lancement : `python -m pictostory.evaluate chemin/du/corpus [--llm qwen2.5:7b]`
"""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

from .annotate import annotate_story
from .coref_llm import ChoiceLLM, OllamaLLM
from .pictograms import InMemoryProvider


def evaluate_story(text: str, gold: dict, llm: ChoiceLLM | None = None) -> list[dict]:
    story = annotate_story(
        text, InMemoryProvider(), known_characters=gold.get("personnages", []), llm=llm
    )
    character_ids = {c.id for c in story.characters}
    results = []
    for item in gold["reprises"]:
        sentence = next(
            (s for s in story.sentences if item["phrase"] in " ".join(s.text.split())), None
        )
        token = None
        if sentence:
            token = next((t for t in sentence.tokens if t.text == item["mot"]), None)
        got = (token.character_id or token.referent) if token else None
        expected = item["attendu"]
        if expected in character_ids or expected is None:
            correct = token is not None and token.character_id == expected and token.referent is None
        else:
            # Référent qui n'est pas un personnage (« herbe ») : il faut le trouver.
            correct = token is not None and token.character_id is None and token.referent == expected
        results.append({**item, "obtenu": got, "trouve": token is not None, "correct": correct})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluer la résolution des reprises.")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--llm", metavar="MODELE", help="modèle Ollama à utiliser, ex. qwen2.5:7b")
    parser.add_argument("--llm-cpu", action="store_true", help="forcer Ollama à calculer sur le processeur")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    llm = OllamaLLM(args.llm, cpu_only=args.llm_cpu) if args.llm else None

    total = correct = 0
    for gold_path in sorted(args.corpus.glob("*.json")):
        text = gold_path.with_suffix(".txt").read_text(encoding="utf-8")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        print(f"\n== {gold_path.stem}")
        for r in evaluate_story(text, gold, llm):
            total += 1
            correct += r["correct"]
            mark = "OK " if r["correct"] else "NON"
            found = "" if r["trouve"] else " (mot introuvable)"
            print(f"  {mark} « {r['mot']} » dans « {r['phrase']}… » : attendu {r['attendu']}, obtenu {r['obtenu']}{found}")
    if total:
        print(f"\nScore : {correct}/{total} ({100 * correct / total:.0f} %)")
    if llm is not None:
        print(llm.summary())


if __name__ == "__main__":
    main()
