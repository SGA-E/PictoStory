# Backend PictoStory (Python + spaCy)

Transforme le texte d'une histoire en histoire annotée : pictogramme pour chaque mot utile, personnages repérés, pronoms (« il », « elle ») reliés au bon personnage, mots ambigus marqués « à valider ».

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
python -m spacy download fr_core_news_md
```

Pour lire des photos : installer [Tesseract](https://github.com/tesseract-ocr/tesseract) avec la langue française, puis `pip install -e ".[ocr]"`.

### LLM local (optionnel, recommandé)

Pour les reprises difficiles (« Et quelle herbe ! **Elle** est savoureuse »), le pipeline peut interroger un LLM qui tourne sur l'ordinateur, sans internet :

1. Installer [Ollama](https://ollama.com/download).
2. Télécharger un modèle, une seule fois : `ollama pull qwen2.5:7b` (environ 5 Go, 8 Go de RAM conseillés).
3. Ajouter `--llm qwen2.5:7b` aux commandes ci-dessous.

Le LLM ne reçoit qu'un court extrait et une liste fermée de choix numérotés : il choisit, il n'invente pas. S'il ne répond pas, le résultat des règles est gardé.

## Utilisation

```bash
pictostory mon_histoire.txt --personnage "Blanquette: chèvre, coureuse" --personnage "monsieur Seguin"
pictostory page.jpg            # photo d'une page, passe par l'OCR
uvicorn pictostory.api:app     # API locale sur http://127.0.0.1:8000/docs
pytest                         # tests
python -m pictostory.evaluate dossier_corpus                    # score avec les règles seules
python -m pictostory.evaluate dossier_corpus --llm qwen2.5:7b   # score avec le LLM local
```

Un personnage se déclare par son nom, suivi si besoin des noms communs qui le désignent aussi dans l'histoire : `"Blanquette: chèvre, coureuse"`. Le format du corpus d'évaluation est décrit dans `evaluate.py`.

## Organisation

| Fichier | Rôle |
| --- | --- |
| `models.py` | Structures de données (histoire, phrase, mot, personnage, pictogramme) |
| `ocr.py` | Lecture du texte d'une photo (Tesseract) |
| `nlp.py` | spaCy : personnages et reprises par règles simples |
| `pictograms.py` | Recherche des pictogrammes ARASAAC, avec cache local pour travailler hors ligne |
| `evaluate.py` | Score de la résolution des reprises sur un corpus annoté |
| `coref_llm.py` | Reprises difficiles confiées au LLM local (Ollama) |
| `wsd.py` | Choix du bon pictogramme selon le contexte (version de base sans IA pour l'instant) |
| `annotate.py` | Enchaîne toutes les étapes |
| `api.py`, `cli.py` | API locale et ligne de commande |

Les pictogrammes viennent d'[ARASAAC](https://arasaac.org) (auteur Sergio Palao, propriété du Gouvernement d'Aragon), sous licence CC BY-NC-SA.
