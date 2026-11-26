# Architecture

Le document d'architecture complet est tenu à jour ici : https://claude.ai/artifact/USMxeTHUaLeaYrxhrCkUTm

Résumé :

1. **Atelier enseignant** (ordinateur, hors ligne) : photo de la page → OCR → relecture du texte → pipeline NLP (spaCy, personnages, reprises, désambiguïsation) → validation des cas douteux.
2. **Ontologie** : mots, sens, pictogrammes ARASAAC, personnages. Elle fournit les candidats et mémorise les choix validés.
3. **LLM local** (Ollama) : consulté seulement pour les mots ambigus et les reprises difficiles.
4. **App enfant** (tablette, Flutter) : lit les histoires annotées sans connexion.

## Étapes du MVP

1. Corpus de quelques histoires réelles, test de l'OCR, modèle de l'ontologie.
2. Pipeline sans IA (ce dépôt, `backend/`) : point de comparaison.
3. Personnages et reprises : comparer règles, Coreferee et LLM local.
4. Désambiguïsation par embeddings puis LLM local.
5. API et écran de validation enseignant.
6. App tablette Flutter.
7. Tests avec des ergothérapeutes et des enseignants.
