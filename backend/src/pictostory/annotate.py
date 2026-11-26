"""Pipeline complet : texte → histoire annotée avec pictogrammes et personnages."""

from __future__ import annotations

from .models import Sentence, Story, Token
from .nlp import (
    CONTENT_POS,
    KnownCharacter,
    add_known_characters,
    find_characters,
    load_nlp,
    resolve_pronouns,
    resolve_speakers,
)
from .coref_llm import ChoiceLLM, refine_with_llm
from .pictograms import PictogramProvider
from .wsd import CONFIDENCE_THRESHOLD, BaselineDisambiguator, Disambiguator


def annotate_story(
    text: str,
    provider: PictogramProvider,
    disambiguator: Disambiguator | None = None,
    known_characters: list[str] | None = None,
    model: str | None = None,
    llm: ChoiceLLM | None = None,
) -> Story:
    nlp = load_nlp(model) if model else load_nlp()
    known = [KnownCharacter.parse(k) for k in known_characters or []]
    add_known_characters(nlp, known)
    disambiguator = disambiguator or BaselineDisambiguator()

    doc = nlp(text)
    characters, token_to_char = find_characters(doc, known)
    references = resolve_pronouns(doc, characters, token_to_char)
    references = resolve_speakers(doc, references)
    referents: dict[int, str] = {}
    if llm is not None:
        references, referents = refine_with_llm(doc, characters, token_to_char, references, llm)

    sentences = []
    for sent in doc.sents:
        tokens = []
        for tok in sent:
            if tok.is_space:
                continue
            token = Token(
                text=tok.text,
                lemma=tok.lemma_,
                pos=tok.pos_,
                character_id=references.get(tok.i),
                referent=referents.get(tok.i),
            )
            if tok.pos_ in CONTENT_POS:
                token.candidates = provider.search(tok.lemma_)
                token.pictogram, token.confidence = disambiguator.choose(
                    tok.lemma_, sent.text, token.candidates
                )
                # Un pictogramme douteux n'est pas affiché tant que l'enseignant ne l'a pas validé.
                token.to_validate = bool(token.candidates) and token.confidence < CONFIDENCE_THRESHOLD
            tokens.append(token)
        sentences.append(Sentence(text=sent.text, tokens=tokens))

    return Story(sentences=sentences, characters=characters)
