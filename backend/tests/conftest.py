import pytest
import spacy.util

from pictostory.nlp import DEFAULT_MODEL


def pytest_collection_modifyitems(config, items):
    if spacy.util.is_package(DEFAULT_MODEL):
        return
    skip = pytest.mark.skip(reason=f"modèle spaCy {DEFAULT_MODEL} non installé")
    for item in items:
        if "needs_model" in item.keywords:
            item.add_marker(skip)
