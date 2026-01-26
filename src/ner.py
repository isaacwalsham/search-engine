from __future__ import annotations
from functools import lru_cache
from typing import Set, Iterable, Tuple

import spacy

_NLP = spacy.load("en_core_web_sm")

_ALLOWED_LABELS = {"PERSON", "GPE", "LOC", "NORP", "LANGUAGE"}

_DEMONYM_TO_GPE = {
    "english": "england",
    "french": "france",
    "spanish": "spain",
    "argentine": "argentina",
    "argentinian": "argentina",
    "italian": "italy",
    "german": "germany",
    "belgian": "belgium",
    "swiss": "switzerland",
    "irish": "ireland",
    "croatian": "croatia",
    "nigerian": "nigeria",
    "swedish": "sweden",
    "brazilian": "brazil",
    "uruguayan": "uruguay",
    "ivorian": "ivory coast",
}

_GPE_TO_DEMONYM = {
    v: k for (k, v) in _DEMONYM_TO_GPE.items()
}

def _normalise_entity(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation-ish edges."""
    if not text:
        return ""

    cleaned = " ".join(text.strip().lower().split())

    while cleaned and cleaned[0] in "\"'`([{<,;:!?/\\":  # fixed escape for backslash
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] in "\"'`)]}>.,;:!?/\\":  # fixed escape for backslash
        cleaned = cleaned[:-1]

    cleaned = " ".join(cleaned.split())
    return cleaned

@lru_cache(maxsize=4096)
def extract_entities(text: str) -> Set[str]:
    """
    Extract a set of normalised named entities from text.
    Cached because we call it often during indexing/search.
    """
    if not text or not text.strip():
        return set()

    doc = _NLP(text)
    ents = set()
    for ent in doc.ents:
        if ent.label_ not in _ALLOWED_LABELS:
            continue

        cleaned = _normalise_entity(ent.text)
        if not cleaned:
            continue

        ents.add(cleaned)

        mapped_gpe = _DEMONYM_TO_GPE.get(cleaned)
        if mapped_gpe:
            ents.add(mapped_gpe)

        mapped_demonym = _GPE_TO_DEMONYM.get(cleaned)
        if mapped_demonym:
            ents.add(mapped_demonym)

    return ents

def extract_query_entities(query: str) -> Set[str]:
    """Extract entities for the query (do NOT cache across different queries)."""
    if not query or not query.strip():
        return set()

    doc = _NLP(query)
    ents: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ not in _ALLOWED_LABELS:
            continue

        cleaned = _normalise_entity(ent.text)
        if not cleaned:
            continue

        ents.add(cleaned)

        mapped_gpe = _DEMONYM_TO_GPE.get(cleaned)
        if mapped_gpe:
            ents.add(mapped_gpe)

        mapped_demonym = _GPE_TO_DEMONYM.get(cleaned)
        if mapped_demonym:
            ents.add(mapped_demonym)

    return ents

def ner_overlap_score(query_entities: Set[str], doc_entities: Set[str]) -> int:
    """Simple overlap count (how many query entities appear in doc entities)."""
    if not query_entities or not doc_entities:
        return 0
    return len(query_entities.intersection(doc_entities))