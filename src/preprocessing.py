import re
from typing import List, Optional, Set

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover
    stopwords = None  

    PorterStemmer = None  

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None  

_STEMMER: Optional[object] = None
_NLP = None
_NLTK_STOPWORDS: Optional[Set[str]] = None

_FALLBACK_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
}

def _get_stemmer():
    global _STEMMER
    if _STEMMER is None:
        if PorterStemmer is None:
            _STEMMER = None
        else:
            _STEMMER = PorterStemmer()
    return _STEMMER

def _get_nltk_stopwords() -> Set[str]:
    """Return NLTK stopwords if available, otherwise fallback set."""
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS is not None:
        return _NLTK_STOPWORDS

    if stopwords is None:
        _NLTK_STOPWORDS = set(_FALLBACK_STOPWORDS)
        return _NLTK_STOPWORDS

    try:
        _NLTK_STOPWORDS = set(stopwords.words("english"))
    except Exception:

        _NLTK_STOPWORDS = set(_FALLBACK_STOPWORDS)

    return _NLTK_STOPWORDS

def _get_spacy_nlp():
    """Lazy-load spaCy model if installed; otherwise return None."""
    global _NLP
    if _NLP is not None:
        return _NLP

    if spacy is None:
        _NLP = None
        return _NLP

    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = None

    return _NLP

CUSTOM_STOPWORDS = {

    "name",
    "position",
    "birthplace",
    "national",
    "team",
    "nationality",
    "club",
    "clubs",
    "country",
}

def simple_tokenize(
    text: str,
    use_nltk_stopwords: bool = True,
    use_custom_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
) -> List[str]:
    """
    Tokenizer:
    - lowercase
    - remove punctuation
    - split into words
    - optional stopword removal (NLTK + custom)
    - optional stemming
    - optional lemmatization
    - if NLTK stopwords are unavailable, a small fallback list is used
    - if SpaCy model is unavailable, lemmatization is skipped
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]

    if use_nltk_stopwords:
        sw = _get_nltk_stopwords()
        tokens = [t for t in tokens if t not in sw]

    if use_custom_stopwords:
        tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]

    if use_stemming:
        stemmer = _get_stemmer()
        if stemmer is not None:
            tokens = [stemmer.stem(t) for t in tokens]

    if use_lemmatization and tokens:
        nlp = _get_spacy_nlp()
        if nlp is not None:
            doc = nlp(" ".join(tokens))
            tokens = [t.lemma_ for t in doc]

    return tokens

if __name__ == "__main__":
    sample = "Name: Antoine Sibierski, Position: Midfield, Birthplace: Lille, France"
    print("Original:", sample)
    print("Tokens (lemmatization - may fallback):", simple_tokenize(sample, use_lemmatization=True))
    print("Tokens (stemming):", simple_tokenize(sample, use_stemming=True))
    print("Tokens (no processing):", simple_tokenize(sample, use_nltk_stopwords=False, use_custom_stopwords=False, use_stemming=False, use_lemmatization=False))