import os
import math
from typing import Dict, List, Set, Optional, Any

import pandas as pd

from preprocessing import simple_tokenize

PREPROCESSING_MODE = os.environ.get("IR_PREPROCESSING_MODE", "lemma").strip().lower() or "lemma"
if PREPROCESSING_MODE not in {"none", "stem", "lemma"}:
    PREPROCESSING_MODE = "lemma"

def tokenize_for_index(text: str) -> List[str]:
    """Wrapper around simple_tokenize that applies a chosen preprocessing mode.

    This makes it easy to switch between different experimental configurations.
    """
    if PREPROCESSING_MODE == "none":
        return simple_tokenize(
            text,
            use_nltk_stopwords=False,
            use_custom_stopwords=False,
            use_stemming=False,
            use_lemmatization=False,
        )
    elif PREPROCESSING_MODE == "stem":
        return simple_tokenize(
            text,
            use_nltk_stopwords=True,
            use_custom_stopwords=True,
            use_stemming=True,
            use_lemmatization=False,
        )
    else:
        return simple_tokenize(
            text,
            use_nltk_stopwords=True,
            use_custom_stopwords=True,
            use_stemming=False,
            use_lemmatization=True,
        )

def extract_player_name(url: str) -> str:
    """Extract a readable player name from the URL string.

    Example:
      'soccer/www.worldfootballers.com/alvaro-recoba-123.html'
      -> 'Alvaro Recoba'
    """
    if not isinstance(url, str):
        return ""

    last_part = url.split("/")[-1]

    if "." in last_part:
        last_part = last_part.split(".")[0]

    parts = last_part.split("-")
    cleaned_parts = []
    for p in parts:
        if p.isdigit():
            break
        cleaned_parts.append(p)

    name = " ".join(cleaned_parts).strip().title()
    return name

def load_soccer_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data", "raw", "soccer.csv")

    df = pd.read_csv(csv_path, dtype=str)
    df = df.fillna("")
    return df

def build_player_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build a list of simple 'document' dicts from the raw dataframe.

    Each document has:
      - doc_id (int)
      - name (str)
      - text (str): combined searchable text
    """
    documents = []

    for idx, row in df.iterrows():
        url = row.get("url", "")
        nationality = row.get("STRING : nationality", "") or ""
        position = row.get("STRING : position", "") or ""
        birthplace = row.get("STRING : birthplace", "") or ""
        national_team = row.get("STRING : national_team", "") or ""

        name = extract_player_name(url)

        parts = []
        if name:
            parts.append(f"Name: {name}")
        if position:
            parts.append(f"Position: {position}")
        if birthplace:
            parts.append(f"Birthplace: {birthplace}")
        if national_team:
            parts.append(f"National team: {national_team}")
        if nationality:
            parts.append(f"Nationality: {nationality}")

        combined_text = ", ".join(parts)

        doc = {
            "doc_id": int(idx),
            "name": name,
            "text": combined_text,
            "url": url,
        }
        documents.append(doc)

    return documents

def build_inverted_index(
    documents: List[Dict[str, Any]],
    return_entities: bool = False,
    compute_doc_norms: bool = True,
):
    """Build a simple inverted index from a list of document dicts.

    Args:
      documents: list of document dicts with keys: doc_id, name, text
      return_entities: if True, also extract named entities per document (SpaCy)
      compute_doc_norms: if True, precompute TF-IDF L2 norms per document for cosine-normalized TF-IDF ranking

    Returns (no NER):
      - vocab: dict[str, int]  (term -> term_id)
      - postings: dict[int, dict[int, int]]  (term_id -> {doc_id: tf})
      - doc_id_to_name: dict[int, str]  (doc_id -> player name)
      - doc_lengths: dict[int, int]  (doc_id -> number of tokens after preprocessing)
      - avgdl: float  (average document length)
      - doc_norms: dict[int, float] (doc_id -> L2 norm of TF-IDF doc vector)

    Returns (with NER):
      - vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl, doc_norms

    NOTE:
      `doc_norms` is used by cosine-normalized TF-IDF scoring. Callers must unpack
      this extra return value.
    """
    vocab: Dict[str, int] = {}
    postings: Dict[int, Dict[int, int]] = {}
    doc_id_to_name: Dict[int, str] = {}
    doc_entities: Dict[int, Set[str]] = {}
    doc_lengths: Dict[int, int] = {}
    total_length = 0
    next_term_id = 0
    doc_norms: Optional[Dict[int, float]] = {} if compute_doc_norms else None

    extract_entities = None
    if return_entities:
        try:
            from ner import extract_entities as _extract_entities

            extract_entities = _extract_entities
        except Exception as e:
            raise RuntimeError(
                "NER extraction requested (return_entities=True) but `ner.py`/SpaCy "
                "is not available. Install SpaCy + model and ensure src/ner.py exists."
            ) from e

    for doc in documents:

        doc_id = int(doc.get("doc_id", 0))
        doc_name = doc.get("name", "")
        doc_id_to_name[doc_id] = doc_name

        text = doc.get("text", "")

        if return_entities and extract_entities is not None:
            doc_entities[doc_id] = extract_entities(text)

        tokens = tokenize_for_index(text)

        doc_len = len(tokens)
        doc_lengths[doc_id] = doc_len
        total_length += doc_len

        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        for term, tf in term_counts.items():
            if term not in vocab:
                vocab[term] = next_term_id
                next_term_id += 1
            term_id = vocab[term]

            if term_id not in postings:
                postings[term_id] = {}

            postings[term_id][doc_id] = tf

    avgdl = (total_length / len(documents)) if documents else 0.0

    if compute_doc_norms:
        N = len(documents)
        if N == 0:
            doc_norms = {}
        else:
            doc_sq_sums: Dict[int, float] = {doc_id: 0.0 for doc_id in doc_lengths.keys()}

            for term_id, doc_tf in postings.items():
                df = len(doc_tf)
                if df == 0:
                    continue
                idf = math.log((N + 1) / (df + 1)) + 1.0

                for doc_id, tf in doc_tf.items():
                    w = tf * idf
                    doc_sq_sums[doc_id] = doc_sq_sums.get(doc_id, 0.0) + (w * w)

            doc_norms = {}
            for doc_id, sq in doc_sq_sums.items():

                doc_norms[doc_id] = math.sqrt(sq) if sq > 0 else 1.0
    else:
        doc_norms = None

    if return_entities:
        return vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl, doc_norms

    return vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_norms

if __name__ == "__main__":
    df = load_soccer_data()

    print("Loaded soccer.csv")
    print("Number of rows:", len(df))

    print("\nColumn names:")
    print(df.columns)

    docs = build_player_documents(df)
    print("\nExample documents (first 3):")
    for d in docs[:3]:
        print("-------------")
        print("doc_id:", d["doc_id"])
        print("name  :", d["name"])
        print("url   :", d["url"])
        print("text  :", d["text"])

    vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_norms = build_inverted_index(
        docs, compute_doc_norms=True
    )

    print("\nInverted index built.")
    print("Vocab size:", len(vocab))
    print("Number of terms with postings:", len(postings))
    print("Average document length (avgdl):", round(avgdl, 2))
    if doc_norms is None:
        print("doc_norms: not computed")
    else:
        print("Example doc_norms[0] (if exists):", round(doc_norms.get(0, 0.0), 4))

    example_term = "uruguay"
    if example_term in vocab:
        term_id = vocab[example_term]
        print(f"\nExample term: '{example_term}' has term_id {term_id}")
        print("Postings (doc_id -> tf):", postings[term_id])
    else:
        print("\nExample term 'uruguay' not found in vocab.")