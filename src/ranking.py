import math
from typing import Dict, List, Tuple, Optional, Set, Literal, Sequence

from indexing import tokenize_for_index as tokenize_query

def compute_tfidf_doc_norms(
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
) -> Dict[int, float]:
    """Pre-compute document vector norms ||d|| for TF-IDF cosine normalisation."""
    N = len(doc_id_to_name)
    if N == 0:
        return {}

    norm_sq: Dict[int, float] = {}

    for _term_id, doc_tf in postings.items():
        df = len(doc_tf)
        if df == 0:
            continue

        idf = math.log((N + 1) / (df + 1)) + 1.0

        for doc_id, tf in doc_tf.items():
            w = tf * idf
            norm_sq[doc_id] = norm_sq.get(doc_id, 0.0) + (w * w)

    return {doc_id: math.sqrt(v) for doc_id, v in norm_sq.items()}

def compute_tfidf_scores(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    doc_norms: Optional[Dict[int, float]] = None,
    normalize: bool = True,
) -> Dict[int, float]:
    """Compute TF-IDF scores.

    If normalize=True, returns cosine similarity scores (dot(d, q) / (||d||*||q||)).
    """
    tokens = tokenize_query(query)
    if not tokens:
        return {}

    N = len(doc_id_to_name)
    if N == 0:
        return {}

    q_tf: Dict[str, int] = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    scores: Dict[int, float] = {}
    q_norm_sq = 0.0

    for term, qf in q_tf.items():
        term_id = vocab.get(term)
        if term_id is None:
            continue

        doc_tf = postings.get(term_id, {})
        df = len(doc_tf)
        if df == 0:
            continue

        idf = math.log((N + 1) / (df + 1)) + 1.0
        q_w = qf * idf
        q_norm_sq += q_w * q_w

        for doc_id, tf in doc_tf.items():
            d_w = tf * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + (d_w * q_w)

    if not normalize:
        return scores

    q_norm = math.sqrt(q_norm_sq)
    if q_norm == 0.0:
        return {}

    if doc_norms is None:
        doc_norms = compute_tfidf_doc_norms(postings, doc_id_to_name)

    for doc_id in list(scores.keys()):
        d_norm = doc_norms.get(doc_id, 0.0)
        if d_norm > 0.0:
            scores[doc_id] /= (d_norm * q_norm)
        else:
            del scores[doc_id]

    return scores

def compute_bm25_scores(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    doc_lengths: Dict[int, int],
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[int, float]:
    """Compute BM25 scores (includes query term frequency)."""
    tokens = tokenize_query(query)
    if not tokens:
        return {}

    N = len(doc_id_to_name)
    if N == 0:
        return {}

    if avgdl <= 0:
        avgdl = 1.0

    scores: Dict[int, float] = {}

    q_tf: Dict[str, int] = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    for term, qf in q_tf.items():
        term_id = vocab.get(term)
        if term_id is None:
            continue

        doc_tf = postings.get(term_id, {})
        df = len(doc_tf)
        if df == 0:
            continue

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id, tf in doc_tf.items():
            dl = doc_lengths.get(doc_id, 0)
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            if denom == 0:
                continue

            score = qf * idf * (tf * (k1 + 1.0) / denom)
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    return scores

def _apply_ner_boost(
    scores: Dict[int, float],
    query: str,
    doc_entities: Optional[Dict[int, Set[str]]],
    use_ner_boost: bool,
    ner_boost_weight: float,
) -> Dict[int, float]:
    """Apply NER overlap boosting if enabled."""
    if not scores or not use_ner_boost or doc_entities is None:
        return scores

    try:
        from ner import extract_query_entities, ner_overlap_score
    except Exception:
        return scores

    query_entities = extract_query_entities(query)
    if not query_entities:
        return scores

    for doc_id in list(scores.keys()):
        overlap = ner_overlap_score(query_entities, doc_entities.get(doc_id, set()))
        if overlap > 0:
            scores[doc_id] *= (1.0 + ner_boost_weight * overlap)

    return scores

def _apply_score_filters(
    ranked: List[Tuple[int, float]],
    *,
    min_score: Optional[float] = None,
    min_score_ratio: Optional[float] = None,
    topk: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Filter a ranked list by an absolute score threshold and/or a ratio-to-best threshold.

    Args:
      ranked: list of (doc_id, score) sorted descending.
      min_score: keep results with score >= min_score.
      min_score_ratio: keep results with score >= (best_score * min_score_ratio).
        (e.g., 0.5 keeps results at least half as good as the best result).
      topk: after filtering, cap the output length.

    Notes:
      - If `ranked` is empty, returns [].
      - Negative thresholds are treated as disabled.
    """
    if not ranked:
        return []

    if min_score is not None and min_score < 0:
        min_score = None
    if min_score_ratio is not None and min_score_ratio < 0:
        min_score_ratio = None

    best_score = ranked[0][1]
    ratio_cutoff = None
    if min_score_ratio is not None:
        ratio_cutoff = best_score * float(min_score_ratio)

    filtered: List[Tuple[int, float]] = []
    for doc_id, score in ranked:
        if min_score is not None and score < float(min_score):
            continue
        if ratio_cutoff is not None and score < ratio_cutoff:
            continue
        filtered.append((doc_id, score))

    if topk is not None:
        topk_i = int(topk)
        if topk_i > 0:
            return filtered[:topk_i]

    return filtered

def tfidf_search(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    doc_norms: Optional[Dict[int, float]] = None,
    doc_entities: Optional[Dict[int, Set[str]]] = None,
    use_ner_boost: bool = False,
    ner_boost_weight: float = 0.25,
    normalize: bool = True,
) -> List[Tuple[int, float]]:
    scores = compute_tfidf_scores(
        query,
        vocab,
        postings,
        doc_id_to_name,
        doc_norms=doc_norms,
        normalize=normalize,
    )
    scores = _apply_ner_boost(scores, query, doc_entities, use_ner_boost, ner_boost_weight)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def bm25_search(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    doc_lengths: Dict[int, int],
    avgdl: float,
    doc_entities: Optional[Dict[int, Set[str]]] = None,
    use_ner_boost: bool = False,
    ner_boost_weight: float = 0.25,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    scores = compute_bm25_scores(
        query,
        vocab,
        postings,
        doc_id_to_name,
        doc_lengths,
        avgdl,
        k1=k1,
        b=b,
    )
    scores = _apply_ner_boost(scores, query, doc_entities, use_ner_boost, ner_boost_weight)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

RankerName = Literal["tfidf", "bm25"]

def search(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    *,
    ranker: RankerName = "tfidf",

    doc_lengths: Optional[Dict[int, int]] = None,
    avgdl: Optional[float] = None,

    doc_norms: Optional[Dict[int, float]] = None,
    tfidf_normalize: bool = True,

    doc_entities: Optional[Dict[int, Set[str]]] = None,
    use_ner_boost: bool = False,
    ner_boost_weight: float = 0.25,

    k1: float = 1.2,
    b: float = 0.75,

    min_score: Optional[float] = None,
    min_score_ratio: Optional[float] = None,
    topk: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Unified search wrapper.

    - ranker='tfidf': cosine-normalised TF-IDF (toggle with tfidf_normalize)
    - ranker='bm25' : Okapi BM25 (requires doc_lengths and avgdl)

    Notes:
      - `k1` and `b` are only used when ranker='bm25'.
      - NER boosting is only applied when `use_ner_boost=True` AND `doc_entities` is provided.
    """
    if ranker == "bm25":
        if doc_lengths is None or avgdl is None:
            raise ValueError(
                "BM25 selected but doc_lengths/avgdl not provided. "
                "Pass doc_lengths and avgdl from indexing.build_inverted_index."
            )

        ranked = bm25_search(
            query,
            vocab,
            postings,
            doc_id_to_name,
            doc_lengths,
            avgdl,
            doc_entities=doc_entities,
            use_ner_boost=use_ner_boost,
            ner_boost_weight=ner_boost_weight,
            k1=k1,
            b=b,
        )
        return _apply_score_filters(ranked, min_score=min_score, min_score_ratio=min_score_ratio, topk=topk)

    ranked = tfidf_search(
        query,
        vocab,
        postings,
        doc_id_to_name,
        doc_norms=doc_norms,
        doc_entities=doc_entities,
        use_ner_boost=use_ner_boost,
        ner_boost_weight=ner_boost_weight,
        normalize=tfidf_normalize,
    )
    return _apply_score_filters(ranked, min_score=min_score, min_score_ratio=min_score_ratio, topk=topk)

def _prf_expand_query_terms(
    query: str,
    *,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    feedback_doc_ids: List[int],
    top_terms: int = 5,
    min_term_len: int = 3,
) -> List[str]:
    """Select PRF expansion terms from the top-ranked documents.

    Lightweight PRF:
      - Pools TF over feedback documents.
      - Scores terms by TF * IDF (smoothed IDF).
      - Returns top terms not already in the query.

    Notes:
      - Uses the same tokenisation pipeline as indexing via `tokenize_query`.
      - Operates over the indexed vocabulary so it stays consistent with retrieval.
      - Uses postings to approximate pooled TF (works fine for small collections).
    """
    tokens = tokenize_query(query)
    query_terms = set(tokens)

    N = len(doc_id_to_name)
    if N == 0 or not feedback_doc_ids:
        return []

    feedback_set = set(feedback_doc_ids)

    pooled_tf: Dict[str, int] = {}

    for term, term_id in vocab.items():
        if term in query_terms:
            continue
        if len(term) < min_term_len:
            continue

        doc_tf = postings.get(term_id)
        if not doc_tf:
            continue

        tf_sum = 0
        for doc_id, tf in doc_tf.items():
            if doc_id in feedback_set:
                tf_sum += tf

        if tf_sum > 0:
            pooled_tf[term] = tf_sum

    if not pooled_tf:
        return []

    scored: List[Tuple[str, float]] = []
    for term, tf_sum in pooled_tf.items():
        term_id = vocab.get(term)
        if term_id is None:
            continue

        df = len(postings.get(term_id, {}))
        if df <= 0:
            continue

        idf = math.log((N + 1) / (df + 1)) + 1.0
        scored.append((term, tf_sum * idf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for (t, _s) in scored[:top_terms]]

def search_with_prf(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    *,
    ranker: RankerName = "tfidf",

    doc_lengths: Optional[Dict[int, int]] = None,
    avgdl: Optional[float] = None,

    doc_norms: Optional[Dict[int, float]] = None,
    tfidf_normalize: bool = True,

    doc_entities: Optional[Dict[int, Set[str]]] = None,
    use_ner_boost: bool = False,
    ner_boost_weight: float = 0.25,

    k1: float = 1.2,
    b: float = 0.75,

    prf_docs: int = 3,
    prf_terms: int = 5,

    prf_feedback_docs: Optional[int] = None,

    use_prf: Optional[bool] = None,

    min_score: Optional[float] = None,
    min_score_ratio: Optional[float] = None,
    topk: Optional[int] = None,
) -> Tuple[List[Tuple[int, float]], str]:
    """Two-pass retrieval with Pseudo-Relevance Feedback (PRF).

    Returns:
      (results, final_query)

    Behaviour:
      - Runs an initial retrieval.
      - Uses the top `prf_docs` docs (or `prf_feedback_docs` if provided) as feedback.
      - Selects `prf_terms` expansion terms from those feedback docs.
      - Re-runs retrieval with the expanded query.

    Notes:
      - If `use_prf` is explicitly set to False, PRF is disabled and this returns
        (first_pass_results, original_query).
      - This function is designed to be compatible with both new and older callers.
    """

    if use_prf is False:
        first_pass = search(
            query,
            vocab,
            postings,
            doc_id_to_name,
            ranker=ranker,
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            doc_norms=doc_norms,
            tfidf_normalize=tfidf_normalize,
            doc_entities=doc_entities,
            use_ner_boost=use_ner_boost,
            ner_boost_weight=ner_boost_weight,
            k1=k1,
            b=b,
            min_score=min_score,
            min_score_ratio=min_score_ratio,
            topk=topk,
        )
        return first_pass, query

    if prf_feedback_docs is not None:
        prf_docs = prf_feedback_docs

    prf_docs = max(1, int(prf_docs))
    prf_terms = max(0, int(prf_terms))

    first_pass = search(
        query,
        vocab,
        postings,
        doc_id_to_name,
        ranker=ranker,
        doc_lengths=doc_lengths,
        avgdl=avgdl,
        doc_norms=doc_norms,
        tfidf_normalize=tfidf_normalize,
        doc_entities=doc_entities,
        use_ner_boost=use_ner_boost,
        ner_boost_weight=ner_boost_weight,
        k1=k1,
        b=b,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        topk=topk,
    )

    if not first_pass or prf_terms == 0:
        return first_pass, query

    feedback_doc_ids = [doc_id for (doc_id, _score) in first_pass[:prf_docs]]

    extra_terms = _prf_expand_query_terms(
        query,
        vocab=vocab,
        postings=postings,
        doc_id_to_name=doc_id_to_name,
        feedback_doc_ids=feedback_doc_ids,
        top_terms=prf_terms,
    )

    if not extra_terms:
        return first_pass, query

    expanded_query = (query + " " + " ".join(extra_terms)).strip()

    second_pass = search(
        expanded_query,
        vocab,
        postings,
        doc_id_to_name,
        ranker=ranker,
        doc_lengths=doc_lengths,
        avgdl=avgdl,
        doc_norms=doc_norms,
        tfidf_normalize=tfidf_normalize,
        doc_entities=doc_entities,
        use_ner_boost=use_ner_boost,
        ner_boost_weight=ner_boost_weight,
        k1=k1,
        b=b,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        topk=topk,
    )

    return second_pass, expanded_query

def _centroid_tfidf_vector(
    doc_ids: List[int],
    *,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
) -> Dict[str, float]:
    """Compute the centroid (average) TF-IDF vector for a set of documents.

    Returns:
      dict[term -> avg tf-idf weight]

    Notes:
      - Uses the same smoothed IDF as TF-IDF scoring.
      - Efficient enough for this coursework-sized collection.
    """
    if not doc_ids:
        return {}

    N = len(doc_id_to_name)
    if N == 0:
        return {}

    target = set(doc_ids)
    accum: Dict[str, float] = {}

    for term, term_id in vocab.items():
        doc_tf = postings.get(term_id)
        if not doc_tf:
            continue

        df = len(doc_tf)
        if df == 0:
            continue

        idf = math.log((N + 1) / (df + 1)) + 1.0

        w_sum = 0.0
        hit = 0
        for d_id, tf in doc_tf.items():
            if d_id in target:
                w_sum += (tf * idf)
                hit += 1

        if hit > 0:

            accum[term] = w_sum / float(len(doc_ids))

    return accum

def _query_tfidf_vector(
    query: str,
    *,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
) -> Dict[str, float]:
    """Build a TF-IDF query vector (term -> weight) using the same IDF as indexing."""
    tokens = tokenize_query(query)
    if not tokens:
        return {}

    N = len(doc_id_to_name)
    if N == 0:
        return {}

    q_tf: Dict[str, int] = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    q_vec: Dict[str, float] = {}
    for term, qf in q_tf.items():
        term_id = vocab.get(term)
        if term_id is None:
            continue

        doc_tf = postings.get(term_id, {})
        df = len(doc_tf)
        if df == 0:
            continue

        idf = math.log((N + 1) / (df + 1)) + 1.0
        q_vec[term] = qf * idf

    return q_vec

def rocchio_expand_query(
    query: str,
    *,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    relevant_doc_ids: Optional[List[int]] = None,
    nonrelevant_doc_ids: Optional[List[int]] = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
    top_terms: int = 5,
    min_term_len: int = 3,

    rocchio_terms: Optional[int] = None,
) -> Tuple[str, Dict[str, float]]:
    """Rocchio relevance feedback to produce an expanded query.

    This uses *explicit* relevance judgements (qrels), unlike PRF.

    Returns:
      (expanded_query, rocchio_weights)

    Implementation choice (coursework-friendly):
      - Build Rocchio term weights in the TF-IDF vector space.
      - Convert back into a plain text query by appending the top positive-weight
        expansion terms that are not already in the query.

    Notes:
      - If no relevant docs are provided, this returns (query, {}).
      - Non-relevant docs are optional; if none are given, the negative term is skipped.
    """

    if rocchio_terms is not None:
        top_terms = rocchio_terms

    relevant_doc_ids = relevant_doc_ids or []
    nonrelevant_doc_ids = nonrelevant_doc_ids or []

    top_terms = max(0, int(top_terms))
    min_term_len = max(1, int(min_term_len))

    if not relevant_doc_ids:
        return query, {}

    q_vec = _query_tfidf_vector(
        query,
        vocab=vocab,
        postings=postings,
        doc_id_to_name=doc_id_to_name,
    )
    rel_centroid = _centroid_tfidf_vector(
        relevant_doc_ids,
        vocab=vocab,
        postings=postings,
        doc_id_to_name=doc_id_to_name,
    )

    nr_centroid: Dict[str, float] = {}
    if nonrelevant_doc_ids and gamma != 0.0:
        nr_centroid = _centroid_tfidf_vector(
            nonrelevant_doc_ids,
            vocab=vocab,
            postings=postings,
            doc_id_to_name=doc_id_to_name,
        )

    weights: Dict[str, float] = {}

    for term, w in q_vec.items():
        weights[term] = weights.get(term, 0.0) + alpha * w

    for term, w in rel_centroid.items():
        weights[term] = weights.get(term, 0.0) + beta * w

    for term, w in nr_centroid.items():
        weights[term] = weights.get(term, 0.0) - gamma * w

    query_terms = set(tokenize_query(query))

    candidates: List[Tuple[str, float]] = []
    for term, w in weights.items():
        if term in query_terms:
            continue
        if len(term) < min_term_len:
            continue
        if w <= 0.0:
            continue
        candidates.append((term, w))

    candidates.sort(key=lambda x: x[1], reverse=True)
    extra_terms = [t for (t, _w) in candidates[:top_terms]]

    if not extra_terms:
        return query, weights

    expanded_query = (query + " " + " ".join(extra_terms)).strip()
    return expanded_query, weights

def search_with_rocchio(
    query: str,
    vocab: Dict[str, int],
    postings: Dict[int, Dict[int, int]],
    doc_id_to_name: Dict[int, str],
    *,
    relevant_doc_ids: Optional[List[int]] = None,
    nonrelevant_doc_ids: Optional[List[int]] = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
    rocchio_terms: int = 5,
    min_term_len: int = 3,

    top_terms: Optional[int] = None,

    ranker: RankerName = "tfidf",
    doc_lengths: Optional[Dict[int, int]] = None,
    avgdl: Optional[float] = None,
    doc_norms: Optional[Dict[int, float]] = None,
    tfidf_normalize: bool = True,
    doc_entities: Optional[Dict[int, Set[str]]] = None,
    use_ner_boost: bool = False,
    ner_boost_weight: float = 0.25,
    k1: float = 1.2,
    b: float = 0.75,

    min_score: Optional[float] = None,
    min_score_ratio: Optional[float] = None,
    topk: Optional[int] = None,
) -> Tuple[List[Tuple[int, float]], str]:
    """Run supervised Rocchio feedback and then retrieve with the selected ranker.

    Returns:
      (results, final_query)

    Notes:
      - Rocchio expansion is computed in TF-IDF space, then applied as query text.
      - Retrieval itself still uses `ranker` (TF-IDF or BM25).
    """

    if top_terms is not None:
        rocchio_terms = top_terms

    relevant_doc_ids = relevant_doc_ids or []
    nonrelevant_doc_ids = nonrelevant_doc_ids or []

    rocchio_terms = max(0, int(rocchio_terms))
    min_term_len = max(1, int(min_term_len))

    if not relevant_doc_ids or rocchio_terms == 0:

        return (
            search(
                query,
                vocab,
                postings,
                doc_id_to_name,
                ranker=ranker,
                doc_lengths=doc_lengths,
                avgdl=avgdl,
                doc_norms=doc_norms,
                tfidf_normalize=tfidf_normalize,
                doc_entities=doc_entities,
                use_ner_boost=use_ner_boost,
                ner_boost_weight=ner_boost_weight,
                k1=k1,
                b=b,
                min_score=min_score,
                min_score_ratio=min_score_ratio,
                topk=topk,
            ),
            query,
        )

    expanded_query, _weights = rocchio_expand_query(
        query,
        vocab=vocab,
        postings=postings,
        doc_id_to_name=doc_id_to_name,
        relevant_doc_ids=relevant_doc_ids,
        nonrelevant_doc_ids=nonrelevant_doc_ids,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        top_terms=rocchio_terms,
        min_term_len=min_term_len,
    )

    results = search(
        expanded_query,
        vocab,
        postings,
        doc_id_to_name,
        ranker=ranker,
        doc_lengths=doc_lengths,
        avgdl=avgdl,
        doc_norms=doc_norms,
        tfidf_normalize=tfidf_normalize,
        doc_entities=doc_entities,
        use_ner_boost=use_ner_boost,
        ner_boost_weight=ner_boost_weight,
        k1=k1,
        b=b,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        topk=topk,
    )

    return results, expanded_query