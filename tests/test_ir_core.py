"""
Core tests for the IR search engine.

Run from project root (ir_search_engine/):

  PYTHONPATH=src pytest -q

Optional: create pytest.ini to avoid PYTHONPATH (see chat).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set

import pytest

from indexing import build_inverted_index
from ranking import search
from query_expansion import expand_query

@pytest.fixture()
def tiny_docs() -> List[Dict[str, Any]]:

    return [
        {
            "doc_id": 0,
            "name": "Diego Placente",
            "text": "Name: Diego Placente, Position: Defender, Birthplace: Capital Federal, Argentina, National team: Argentina",
            "url": "soccer/www.worldfootballers.com/diego-placente-12.html",
        },
        {
            "doc_id": 1,
            "name": "Sylvain Wiltord",
            "text": "Name: Sylvain Wiltord, Position: Forward, Birthplace: Neuilly-sur-Marne, France, National team: France",
            "url": "soccer/www.worldfootballers.com/sylvain-wiltord-53.html",
        },
        {
            "doc_id": 2,
            "name": "Benito Carbone",
            "text": "Name: Benito Carbone, Position: Midfield, Birthplace: Bagnara Calabra, Italy, National team: Italy",
            "url": "soccer/www.worldfootballers.com/benito-carbone-2.html",
        },
    ]

@pytest.fixture()
def tiny_index(tiny_docs):

    vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_norms = build_inverted_index(
        tiny_docs, return_entities=False, compute_doc_norms=True
    )
    return {
        "docs": tiny_docs,
        "vocab": vocab,
        "postings": postings,
        "doc_id_to_name": doc_id_to_name,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "doc_norms": doc_norms,
    }

def test_build_inverted_index_shapes(tiny_index):
    assert isinstance(tiny_index["vocab"], dict)
    assert isinstance(tiny_index["postings"], dict)
    assert isinstance(tiny_index["doc_id_to_name"], dict)
    assert isinstance(tiny_index["doc_lengths"], dict)
    assert isinstance(tiny_index["avgdl"], float)
    assert isinstance(tiny_index["doc_norms"], dict)

    assert len(tiny_index["doc_id_to_name"]) == 3
    assert all(isinstance(k, int) for k in tiny_index["doc_lengths"].keys())
    assert tiny_index["avgdl"] > 0

def test_doc_norms_positive(tiny_index):
    norms = tiny_index["doc_norms"]
    assert set(norms.keys()) == {0, 1, 2}
    assert all(norms[d] > 0 for d in norms)

def test_expand_query_returns_string():
    q = "fast left sided defender"
    out = expand_query(q, enable=True)
    assert isinstance(out, str)
    assert len(out.strip()) > 0

def test_expand_query_disable_is_passthrough():
    import re

    def tokens(s: str):

        return {t.lower() for t in re.findall(r"\w+", s)}

    q = "Placente, Diego"
    out = expand_query(q, enable=False)

    assert tokens(q).issubset(tokens(out))

def _ids(results: List[Tuple[int, float]]) -> List[int]:
    return [doc_id for doc_id, _score in results]

def test_search_tfidf_finds_exact_player(tiny_index):
    res = search(
        "Placente Diego",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="tfidf",
        doc_norms=tiny_index["doc_norms"],
        tfidf_normalize=True,
        topk=5,
    )
    assert res, "Expected non-empty results"
    assert _ids(res)[0] == 0  

def test_search_bm25_finds_exact_player(tiny_index):
    res = search(
        "Sylvain Wiltord",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        k1=1.2,
        b=0.75,
        topk=5,
    )
    assert res, "Expected non-empty results"
    assert _ids(res)[0] == 1  

def test_search_topk_respected(tiny_index):
    res = search(
        "France",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=1,
    )
    assert len(res) == 1

def test_search_deterministic(tiny_index):
    q = "Defender Argentina"
    r1 = search(
        q,
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=5,
    )
    r2 = search(
        q,
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=5,
    )
    assert r1 == r2

def test_threshold_min_score_filters_if_supported(tiny_index):

    import inspect

    sig = inspect.signature(search)
    if "min_score" not in sig.parameters:
        pytest.skip("ranking.search does not support min_score/min_score_ratio parameters")

    res_all = search(
        "Diego",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=5,
        min_score=None,
        min_score_ratio=None,
    )
    assert res_all

    res_none = search(
        "Diego",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=5,
        min_score=1e9,
        min_score_ratio=None,
    )
    assert res_none == []

def test_threshold_ratio_filters_if_supported(tiny_index):
    import inspect

    sig = inspect.signature(search)
    if "min_score_ratio" not in sig.parameters:
        pytest.skip("ranking.search does not support min_score/min_score_ratio parameters")

    res = search(
        "France",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=5,
        min_score=None,
        min_score_ratio=None,
    )
    assert res

    best = res[0][1]

    res_ratio = search(
        "France",
        tiny_index["vocab"],
        tiny_index["postings"],
        tiny_index["doc_id_to_name"],
        ranker="bm25",
        doc_lengths=tiny_index["doc_lengths"],
        avgdl=tiny_index["avgdl"],
        topk=5,
        min_score=None,
        min_score_ratio=1.0,
    )
    assert res_ratio
    assert all(score == best for _doc_id, score in res_ratio)