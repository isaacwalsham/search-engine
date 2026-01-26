from __future__ import annotations

from collections import Counter
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def default_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())

DEFAULT_PRF_EXCLUDE: Set[str] = {
    "name",
    "position",
    "birthplace",
    "national",
    "team",
    "nationality",
}

QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "english": ["england"],
    "british": ["england"],
    "scottish": ["scotland"],
    "welsh": ["wales"],
    "french": ["france"],
    "spanish": ["spain"],
    "italian": ["italy"],
    "german": ["germany"],
    "irish": ["ireland"],
    "argentinian": ["argentina"],
    "argentine": ["argentina"],

    "midfielder": ["midfield"],
    "forwards": ["forward"],
    "strikers": ["forward"],
    "striker": ["forward"],
    "defenders": ["defender"],

    "fast": ["quick", "pace"],
    "quick": ["fast"],
    "left": ["left-sided", "leftsided"],
    "wing": ["wide"],
}

def expand_query(
    query: str,
    enable: bool = True,
    *,
    tokenize: Optional["TokenizeFn"] = None,
) -> str:
    """Expand a raw query string using a simple synonym/expansion dictionary.

    - Lowercases the query.
    - Tokenises the query (defaults to `.lower().split()` unless a tokenizer is provided).
    - For each token, adds any extra terms defined in QUERY_EXPANSIONS.
    - Returns a new, expanded query string.

    If `enable` is False, the original query is returned unchanged.

    NOTE:
      This function is intentionally simple and deterministic, matching the lecture/lab
      style approach of dictionary-based query expansion.

    Args:
      query: raw query string
      enable: whether to apply dictionary expansion
      tokenize: optional tokenization function (e.g., `indexing.tokenize_for_index`)

    Returns:
      Expanded query string.
    """
    if not enable or not isinstance(query, str):
        return query

    if tokenize is None:
        tokenize = default_tokenize

    original_tokens = [t.lower() for t in tokenize(query) if t]
    expanded_tokens: List[str] = []

    for token in original_tokens:
        expanded_tokens.append(token)
        extras = QUERY_EXPANSIONS.get(token, [])
        expanded_tokens.extend(extras)

    return _dedupe_keep_order(expanded_tokens)

TokenizeFn = Callable[[str], List[str]]

def pseudo_relevance_feedback(
    query: str,
    top_docs: Sequence[str],
    *,
    tokenize: Optional[TokenizeFn] = None,
    fb_docs: Optional[int] = None,
    fb_terms: int = 5,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Return PRF expansion terms from the top retrieved documents.

    This implements a simple PRF variant suitable for coursework:
      - Assume the top `fb_docs` documents are relevant.
      - Extract candidate terms by tokenising those docs.
      - Rank candidate terms by frequency across feedback docs.
      - Return the top `fb_terms` terms (excluding the original query terms).

    Args:
      query: original query string
      top_docs: a sequence of document texts (already retrieved by the ranker)
      tokenize: function to tokenize text. If None, uses `.lower().split()`.
      fb_docs: number of feedback docs to use. If None, uses all provided.
      fb_terms: number of feedback terms to return
      exclude: optional set of additional terms to exclude

    Returns:
      A list of expansion terms (strings) to append to the query.

    Why this design:
      - Keeps PRF independent of ranking/indexing modules (avoids circular imports).
      - Lets `evaluation.py`/`main.py` decide what the "top docs" are.
      - Uses the same tokenizer as your index when you pass `indexing.tokenize_for_index`.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    if not top_docs:
        return []

    if fb_terms <= 0:
        return []

    if tokenize is None:
        tokenize = default_tokenize

    q_terms = set(tokenize(query))

    q_terms |= set(DEFAULT_PRF_EXCLUDE)

    if exclude:
        q_terms |= set(exclude)

    docs_to_use = top_docs[:fb_docs] if fb_docs is not None else top_docs

    counts: Counter[str] = Counter()
    for text in docs_to_use:
        if not text:
            continue
        for tok in tokenize(text):
            if not tok:
                continue

            if len(tok) <= 2:
                continue
            if tok in q_terms:
                continue
            counts[tok] += 1

    if not counts:
        return []

    ranked: List[Tuple[str, int]] = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ranked[:fb_terms]]

def expand_query_prf(
    query: str,
    *,
    enable_dict: bool = True,
    enable_prf: bool = False,
    top_docs: Optional[Sequence[str]] = None,
    tokenize: Optional[TokenizeFn] = None,
    fb_docs: int = 5,
    fb_terms: int = 5,
) -> str:
    """Combined expansion: dictionary expansion + optional PRF.

    This is the function you will typically call from `main.py` / `evaluation.py`:
      1) Apply dictionary expansion if enabled.
      2) If PRF is enabled and `top_docs` provided, append PRF terms.

    Returns a single expanded query string.
    """
    if not isinstance(query, str):
        return query

    expanded = expand_query(query, enable=enable_dict, tokenize=tokenize)

    if enable_prf and top_docs:
        prf_terms = pseudo_relevance_feedback(
            expanded,
            top_docs,
            tokenize=tokenize,
            fb_docs=fb_docs,
            fb_terms=fb_terms,
        )
        if prf_terms:
            expanded = " ".join([expanded, " ".join(prf_terms)])

    if tokenize is None:
        tokens = default_tokenize(expanded)
    else:
        tokens = [t.lower() for t in tokenize(expanded) if t]

    return _dedupe_keep_order(tokens)

def _dedupe_keep_order(tokens: Iterable[str]) -> str:
    seen: Set[str] = set()
    out: List[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    return " ".join(out)

if __name__ == "__main__":
    examples = [
        "english players",
        "fast left sided defender",
        "french forward",
        "spanish midfielder",
    ]

    for q in examples:
        expanded = expand_query(q)
        print(f"Original: {q}")
        print(f"Expanded: {expanded}")
        print("-" * 40)

    demo_query = "english players"
    demo_top_docs = [
        "Name: Darren Anderton, Position: Midfield, Birthplace: Southampton, England, National team: England",
        "Name: Jamie Redknapp, Position: Midfield, Birthplace: Barton-on-Sea, England, National team: England",
    ]
    prf_expanded = expand_query_prf(
        demo_query,
        enable_dict=True,
        enable_prf=True,
        top_docs=demo_top_docs,
        fb_docs=2,
        fb_terms=5,
    )
    print("PRF demo")
    print(f"Original: {demo_query}")
    print(f"Expanded: {prf_expanded}")

