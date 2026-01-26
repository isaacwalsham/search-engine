import argparse
import csv
import json
from datetime import datetime
from typing import List, Set, Optional, Dict, Any, Iterable, Tuple

from indexing import load_soccer_data, build_player_documents, build_inverted_index
from query_expansion import expand_query
from ranking import search, search_with_prf, search_with_rocchio

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a user-provided path. If relative, resolve it against PROJECT_ROOT."""
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p)
    return str(p.resolve())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation for the IR engine with toggles for TF-IDF/BM25, query expansion, "
            "Pseudo Relevance Feedback (PRF), NER boosting, TF-IDF cosine normalisation, "
            "thresholds, and optional batch metrics."
        )
    )

    parser.add_argument(
        "--ranker",
        choices=["tfidf", "bm25"],
        default="tfidf",
        help="Ranking function to use.",
    )

    parser.add_argument("--qe", dest="use_qe", action="store_true", help="Enable query expansion.")
    parser.add_argument("--no-qe", dest="use_qe", action="store_false", help="Disable query expansion.")
    parser.set_defaults(use_qe=True)

    parser.add_argument(
        "--prf",
        dest="use_prf",
        action="store_true",
        help="Enable pseudo-relevance feedback (two-pass retrieval).",
    )
    parser.add_argument(
        "--no-prf",
        dest="use_prf",
        action="store_false",
        help="Disable pseudo-relevance feedback.",
    )
    parser.set_defaults(use_prf=False)

    parser.add_argument(
        "--prf-docs",
        type=int,
        default=5,
        help="Number of top documents to assume relevant for PRF (default: 5).",
    )
    parser.add_argument(
        "--prf-terms",
        type=int,
        default=5,
        help="Number of expansion terms to add from pseudo-relevant docs (default: 5).",
    )

    parser.add_argument(
        "--ner",
        dest="use_ner",
        action="store_true",
        help="Enable NER extraction + NER overlap boosting.",
    )
    parser.add_argument(
        "--no-ner",
        dest="use_ner",
        action="store_false",
        help="Disable NER extraction + boosting.",
    )
    parser.set_defaults(use_ner=False)

    parser.add_argument("--k", type=int, default=10, help="k for metrics@k (default: 10).")
    parser.add_argument("--topk", type=int, default=10, help="How many results to show before judging relevance.")

    parser.add_argument(
        "--rocchio",
        dest="use_rocchio",
        action="store_true",
        help="Enable Rocchio relevance feedback (requires --batch and --qrels).",
    )
    parser.add_argument(
        "--no-rocchio",
        dest="use_rocchio",
        action="store_false",
        help="Disable Rocchio relevance feedback.",
    )
    parser.set_defaults(use_rocchio=False)

    parser.add_argument("--rocchio-alpha", type=float, default=1.0, help="Rocchio alpha.")
    parser.add_argument("--rocchio-beta", type=float, default=0.75, help="Rocchio beta.")
    parser.add_argument("--rocchio-gamma", type=float, default=0.15, help="Rocchio gamma.")
    parser.add_argument("--rocchio-terms", type=int, default=5, help="How many expansion terms to add (default 5).")
    parser.add_argument(
        "--rocchio-nonrel",
        type=int,
        default=5,
        help="How many top retrieved docs to treat as non-relevant (excluding relevant qrels) (default 5).",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Hide results with score below this absolute threshold (applied after top-k).",
    )
    parser.add_argument(
        "--min-score-ratio",
        type=float,
        default=None,
        help="Hide results scoring below this fraction of the best score (applied after top-k).",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Optional path to write a JSONL log (one JSON object per query evaluation). "
            "If the file exists, logs are appended."
        ),
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Run batch evaluation from a queries file (+ optional qrels), instead of interactive mode. "
            "Use --queries to provide the query list."
        ),
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to a text file containing one query per line (blank lines ignored). Used with --batch.",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help=(
            "Optional JSON qrels mapping query -> relevant doc_ids (binary) OR doc_id->relevance (graded).\n"
            "Binary example: {\"English players\": [9, 23, 39, 49]}\n"
            "Graded example: {\"English players\": {\"9\": 2, \"23\": 2, \"39\": 1}}"
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write a CSV of per-query metrics in batch mode (overwritten).",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=None,
        help="Optional path to write a JSON summary of aggregate metrics in batch mode (overwritten).",
    )

    parser.add_argument(
        "--tfidf-norm",
        dest="use_tfidf_norm",
        action="store_true",
        help="Enable TF-IDF cosine normalisation (requires doc_norms from indexing).",
    )
    parser.add_argument(
        "--no-tfidf-norm",
        dest="use_tfidf_norm",
        action="store_false",
        help="Disable TF-IDF cosine normalisation.",
    )
    parser.set_defaults(use_tfidf_norm=True)

    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter (passed through to ranking).")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (passed through to ranking).")

    parser.add_argument(
        "--ner-weight",
        type=float,
        default=0.25,
        help="Strength of NER overlap boosting (multiplier per overlapping entity).",
    )

    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=None,
        help="Compute nDCG@ndcg_k in batch mode when qrels are provided (defaults to --k if omitted).",
    )

    return parser.parse_args()

def build_index(use_ner: bool):
    """Load data, build documents, and construct the inverted index.

    Returns:
      docs, vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_entities, doc_norms

    Notes:
      - doc_norms may be None if indexing.build_inverted_index does not return it yet.
    """
    df = load_soccer_data()
    docs = build_player_documents(df)

    doc_norms: Optional[Dict[int, float]] = None

    if use_ner:
        out = build_inverted_index(docs, return_entities=True)

        if len(out) == 7:
            vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl, doc_norms = out
        elif len(out) == 6:
            vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl = out
        else:
            raise ValueError(
                f"Unexpected return size from build_inverted_index(..., return_entities=True): {len(out)}"
            )
    else:
        out = build_inverted_index(docs)

        if len(out) == 6:
            vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_norms = out
        elif len(out) == 5:
            vocab, postings, doc_id_to_name, doc_lengths, avgdl = out
        else:
            raise ValueError(
                f"Unexpected return size from build_inverted_index(...): {len(out)}"
            )
        doc_entities = None

    return docs, vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_entities, doc_norms

def _iter_nonempty_lines(path: str) -> Iterable[str]:
    resolved = _resolve_path(path)
    if resolved is None:
        raise FileNotFoundError("No path provided")
    with open(resolved, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s

def load_queries_file(path: str) -> List[str]:
    return list(_iter_nonempty_lines(path))

Qrels = Dict[str, Dict[int, int]]  

def load_qrels_json(path: str) -> Qrels:
    """Load qrels as a mapping query -> {doc_id: relevance}.

    Supports:
      - Binary list format: {"q": [12, 20]}
      - Graded dict format: {"q": {"12": 2, "20": 1}}
    """
    with open(_resolve_path(path), "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("qrels JSON must be an object mapping query -> list/dict of relevance judgements")

    out: Qrels = {}
    for q, v in raw.items():
        q = str(q)
        if isinstance(v, list):
            out[q] = {int(doc_id): 1 for doc_id in v}
        elif isinstance(v, dict):
            d: Dict[int, int] = {}
            for doc_id_str, rel in v.items():
                d[int(doc_id_str)] = int(rel)
            out[q] = d
        else:
            raise ValueError(f"qrels entry for query '{q}' must be a list[int] or dict[doc_id->rel]")
    return out

def _apply_thresholds(
    results: List[Tuple[int, float]],
    *,
    topk: int,
    min_score: Optional[float],
    min_score_ratio: Optional[float],
) -> List[Tuple[int, float]]:
    """Option 3: take top-k first, then apply absolute/relative score thresholds."""
    shown = results[:topk]

    best_score = shown[0][1] if shown else None

    if min_score is not None:
        shown = [(doc_id, score) for (doc_id, score) in shown if score >= min_score]

    if min_score_ratio is not None and best_score is not None:
        cutoff = best_score * min_score_ratio
        shown = [(doc_id, score) for (doc_id, score) in shown if score >= cutoff]

    return shown

def precision_at_k(relevant: Set[int], retrieved: List[int], k: int) -> float:
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / min(k, len(top_k))

def recall_at_k(relevant: Set[int], retrieved: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(relevant)

def f1_at_k(relevant: Set[int], retrieved: List[int], k: int) -> float:
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    if (p + r) == 0:
        return 0.0
    return 2.0 * p * r / (p + r)

def average_precision(relevant: Set[int], ranked: List[int], k: Optional[int] = None) -> float:
    """Average Precision (AP). Uses binary relevance."""
    if not relevant:
        return 0.0

    if k is None:
        k = len(ranked)

    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / i

    denom = min(len(relevant), k)
    return sum_prec / denom

def reciprocal_rank(relevant: Set[int], ranked: List[int], k: Optional[int] = None) -> float:
    """Reciprocal Rank (RR)."""
    if not relevant:
        return 0.0
    if k is None:
        k = len(ranked)

    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

def math_log2(x: float) -> float:
    import math

    return math.log2(x)

def ndcg_at_k(graded_rels: Dict[int, int], ranked: List[int], k: int) -> float:
    """nDCG@k using graded relevance. If qrels are binary, rels are 0/1."""
    if k <= 0:
        return 0.0

    def dcg(items: List[int]) -> float:
        score = 0.0
        for i, doc_id in enumerate(items[:k], start=1):
            rel = graded_rels.get(doc_id, 0)
            if rel > 0:
                gain = (2**rel) - 1
                score += gain / math_log2(i + 1)
        return score

    ideal_order = sorted(graded_rels.items(), key=lambda x: x[1], reverse=True)
    ideal_docs = [doc_id for doc_id, _rel in ideal_order]
    idcg = dcg(ideal_docs)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked) / idcg

def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create parent directories for a file path if needed (relative paths resolved from PROJECT_ROOT)."""
    if not path:
        return
    resolved = _resolve_path(path)
    if resolved is None:
        return
    parent = Path(resolved).parent
    parent.mkdir(parents=True, exist_ok=True)

def write_batch_csv(out_csv_path: str, rows: List[Dict[str, Any]]) -> None:
    out_csv_path = _resolve_path(out_csv_path) or out_csv_path
    _ensure_parent_dir(out_csv_path)
    if not rows:
        with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "query,expanded_query,final_query,ranker,use_qe,use_prf,prf_docs,prf_terms,"
                "use_rocchio,rocchio_alpha,rocchio_beta,rocchio_gamma,rocchio_terms,rocchio_nonrel,"
                "use_ner,use_tfidf_norm,k,topk,min_score,min_score_ratio,"
                "p_at_k,r_at_k,f1_at_k,ap,rr,ndcg_at_k,num_shown,shown_doc_ids\n"
            )
        return

    preferred = [
        "query",
        "expanded_query",
        "final_query",
        "ranker",
        "use_qe",
        "use_prf",
        "prf_docs",
        "prf_terms",
        "use_rocchio",
        "rocchio_alpha",
        "rocchio_beta",
        "rocchio_gamma",
        "rocchio_terms",
        "rocchio_nonrel",
        "use_ner",
        "use_tfidf_norm",
        "k",
        "topk",
        "min_score",
        "min_score_ratio",
        "p_at_k",
        "r_at_k",
        "f1_at_k",
        "ap",
        "rr",
        "ndcg_at_k",
        "num_shown",
        "shown_doc_ids",
    ]
    fieldnames = preferred + [k for k in rows[0].keys() if k not in preferred]

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    path = _resolve_path(path) or path
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def interactive_evaluation(args: argparse.Namespace) -> None:
    print("Building index for evaluation...")
    docs, vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_entities, doc_norms = build_index(args.use_ner)
    print("Index built.")
    print(f"Documents: {len(docs)}")

    tfidf_norm_label = str(args.use_tfidf_norm) if args.ranker == "tfidf" else "n/a"
    print(
        f"Ranker: {args.ranker} | Query expansion: {args.use_qe} | PRF: {args.use_prf} | NER boost: {args.use_ner} | "
        f"TF-IDF norm: {tfidf_norm_label} | k: {args.k} | topk: {args.topk}"
    )

    if args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is None:
        print(
            "[WARN] TF-IDF cosine normalisation is enabled, but doc_norms were not provided by indexing. "
            "Proceeding without normalisation."
        )

    if args.use_rocchio:
        print("[ERROR] Rocchio feedback requires qrels. Run batch mode with --batch --qrels <path>.")
        return

    if args.use_prf and args.use_rocchio:
        print("[ERROR] Please choose either PRF (--prf) or Rocchio (--rocchio), not both.")
        return

    while True:
        query = input("\nEnter evaluation query (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Exiting evaluation.")
            break

        if not query:
            print("Please type a query.")
            continue

        expanded_query = expand_query(query, enable=args.use_qe)
        final_query = expanded_query

        if args.use_prf:
            fb_docs = max(1, int(args.prf_docs))
            fb_terms = max(1, int(args.prf_terms))
            results, final_query = search_with_prf(
                expanded_query,
                vocab,
                postings,
                doc_id_to_name,
                ranker=args.ranker,
                doc_lengths=doc_lengths if args.ranker == "bm25" else None,
                avgdl=avgdl if args.ranker == "bm25" else None,
                k1=args.k1,
                b=args.b,
                doc_norms=(
                    doc_norms
                    if (args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is not None)
                    else None
                ),
                doc_entities=doc_entities,
                use_ner_boost=args.use_ner,
                ner_boost_weight=args.ner_weight,
                tfidf_normalize=bool(args.use_tfidf_norm),
                prf_docs=fb_docs,
                prf_terms=fb_terms,
            )
        else:
            results = search(
                expanded_query,
                vocab,
                postings,
                doc_id_to_name,
                ranker=args.ranker,
                doc_lengths=doc_lengths if args.ranker == "bm25" else None,
                avgdl=avgdl if args.ranker == "bm25" else None,
                k1=args.k1,
                b=args.b,
                doc_norms=(
                    doc_norms
                    if (args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is not None)
                    else None
                ),
                doc_entities=doc_entities,
                use_ner_boost=args.use_ner,
                ner_boost_weight=args.ner_weight,
                tfidf_normalize=bool(args.use_tfidf_norm),
            )

        if not results:
            print("No results found.")
            continue

        print(f"\nTop results for: \"{query}\"")
        if args.use_qe and expanded_query != query:
            print(f"(expanded to: \"{expanded_query}\")")
        if args.use_prf and final_query != expanded_query:
            print(f"(PRF expanded to: \"{final_query}\")")

        shown = _apply_thresholds(
            results,
            topk=args.topk,
            min_score=args.min_score,
            min_score_ratio=args.min_score_ratio,
        )

        if not shown:
            print("No results found (after applying thresholds).")
            continue

        retrieved_ids: List[int] = []
        for rank, (doc_id, score) in enumerate(shown, start=1):
            retrieved_ids.append(doc_id)
            doc = docs[doc_id]
            print(f"{rank}. doc_id={doc_id}, {doc.get('name','')} (score={score:.4f})")
            print(f"   {doc.get('text','')}")
            print()

        raw_input_str = input(
            "Enter RELEVANT results as ranks (e.g. 1,3,5) OR names (e.g. 'Diego Placente'). Leave blank if none: "
        ).strip()

        relevant_ranks: List[int] = []
        relevant_doc_ids: Set[int] = set()

        def _norm(s: str) -> str:
            return " ".join(s.lower().strip().split())

        if raw_input_str:

            try:
                relevant_ranks = [int(x.strip()) for x in raw_input_str.split(",") if x.strip()]
                relevant_doc_ids = {
                    retrieved_ids[r - 1] for r in relevant_ranks if 1 <= r <= len(retrieved_ids)
                }
            except ValueError:

                wanted = [_norm(x) for x in raw_input_str.split(",") if x.strip()]
                name_to_rank: Dict[str, int] = {}
                for r, doc_id in enumerate(retrieved_ids, start=1):
                    name_to_rank[_norm(docs[doc_id].get("name", ""))] = r

                matched_ranks: Set[int] = set()
                for w in wanted:

                    if w in name_to_rank:
                        matched_ranks.add(name_to_rank[w])
                        continue

                    for nm, rnk in name_to_rank.items():
                        if w and (w in nm or nm in w):
                            matched_ranks.add(rnk)

                if not matched_ranks:
                    print(
                        "Invalid input. Please enter ranks like 1,3,5 or a player name from the shown list. Treating as no relevant results."
                    )
                else:
                    relevant_ranks = sorted(matched_ranks)
                    relevant_doc_ids = {retrieved_ids[r - 1] for r in relevant_ranks}

        p_at_k = precision_at_k(relevant_doc_ids, retrieved_ids, k=args.k)
        r_at_k = recall_at_k(relevant_doc_ids, retrieved_ids, k=args.k)
        f1 = f1_at_k(relevant_doc_ids, retrieved_ids, k=args.k)
        ap = average_precision(relevant_doc_ids, retrieved_ids, k=args.k)
        rr = reciprocal_rank(relevant_doc_ids, retrieved_ids, k=args.k)
        graded = {doc_id: 1 for doc_id in relevant_doc_ids}
        ndcg = ndcg_at_k(graded, retrieved_ids, k=args.k)

        print(
            f"\nMetrics@{args.k} for query \"{query}\": "
            f"P={p_at_k:.2f} | R={r_at_k:.2f} | F1={f1:.2f} | AP={ap:.2f} | RR={rr:.2f} | nDCG={ndcg:.2f}"
        )

        if args.out:
            log_obj: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": "interactive",
                "ranker": args.ranker,
                "use_qe": args.use_qe,
                "use_prf": bool(args.use_prf),
                "prf_docs": int(args.prf_docs),
                "prf_terms": int(args.prf_terms),
                "use_ner": args.use_ner,
                "use_tfidf_norm": ("n/a" if args.ranker == "bm25" else bool(args.use_tfidf_norm)),
                "k": args.k,
                "topk": args.topk,
                "min_score": args.min_score,
                "min_score_ratio": args.min_score_ratio,
                "k1": args.k1,
                "b": args.b,
                "ner_weight": args.ner_weight,
                "query": query,
                "expanded_query": expanded_query,
                "final_query": final_query,
                "shown": [
                    {
                        "rank": r,
                        "doc_id": doc_id,
                        "name": docs[doc_id].get("name", ""),
                        "url": docs[doc_id].get("url", ""),
                        "score": float(score),
                    }
                    for r, (doc_id, score) in enumerate(shown, start=1)
                ],
                "relevant_ranks": sorted(relevant_ranks),
                "metrics": {
                    "p_at_k": float(p_at_k),
                    "r_at_k": float(r_at_k),
                    "f1_at_k": float(f1),
                    "ap": float(ap),
                    "rr": float(rr),
                    "ndcg_at_k": float(ndcg),
                },
            }
            _append_jsonl(args.out, log_obj)

def batch_evaluation(args: argparse.Namespace) -> None:
    if not args.queries:
        raise SystemExit("[ERROR] --batch requires --queries <path-to-queries.txt>")

    queries = load_queries_file(args.queries)
    if not queries:
        raise SystemExit("[ERROR] No queries found in the provided --queries file.")

    qrels: Optional[Qrels] = None
    if args.qrels:
        qrels = load_qrels_json(args.qrels)

    if args.use_rocchio and qrels is None:
        raise SystemExit("[ERROR] --rocchio requires --qrels (Rocchio needs relevance judgements).")

    if args.use_prf and args.use_rocchio:
        raise SystemExit("[ERROR] Please choose either PRF (--prf) or Rocchio (--rocchio), not both.")

    prf_docs = max(1, int(args.prf_docs))
    prf_terms = max(1, int(args.prf_terms))
    rocchio_terms = max(0, int(args.rocchio_terms))
    rocchio_nonrel = max(0, int(args.rocchio_nonrel))

    print("Building index for batch evaluation...")
    docs, vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_entities, doc_norms = build_index(args.use_ner)
    print("Index built.")
    print(f"Documents: {len(docs)}")

    tfidf_norm_label = str(args.use_tfidf_norm) if args.ranker == "tfidf" else "n/a"
    print(
        f"Batch mode | Ranker: {args.ranker} | QE: {args.use_qe} | PRF: {args.use_prf} | Rocchio: {args.use_rocchio} | NER: {args.use_ner} | "
        f"TF-IDF norm: {tfidf_norm_label} | k: {args.k} | topk: {args.topk}"
    )

    if args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is None:
        print(
            "[WARN] TF-IDF cosine normalisation is enabled, but doc_norms were not provided by indexing. "
            "Proceeding without normalisation."
        )

    ndcg_k = args.ndcg_k if args.ndcg_k is not None else args.k

    per_query_rows: List[Dict[str, Any]] = []

    p_list: List[float] = []
    r_list: List[float] = []
    f1_list: List[float] = []
    ap_list: List[float] = []
    rr_list: List[float] = []
    ndcg_list: List[float] = []

    for query in queries:
        expanded_query = expand_query(query, enable=args.use_qe)
        final_query = expanded_query

        if args.use_prf:
            results, final_query = search_with_prf(
                expanded_query,
                vocab,
                postings,
                doc_id_to_name,
                ranker=args.ranker,
                doc_lengths=doc_lengths if args.ranker == "bm25" else None,
                avgdl=avgdl if args.ranker == "bm25" else None,
                k1=args.k1,
                b=args.b,
                doc_norms=(
                    doc_norms
                    if (args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is not None)
                    else None
                ),
                doc_entities=doc_entities,
                use_ner_boost=args.use_ner,
                ner_boost_weight=args.ner_weight,
                tfidf_normalize=bool(args.use_tfidf_norm),
                prf_docs=prf_docs,
                prf_terms=prf_terms,
            )

        elif args.use_rocchio:

            baseline_results = search(
                expanded_query,
                vocab,
                postings,
                doc_id_to_name,
                ranker=args.ranker,
                doc_lengths=doc_lengths if args.ranker == "bm25" else None,
                avgdl=avgdl if args.ranker == "bm25" else None,
                k1=args.k1,
                b=args.b,
                doc_norms=(
                    doc_norms
                    if (args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is not None)
                    else None
                ),
                doc_entities=doc_entities,
                use_ner_boost=args.use_ner,
                ner_boost_weight=args.ner_weight,
                tfidf_normalize=bool(args.use_tfidf_norm),
            )

            graded_for_query = (qrels or {}).get(query) or (qrels or {}).get(expanded_query) or {}
            relevant_ids = {doc_id for doc_id, rel in graded_for_query.items() if rel > 0}

            nonrel_ids: List[int] = []
            if rocchio_nonrel > 0:
                for doc_id, _score in baseline_results:
                    if doc_id not in relevant_ids:
                        nonrel_ids.append(doc_id)
                    if len(nonrel_ids) >= rocchio_nonrel:
                        break

            results, final_query = search_with_rocchio(
                expanded_query,
                vocab,
                postings,
                doc_id_to_name,
                relevant_doc_ids=sorted(relevant_ids),
                nonrelevant_doc_ids=nonrel_ids,
                alpha=float(args.rocchio_alpha),
                beta=float(args.rocchio_beta),
                gamma=float(args.rocchio_gamma),
                rocchio_terms=rocchio_terms,
                ranker=args.ranker,
                doc_lengths=doc_lengths if args.ranker == "bm25" else None,
                avgdl=avgdl if args.ranker == "bm25" else None,
                k1=args.k1,
                b=args.b,
                doc_norms=(
                    doc_norms
                    if (args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is not None)
                    else None
                ),
                doc_entities=doc_entities,
                use_ner_boost=args.use_ner,
                ner_boost_weight=args.ner_weight,
                tfidf_normalize=bool(args.use_tfidf_norm),
            )

        else:
            results = search(
                expanded_query,
                vocab,
                postings,
                doc_id_to_name,
                ranker=args.ranker,
                doc_lengths=doc_lengths if args.ranker == "bm25" else None,
                avgdl=avgdl if args.ranker == "bm25" else None,
                k1=args.k1,
                b=args.b,
                doc_norms=(
                    doc_norms
                    if (args.ranker == "tfidf" and args.use_tfidf_norm and doc_norms is not None)
                    else None
                ),
                doc_entities=doc_entities,
                use_ner_boost=args.use_ner,
                ner_boost_weight=args.ner_weight,
                tfidf_normalize=bool(args.use_tfidf_norm),
            )

        shown = _apply_thresholds(
            results,
            topk=args.topk,
            min_score=args.min_score,
            min_score_ratio=args.min_score_ratio,
        )
        retrieved_ids = [doc_id for (doc_id, _score) in shown]

        p_at_k: Optional[float] = None
        r_at_k: Optional[float] = None
        f1: Optional[float] = None
        ap: Optional[float] = None
        rr: Optional[float] = None
        ndcg: Optional[float] = None

        if qrels is not None:
            graded_for_query = qrels.get(query) or qrels.get(expanded_query) or qrels.get(final_query) or {}
            binary_rel = {doc_id for doc_id, rel in graded_for_query.items() if rel > 0}

            p_at_k = precision_at_k(binary_rel, retrieved_ids, k=args.k)
            r_at_k = recall_at_k(binary_rel, retrieved_ids, k=args.k)
            f1 = f1_at_k(binary_rel, retrieved_ids, k=args.k)

            ap = average_precision(binary_rel, retrieved_ids, k=args.k)
            rr = reciprocal_rank(binary_rel, retrieved_ids, k=args.k)

            ndcg = ndcg_at_k(graded_for_query, retrieved_ids, k=ndcg_k)

            p_list.append(float(p_at_k))
            r_list.append(float(r_at_k))
            f1_list.append(float(f1))
            ap_list.append(float(ap))
            rr_list.append(float(rr))
            ndcg_list.append(float(ndcg))

        per_query_rows.append(
            {
                "query": query,
                "expanded_query": expanded_query,
                "final_query": final_query,
                "ranker": args.ranker,
                "use_qe": bool(args.use_qe),
                "use_prf": bool(args.use_prf),
                "prf_docs": prf_docs,
                "prf_terms": prf_terms,
                "use_rocchio": bool(args.use_rocchio),
                "rocchio_alpha": float(args.rocchio_alpha),
                "rocchio_beta": float(args.rocchio_beta),
                "rocchio_gamma": float(args.rocchio_gamma),
                "rocchio_terms": rocchio_terms,
                "rocchio_nonrel": rocchio_nonrel,
                "use_ner": bool(args.use_ner),
                "use_tfidf_norm": ("n/a" if args.ranker == "bm25" else bool(args.use_tfidf_norm)),
                "k": int(args.k),
                "topk": int(args.topk),
                "min_score": args.min_score,
                "min_score_ratio": args.min_score_ratio,
                "p_at_k": ("" if p_at_k is None else float(p_at_k)),
                "r_at_k": ("" if r_at_k is None else float(r_at_k)),
                "f1_at_k": ("" if f1 is None else float(f1)),
                "ap": ("" if ap is None else float(ap)),
                "rr": ("" if rr is None else float(rr)),
                "ndcg_at_k": ("" if ndcg is None else float(ndcg)),
                "num_shown": len(shown),
                "shown_doc_ids": " ".join(str(d) for d in retrieved_ids),
            }
        )

        if args.out:
            log_obj: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": "batch",
                "ranker": args.ranker,
                "use_qe": args.use_qe,
                "use_prf": bool(args.use_prf),
                "prf_docs": prf_docs,
                "prf_terms": prf_terms,
                "use_rocchio": bool(args.use_rocchio),
                "rocchio_alpha": float(args.rocchio_alpha),
                "rocchio_beta": float(args.rocchio_beta),
                "rocchio_gamma": float(args.rocchio_gamma),
                "rocchio_terms": rocchio_terms,
                "rocchio_nonrel": rocchio_nonrel,
                "use_ner": args.use_ner,
                "use_tfidf_norm": ("n/a" if args.ranker == "bm25" else bool(args.use_tfidf_norm)),
                "k": args.k,
                "topk": args.topk,
                "min_score": args.min_score,
                "min_score_ratio": args.min_score_ratio,
                "k1": args.k1,
                "b": args.b,
                "ner_weight": args.ner_weight,
                "query": query,
                "expanded_query": expanded_query,
                "final_query": final_query,
                "shown": [
                    {
                        "rank": r,
                        "doc_id": doc_id,
                        "name": docs[doc_id].get("name", ""),
                        "url": docs[doc_id].get("url", ""),
                        "score": float(score),
                    }
                    for r, (doc_id, score) in enumerate(shown, start=1)
                ],
                "metrics": {
                    "p_at_k": None if p_at_k is None else float(p_at_k),
                    "r_at_k": None if r_at_k is None else float(r_at_k),
                    "f1_at_k": None if f1 is None else float(f1),
                    "ap": None if ap is None else float(ap),
                    "rr": None if rr is None else float(rr),
                    "ndcg_at_k": None if ndcg is None else float(ndcg),
                },
            }
            _append_jsonl(args.out, log_obj)

    if args.out_csv:
        write_batch_csv(args.out_csv, per_query_rows)
        print(f"Wrote CSV: {_resolve_path(args.out_csv) or args.out_csv}")

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ranker": args.ranker,
        "use_qe": bool(args.use_qe),
        "use_prf": bool(args.use_prf),
        "prf_docs": prf_docs,
        "prf_terms": prf_terms,
        "use_rocchio": bool(args.use_rocchio),
        "rocchio_alpha": float(args.rocchio_alpha),
        "rocchio_beta": float(args.rocchio_beta),
        "rocchio_gamma": float(args.rocchio_gamma),
        "rocchio_terms": rocchio_terms,
        "rocchio_nonrel": rocchio_nonrel,
        "use_ner": bool(args.use_ner),
        "use_tfidf_norm": ("n/a" if args.ranker == "bm25" else bool(args.use_tfidf_norm)),
        "k": int(args.k),
        "topk": int(args.topk),
        "min_score": args.min_score,
        "min_score_ratio": args.min_score_ratio,
        "k1": float(args.k1),
        "b": float(args.b),
        "ner_weight": float(args.ner_weight),
        "num_queries": len(queries),
        "has_qrels": bool(qrels is not None),
    }

    if qrels is not None and p_list:
        summary.update(
            {
                "mean_p_at_k": sum(p_list) / len(p_list),
                "mean_r_at_k": sum(r_list) / len(r_list),
                "mean_f1_at_k": sum(f1_list) / len(f1_list),
                "map_at_k": sum(ap_list) / len(ap_list),
                "mrr_at_k": sum(rr_list) / len(rr_list),
                "mean_ndcg_at_k": sum(ndcg_list) / len(ndcg_list),
                "ndcg_k": int(ndcg_k),
            }
        )

    if args.out_summary:
        out_summary_path = _resolve_path(args.out_summary) or args.out_summary
        _ensure_parent_dir(out_summary_path)
        with open(out_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote summary JSON: {out_summary_path}")
    else:
        if qrels is None:
            print("Summary: qrels not provided, so only result lists were generated (no automatic metrics).")
        else:
            print(
                f"Summary: P@{args.k}={summary.get('mean_p_at_k', 0):.3f} | "
                f"R@{args.k}={summary.get('mean_r_at_k', 0):.3f} | "
                f"F1@{args.k}={summary.get('mean_f1_at_k', 0):.3f} | "
                f"MAP@{args.k}={summary.get('map_at_k', 0):.3f} | "
                f"MRR@{args.k}={summary.get('mrr_at_k', 0):.3f} | "
                f"nDCG@{ndcg_k}={summary.get('mean_ndcg_at_k', 0):.3f}"
            )

if __name__ == "__main__":
    args = parse_args()
    if args.batch:
        batch_evaluation(args)
    else:
        interactive_evaluation(args)