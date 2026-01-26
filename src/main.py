import argparse
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from indexing import build_inverted_index, build_player_documents, load_soccer_data
from query_expansion import expand_query, expand_query_prf
from ranking import search

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive IR search engine (TF-IDF / BM25) with QE, PRF, NER, and thresholds."
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
        help="Enable pseudo relevance feedback (PRF) expansion (two-pass retrieval).",
    )
    parser.add_argument(
        "--no-prf",
        dest="use_prf",
        action="store_false",
        help="Disable pseudo relevance feedback (PRF) expansion.",
    )
    parser.set_defaults(use_prf=False)

    parser.add_argument(
        "--prf-docs",
        type=int,
        default=5,
        help="Number of top documents to use as feedback for PRF (default: 5).",
    )
    parser.add_argument(
        "--prf-terms",
        type=int,
        default=5,
        help="Number of feedback terms to add during PRF (default: 5).",
    )

    parser.add_argument("--ner", dest="use_ner", action="store_true", help="Enable NER boosting.")
    parser.add_argument("--no-ner", dest="use_ner", action="store_false", help="Disable NER boosting.")
    parser.set_defaults(use_ner=False)

    parser.add_argument(
        "--tfidf-normalize",
        dest="tfidf_normalize",
        action="store_true",
        help="Enable cosine normalisation for TF-IDF.",
    )
    parser.add_argument(
        "--no-tfidf-normalize",
        dest="tfidf_normalize",
        action="store_false",
        help="Disable cosine normalisation for TF-IDF.",
    )
    parser.set_defaults(tfidf_normalize=True)

    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")

    parser.add_argument("--topk", type=int, default=10, help="Maximum number of results to display.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Only show results with score >= min-score.",
    )
    parser.add_argument(
        "--min-score-ratio",
        type=float,
        default=None,
        help="Only show results with score >= ratio * best_score.",
    )
    parser.add_argument(
        "--hide-below-threshold",
        action="store_true",
        help="Hide results that do not meet score thresholds.",
    )

    parser.add_argument(
        "--show-url",
        action="store_true",
        help="Include the document URL column in the results table.",
    )

    parser.add_argument(
        "--wrap",
        type=int,
        default=70,
        help="Wrap width for the text/snippet column in table output.",
    )

    parser.add_argument(
        "--ner-weight",
        type=float,
        default=0.25,
        help="NER overlap boost strength.",
    )

    args = parser.parse_args()
    if args.min_score_ratio is not None and args.min_score_ratio < 0:
        parser.error("--min-score-ratio must be >= 0")
    if args.topk <= 0:
        parser.error("--topk must be >= 1")
    return args

def build_index(use_ner: bool):
    """Build index and return everything needed for search.

    Returns:
      docs, vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_entities, doc_norms

    Notes:
      `indexing.build_inverted_index` may return optional doc_norms (for TF-IDF cosine normalisation).

      Expected patterns:
        - without NER: 5 values (vocab, postings, doc_id_to_name, doc_lengths, avgdl)
                      or 6 values (+ doc_norms)
        - with NER:    6 values (vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl)
                      or 7 values (+ doc_norms)
    """
    df = load_soccer_data()
    docs = build_player_documents(df)

    doc_entities = None
    doc_norms = None

    if use_ner:
        res = build_inverted_index(docs, return_entities=True)
        if len(res) == 7:
            vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl, doc_norms = res
        elif len(res) == 6:
            vocab, postings, doc_id_to_name, doc_entities, doc_lengths, avgdl = res
        else:
            raise ValueError(
                f"Unexpected return size from build_inverted_index(..., return_entities=True): {len(res)}"
            )
    else:
        res = build_inverted_index(docs)
        if len(res) == 6:
            vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_norms = res
        elif len(res) == 5:
            vocab, postings, doc_id_to_name, doc_lengths, avgdl = res
        else:
            raise ValueError(
                f"Unexpected return size from build_inverted_index(...): {len(res)}"
            )

    return docs, vocab, postings, doc_id_to_name, doc_lengths, avgdl, doc_entities, doc_norms

def _format_results_table(
    docs: List[Dict[str, Any]],
    results: List[Tuple[int, float]],
    *,
    show_url: bool = False,
    wrap: int = 70,
) -> str:
    """Return a pretty, console-friendly table for ranked results."""

    if not results:
        return ""

    rows: List[Dict[str, str]] = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = docs[doc_id]
        name = str(doc.get("name", ""))
        text = str(doc.get("text", ""))
        url = str(doc.get("url", ""))

        wrap_width = max(20, int(wrap))
        wrapped_text = "\n".join(textwrap.wrap(text, width=wrap_width)) if text else ""

        row = {
            "Rank": str(rank),
            "DocID": str(doc_id),
            "Score": f"{score:.4f}",
            "Name": name,
            "Text": wrapped_text,
        }
        if show_url:
            row["URL"] = url
        rows.append(row)

    headers = ["Rank", "DocID", "Score", "Name", "Text"]
    if show_url:
        headers.append("URL")

    def cell_width(s: str) -> int:
        if not s:
            return 0
        return max(len(line) for line in s.splitlines())

    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], cell_width(r.get(h, "")))

    widths["Name"] = min(widths["Name"], 28)
    widths["Text"] = min(widths["Text"], max(30, int(wrap)))
    if show_url:
        widths["URL"] = min(widths["URL"], 45)

    def clamp(s: str, max_w: int) -> str:
        s = s or ""
        if len(s) <= max_w:
            return s
        if max_w <= 1:
            return s[:max_w]
        return s[: max_w - 1] + "…"

    out_lines: List[str] = []

    def sep(char: str = "-") -> str:
        return "+" + "+".join(char * (widths[h] + 2) for h in headers) + "+"

    out_lines.append(sep("-"))
    out_lines.append("|" + "|".join(f" {h.ljust(widths[h])} " for h in headers) + "|")
    out_lines.append(sep("="))

    for r in rows:
        r2 = dict(r)
        r2["Name"] = clamp(r2.get("Name", ""), widths["Name"])
        if show_url:
            r2["URL"] = clamp(r2.get("URL", ""), widths["URL"])

        text_lines = (r2.get("Text", "") or "").splitlines() or [""]

        first = dict(r2)
        first["Text"] = text_lines[0]
        out_lines.append("|" + "|".join(f" {first.get(h, '').ljust(widths[h])} " for h in headers) + "|")

        for tline in text_lines[1:]:
            cont = {h: "" for h in headers}
            cont["Text"] = tline
            out_lines.append("|" + "|".join(f" {cont.get(h, '').ljust(widths[h])} " for h in headers) + "|")

        out_lines.append(sep("-"))

    return "\n".join(out_lines)

def main() -> None:
    args = parse_args()

    tfidf_norm_enabled = bool(args.ranker == "tfidf" and args.tfidf_normalize)

    print("Building index...")
    (
        docs,
        vocab,
        postings,
        doc_id_to_name,
        doc_lengths,
        avgdl,
        doc_entities,
        doc_norms,
    ) = build_index(args.use_ner)

    bm25_part = ""
    if args.ranker == "bm25":
        bm25_part = f" | k1={args.k1} | b={args.b}"

    thresh_part = ""
    if args.min_score is not None or args.min_score_ratio is not None:
        thresh_part = (
            f" | min_score={args.min_score} | min_ratio={args.min_score_ratio}"
            f" | hide_below={args.hide_below_threshold}"
        )

    print(
        f"Index built | docs={len(docs)} | ranker={args.ranker} | "
        f"QE={args.use_qe} | PRF={args.use_prf} | NER={args.use_ner} | "
        f"tfidf_norm={tfidf_norm_enabled if args.ranker == 'tfidf' else 'n/a'}"
        f"{bm25_part}{thresh_part}"
    )

    if args.use_prf:
        print(f"PRF settings: prf_docs={args.prf_docs} | prf_terms={args.prf_terms}")

    if args.min_score is not None or args.min_score_ratio is not None:
        print(
            "Thresholds active: "
            f"min_score={args.min_score} | min_score_ratio={args.min_score_ratio} | "
            f"hide_below_threshold={args.hide_below_threshold}"
        )

    while True:
        query = input("\nEnter search query (or 'quit'): ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not query:
            print("Please type a query (or 'quit').")
            continue

        expanded_query = expand_query(query, enable=args.use_qe)

        first_pass_results = search(
            expanded_query,
            vocab,
            postings,
            doc_id_to_name,
            ranker=args.ranker,
            doc_lengths=doc_lengths if args.ranker == "bm25" else None,
            avgdl=avgdl if args.ranker == "bm25" else None,
            doc_norms=doc_norms if (tfidf_norm_enabled and doc_norms is not None) else None,
            tfidf_normalize=args.tfidf_normalize,
            doc_entities=doc_entities,
            use_ner_boost=args.use_ner,
            ner_boost_weight=args.ner_weight,
            k1=args.k1,
            b=args.b,
            topk=max(args.topk, int(args.prf_docs), 10) if args.use_prf else args.topk,
            min_score=None,
            min_score_ratio=None,
        )

        if not first_pass_results:
            print("No results found.")
            continue

        final_query = expanded_query
        if args.use_prf:
            fb_docs = max(1, int(args.prf_docs))
            fb_terms = max(1, int(args.prf_terms))

            top_docs_text: List[str] = []
            for doc_id, _score in first_pass_results[:fb_docs]:
                top_docs_text.append(str(docs[doc_id].get("text", "")))

            final_query = expand_query_prf(
                expanded_query,
                enable_dict=args.use_qe,
                enable_prf=True,
                top_docs=top_docs_text,
                fb_docs=fb_docs,
                fb_terms=fb_terms,
            )

        use_min_score = args.min_score if args.hide_below_threshold else None
        use_min_ratio = args.min_score_ratio if args.hide_below_threshold else None

        results = search(
            final_query,
            vocab,
            postings,
            doc_id_to_name,
            ranker=args.ranker,
            doc_lengths=doc_lengths if args.ranker == "bm25" else None,
            avgdl=avgdl if args.ranker == "bm25" else None,
            doc_norms=doc_norms if (tfidf_norm_enabled and doc_norms is not None) else None,
            tfidf_normalize=args.tfidf_normalize,
            doc_entities=doc_entities,
            use_ner_boost=args.use_ner,
            ner_boost_weight=args.ner_weight,
            k1=args.k1,
            b=args.b,
            topk=args.topk,
            min_score=use_min_score,
            min_score_ratio=use_min_ratio,
        )

        print(f"\nResults for: \"{query}\"")
        if expanded_query != query:
            print(f"(QE expanded: \"{expanded_query}\")")
        if args.use_prf and final_query != expanded_query:
            print(f"(PRF expanded: \"{final_query}\")")

        if not results:
            if args.hide_below_threshold and (args.min_score is not None or args.min_score_ratio is not None):
                print("No results passed the threshold(s).")
            else:
                print("No results found.")
            continue

        if (args.min_score is not None or args.min_score_ratio is not None) and not args.hide_below_threshold:
            best_score = results[0][1]
            below = 0
            for _doc_id, score in results:
                if args.min_score is not None and score < args.min_score:
                    below += 1
                    continue
                if args.min_score_ratio is not None and score < best_score * args.min_score_ratio:
                    below += 1
                    continue
            if below > 0:
                print(
                    f"Note: {below} of the displayed {len(results)} result(s) are below the threshold(s). "
                    "(Use --hide-below-threshold to filter them out.)"
                )

        table = _format_results_table(
            docs,
            results,
            show_url=args.show_url,
            wrap=args.wrap,
        )
        print(table)

if __name__ == "__main__":
    main()