"""
format_results.py

Reads one or more batch summary JSON files (produced by evaluation.py --batch)
and writes:
  - a comparison CSV
  - a comparison Markdown table
  - a combined JSON (for reuse later in the report/log)

Example:
  python3 src/format_results.py \
    --inputs logs/batch_bm25_qe.json logs/batch_bm25_qe_prf.json logs/summary_rocchio.json \
    --out-csv logs/compare.csv \
    --out-md logs/compare.md \
    --out-json logs/compare.json
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

METRIC_KEYS = [
    ("mean_p_at_k", "P@k"),
    ("mean_r_at_k", "R@k"),
    ("mean_f1_at_k", "F1@k"),
    ("map_at_k", "MAP@k"),
    ("mrr_at_k", "MRR@k"),
    ("mean_ndcg_at_k", "nDCG@k"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _resolve_path(path: str) -> str:
    """Resolve a user-provided path. If relative, resolve it against PROJECT_ROOT."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())

def _load_json(path: str) -> Dict[str, Any]:
    path = _resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _fmt(x: Optional[float], dp: int = 3) -> str:
    if x is None:
        return ""
    return f"{x:.{dp}f}"

def _default_system_name(summary: Dict[str, Any], path: str) -> str:

    if isinstance(summary.get("system_name"), str) and summary["system_name"].strip():
        return summary["system_name"].strip()

    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    name = name.replace("summary_", "").replace("batch_", "")
    name = name.replace("__", "_").replace("_", " ").strip()
    return name or base

def _system_flags(summary: Dict[str, Any]) -> str:
    parts: List[str] = []

    ranker = summary.get("ranker")
    if ranker:
        r = str(ranker).lower().strip()
        if r == "tfidf":
            parts.append("TF-IDF")
        elif r == "bm25":
            parts.append("BM25")
        else:
            parts.append(str(ranker).upper())

    if str(ranker).lower().strip() == "bm25":
        k1 = summary.get("k1")
        b = summary.get("b")
        if k1 is not None and b is not None:
            parts.append(f"k1={k1}")
            parts.append(f"b={b}")

    if summary.get("use_qe"):
        parts.append("QE")

    if summary.get("use_prf"):
        parts.append(f"PRF(d={summary.get('prf_docs')},t={summary.get('prf_terms')})")

    if summary.get("use_rocchio"):
        parts.append(f"ROCCHIO(t={summary.get('rocchio_terms')},nr={summary.get('rocchio_nonrel')})")

    if summary.get("use_ner"):
        nw = summary.get("ner_weight")
        if nw is not None:
            parts.append(f"NER(w={nw})")
        else:
            parts.append("NER")

    min_score = summary.get("min_score")
    min_ratio = summary.get("min_score_ratio")
    if min_score is not None or min_ratio is not None:
        ms = "" if min_score is None else str(min_score)
        mr = "" if min_ratio is None else str(min_ratio)
        parts.append(f"THRESH(ms={ms},mr={mr})")

    return " + ".join([p for p in parts if p]) if parts else ""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Format evaluation summary JSONs into comparison tables.")
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="List of summary JSON files. If omitted, use --glob.",
    )
    p.add_argument(
        "--glob",
        default="logs/*.json",
        help="Glob pattern to find summary JSONs (default: logs/*.json).",
    )
    p.add_argument("--out-csv", default="logs/compare.csv", help="Output CSV path.")
    p.add_argument("--out-md", default="logs/compare.md", help="Output Markdown path.")
    p.add_argument("--out-json", default="logs/compare.json", help="Output JSON path.")
    p.add_argument("--dp", type=int, default=3, help="Decimal places for metrics.")
    p.add_argument(
        "--include-no-metrics",
        action="store_true",
        help="Include summary JSONs even if they don’t contain mean_* metrics (cells will be blank).",
    )
    p.add_argument(
        "--sort-by",
        default="P@k",
        help="Sort by this column label (default: P@k). Use one of: P@k, MAP@k, MRR@k, nDCG@k, R@k, F1@k",
    )
    p.add_argument(
        "--sort-secondary",
        default="MAP@k",
        help="Secondary sort column label (default: MAP@k).",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    paths: List[str] = []
    if args.inputs:
        paths = [str(p) for p in args.inputs]
    else:

        glob_pat = args.glob
        if not os.path.isabs(glob_pat):
            glob_pat = str((PROJECT_ROOT / glob_pat))
        paths = sorted(glob.glob(glob_pat))

    summaries: List[Dict[str, Any]] = []
    used_paths: List[str] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            s = _load_json(path)
        except Exception:
            continue
        if ("mean_p_at_k" not in s) and (not args.include_no_metrics):
            continue
        summaries.append(s)
        used_paths.append(_resolve_path(path))

    if not summaries:
        print("No valid summary JSON files found.")
        print("Tip: pass explicit files with --inputs logs/summary_*.json logs/batch_*.json")
        return

    rows: List[Dict[str, Any]] = []
    for s, path in zip(summaries, used_paths):
        row: Dict[str, Any] = {}
        row["system"] = _default_system_name(s, path)
        row["flags"] = _system_flags(s)
        row["k"] = s.get("k", s.get("ndcg_k", ""))
        row["num_queries"] = s.get("num_queries", "")
        for key, label in METRIC_KEYS:
            row[label] = _safe_float(s.get(key))
        rows.append(row)

    sort_primary = str(args.sort_by).strip()
    sort_secondary = str(args.sort_secondary).strip()

    def _sort_val(r: Dict[str, Any], key: str) -> float:
        v = r.get(key)
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    rows.sort(key=lambda r: (-_sort_val(r, sort_primary), -_sort_val(r, sort_secondary)))

    args.out_csv = _resolve_path(args.out_csv)
    args.out_md = _resolve_path(args.out_md)
    args.out_json = _resolve_path(args.out_json)

    for out_path in [args.out_csv, args.out_md, args.out_json]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": used_paths,
                "rows": rows,
                "metric_keys": [{"key": k, "label": lab} for k, lab in METRIC_KEYS],
            },
            f,
            indent=2,
        )

    import csv

    fieldnames = ["system", "flags", "k", "num_queries"] + [label for _, label in METRIC_KEYS]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for _, label in METRIC_KEYS:
                out[label] = _fmt(out.get(label), dp=args.dp)
            w.writerow(out)

    headers = ["System", "Flags", "k", "#Q"] + [label for _, label in METRIC_KEYS]
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("system", "")),
                    str(r.get("flags", "")),
                    str(r.get("k", "")),
                    str(r.get("num_queries", "")),
                    *[_fmt(r.get(label), dp=args.dp) for _, label in METRIC_KEYS],
                ]
            )
            + " |"
        )
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Wrote CSV:  {args.out_csv}")
    print(f"Wrote MD:   {args.out_md}")
    print(f"Wrote JSON: {args.out_json}")
    print(f"Included {len(rows)} system(s).")

if __name__ == "__main__":
    main()