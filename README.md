# Information Retrieval Coursework – Search Engine

## Project Overview

This project implements an interactive and evaluative **Information Retrieval (IR) system** designed to explore both core and advanced IR concepts through practical experimentation.

The system supports multiple ranking models, query expansion techniques, relevance feedback mechanisms, and a full evaluation pipeline. A detailed research log accompanies the codebase, documenting the technical decisions, experiments, and learning outcomes.

The project is implemented in **Python** and operates over a **soccer player dataset**, enabling controlled experimentation with retrieval algorithms and evaluation metrics.

---

## Repository Structure

```text
ir_search_engine/
├── src/
│   ├── __init__.py            # Marks src as a package
│   ├── main.py                # Interactive CLI search engine
│   ├── indexing.py            # Index construction and preprocessing
│   ├── ranking.py             # TF-IDF and BM25 ranking implementations
│   ├── query_expansion.py     # QE, PRF, and Rocchio logic
│   ├── evaluation.py          # Batch evaluation pipeline
│   ├── format_results.py      # Aggregates and formats evaluation results
│   ├── preprocessing.py       # Tokenisation, stopwords, lemmatisation
│   ├── ner.py                 # Named Entity Recognition (optional)
│   └── queries.txt            # Evaluation queries
│
├── tests/
│   └── test_ir_core.py        # Core unit tests (pytest)
│
├── data/
│   └── raw/
│       └── soccer.csv         # Soccer player dataset
│
├── logs/
│   ├── final_*.json           # Final evaluation summaries
│   ├── final_*.csv            # Per-query evaluation results
│   └── final_compare.md       # Comparative evaluation table
│
├── qrels.json                 # Relevance judgements
├── README.md                  # Project README
└── 100494222_research_log.doc # Research log