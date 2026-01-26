Information Retrieval Coursework – README

Project Overview

This project implements an interactive and evaluative Information Retrieval (IR) system designed to explore core and advanced IR concepts through practical experimentation. The system supports multiple ranking models, query expansion techniques, relevance feedback mechanisms, and evaluation pipelines. It is accompanied by a research log that documents the full technical development process.

The project is implemented in Python and operates over a soccer player dataset, enabling controlled experimentation with retrieval algorithms and evaluation metrics.

Repository Structure

ir_search_engine/
│
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


System Features
	•	Indexing: Inverted index with configurable preprocessing (none / stemming / lemmatisation)
	•	Ranking Models:
	•	TF-IDF (with optional cosine normalisation)
	•	BM25 (configurable k1 and b parameters)
	•	Query Expansion Techniques:
	•	Dictionary-based Query Expansion (QE)
	•	Pseudo-Relevance Feedback (PRF)
	•	Rocchio relevance feedback
	•	Named Entity Recognition (NER):
	•	Optional entity overlap boosting
	•	Thresholding:
	•	Minimum score and score-ratio filtering
	•	Evaluation:
	•	Precision@k, Recall@k, F1@k, MAP@k, MRR@k, nDCG@k
	•	Testing:
	•	Unit tests using pytest

⸻

Installation and Setup

1. Create a virtual environment (recommended)

python3 -m venv venv
source venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

If no requirements file is provided:

pip install pandas numpy pytest spacy
python -m spacy download en_core_web_sm


⸻

Running the Interactive Search Engine

Run the system in interactive CLI mode using:

python3 src/main.py

Common configurations

TF-IDF with Query Expansion

python3 src/main.py --ranker tfidf --qe

BM25 with Query Expansion

python3 src/main.py --ranker bm25 --qe

BM25 with thresholds applied

python3 src/main.py --ranker bm25 --qe --min-score-ratio 0.6 --hide-below-threshold

The system will prompt for queries until quit or exit is entered.

⸻

Batch Evaluation (Reproducibility)

All experimental results reported in the research log can be reproduced using the batch evaluation pipeline.

Example: BM25 without Query Expansion

python3 src/evaluation.py \
  --batch \
  --ranker bm25 \
  --no-qe \
  --no-ner \
  --queries src/queries.txt \
  --qrels qrels.json \
  --out-summary logs/final_bm25_noqe.json \
  --out-csv logs/final_bm25_noqe.csv

Example: BM25 + QE + PRF

python3 src/evaluation.py \
  --batch \
  --ranker bm25 \
  --qe \
  --prf \
  --prf-docs 5 \
  --prf-terms 5 \
  --queries src/queries.txt \
  --qrels qrels.json \
  --out-summary logs/final_bm25_qe_prf.json

Aggregating Results

python3 src/format_results.py \
  --inputs logs/final_bm25_noqe.json logs/final_bm25_qe.json logs/final_bm25_qe_prf.json logs/final_rocchio.json logs/final_bm25_qe_ner.json \
  --out-md logs/final_compare.md \
  --out-csv logs/final_compare.csv \
  --out-json logs/final_compare.json


⸻

Testing

All core components are unit tested using pytest.

Run tests with:

PYTHONPATH=src pytest -q

All tests should pass. A documented test failure and fix is discussed in the research log as part of the reflective analysis.

⸻

Research Log

The Research Log should be read alongside the code. It provides:
	•	Dataset and domain justification
	•	Conceptual investigation of IR models
	•	Design rationale for algorithms and data structures
	•	Experimental records and evaluation results
	•	Critical reflection on failures, debugging, and learning

Author

Isaac
Information Retrieval Coursework (2025)# search-engine
