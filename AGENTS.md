# AGENTS.md

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install spacy fastapi uvicorn
python3 -m spacy download es_core_news_sm
python3 pipeline_nlp.py              # CLI pipeline (fixed input)
python3 -m uvicorn api:app --port 8000  # API server for frontend
```

## Requirements

- `requirements.txt` lists `spacy` pip package + the URL wheel for `es_core_news_sm`.
- The model must be downloaded separately (`python3 -m spacy download es_core_news_sm`); it is **not** installed by `pip install -r requirements.txt` alone.

## Project structure

| File | Purpose |
|---|---|
| `pipeline_nlp.py` | Single-file classical NLP pipeline (5 stages: sentence segmentation, tokenization, lemmatization, POS tagging, stopword removal) |
| `api.py` | FastAPI server exposing POST /analyze endpoint using spaCy |
| `index.html` | Single-page frontend for GitHub Pages, calls the API |
| `requirements.txt` | Python dependencies |
| `.venv/` | Local virtualenv (created on first setup) |

## Important constraints

- The input text in `pipeline_nlp.py` must **not** be changed — it is the fixed deliverable.
- All 5 stages must produce clear printed output with headers (they are part of the deliverable log).
- The script uses `spacy.load("es_core_news_sm")` explicitly; multi-language or other model variants are not substitutes.

## Commands

```bash
# run pipeline
python3 pipeline_nlp.py

# reinstall model if missing
python3 -m spacy download es_core_news_sm
```
