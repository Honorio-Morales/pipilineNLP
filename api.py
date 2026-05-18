import sys

import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="NLP Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("ERROR: Model 'es_core_news_sm' not found.", file=sys.stderr)
    print("Run: python -m spacy download es_core_news_sm", file=sys.stderr)
    sys.exit(1)

POS_DESCRIPTIONS = {
    "ADJ": "Adjetivo",
    "ADP": "Preposicion / Postposicion",
    "ADV": "Adverbio",
    "AUX": "Verbo auxiliar",
    "CCONJ": "Conjuncion coordinante",
    "DET": "Determinante",
    "INTJ": "Interjeccion",
    "NOUN": "Sustantivo",
    "NUM": "Numeral",
    "PART": "Particula",
    "PRON": "Pronombre",
    "PROPN": "Nombre propio",
    "PUNCT": "Puntuacion",
    "SCONJ": "Conjuncion subordinante",
    "SYM": "Simbolo",
    "VERB": "Verbo",
    "X": "Otro",
}


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    doc = nlp(req.text)

    sentences = [sent.text.strip() for sent in doc.sents]

    tokens = [token.text for token in doc]

    lemmas = []
    for token in doc:
        lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
        lemmas.append({"token": token.text, "lemma": lemma})

    pos_tags = []
    for token in doc:
        pos_tags.append(
            {
                "token": token.text,
                "pos": token.pos_,
                "description": POS_DESCRIPTIONS.get(token.pos_, token.pos_),
            }
        )

    removed = []
    kept = []
    for token in doc:
        if not token.is_space:
            entry = {
                "token": token.text,
                "pos": token.pos_,
                "description": POS_DESCRIPTIONS.get(token.pos_, token.pos_),
            }
            if token.is_stop or token.is_punct:
                removed.append(entry)
            else:
                kept.append(entry)

    return {
        "sentences": sentences,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "stopwords": {"removed": removed, "kept": kept},
        "summary": {
            "total_tokens": len(tokens),
            "total_sentences": len(sentences),
            "tokens_after_stopword_removal": len(kept),
        },
    }
