import spacy
import sys

TEXTO = (
    "Netflix ha encontrado en el juego del calamar su nuevo fenómeno mundial "
    "ni siquiera en la propia plataforma contaban con ello como seguro que "
    "tampoco esperaban recibir multitud de quejas por una escena del cuarto "
    "episodio sin embargo han estado rápidos para responder a la indignación "
    "del público y ha introducido un cambio en el equipo"
)

POS_DESCRIPCIONES = {
    "ADJ": "Adjetivo",
    "ADP": "Preposición / Postposición",
    "ADV": "Adverbio",
    "AUX": "Verbo auxiliar",
    "CCONJ": "Conjunción coordinante",
    "DET": "Determinante",
    "INTJ": "Interjección",
    "NOUN": "Sustantivo",
    "NUM": "Numeral",
    "PART": "Partícula",
    "PRON": "Pronombre",
    "PROPN": "Nombre propio",
    "PUNCT": "Puntuación",
    "SCONJ": "Conjunción subordinante",
    "SYM": "Símbolo",
    "VERB": "Verbo",
    "X": "Otro",
}


def encabezado(titulo: str) -> None:
    print()
    print("=" * 72)
    print(f"  {titulo}")
    print("=" * 72)


def etapa_segmentacion(nlp, texto: str):
    encabezado("ETAPA 1: Segmentación en Frases (Sentence Segmentation)")
    doc = nlp(texto)
    for i, sent in enumerate(doc.sents, 1):
        print(f"  Oración {i:2d}: {sent.text}")
    print(f"\n  → Total de oraciones detectadas: {i}")
    print()
    for sent in doc.sents:
        print("  " + " | ".join(token.text for token in sent))
    return doc


def etapa_tokenizacion(doc):
    encabezado("ETAPA 2: Tokenización (Tokenization)")
    for i, token in enumerate(doc, 1):
        print(f"  Token {i:2d}: {token.text}")
    print(f"\n  → Total de tokens: {len(doc)}")
    print()
    for sent in doc.sents:
        print("  " + " | ".join(token.text for token in sent))


def etapa_lematizacion(doc):
    encabezado("ETAPA 3: Lematización (Lemmatization)")
    for i, token in enumerate(doc, 1):
        lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
        print(f"  Token {i:2d}: {token.text:30s} →  lema: {lemma}")
    print(f"\n  → Total de tokens: {len(doc)}")
    print()
    for sent in doc.sents:
        lemmas = [
            token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
            for token in sent
        ]
        print("  " + " | ".join(lemmas))


def etapa_pos(doc):
    encabezado("ETAPA 4: POS Tagging (Part-of-Speech)")
    for i, token in enumerate(doc, 1):
        desc = POS_DESCRIPCIONES.get(token.pos_, token.pos_)
        print(f"  Token {i:2d}: {token.text:30s}  POS: {token.pos_:6s}  ({desc})")
    print(f"\n  → Total de tokens: {len(doc)}")
    print()
    for sent in doc.sents:
        print("  " + " | ".join(f"{token.text} [{token.pos_}]" for token in sent))


def etapa_stopwords(doc):
    encabezado("ETAPA 5: Quitar Stopwords (Stopword Removal)")
    for i, token in enumerate(doc, 1):
        if not token.is_stop and not token.is_punct and not token.is_space:
            desc = POS_DESCRIPCIONES.get(token.pos_, token.pos_)
            print(f"  Token {i:2d}: {token.text:30s}  POS: {token.pos_:6s}  ({desc})")
    print(f"\n  → Total de tokens: {len(doc)}")
    print()
    for sent in doc.sents:
        kept = [
            token.text
            for token in sent
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        if kept:
            print("  " + " | ".join(kept))


def main():
    print("=" * 72)
    print("  PIPELINE DE PROCESAMIENTO DE LENGUAJE NATURAL (NLP)")
    print("  Modelo: spaCy es_core_news_sm")
    print("=" * 72)

    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        print(
            "\n  ERROR: No se encuentra el modelo 'es_core_news_sm'.\n"
            "  Ejecute: python -m spacy download es_core_news_sm\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n  Texto de entrada:")
    print(f"  {TEXTO[:80]}...")
    print(f"  ({len(TEXTO)} caracteres)")

    doc = etapa_segmentacion(nlp, TEXTO)
    etapa_tokenizacion(doc)
    etapa_lematizacion(doc)
    etapa_pos(doc)
    etapa_stopwords(doc)

    print()
    print("=" * 72)
    print("  Pipeline completado exitosamente.")
    print("=" * 72)


if __name__ == "__main__":
    main()
