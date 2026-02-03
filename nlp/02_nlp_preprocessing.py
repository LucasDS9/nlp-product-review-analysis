import spacy
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "data_cleaned.csv"
OUTPUT_PATH = DATA_DIR / "text_preprocessed.csv"


nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]
)


data = pd.read_csv(INPUT_PATH)


def preprocess_texts(texts, batch_size=1000):
    processed = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and not token.is_punct
        ]
        processed.append(" ".join(tokens))

    return processed


index_chunks = np.array_split(data.index, 10)
processed_chunks = []

for i, idx in enumerate(index_chunks):
    print(f"Processing chunk {i+1}/10")
    chunk = data.loc[idx].copy()
    chunk["Text_processed"] = preprocess_texts(chunk["Text"])
    processed_chunks.append(chunk)

data = pd.concat(processed_chunks, ignore_index=True)


data.to_csv(OUTPUT_PATH, index=False)

print(f"Arquivo salvo em: {OUTPUT_PATH}")
