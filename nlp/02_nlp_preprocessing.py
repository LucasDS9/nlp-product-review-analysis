import spacy
import pandas as pd
import numpy as np

nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]
)

data = pd.read_csv("../data/data_cleaned.csv")

import numpy as np

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

chunks = np.array_split(data, 10)

processed_chunks = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/10")
    chunk["Text_processed"] = preprocess_texts(chunk["Text"])
    processed_chunks.append(chunk)

data = pd.concat(processed_chunks)


data.to_csv("../data/preprocessed.csv")
