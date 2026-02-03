from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "text_preprocessed.csv"
OUTPUT_PATH = BASE_DIR / "data" / "metrics_for_llm.csv"



data = pd.read_csv(DATA_PATH)

required_cols = ["ProductId", "Score", "Text_processed"]
missing = set(required_cols) - set(data.columns)
if missing:
    raise ValueError(f"Colunas ausentes no dataset: {missing}")


def review_sentiment(score: float) -> str:
    if score <= 2.5:
        return "negative"
    elif score < 4.5:
        return "neutral"
    else:
        return "positive"


data["review_sentiment"] = data["Score"].apply(review_sentiment)


product_metrics = (
    data
    .groupby("ProductId")
    .agg(
        n_reviews=("Score", "count"),
        avg_score=("Score", "mean")
    )
    .reset_index()
)


sentiment_counts = (
    data
    .groupby(["ProductId", "review_sentiment"])
    .size()
    .unstack(fill_value=0)
)

sentiment_perc = (
    sentiment_counts
    .div(sentiment_counts.sum(axis=1), axis=0)
    .reset_index()
)

sentiment_perc.rename(columns={
    "negative": "perc_negative",
    "neutral": "perc_neutral",
    "positive": "perc_positive"
}, inplace=True)


def product_sentiment(row) -> str:
    if row["perc_negative"] > row["perc_positive"]:
        return "negativo"
    elif row["perc_positive"] > 0.5:
        return "positivo"
    else:
        return "normal"


sentiment_perc["product_sentiment"] = sentiment_perc.apply(
    product_sentiment, axis=1
)


def extract_tfidf_keywords(df: pd.DataFrame, n_terms: int = 8) -> pd.Series:
    if len(df) < 3:
        return pd.Series({"tfidf_keywords": None})

    tfidf = TfidfVectorizer(
        min_df=2,
        max_df=0.8,
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        X = tfidf.fit_transform(df["Text_processed"])
    except ValueError:
        return pd.Series({"tfidf_keywords": None})

    terms = tfidf.get_feature_names_out()
    scores = X.mean(axis=0).A1

    keywords = (
        pd.Series(scores, index=terms)
        .sort_values(ascending=False)
        .head(n_terms)
        .index
        .tolist()
    )

    return pd.Series({
        "tfidf_keywords": ", ".join(keywords)
    })


tfidf_keywords = (
    data
    .groupby("ProductId")
    .apply(extract_tfidf_keywords)
    .reset_index()
)


final_df = (
    product_metrics
    .merge(sentiment_perc, on="ProductId", how="left")
    .merge(tfidf_keywords, on="ProductId", how="left")
)


final_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Métricas prontas para LLM!")
print(f"📁 Arquivo salvo em: {OUTPUT_PATH}")
