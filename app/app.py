import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
from llm.analyze_product import call_llama, build_prompt

st.set_page_config(page_title="Product Review NLP", layout="centered")

st.title("📦 Product Review Analysis")

df = pd.read_csv(ROOT_DIR / "data" / "metrics_for_llm.csv")

product_id = st.selectbox(
    "Selecione um produto",
    df["ProductId"].unique()
)

if st.button("Gerar análise"):
    with st.spinner("Gerando insights com LLM..."):
        prompt = build_prompt(product_id)
        response = call_llama(prompt)

    st.markdown("## 📊 Análise do Produto")
    st.markdown(response)
