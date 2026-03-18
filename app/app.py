import gradio as gr
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH   = ROOT_DIR / "data" / "metrics_for_llm.csv"
PROMPT_PATH = ROOT_DIR / "llm"  / "prompt_product_overview.txt"

df = pd.read_csv(DATA_PATH)

LLM_MODEL = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModelForCausalLM.from_pretrained(LLM_MODEL)


def build_prompt(product_id: str) -> str:
    row = df[df["ProductId"] == product_id]
    if row.empty:
        raise ValueError(f"Product not found: {product_id}")

    product_data = row.to_dict(orient="records")[0]

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    return base_prompt.format(
        product_data=json.dumps(product_data, indent=2, ensure_ascii=False)
    )


def get_preview(product_id: str) -> str:
    WRAPPER_OPEN = """
    <div style="min-height:260px;height:260px;overflow:hidden;
                border-radius:12px;border:1px solid rgba(255,255,255,0.08);
                padding:1.25rem 1.5rem;background:rgba(255,255,255,0.04);
                color:#e8e8e8;font-family:sans-serif;box-sizing:border-box;">
    """
    WRAPPER_CLOSE = "</div>"

    if not product_id:
        return (
            WRAPPER_OPEN
            + '<div style="height:100%;display:flex;align-items:center;justify-content:center;'
            + 'color:rgba(180,180,180,0.35);font-size:13px">Select a product to see its metrics</div>'
            + WRAPPER_CLOSE
        )

    row = df[df["ProductId"] == product_id].iloc[0]

    perc_pos = row["perc_positive"] * 100
    perc_neu = row["perc_neutral"]  * 100
    perc_neg = row["perc_negative"] * 100
    avg      = row["avg_score"]
    n        = int(row["n_reviews"])
    keywords = row["tfidf_keywords"] if pd.notna(row["tfidf_keywords"]) else "—"

    tags_html = "".join(
        f'<span style="font-size:12px;padding:3px 10px;border-radius:20px;'
        f'background:rgba(55,138,221,0.15);color:#378ADD;'
        f'margin:2px;display:inline-block;border:1px solid rgba(55,138,221,0.25)">'
        f'{k.strip()}</span>'
        for k in keywords.split(",")
    )

    def bar(label: str, pct: float, color: str) -> str:
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'  <span style="font-size:11px;color:rgba(180,180,180,0.8);width:52px;text-align:right">{label}</span>'
            f'  <div style="flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,0.08)">'
            f'    <div style="width:{pct:.0f}%;height:100%;border-radius:3px;background:{color}"></div>'
            f'  </div>'
            f'  <span style="font-size:11px;color:rgba(180,180,180,0.8);width:34px">{pct:.0f}%</span>'
            f'</div>'
        )

    content = f"""
      <div style="font-size:11px;font-weight:600;color:rgba(180,180,180,0.6);
                  font-family:monospace;letter-spacing:0.05em;margin-bottom:14px">
        {product_id}
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
        <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:12px 14px;
                    border:1px solid rgba(255,255,255,0.08)">
          <div style="font-size:11px;color:rgba(180,180,180,0.6);margin-bottom:4px">avg score</div>
          <div style="font-size:22px;font-weight:600;color:#e8e8e8">{avg:.2f}
            <span style="font-size:13px;color:rgba(180,180,180,0.5)">/ 5</span>
          </div>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:12px 14px;
                    border:1px solid rgba(255,255,255,0.08)">
          <div style="font-size:11px;color:rgba(180,180,180,0.6);margin-bottom:4px">total reviews</div>
          <div style="font-size:22px;font-weight:600;color:#e8e8e8">{n}</div>
        </div>
      </div>

      {bar("positive", perc_pos, "#1D9E75")}
      {bar("neutral",  perc_neu, "#888780")}
      {bar("negative", perc_neg, "#D85A30")}

      <div style="margin-top:12px">
        <div style="font-size:11px;color:rgba(180,180,180,0.6);margin-bottom:6px">top keywords</div>
        <div>{tags_html}</div>
      </div>
    """

    return WRAPPER_OPEN + content + WRAPPER_CLOSE


LOADING_HTML = """
<div style="display:flex;align-items:center;gap:10px;padding:8px 0;
            color:rgba(180,180,180,0.8);font-family:sans-serif;font-size:13px">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
       stroke="#1D9E75" stroke-width="2.5" stroke-linecap="round">
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
             M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83">
      <animateTransform attributeName="transform" type="rotate"
        from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
    </path>
  </svg>
  Generating analysis with Qwen 0.5B...
</div>
"""

EMPTY_HTML = ""


def run_analysis(product_id: str):
    if not product_id:
        yield EMPTY_HTML, "⚠️ Select a product first."
        return

    try:
        prompt = build_prompt(product_id)
    except ValueError as e:
        yield EMPTY_HTML, f"❌ {e}"
        return

    yield LOADING_HTML, ""

    try:
        messages = [
            {"role": "system", "content": "You are a product review analyst. Be objective and concise."},
            {"role": "user",   "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs  = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    except Exception as e:
        yield EMPTY_HTML, f"❌ Generation error: {e}"
        return

    yield EMPTY_HTML, response


# ─── Layout ──────────────────────────────────────────────────────────────────
product_ids = df["ProductId"].tolist()

with gr.Blocks(
    title="Product Review Analysis",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
    ),
    css="""
        .gradio-container { max-width: 860px !important; margin: auto; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("# 📦 Product Review Analysis")
    gr.Markdown("Choose a product to preview its metrics and generate an LLM analysis.")

    dropdown = gr.Dropdown(
        choices=product_ids,
        label="Product",
        interactive=True,
    )

    preview = gr.HTML(value=get_preview(""))
    btn     = gr.Button("Generate analysis", variant="primary", size="lg")
    status  = gr.HTML()
    output  = gr.Markdown(label="Analysis")

    dropdown.change(fn=get_preview, inputs=dropdown, outputs=preview)
    btn.click(fn=run_analysis, inputs=dropdown, outputs=[status, output])


if __name__ == "__main__":
    demo.launch()