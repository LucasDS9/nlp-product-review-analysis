from pathlib import Path
import pandas as pd
import subprocess
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
sys.stderr.reconfigure(encoding="utf-8", errors="ignore")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "metrics_for_llm.csv"
PROMPT_PATH = BASE_DIR / "llm" / "prompt_product_overview.txt"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)


def list_products():
    print("\n📦 Produtos disponíveis:\n")
    for pid in df["ProductId"].unique():
        print(f"- {pid}")


def build_prompt(product_id: str) -> str:
    row = df[df["ProductId"] == product_id]

    if row.empty:
        raise ValueError("❌ Produto não encontrado.")

    product_data = row.to_dict(orient="records")[0]

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    return base_prompt.format(
        product_data=json.dumps(product_data, indent=2, ensure_ascii=False)
    )


def call_llama(prompt: str, model: str = "llama3.1:8b") -> str:
    process = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if process.stderr:
        print("⚠️ Avisos do Ollama:")
        print(process.stderr)

    return process.stdout.strip()

if __name__ == "__main__":
    list_products()

    product_id = input("\nDigite o ProductId para análise: ").strip()

    prompt = build_prompt(product_id)

    print("\n🤖 Gerando análise com LLaMA...\n")

    response = call_llama(prompt)

    report_path = REPORTS_DIR / f"product_{product_id}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Product Analysis: {product_id}\n\n")
        f.write(response)

    print(f"✅ Relatório gerado em: {report_path}")
