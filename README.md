# 🛒 NLP Product Review Analysis

> Pipeline completo de **Processamento de Linguagem Natural (NLP)** voltado à extração de insights estruturados a partir de avaliações textuais do **Amazon Fine Foods dataset**. O objetivo é transformar reviews não estruturadas em informações estratégicas por produto, combinando técnicas clássicas de NLP com análise via Large Language Models.

🌐 **[Acesse o portfólio](https://LucasDS9.github.io)** · 🚀 **[Testar o modelo](https://huggingface.co/spaces/LucasDS9/nlp-product-review-analysis)**

---

## 💼 Aplicações Reais

| Caso de uso | Descrição |
|---|---|
| 🔍 Monitoramento de produtos | Detectar quedas na percepção de qualidade via sentimento |
| 👤 Experiência do cliente | Compreender dores e expectativas de forma automatizada |
| 📋 Apoio à decisão | Relatórios claros e objetivos para gestores e stakeholders |
| ⚡ Escalabilidade analítica | Substituir análises manuais por pipeline NLP + LLM |

---

## 📌 Problema

Avaliações de clientes são uma das fontes de dados mais valiosas e menos exploradas pelas empresas. Elas contêm feedback direto sobre qualidade do produto, satisfação do cliente, problemas recorrentes e expectativas não atendidas — porém em um formato não estruturado e difícil de analisar em escala.

Este projeto demonstra como técnicas de NLP podem:
- Converter milhares de textos livres em **métricas estruturadas**
- Identificar principais **reclamações e pontos fortes** dos produtos
- Quantificar o **sentimento dos clientes** de forma objetiva
- Apoiar decisões estratégicas **sem leitura manual** das avaliações

---

## 📁 Estrutura do Projeto

```text
📦 nlp-product-review-analysis
├── 📁 data
│   └── reviews.csv              # Amazon Fine Foods dataset
│
├── 📁 src
│   ├── preprocessing.py         # Limpeza textual e feature engineering
│   ├── vectorization.py         # TF-IDF e Word2Vec
│   ├── sentiment.py             # Agregação de sentimento por produto
│   └── llm_insights.py          # Geração de insights com Qwen 0.5B
│
├── app.py                       # Interface Gradio/Streamlit
├── requirements.txt
└── README.md
```

---

## 🧱 Etapas do Projeto

### 1️⃣ Ingestão e conhecimento dos dados
- Leitura do **Amazon Fine Foods dataset**
- Inspeções iniciais: estrutura, tipos de variáveis e distribuição das avaliações
- Entendimento do volume e qualidade dos reviews por produto

### 2️⃣ Limpeza textual e feature engineering
- Remoção de stopwords, pontuação e caracteres especiais com **spaCy**
- Normalização e lematização dos textos
- Criação de features derivadas para enriquecer a análise

### 3️⃣ Vetorização de texto
- **TF-IDF**: extração das palavras-chave mais relevantes por produto
- **Word2Vec**: representação semântica densa dos reviews para capturar relações entre termos

### 4️⃣ Análise de sentimento e agregação por produto
- Classificação dos reviews em positivo, neutro e negativo
- Agregação das métricas por produto: score médio, distribuição de sentimentos e keywords principais

### 5️⃣ Geração de insights com LLM
- **Qwen 0.5B** com prompt engineering estruturado para gerar relatórios executivos por produto
- Análise automática de pontos fortes, críticas recorrentes e recomendações de melhoria
- Output formatado e pronto para consumo por gestores e stakeholders

---

## 📊 Exemplo de Output do Modelo

Produto analisado: *Chocolate Almond Bar*

| Métrica | Valor |
|---|---|
| Sentimento geral | Neutro |
| Reviews negativas | 19,5% |
| Reviews neutras | 47,2% |
| Reviews positivas | 33,3% |
| Score médio | 3,69 / 5 |

**Palavras-chave TF-IDF:** chocolate · amêndoa · cacau · sabor · nozes

**Visão executiva gerada pelo modelo:**
> Produto com qualidade satisfatória e score ligeiramente acima da média. Aproximadamente um terço dos clientes relata experiência positiva, enquanto 19,5% demonstram insatisfação — principalmente relacionada à consistência de sabor e ingredientes. Manter padrões de qualidade e endereçar as reclamações recorrentes é crucial para melhorar a satisfação geral.

**Riscos e fraquezas identificados:**
- Alto percentual de reviews neutras indica ausência de diferencial percebido
- Reclamações sobre sabor e textura aparecem de forma recorrente
- Ingrediente amêndoa pode representar risco de alergia para parte dos consumidores

---

## 🚀 Conclusão

O projeto demonstra como NLP tradicional e LLMs podem ser integrados para transformar dados textuais não estruturados em **insights acionáveis e escaláveis**. O resultado final é um sistema projetado para gerar informações prontas para consumo executivo, indo além de análises puramente técnicas.

---

## 🛠 Tecnologias Utilizadas

| Tecnologia | Função |
|---|---|
| 🐍 **Python** | Linguagem principal do projeto |
| 🧮 **Pandas / NumPy** | Manipulação e análise de dados |
| 🔤 **spaCy** | Limpeza textual, tokenização e lematização |
| 📐 **TF-IDF (Scikit-learn)** | Extração de palavras-chave por produto |
| 🔢 **Word2Vec (Gensim)** | Representação semântica densa dos reviews |
| 🤖 **Qwen 0.5B (Transformers)** | Geração de insights executivos via LLM |
| 💬 **Prompt Engineering** | Estruturação dos outputs do LLM |
| 🚀 **Gradio / Streamlit** | Interface web e deploy da aplicação |
| 🤗 **Hugging Face Spaces** | Deploy e hospedagem do modelo |