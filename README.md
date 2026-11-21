# InsightForge: AI-Powered Business Intelligence Assistant

A conversational BI assistant for SMEs, built with LangChain (RAG + memory), FAISS, pandas, Plotly and Streamlit, and evaluated with `QAEvalChain`.

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env        # set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
streamlit run app.py
python -m insightforge.evaluation
```
The LLM (Llama 3.3 70B) and embeddings (BGE-base) are served by Cloudflare Workers AI through its OpenAI-compatible API.

## Project layout
| Path | Purpose |
|---|---|
| `data/sales_data.csv` | Sales dataset (replace with the course dataset if you have it; same schema) |
| `data/generate_sample_data.py` | Reproducible generator for the sample dataset |
| `insightforge/data_loader.py` | Loading and derived columns |
| `insightforge/analytics.py` | pandas KPIs, period, product, region and demographic statistics |
| `insightforge/knowledge_base.py` | Statistic documents and FAISS index |
| `insightforge/retriever.py` | Custom hybrid retriever (live pandas + topic routing + semantic search) |
| `insightforge/assistant.py` | Prompt chains (rephrase → retrieve → answer) and conversation memory |
| `insightforge/evaluation.py` | QAEvalChain evaluation, retrieval hit rate, latency |
| `insightforge/visualizations.py` | Plotly charts |
| `app.py` | Streamlit UI |
