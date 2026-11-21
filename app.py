"""InsightForge - Streamlit UI for the AI-powered Business Intelligence Assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from insightforge import analytics as A
from insightforge import visualizations as V
from insightforge.assistant import InsightForgeAssistant
from insightforge.data_loader import load_data, prepare
from insightforge.evaluation import RESULTS_PATH, run_evaluation
from insightforge.llm import provider_name

st.set_page_config(page_title="InsightForge BI Assistant", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem}
div[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px}
</style>""", unsafe_allow_html=True)

SAMPLE_QUESTIONS = [
    "Give me an overview of business performance",
    "Which product has the highest total sales?",
    "What about Widget C?",
    "Which region has the lowest sales?",
    "Show the yearly sales trend",
    "How do sales vary by age group?",
    "What is the median and standard deviation of sales?",
    "Which product has the highest customer satisfaction?",
]

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("📈 InsightForge")
    st.caption("AI-powered Business Intelligence Assistant")
    st.info(f"**LLM:** {provider_name()}")
    upload = st.file_uploader("Upload your own sales CSV (same schema)", type="csv")
    st.markdown("**Try a question:**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_q"] = q
    if st.button("🧹 Clear conversation memory", use_container_width=True):
        st.session_state.get("assistant") and st.session_state["assistant"].reset_memory()
        st.session_state["chat"] = []


@st.cache_data
def _load(file_bytes: bytes | None) -> pd.DataFrame:
    if file_bytes:
        import io
        return prepare(pd.read_csv(io.BytesIO(file_bytes)))
    return load_data()


df = _load(upload.getvalue() if upload else None)
data_key = (len(df), float(df["Sales"].sum()))
if st.session_state.get("data_key") != data_key:
    with st.spinner("Building knowledge base & FAISS index..."):
        st.session_state["assistant"] = InsightForgeAssistant(df)
    st.session_state["data_key"] = data_key
    st.session_state["chat"] = []
assistant: InsightForgeAssistant = st.session_state["assistant"]

k = A.kpis(df)
st.header("InsightForge — Business Intelligence Assistant")
c = st.columns(6)
c[0].metric("Total sales", f"${k['total_sales']/1e6:.2f}M")
c[1].metric("Transactions", f"{k['transactions']:,}")
c[2].metric("Avg sale", A.money(k["avg_sale"]))
c[3].metric("Median sale", A.money(k["median_sale"]))
c[4].metric("Std deviation", A.money(k["std_sale"]))
c[5].metric("Avg satisfaction", f"{k['avg_satisfaction']:.2f} / 5")

tab_chat, tab_dash, tab_explore, tab_eval, tab_mem = st.tabs(
    ["💬 AI Assistant", "📊 Insights Dashboard", "🔎 Data Explorer", "🧪 Model Evaluation", "🧠 Memory & Logs"])

# -------------------------------------------------------------------- chat --
with tab_chat:
    st.caption(f"Knowledge base: {len(assistant.docs)} statistical documents indexed in FAISS · "
               f"custom retriever = live pandas stats + topic routing + semantic search · memory window = {assistant.memory_window} turns")
    for msg in st.session_state.get("chat", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True, theme=None, key=f"c{id(msg)}")
            if msg.get("sources"):
                with st.expander(f"Retrieved context ({len(msg['sources'])} documents) · standalone question: “{msg['standalone']}”"):
                    for d in msg["sources"]:
                        st.markdown(f"- `{d.metadata.get('topic')}` {d.page_content}")
    question = st.chat_input("Ask about sales, products, regions, customers...") or st.session_state.pop("pending_q", None)
    if question:
        try:
            turn = assistant.ask(question)
            st.session_state.setdefault("chat", []).append({"role": "user", "content": question})
            st.session_state["chat"].append({"role": "assistant", "content": turn.answer, "sources": turn.sources,
                                             "standalone": turn.standalone, "chart": V.chart_for_topics(df, turn.topics)})
            st.rerun()
        except Exception as e:  # e.g. Cloudflare quota / network errors
            st.error(f"LLM request failed: {type(e).__name__}: {e}")

# --------------------------------------------------------------- dashboard --
with tab_dash:
    st.subheader("Sales trends over time")
    st.plotly_chart(V.sales_trend(df), use_container_width=True, theme=None)
    a, b = st.columns(2)
    a.plotly_chart(V.yearly_quarterly(df), use_container_width=True, theme=None)
    b.plotly_chart(V.product_trend(df), use_container_width=True, theme=None)
    st.subheader("Product performance comparison")
    a, b = st.columns(2)
    a.plotly_chart(V.product_performance(df), use_container_width=True, theme=None)
    b.plotly_chart(V.product_distribution(df), use_container_width=True, theme=None)
    st.subheader("Regional analysis")
    a, b = st.columns(2)
    a.plotly_chart(V.regional_analysis(df), use_container_width=True, theme=None)
    b.plotly_chart(V.product_region_heatmap(df), use_container_width=True, theme=None)
    st.subheader("Customer demographics & segmentation")
    a, b = st.columns(2)
    a.plotly_chart(V.age_group_sales(df), use_container_width=True, theme=None)
    b.plotly_chart(V.gender_split(df), use_container_width=True, theme=None)
    a, b = st.columns(2)
    a.plotly_chart(V.satisfaction_by_product(df), use_container_width=True, theme=None)
    b.plotly_chart(V.age_vs_sales(df), use_container_width=True, theme=None)

# ---------------------------------------------------------------- explorer --
with tab_explore:
    f1, f2, f3, f4 = st.columns(4)
    prods = f1.multiselect("Product", sorted(df["Product"].unique()), default=sorted(df["Product"].unique()))
    regs = f2.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    years = f3.multiselect("Year", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
    dim = f4.selectbox("Group by", ["Product", "Region", "Year", "Quarter", "Age_Group", "Customer_Gender"])
    sub = df[df["Product"].isin(prods) & df["Region"].isin(regs) & df["Year"].isin(years)]
    st.markdown(f"**{len(sub):,} transactions selected** · total {A.money(sub['Sales'].sum())}")
    st.dataframe(A.group_summary(sub, dim), use_container_width=True)
    st.markdown("**Statistical measures**")
    st.dataframe(sub[["Sales", "Customer_Age", "Customer_Satisfaction"]].describe().T.assign(
        median=sub[["Sales", "Customer_Age", "Customer_Satisfaction"]].median()).round(2), use_container_width=True)
    with st.expander("Raw data"):
        st.dataframe(sub, use_container_width=True, height=300)
    st.download_button("Download filtered CSV", sub.to_csv(index=False), "insightforge_filtered.csv")

# -------------------------------------------------------------- evaluation --
with tab_eval:
    st.markdown("Ground-truth answers are computed with **pandas** and the assistant's answers are graded with "
                "LangChain's **QAEvalChain** (LLM-as-a-judge: CORRECT / INCORRECT). Retrieval hit-rate checks that the "
                "custom retriever returned documents for the expected topic.")
    if st.button("▶️ Run QAEvalChain evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            res, summary = run_evaluation(df)
        st.session_state["eval"] = (res, summary)
    elif "eval" not in st.session_state and RESULTS_PATH.with_name("eval_summary.json").exists():
        import json
        st.session_state["eval"] = (pd.read_csv(RESULTS_PATH), json.loads(RESULTS_PATH.with_name("eval_summary.json").read_text()))
        st.caption("Showing the results of the last evaluation run (evaluation/eval_results.csv). Click the button to re-run.")
    if "eval" in st.session_state:
        res, s = st.session_state["eval"]
        m = st.columns(5)
        m[4].metric("Wrong answers caught by grader", s.get("grader_negative_controls_caught", "–"),
                    help="Negative control: deliberately wrong answers that the QAEvalChain judge must mark INCORRECT")
        m[0].metric("Test questions", s["n_tests"])
        m[1].metric("QAEvalChain accuracy", f"{s['qa_accuracy_%']}%")
        m[2].metric("Retrieval hit rate", f"{s['retrieval_hit_rate_%']}%")
        m[3].metric("Avg latency", f"{s['avg_latency_s']}s")
        st.dataframe(res[["question", "reference", "grade", "retrieval_hit", "latency_s"]], use_container_width=True, height=540)
        with st.expander("Full predictions"):
            for _, r in res.iterrows():
                st.markdown(f"**Q:** {r.question}  \n**Reference:** {r.reference}  \n**Grade:** {r.grade}")
                st.markdown(r.prediction)
                st.divider()

# ------------------------------------------------------------------ memory --
with tab_mem:
    st.markdown("Conversation memory (sliding window) is injected into the **rephrase** prompt so that follow-ups such as "
                "*“What about Widget C?”* are rewritten into standalone questions before retrieval.")
    st.markdown("**Interaction log (question -> rewritten standalone question -> retrieval)**")
    if True:
        if assistant.turns:
            st.dataframe(pd.DataFrame([{"question": t.question, "standalone_question": t.standalone,
                                        "topics": ", ".join(t.topics), "docs": len(t.sources), "latency_s": t.latency_s}
                                       for t in assistant.turns]), use_container_width=True)
        else:
            st.caption("No questions asked yet.")
    st.markdown("**Memory buffer sent to the LLM**")
    st.code(assistant.history_text(), language="text", wrap_lines=True, height=420)
