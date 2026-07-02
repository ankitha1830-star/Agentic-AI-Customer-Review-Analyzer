import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import os
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
import faiss
from fpdf import FPDF
from langchain_core.prompts import PromptTemplate




# ---------------- NLTK DOWNLOADS ----------------
for pkg in ["stopwords", "wordnet", "punkt"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agentic AI Customer Review Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    color: white;
}
.metric-card .val { font-size: 2rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }
.insight-box {
    background: #1e293b;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
}
.reply-box {
    background: #0f2d1e;
    border: 1px solid #16a34a;
    border-radius: 8px;
    padding: 14px;
    color: #bbf7d0;
    font-style: italic;
}
.anomaly-box {
    background: #2d1515;
    border: 1px solid #dc2626;
    border-radius: 8px;
    padding: 14px;
    color: #fca5a5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
    )

@st.cache_resource
def load_emotion_model():
    try:
        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )
    except Exception:
        return None

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

sentiment_pipeline = load_sentiment_model()
emotion_model = load_emotion_model()
sbert_model = load_sbert()

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

lemmatizer = WordNetLemmatizer()

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = []
    for w in text.split():
        if w not in stop_words and len(w) > 2:
            words.append(lemmatizer.lemmatize(w))
    return " ".join(words)


def get_sentiment(text: str) -> tuple[str, float]:
    text = str(text).strip()
    if not text:
        return "NEUTRAL", 0.5
    try:
        r = sentiment_pipeline(text[:512])[0]
        return r["label"], round(float(r["score"]), 3)
    except Exception:
        return "NEUTRAL", 0.5


def get_emotion(text: str) -> str:
    if emotion_model is None:
        return "unknown"
    text = str(text).strip()
    if not text:
        return "unknown"
    try:
        r = emotion_model(text[:512])[0]
        return r[0]["label"] if isinstance(r, list) else r["label"]
    except Exception:
        return "unknown"


def generate_recommendation(sentiment: str, review: str) -> str:
    review = str(review).lower()
    if sentiment == "NEGATIVE":
        rules = [
            (["size", "fit", "tight", "loose"], "Improve size chart accuracy and fitting guidance."),
            (["quality", "fabric", "material"], "Enhance product material quality control."),
            (["delivery", "shipping", "late"], "Improve delivery speed and packaging process."),
            (["price", "expensive", "cheap"], "Review pricing and add suitable offers."),
            (["return", "refund", "exchange"], "Simplify return, refund, and exchange process."),
            (["customer service", "support"], "Improve customer support response quality."),
        ]
        for keywords, message in rules:
            if any(k in review for k in keywords):
                return message
        return "Investigate complaint pattern and improve customer experience."
    if sentiment == "POSITIVE":
        return "Maintain quality and promote positive customer experience."
    return "Collect more reviews to understand customer needs better."


def detect_complaints(df: pd.DataFrame) -> pd.DataFrame:
    complaint_words = [
        "bad", "poor", "small", "large", "tight", "loose", "quality", "return", "refund",
        "damaged", "late", "uncomfortable", "cheap", "wrong", "disappointed", "broken",
        "fake", "slow", "ugly", "smell", "colour", "color", "missing", "defective",
    ]
    counts = {}
    for review in df.get("Cleaned_Review", pd.Series(dtype=str)).fillna(""):
        for word in complaint_words:
            if word in str(review).split():
                counts[word] = counts.get(word, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["Complaint", "Count"])
    return pd.DataFrame(counts.items(), columns=["Complaint", "Count"]).sort_values("Count", ascending=False)


def extract_topics_simple(df: pd.DataFrame, top_n: int = 15) -> list[str]:
    reviews = df["Cleaned_Review"].dropna().astype(str)
    reviews = reviews[reviews.str.strip() != ""]
    if reviews.empty:
        return ["No meaningful text found"]
    try:
        vec = CountVectorizer(max_features=top_n)
        vec.fit_transform(reviews)
        return list(vec.get_feature_names_out())
    except ValueError:
        return ["No meaningful text found"]


def run_lda(df: pd.DataFrame, n_topics: int = 5, n_words: int = 8) -> list[dict]:
    reviews = df["Cleaned_Review"].dropna().astype(str)
    reviews = reviews[reviews.str.strip() != ""]

    if reviews.empty:
        return [{"id": 1, "words": ["No meaningful text found"]}]

    try:
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        X = tfidf.fit_transform(reviews)

        if X.shape[0] == 0 or X.shape[1] == 0:
            return [{"id": 1, "words": ["No meaningful text found"]}]

        actual_topics = min(n_topics, X.shape[0], X.shape[1])
        actual_topics = max(1, actual_topics)

        lda = LatentDirichletAllocation(n_components=actual_topics, random_state=42)
        lda.fit(X)
        feature_names = tfidf.get_feature_names_out()

        topics = []
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append({"id": idx + 1, "words": top_words})
        return topics
    except ValueError:
        return [{"id": 1, "words": ["No meaningful text found"]}]
    except Exception as e:
        return [{"id": 1, "words": [f"Topic modeling skipped: {e}"]}]


def build_faiss_index(df: pd.DataFrame):
    texts = df["Review_Text"].dropna().astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if not texts:
        return None, []
    embeddings = sbert_model.encode(texts, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts


def semantic_search(query: str, index, texts: list[str], k: int = 5) -> list[str]:
    if index is None or not texts or not query.strip():
        return []
    q_emb = sbert_model.encode([query]).astype("float32")
    _, I = index.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]


def detect_anomalies(df: pd.DataFrame, window: int = 50) -> list[str]:
    alerts = []
    if "Sentiment" not in df.columns or len(df) < 10:
        return alerts
    negative_rate = df["Sentiment"].eq("NEGATIVE").mean()
    if negative_rate > 0.5:
        alerts.append("🚨 Overall negative reviews are more than 50%. Product/service improvement is needed.")
    if len(df) >= window * 2:
        neg_rolling = df["Sentiment"].eq("NEGATIVE").rolling(window).mean()
        spikes = neg_rolling[neg_rolling > 0.6].index.tolist()
        if spikes:
            alerts.append(f"⚠️ Negative sentiment spike detected near rows {spikes[:3]}.")
    return alerts


def rule_based_reply(review_text: str) -> str:
    lower = str(review_text).lower()
    if any(w in lower for w in ["size", "fit", "tight", "loose"]):
        return "Thank you for your feedback. We are sorry the size did not meet your expectation. Please contact support with your order number for an exchange or refund."
    if any(w in lower for w in ["quality", "fabric", "material", "cheap"]):
        return "We sincerely apologise for the quality issue. This is not our standard. Please share your order details so we can help you quickly."
    if any(w in lower for w in ["delivery", "shipping", "late", "slow"]):
        return "We are sorry for the delivery delay. Please share your order number and our team will check it immediately."
    return "Thank you for sharing your feedback. We are sorry for your experience and would like to make it right. Please contact our support team."


def draft_reply(review_text: str, groq_key: str) -> str:
    if not groq_key:
        return rule_based_reply(review_text)
    try:
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "Write a short professional customer support reply."},
                {"role": "user", "content": str(review_text)},
            ],
            "max_tokens": 180,
            "temperature": 0.7,
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return rule_based_reply(review_text)


def generate_pdf_report(df: pd.DataFrame, sentiment_counts: dict, topics: list[str], anomalies: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Customer Review Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Total reviews analysed: {len(df)}", new_x="LMARGIN", new_y="NEXT")
    for label, count in sentiment_counts.items():
        pct = round(100 * count / max(len(df), 1), 1)
        pdf.cell(0, 8, f"{label}: {count} ({pct}%)", new_x="LMARGIN", new_y="NEXT")

    if anomalies:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Anomalies", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for a in anomalies:
            pdf.multi_cell(0, 8, f"- {a}".encode("latin-1", "ignore").decode("latin-1"))

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Key Topics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, ", ".join(topics).encode("latin-1", "ignore").decode("latin-1"))

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Sample Reviews", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, row in df.head(10).iterrows():
        text = str(row.get("Review_Text", ""))[:220].replace("\n", " ")
        sent = str(row.get("Sentiment", ""))
        line = f"[{sent}] {text}".encode("latin-1", "ignore").decode("latin-1")
        pdf.multi_cell(0, 7, line)
        pdf.ln(1)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)


def langchain_summary(sentiment_counts, topics) -> str:
    template = PromptTemplate(
        input_variables=["sentiments", "topics"],
        template=(
            "Customer review analysis summary:\n"
            "Sentiment distribution: {sentiments}\n"
            "Key topics identified: {topics}\n\n"
            "Business Insight:\n"
            "The analysis shows customer satisfaction trends, repeated issues, and improvement areas."
        ),
    )
    return template.format(sentiments=str(sentiment_counts), topics=", ".join(topics))






def guess_text_column(df: pd.DataFrame) -> int:
    names = [str(c).lower() for c in df.columns]
    preferred = ["review text", "review", "text", "content", "comment", "description", "title"]
    for p in preferred:
        for i, name in enumerate(names):
            if p == name or p in name:
                return i
    object_cols = [i for i, c in enumerate(df.columns) if df[c].dtype == "object"]
    return object_cols[0] if object_cols else 0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🧠 Review Analyzer")
    st.markdown("### ⚙️ Settings")
    max_rows = st.slider("Max reviews to analyze", 10, 5000, 1000, step=50)
    n_topics_lda = st.slider("LDA topic count", 2, 10, 5)
    enable_emotion = st.checkbox("Enable emotion detection", value=True)
    enable_semantic = st.checkbox("Enable semantic search", value=True)
    enable_replies = st.checkbox("Auto-draft replies for negative reviews", value=False)

    st.markdown("### 🔑 API Key Optional")
    groq_key = st.text_input("Groq API key", type="password")

   

# ---------------- MAIN ----------------
st.title("🧠 Agentic AI Customer Review Analyzer")
st.write("Upload customer reviews and run sentiment analysis, topic modeling, semantic search, complaint detection, recommendations, and report export.")

st.info("Important: Select the column that contains actual review sentences, for example Review Text. Do not select numeric columns like Positive Feedback Count.")

tab_upload = st.tabs(["📂 Upload CSV"])[0]

df = pd.DataFrame()

with tab_upload:
    uploaded_file = st.file_uploader("Upload Customer Review CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")
        st.success(f"Loaded {len(df):,} rows")
        st.dataframe(df.head(), use_container_width=True)

        

# ---------------- ANALYSIS ----------------
if not df.empty:
    st.markdown("---")
    default_index = guess_text_column(df)
    text_column = st.selectbox("Select review text column", df.columns, index=default_index)

    if not pd.api.types.is_string_dtype(df[text_column]) and df[text_column].nunique() < 20:
        st.warning("This selected column looks numeric/categorical. Please choose a text review column.")

    df = df.head(max_rows).copy()
    df["Review_Text"] = df[text_column].fillna("").astype(str)
    df = df[df["Review_Text"].str.strip() != ""].copy()

    if df.empty:
        st.error("Selected column has no valid review text.")
        st.stop()

    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting...")

        progress.progress(10, "Preprocessing text...")
        df["Cleaned_Review"] = df["Review_Text"].apply(preprocess_text)

        progress.progress(25, "Running sentiment analysis...")
        results = df["Review_Text"].apply(get_sentiment)
        df["Sentiment"] = results.apply(lambda x: x[0])
        df["Sentiment_Score"] = results.apply(lambda x: x[1])

        if enable_emotion:
            progress.progress(40, "Detecting emotions...")
            df["Emotion"] = df["Review_Text"].apply(get_emotion)

        progress.progress(55, "Generating recommendations...")
        df["Recommendation"] = df.apply(lambda r: generate_recommendation(r["Sentiment"], r["Cleaned_Review"]), axis=1)

        progress.progress(65, "Extracting topics...")
        lda_topics = run_lda(df, n_topics=n_topics_lda)

        if enable_semantic:
            progress.progress(75, "Building semantic search index...")
            faiss_index, review_texts = build_faiss_index(df)
            st.session_state["faiss_index"] = faiss_index
            st.session_state["review_texts"] = review_texts

        progress.progress(85, "Detecting anomalies...")
        anomalies = detect_anomalies(df)

        if enable_replies:
            progress.progress(92, "Drafting replies...")
            neg_df = df[df["Sentiment"] == "NEGATIVE"].head(5).copy()
            neg_df["Draft_Reply"] = neg_df["Review_Text"].apply(lambda t: draft_reply(t, groq_key))
            st.session_state["neg_df"] = neg_df

        progress.progress(100, "Done")
        st.session_state["df"] = df
        st.session_state["lda_topics"] = lda_topics
        st.session_state["anomalies"] = anomalies
        st.success("✅ Analysis complete")

# ---------------- RESULTS ----------------
if "df" in st.session_state:
    df = st.session_state["df"]
    lda_topics = st.session_state.get("lda_topics", [])
    anomalies = st.session_state.get("anomalies", [])

    sentiment_counts = df["Sentiment"].value_counts()
    total = len(df)
    pos = int(sentiment_counts.get("POSITIVE", 0))
    neg = int(sentiment_counts.get("NEGATIVE", 0))
    neu = int(sentiment_counts.get("NEUTRAL", 0))
    avg_score = round(float(df["Sentiment_Score"].mean()), 3)

    st.markdown("### 📊 Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [(c1, total, "Total Reviews"), (c2, pos, "Positive"), (c3, neg, "Negative"), (c4, neu, "Neutral"), (c5, avg_score, "Avg Confidence")]
    for col, val, label in cards:
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

    if anomalies:
        st.markdown("### 🚨 Anomaly Detection")
        for a in anomalies:
            st.markdown(f'<div class="anomaly-box">{a}</div>', unsafe_allow_html=True)

    st.markdown("### 📈 Sentiment Analysis")
    sentiment_df = sentiment_counts.reset_index()
    sentiment_df.columns = ["Sentiment", "Count"]
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(sentiment_df, names="Sentiment", values="Count", hole=0.4, title="Sentiment Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(sentiment_df, x="Sentiment", y="Count", title="Review Count by Sentiment")
        st.plotly_chart(fig_bar, use_container_width=True)

    fig_hist = px.histogram(df, x="Sentiment_Score", color="Sentiment", nbins=30, title="Confidence Score Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

    if "Emotion" in df.columns:
        st.markdown("### 🎭 Emotion Detection")
        emotion_df = df["Emotion"].value_counts().reset_index()
        emotion_df.columns = ["Emotion", "Count"]
        fig_emo = px.bar(emotion_df, x="Emotion", y="Count", title="Customer Emotion Breakdown")
        st.plotly_chart(fig_emo, use_container_width=True)

        heat_data = df.groupby(["Emotion", "Sentiment"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(heat_data, title="Emotion × Sentiment Heatmap", aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### ⚠️ Complaint Detection")
    complaint_df = detect_complaints(df)
    if complaint_df.empty:
        st.info("No common complaint keywords found.")
    else:
        col3, col4 = st.columns([1, 2])
        with col3:
            st.dataframe(complaint_df.head(15), use_container_width=True)
        with col4:
            fig_comp = px.bar(complaint_df.head(12), x="Complaint", y="Count", title="Top Customer Complaints")
            st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### 🧩 Topic Modeling")
    for t in lda_topics:
        words = t.get("words", [])
        st.markdown(f'<div class="insight-box"><b>Topic {t.get("id", 1)}:</b> {", ".join(words)}</div>', unsafe_allow_html=True)

    st.markdown("### ☁️ Word Cloud")
    col5, col6 = st.columns(2)
    for col, stype, title in [(col5, "POSITIVE", "Positive Word Cloud"), (col6, "NEGATIVE", "Negative Word Cloud")]:
        with col:
            sub = df[df["Sentiment"] == stype]["Cleaned_Review"].dropna().astype(str)
            text_data = " ".join(sub)
            if not text_data.strip():
                st.info(f"No {stype.lower()} reviews found.")
            else:
                wc = WordCloud(width=700, height=350, background_color="white", max_words=80).generate(text_data)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.imshow(wc)
                ax.axis("off")
                ax.set_title(title)
                st.pyplot(fig)

    if enable_semantic and "faiss_index" in st.session_state:
        st.markdown("### 🔍 Semantic Search")
        query = st.text_input("Search reviews by meaning", placeholder="Example: product quality is bad")
        if query:
            search_results = semantic_search(query, st.session_state["faiss_index"], st.session_state["review_texts"], k=6)
            if not search_results:
                st.info("No semantic search results found.")
            for i, r in enumerate(search_results, 1):
                st.markdown(f'<div class="insight-box"><b>#{i}</b> {r[:400]}</div>', unsafe_allow_html=True)

    if "neg_df" in st.session_state:
        st.markdown("### 💬 Auto-Drafted Replies")
        for _, row in st.session_state["neg_df"].iterrows():
            with st.expander(str(row["Review_Text"])[:100]):
                st.markdown(f'<div class="reply-box">{row["Draft_Reply"]}</div>', unsafe_allow_html=True)

    st.markdown("### 🔗 LangChain Business Summary")
    topics_flat = [w for t in lda_topics for w in t.get("words", [])[:3]]
    summary = langchain_summary(sentiment_counts.to_dict(), topics_flat)
    st.markdown(f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True)

    

    st.markdown("### 📋 Full Analyzed Dataset")
    display_cols = ["Review_Text", "Sentiment", "Sentiment_Score", "Recommendation"]
    if "Emotion" in df.columns:
        display_cols.insert(2, "Emotion")
    st.dataframe(df[display_cols].head(100), use_container_width=True)

    st.markdown("### 📤 Export Results")
    col7, col8 = st.columns(2)
    with col7:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="review_analysis_results.csv", mime="text/csv", use_container_width=True)
    with col8:
        try:
            pdf_bytes = generate_pdf_report(df, sentiment_counts.to_dict(), topics_flat, anomalies)
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="review_analysis_report.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.warning(f"PDF generation error: {e}")




else:
    st.info("Upload a CSV file to start analysis.")
