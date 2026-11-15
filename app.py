import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from keybert import KeyBERT
import PyPDF2
st.set_page_config(
    page_title="KeySense - Keyword & Keyphrase Extraction",
    page_icon="🔍",
    layout="wide"
)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_sm")
    kw_model = KeyBERT()
    return nlp, kw_model
with st.spinner("Loading NLP semantic models..."):
    nlp, kw_model = load_nlp_models()
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text
def extract_tfidf(text, top_n):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [(kw, round(float(score), 4)) for kw, score in sorted_scores]
def extract_rake(text, top_n):
    r = Rake()
    r.extract_keywords_from_text(text)
    scored = r.get_ranked_phrases_with_scores()[:top_n]
    return [(phrase, round(score, 2)) for score, phrase in scored]
def extract_keybert(text, top_n):
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3), 
        stop_words="english", 
        use_mmr=True, 
        diversity=0.5, 
        top_n=top_n
    )
    return [(kw, round(score, 4)) for kw, score in keywords]
st.title("🔍 KeySense: Advanced Keyword & Keyphrase Intelligence")
st.markdown("Extract core technical topics, phrases, and document metadata using Statistical, Co-occurrence, and Semantic Transformer algorithms[cite: 10].")
st.markdown("---")
with st.sidebar:
    st.header("⚙️ Configuration")
    top_n = st.slider("Top-N Keywords", min_value=3, max_value=20, value=8)
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Upload Document (.txt or .pdf)", type=["txt", "pdf"])
    st.info("Tip: Upload research papers or reports to test automated extraction.")
tab1, tab2 = st.tabs(["🚀 Single Algorithm Analysis", "📊 Multi-Model Comparison (Presentation View)"])
default_text = (
    "Transformer architectures have revolutionized natural language processing by replacing recurrent "
    "layers with self-attention mechanisms. Models such as BERT, BART, and T5 achieve state-of-the-art "
    "performance across abstractive text summarization, named entity recognition, and semantic keyphrase extraction."
)
with tab1:
    col1, col2 = st.columns([1.2, 0.8])   
    with col1:
        st.subheader("📄 Input Document")
        input_text = st.text_area("Paste text or load document:", value=default_text, height=260)
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                input_text = extract_text_from_pdf(uploaded_file)
            else:
                input_text = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded `{uploaded_file.name}` ({len(input_text.split())} words)")
    with col2:
        st.subheader("🛠️ Parameters")
        algorithm = st.selectbox(
            "Select Extraction Method", 
            ["KeyBERT (Semantic Embeddings)", "RAKE (Co-occurrence Graph)", "TF-IDF (Statistical Baseline)"]
        )
        st.write("")
        run_btn = st.button("Extract Keywords", type="primary")
    if run_btn:
        if not input_text.strip():
            st.warning("Please enter text or upload a document.")
        else:
            with st.spinner("Extracting keyphrases..."):
                if "KeyBERT" in algorithm:
                    results = extract_keybert(input_text, top_n)
                elif "RAKE" in algorithm:
                    results = extract_rake(input_text, top_n)
                else:
                    results = extract_tfidf(input_text, top_n)
            st.markdown("---")
            st.subheader(f"Results using {algorithm}")
            res_col1, res_col2 = st.columns(2)
            df = pd.DataFrame(results, columns=["Keyword / Keyphrase", "Score"])
            with res_col1:
                st.dataframe(df, use_container_width=True)
            with res_col2:
                st.bar_chart(df.set_index("Keyword / Keyphrase"))
with tab2:
    st.subheader("🔬 Compare All 3 Extraction Engines Side-by-Side")
    st.markdown("Great for demonstrating different NLP methodologies during your project presentation.")
    comp_text = st.text_area("Enter text for comparative analysis:", value=default_text, height=150)
    if st.button("Run Comparative Analysis", type="primary"):
        if not comp_text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Running TF-IDF, RAKE, and KeyBERT simultaneously..."):
                tfidf_res = extract_tfidf(comp_text, 5)
                rake_res = extract_rake(comp_text, 5)
                kbert_res = extract_keybert(comp_text, 5)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("#### TF-IDF (Statistical)")
                st.dataframe(pd.DataFrame(tfidf_res, columns=["Term", "Score"]), use_container_width=True)
            with col_b:
                st.markdown("#### RAKE (Graph)")
                st.dataframe(pd.DataFrame(rake_res, columns=["Term", "Score"]), use_container_width=True)                
            with col_c:
                st.markdown("#### KeyBERT (Semantic)")
                st.dataframe(pd.DataFrame(kbert_res, columns=["Term", "Score"]), use_container_width=True)
