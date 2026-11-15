import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
st.set_page_config(
    page_title="BriefAI - Abstractive Summarizer",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { 
        font-size: 2.5rem; 
        color: #60A5FA; 
        font-weight: 700; 
        margin-bottom: 0.2rem;
    }
    .sub-title { 
        font-size: 1.1rem; 
        color: #94A3B8; 
        margin-bottom: 2rem; 
    }
    .summary-card { 
        background-color: #1E293B; 
        color: #F8FAFC; 
        padding: 1.5rem; 
        border-radius: 0.75rem; 
        border: 1px solid #334155;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        font-size: 1.05rem;
        line-height: 1.65;
        margin-bottom: 1.2rem;
    }
    .summary-label {
        color: #60A5FA;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model
with st.spinner("Loading Transformer Model (`distilbart-cnn-12-6`). Please wait..."):
    tokenizer, model = load_model()
st.markdown('<p class="main-title">📝 BriefAI: Abstractive Text Summarization</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Project 13: Generate concise, coherent, and context-aware summaries using Sequence-to-Sequence Transformer models.</p>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.markdown("### 📥 Input Document")
    text_input = st.text_area(
        "Paste your article, report, or long text here:",
        height=300,
        placeholder="Paste text here to summarize..."
    )
    st.markdown("#### Summary Parameters")
    max_len = st.slider("Maximum Summary Length", 50, 200, 130)
    min_len = st.slider("Minimum Summary Length", 20, 80, 30)
    run_btn = st.button("Generate Summary", type="primary", use_container_width=True)
with col2:
    st.markdown("### 📊 Generated Abstractive Summary")
    if run_btn:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to summarize.")
        else:
            with st.spinner("Synthesizing and generating summary..."):
                try:
                    inputs = tokenizer(text_input, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )
                    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    st.success("Summary Generated Successfully!")
                    st.markdown(f"""
                        <div class="summary-card">
                            <span class="summary-label">✨ Abstractive Summary</span>
                            {summary_text}
                        </div>
                    """, unsafe_allow_html=True)
                    word_count_orig = len(text_input.split())
                    word_count_summ = len(summary_text.split())
                    reduction = ((word_count_orig - word_count_summ) / word_count_orig) * 100                    
                    st.metric(
                        label="Compression Efficiency", 
                        value=f"{reduction:.1f}% Reduction", 
                        delta=f"{word_count_orig} words ➔ {word_count_summ} words"
                    )
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")
    else:
        st.info("👈 Enter your text on the left and click **Generate Summary** to see the results here.")