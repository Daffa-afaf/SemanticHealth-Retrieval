"""
Streamlit app for health IR: TF-IDF, BM25, and Semantic Search.
Loads artifacts from ./artifacts. Simple, consistent UI.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pickle
from pathlib import Path
import threading

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.preprocessing import preprocess_query

# Paths
ARTIFACTS_DIR = Path("./artifacts")
TFIDF_DIR = ARTIFACTS_DIR / "tfidf"
BM25_DIR = ARTIFACTS_DIR / "bm25"
SEMANTIC_DIR = ARTIFACTS_DIR / "semantic"

# UI styling
PRIMARY = "#0F766E"  # teal
ACCENT = "#F97316"   # orange
CARD_BG = "#0B1C26"
TEXT_MUTED = "#8CA3AF"
PAPER = "#0D1620"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {{
            --primary: {PRIMARY};
            --accent: {ACCENT};
            --paper: {PAPER};
        }}

        /* App background */
        .stApp {{
            background: radial-gradient(circle at 20% 20%, rgba(15,118,110,0.08), transparent 28%),
                        radial-gradient(circle at 80% 0%, rgba(249,115,22,0.08), transparent 28%),
                        #050b12;
            font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif;
            color: #E5E7EB;
        }}

        /* Base layout */
        .block-container {{padding: 1.25rem 2.25rem 2.5rem 2.25rem;}}
        /* Header */
        .hero {{
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 1.2rem;
            align-items: center;
            padding: 1rem 1.25rem;
            border: 1px solid #1F2937;
            border-radius: 16px;
            background: linear-gradient(120deg, rgba(15,118,110,0.12), rgba(11,28,38,0.85));
            box-shadow: 0 20px 50px rgba(0,0,0,0.28);
        }}
        .hero h1 {{margin: 0 0 0.2rem 0; color: #F9FAFB; font-size: 2rem; letter-spacing: -0.02em;}}
        .hero .eyebrow {{letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.78rem; color: {TEXT_MUTED};}}
        .hero .lede {{color: #cbd5e1; margin: 0.35rem 0 0.75rem 0;}}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            background: #0b1722;
            border: 1px solid #1f2f3f;
            color: #e2e8f0;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .pill .dot {{width: 8px; height: 8px; border-radius: 50%; background: var(--primary);}}
        .pill + .pill {{margin-left: 0.5rem;}}

        .stat-grid {{display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 0.8rem;}}
        .stat-card {{
            background: #0b121a;
            border: 1px solid #1f2a38;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }}
        .stat-label {{color: {TEXT_MUTED}; font-size: 0.85rem; margin-bottom: 0.15rem; display:block;}}
        .stat-value {{color: #F9FAFB; font-size: 1.35rem; font-weight: 700;}}
        .stat-sub {{color: {TEXT_MUTED}; font-size: 0.8rem;}}

        .metric-pill {{
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: {PRIMARY}1A;
            color: {PRIMARY};
            font-weight: 600;
            margin-right: 0.4rem;
            font-size: 0.9rem;
        }}
        /* Cards */
        .result-card {{
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            background: {CARD_BG};
            color: #E5E7EB;
            border: 1px solid #1F2937;
            box-shadow: 0 6px 16px rgba(0,0,0,0.25);
        }}
        .result-card h4 {{margin: 0 0 0.35rem 0;}}
        .result-meta {{color: {TEXT_MUTED}; font-size: 0.9rem;}}
        .score-chip {{
            background: {ACCENT}1A;
            color: {ACCENT};
            padding: 0.25rem 0.55rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
        }}
        .control-card {{
            border-radius: 14px;
            padding: 1rem 1.2rem;
            background: rgba(11,22,32,0.88);
            border: 1px solid #1F2937;
            box-shadow: 0 10px 28px rgba(0,0,0,0.3);
        }}
        /* Comparison specific */
        .comparison-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 1.5rem 0 0.8rem 0;
            padding: 0.75rem 1rem;
            background: rgba(15,118,110,0.08);
            border-left: 3px solid var(--primary);
            border-radius: 8px;
        }}
        .comparison-header h3 {{
            margin: 0;
            color: #f9fafb;
            font-size: 1.15rem;
            font-weight: 600;
        }}
        .model-badge {{
            padding: 0.35rem 0.7rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .model-badge.tfidf {{background: rgba(59,130,246,0.2); color: #60a5fa;}}
        .model-badge.bm25 {{background: rgba(249,115,22,0.2); color: #fb923c;}}
        .model-badge.semantic {{background: rgba(168,85,247,0.2); color: #c084fc;}}
        .overlap-badge {{
            background: rgba(34,197,94,0.2);
            color: #4ade80;
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}
        /* Sidebar hidden for full-bleed layout */
        section[data-testid="stSidebar"] {{display: none;}}
        /* Stretch main when sidebar is removed */
        .block-container {{max-width: 1200px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load TF-IDF artifacts if available
@st.cache_resource(show_spinner=False)
def load_tfidf():
    if not TFIDF_DIR.exists():
        return None
    try:
        with open(TFIDF_DIR / "vectorizer.pkl", "rb") as f:
            vectorizer: TfidfVectorizer = pickle.load(f)
        tfidf_matrix = pickle.load(open(TFIDF_DIR / "tfidf_matrix.pkl", "rb"))
        df_meta = pd.read_pickle(TFIDF_DIR / "corpus_meta.pkl")
        return vectorizer, tfidf_matrix, df_meta
    except Exception as e:
        st.warning(f"Failed to load TF-IDF artifacts: {e}")
        return None


# Load BM25 artifacts
@st.cache_resource(show_spinner=False)
def load_bm25():
    if not BM25_DIR.exists():
        return None
    try:
        with open(BM25_DIR / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        df_meta = pd.read_pickle(BM25_DIR / "corpus_meta.pkl")
        with open(BM25_DIR / "stopwords.json", "r") as f:
            stopwords = json.load(f)
        with open(BM25_DIR / "bm25_config.json", "r") as f:
            cfg = json.load(f)
        return bm25, df_meta, stopwords, cfg
    except Exception as e:
        st.warning(f"Failed to load BM25 artifacts: {e}")
        return None


# Load Semantic Search artifacts
@st.cache_resource(show_spinner=False)
def load_semantic():
    """Load semantic search model and embeddings"""
    if not SEMANTIC_DIR.exists():
        return None
    try:
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        
        # Load model
        model_path = SEMANTIC_DIR / "model"
        if model_path.exists():
            # Force CPU for broader compatibility and faster init on servers
            model = SentenceTransformer(str(model_path), device="cpu")
        else:
            # Fallback to default model - disable for now if PyTorch fails
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device="cpu")
        
        # Load precomputed embeddings and metadata
        # Memory-map to avoid loading full array into RAM at startup
        embeddings = np.load(SEMANTIC_DIR / "embeddings.npy", mmap_mode='r')
        df_meta = pd.read_pickle(SEMANTIC_DIR / "corpus_meta.pkl")
        
        return model, embeddings, df_meta
    except Exception as e:
        # Silently fail - Semantic Search will be disabled
        return None


def is_semantic_available() -> bool:
    """Check presence of semantic artifacts without loading heavy resources."""
    if not SEMANTIC_DIR.exists():
        return False
    try:
        required = [SEMANTIC_DIR / "embeddings.npy", SEMANTIC_DIR / "corpus_meta.pkl"]
        return all(p.exists() for p in required)
    except Exception:
        return False


def read_docs_count() -> int:
    """Lightweight doc count by reading any available corpus_meta.pkl."""
    candidates = [TFIDF_DIR / "corpus_meta.pkl", BM25_DIR / "corpus_meta.pkl", SEMANTIC_DIR / "corpus_meta.pkl"]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_pickle(p)
                return int(df.shape[0])
            except Exception:
                continue
    return 0


def search_tfidf(query: str, vectorizer, tfidf_matrix, df_meta, top_k=10):
    processed = preprocess_query(query)
    if not processed:
        return pd.DataFrame()
    qv = vectorizer.transform([processed])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    results = df_meta.iloc[top_idx].copy()
    results["score"] = sims[top_idx]
    results["rank"] = range(1, len(results) + 1)
    return results


def search_bm25(query: str, bm25, df_meta, top_k=10):
    processed = preprocess_query(query)
    if not processed:
        return pd.DataFrame()
    tokens = processed.split()
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[-top_k:][::-1]
    results = df_meta.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    results["rank"] = range(1, len(results) + 1)
    return results


def search_semantic(query: str, model, embeddings, df_meta, top_k=10):
    """Semantic search using sentence embeddings"""
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, embeddings).flatten()
        
        # Get top-k results
        top_idx = similarities.argsort()[-top_k:][::-1]
        results = df_meta.iloc[top_idx].copy()
        results["score"] = similarities[top_idx]
        results["rank"] = range(1, len(results) + 1)
        return results
    except Exception as e:
        st.error(f"Semantic search error: {e}")
        return pd.DataFrame()


def render_results(results: pd.DataFrame, model_name: str = "", highlight_ids: set = None):
    """Render results with optional highlighting for overlapping documents"""
    for _, row in results.iterrows():
        is_overlap = highlight_ids and row.name in highlight_ids
        card_style = "border: 2px solid #4ade80;" if is_overlap else ""
        overlap_marker = "✓ " if is_overlap else ""
        
        st.markdown(
            f"""
            <div class="result-card" style="{card_style}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h4>{overlap_marker}{row['rank']}. {row['title']}</h4>
                    <span class="score-chip">{row['score']:.4f}</span>
                </div>
                <div class="result-meta">Topic: {row.get('topic_set','-')} | Year: {row.get('year','-')}</div>
                <div style="margin-top:0.4rem; line-height:1.5;">{row.get('answer','')[:320]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_comparison_header(model_name: str, overlap_count: int = None):
    """Render header for comparison columns"""
    model_lower = model_name.lower()
    if model_lower == "tfidf":
        badge_class = "tfidf"
    elif model_lower == "bm25":
        badge_class = "bm25"
    else:
        badge_class = "semantic"
    
    overlap_html = f'<span class="overlap-badge">{overlap_count} overlap</span>' if overlap_count is not None else ""
    
    st.markdown(
        f"""
        <div class="comparison-header">
            <h3>{model_name.upper()} Results</h3>
            <div>
                <span class="model-badge {badge_class}">{model_name}</span>
                {overlap_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Health IR - TF-IDF / BM25 / Semantic",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    tfidf_bundle = load_tfidf()
    bm25_bundle = load_bm25()
    # Lazy load semantic to avoid slow cold start
    semantic_bundle = None

    tfidf_docs = tfidf_bundle[1].shape[0] if tfidf_bundle else 0
    bm25_docs = bm25_bundle[1].shape[0] if bm25_bundle else 0
    semantic_available = is_semantic_available()

    models = []
    if tfidf_bundle:
        models.append("tfidf")
    if bm25_bundle:
        models.append("bm25")
    if semantic_available:
        models.append("semantic")

    if not models:
        st.error("No artifacts found. Please place TF-IDF/BM25/Semantic artifacts in ./artifacts.")
        return

    # Ambil total dokumen (asumsi semua model punya jumlah sama)
    total_docs = tfidf_docs or bm25_docs or read_docs_count()
    
    st.markdown(
        f"""
        <div class="hero">
            <div>
                <div class="eyebrow">Information Retrieval</div>
                <h1>Health IR System</h1>
                <p class="lede">Bandingkan performa TF-IDF, BM25, dan Semantic Search untuk menjawah pertanyaan kesehatan dalam bahasa Indonesia.</p>
                <div class="pill-group">
                    <span class="pill"><span class="dot"></span>{len(models)} Models Ready</span>
                    <span class="pill"><span class="dot" style="background: var(--accent);"></span>{total_docs:,} Dokumen</span>
                </div>
            </div>
            <div class="stat-grid">
                <div class="stat-card">
                    <span class="stat-label">Model Tersedia</span>
                    <span class="stat-value">{len(models)}</span>
                    <span class="stat-sub">{', '.join([m.upper() for m in models])}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Total Korpus</span>
                    <span class="stat-value">{total_docs:,}</span>
                    <span class="stat-sub">Indo Health Consultation</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin: 1.1rem 0 0.6rem 0; color:#cbd5e1; font-weight:600;'>Jalankan pencarian</div>", unsafe_allow_html=True)
    left, right = st.columns([3, 1.35])
    with left:
        st.markdown("<div class='control-card'>", unsafe_allow_html=True)
        with st.form("search_form"):
            # Add comparison mode toggle
            comparison_mode = st.checkbox(
                "Mode Perbandingan (Side-by-Side Comparison)", 
                value=False,
                help="Tampilkan hasil dari semua model secara berdampingan"
            )
            
            if not comparison_mode:
                model_choice = st.selectbox(
                    "Pilih model", 
                    models, 
                    index=0,
                    format_func=lambda x: x.upper()
                )
            else:
                model_choice = None  # Will use all available models
                
            query = st.text_area("Masukkan pertanyaan/gejala", "anak demam tinggi dan batuk", height=120)
            top_k = st.slider("Top K", 3, 20, 10)
            search_btn = st.form_submit_button("🔍 Run Search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='control-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-pill'>Status Model</div>", unsafe_allow_html=True)
        
        model_status = []
        if tfidf_bundle:
            model_status.append("✅ TF-IDF")
        else:
            model_status.append("❌ TF-IDF")
            
        if bm25_bundle:
            model_status.append("✅ BM25")
        else:
            model_status.append("❌ BM25")
            
        if semantic_available:
            model_status.append("✅ Semantic")
        else:
            model_status.append("❌ Semantic")
        
        for status in model_status:
            st.markdown(f"<div style='padding: 0.3rem 0; color: #e5e7eb;'>{status}</div>", unsafe_allow_html=True)
        
        total_docs = tfidf_docs or bm25_docs or read_docs_count()
        st.markdown(f"<div style='margin-top: 1rem; padding: 0.5rem; background: rgba(15,118,110,0.1); border-radius: 8px; text-align: center;'><strong style='color: #4ade80;'>{total_docs:,}</strong><br/><span style='font-size: 0.85rem; color: #94a3b8;'>Total Dokumen</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if search_btn:
        if not comparison_mode:
            # Single model mode
            if model_choice == "tfidf":
                vectorizer, tfidf_matrix, df_meta = tfidf_bundle
                results = search_tfidf(query, vectorizer, tfidf_matrix, df_meta, top_k)
            elif model_choice == "bm25":
                bm25, df_meta, stopwords, cfg = bm25_bundle
                results = search_bm25(query, bm25, df_meta, top_k)
            else:  # semantic
                # Load semantic on demand
                semantic_bundle = load_semantic()
                if not semantic_bundle:
                    st.warning("Semantic artifacts tidak tersedia atau gagal diload.")
                    results = pd.DataFrame()
                else:
                    model, embeddings, df_meta = semantic_bundle
                results = search_semantic(query, model, embeddings, df_meta, top_k)

            if results.empty:
                st.info("Tidak ada hasil.")
            else:
                render_results(results)
        else:
            # Comparison mode: run all available models
            results_dict = {}
            
            if tfidf_bundle:
                vectorizer, tfidf_matrix, df_meta_tfidf = tfidf_bundle
                results_dict["TF-IDF"] = search_tfidf(query, vectorizer, tfidf_matrix, df_meta_tfidf, top_k)
            
            if bm25_bundle:
                bm25, df_meta_bm25, stopwords, cfg = bm25_bundle
                results_dict["BM25"] = search_bm25(query, bm25, df_meta_bm25, top_k)
            
            if semantic_available:
                # Ensure semantic is loaded when needed
                semantic_bundle = semantic_bundle or load_semantic()
                if semantic_bundle:
                    model, embeddings, df_meta_sem = semantic_bundle
                    results_dict["Semantic"] = search_semantic(query, model, embeddings, df_meta_sem, top_k)
                else:
                    st.info("Semantic belum siap atau gagal diload.")
            
            if not results_dict or all(r.empty for r in results_dict.values()):
                st.info("Tidak ada hasil dari model manapun.")
            else:
                # Calculate overlap across all models
                all_ids = [set(r.index) for r in results_dict.values() if not r.empty]
                if len(all_ids) > 1:
                    overlap_ids = set.intersection(*all_ids)
                    overlap_count = len(overlap_ids)
                else:
                    overlap_ids = set()
                    overlap_count = 0
                
                # Display summary
                st.markdown(
                    f"""
                    <div style='background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); 
                                border-radius: 10px; padding: 0.8rem 1rem; margin: 1rem 0;'>
                        <strong style='color: #4ade80;'>📊 Perbandingan {len(results_dict)} Model:</strong>
                        <span style='color: #e5e7eb; margin-left: 1rem;'>
                            {overlap_count} dokumen muncul di semua model ({overlap_count/top_k*100:.1f}% overlap)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Visualisasi Score Distribution
                st.markdown("<h3 style='color: #e5e7eb; margin-top: 2rem;'>📈 Distribusi Score</h3>", unsafe_allow_html=True)
                chart_cols = st.columns(len(results_dict))
                
                for idx, (model_name, results) in enumerate(results_dict.items()):
                    with chart_cols[idx]:
                        if not results.empty:
                            # Create simple bar chart data
                            chart_data = results[['rank', 'score']].head(10)
                            st.bar_chart(chart_data.set_index('rank')['score'], height=200)
                            st.caption(f"{model_name} - Avg: {results['score'].mean():.4f}")
                
                st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                
                # Create columns based on number of models
                cols = st.columns(len(results_dict))
                
                for idx, (model_name, results) in enumerate(results_dict.items()):
                    with cols[idx]:
                        render_comparison_header(model_name, overlap_count)
                        if results.empty:
                            st.info(f"Tidak ada hasil {model_name}")
                        else:
                            render_results(results, model_name.lower(), overlap_ids)

    # Sidebar intentionally removed for full-screen layout
    # Background warmup for semantic (non-blocking) to improve first query performance
    try:
        if semantic_available and not st.session_state.get("semantic_warmup_started"):
            st.session_state["semantic_warmup_started"] = True
            threading.Thread(target=load_semantic, daemon=True).start()
    except Exception:
        pass


if __name__ == "__main__":
    main()
