import time
import streamlit as st
from rag_pipeline import RAGPipeline
from config import OPENAI_MODEL, RERANKER_MODEL

st.set_page_config(
    page_title="Striver A2Z DSA Navigator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
        index_path="lecture_embeddings/faiss_index.bin",
    )


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 3.5rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

.header-container {
    text-align: center;
    margin-bottom: 2rem;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 75%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.5;
    max-width: 680px;
    margin: 0 auto;
}

.video-card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0.85rem 0 0.85rem 0;
    line-height: 1.4;
}

div[data-testid="stVideo"] {
    border-radius: 12px;
    overflow: hidden;
}

.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0f172a80;
    backdrop-filter: blur(12px);
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 8px 0;
    border-top: 1px solid #33415540;
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)

with st.spinner("Initializing Project Gandalf..."):
    pipeline = load_pipeline()

st.markdown("""
<div class="header-container">
    <div class="main-title">🎯 Striver A2Z DSA Navigator</div>
    <div class="main-subtitle">
        Instantly pinpoint the exact lecture timestamp where Striver explained that concept/question.
    </div>
</div>
""", unsafe_allow_html=True)

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""


def set_query(query_text):
    st.session_state["query_input"] = query_text
    st.session_state["auto_submit"] = True


_, col_search, _ = st.columns([1, 6, 1])

with col_search:
    with st.form("search_form"):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "Search Query or LeetCode URL:",
                value=st.session_state["query_input"],
                placeholder="Search DSA concept (e.g. 'Trapping Rain Water Optimal') or paste LeetCode URL...",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)

    with chip_col1:
        if st.button("🔢 4Sum Optimal", use_container_width=True):
            set_query("4Sum Optimal Solution")
            st.rerun()

    with chip_col2:
        if st.button("🛣️ Dijkstra with PQ", use_container_width=True):
            set_query("Dijkstra Algorithm with Priority Queue")
            st.rerun()

    with chip_col3:
        if st.button("💾 LRU Cache", use_container_width=True):
            set_query("LRU Cache Implementation")
            st.rerun()

    with chip_col4:
        if st.button("🔗 Add Two Numbers", use_container_width=True):
            set_query("https://leetcode.com/problems/add-two-numbers/")
            st.rerun()

auto_submitted = st.session_state.pop("auto_submit", False)
if submitted or auto_submitted:
    if not query.strip():
        st.warning("Please enter a search query or LeetCode URL.")
    else:
        try:
            with st.spinner("Finding matching lecture video..."):
                result = pipeline.search_details(query)

            recommendations = result.get("recommendations", [])[:1]

            if not recommendations:
                st.info("No matching lecture found. Try rephrasing your search query.")
            else:
                rec = recommendations[0]
                with col_search:
                    st.write("")
                    with st.container(border=True):
                        if rec.get("url"):
                            st.video(rec["url"])
                        st.markdown(f'<div class="video-card-title">{rec["title"]}</div>', unsafe_allow_html=True)
                        if rec.get("url"):
                            st.link_button("Open Video", rec["url"], use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while searching: {e}")

st.markdown("""
<div class="custom-footer">
    Project Gandalf • Built for Striver's A2Z DSA Course learners
</div>
""", unsafe_allow_html=True)