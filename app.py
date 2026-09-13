import time
import streamlit as st
from rag_pipeline import RAGPipeline
from config import OPENAI_MODEL, RERANKER_MODEL

st.set_page_config(
    page_title="Project Gandalf - Striver A2Z DSA Navigator",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
        index_path="lecture_embeddings/faiss_index.bin",
    )


st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-med {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-mod {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff10;
        backdrop-filter: blur(8px);
        text-align: center;
        color: #888888;
        font-size: 13px;
        padding: 6px 0;
        border-top: 1px solid #33333320;
    }
</style>
""", unsafe_allow_html=True)

with st.spinner("Initializing Project Gandalf (BGE-M3 + FAISS + BM25)..."):
    pipeline = load_pipeline()

st.markdown("""
<div class="main-header">
    <h1>🎯 Striver A2Z DSA Lecture Navigator</h1>
    <p style="color: #666; font-size: 1.1rem;">
        Pinpoint the exact timestamp & video where Striver explains any DSA concept across 315 lectures.
    </p>
</div>
""", unsafe_allow_html=True)

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""


def set_query(query_text):
    st.session_state["query_input"] = query_text
    st.session_state["auto_submit"] = True


st.markdown("##### 💡 Try Popular Searches:")
chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)

with chip_col1:
    if st.button("🔢 4Sum Optimal Solution", use_container_width=True):
        set_query("4Sum Optimal Solution")
        st.rerun()

with chip_col2:
    if st.button("🛣️ Dijkstra Algorithm with PQ", use_container_width=True):
        set_query("Dijkstra Algorithm with Priority Queue")
        st.rerun()

with chip_col3:
    if st.button("💾 LRU Cache Implementation", use_container_width=True):
        set_query("LRU Cache Implementation")
        st.rerun()

with chip_col4:
    if st.button("🔗 Add Two Numbers (LeetCode)", use_container_width=True):
        set_query("https://leetcode.com/problems/add-two-numbers/")
        st.rerun()

st.write("")

with st.form("search_form"):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Search Query or LeetCode URL:",
            value=st.session_state["query_input"],
            placeholder="e.g. 'Trapping Rain Water Optimal' or https://leetcode.com/problems/rotate-image/",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)

auto_submitted = st.session_state.pop("auto_submit", False)
if submitted or auto_submitted:
    if not query.strip():
        st.warning("⚠️ Please enter a search query or LeetCode URL.")
    else:
        try:
            with st.spinner("Analyzing 315 lectures with Hybrid Search & Reranking..."):
                result = pipeline.search_details(query)

            timings = result["timings"]
            recommendations = result["recommendations"]
            norm_query = result.get("normalized_query", query)

            if norm_query != query:
                st.info(f"🔗 Detected LeetCode URL! Searching for: **{norm_query}**")

            st.caption(
                f"⚡ Retrieved in **{timings['total']:.2f}s** "
                f"(Retrieval: {timings['retrieval']:.2f}s | "
                f"Reranking: {timings['rerank']:.2f}s | "
                f"LLM: {timings['llm']:.2f}s)"
            )

            if not recommendations:
                st.error("No relevant lecture segments found for your query. Try rephrasing.")
            else:
                st.write(f"### 📍 Recommended Lecture Timestamps ({len(recommendations)})")

                for idx, rec in enumerate(recommendations, start=1):
                    with st.container(border=True):
                        col_info, col_badge = st.columns([4, 1])

                        with col_info:
                            st.subheader(f"📺 {rec['title']}")
                            if rec.get("timestamp"):
                                st.markdown(f"**🕒 Jump to:** `{rec['timestamp']}`")

                        with col_badge:
                            rerank_sc = rec.get("rerank_score", 0.0)
                            if rerank_sc >= 2.0:
                                st.markdown('<span class="badge-high">🟢 High Relevance</span>', unsafe_allow_html=True)
                            elif rerank_sc >= 0.0:
                                st.markdown('<span class="badge-med">🟡 Moderate Match</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-mod">⚪ Context Match</span>', unsafe_allow_html=True)

                        if rec.get("url"):
                            st.video(rec["url"])
                            st.link_button(
                                "▶ Open in YouTube (New Tab)",
                                rec["url"],
                                use_container_width=True,
                            )

                        snippet = rec.get("text_snippet")
                        if snippet:
                            with st.expander("📝 View Transcript Snippet from Lecture"):
                                st.markdown(f"> *\"{snippet}\"*")

                        if rec.get("description") and not rec.get("url"):
                            st.write(rec["description"])

        except Exception as e:
            st.error(f"❌ An error occurred while searching: {e}")
            st.info("Tip: If you're missing an OpenAI API Key, configure `OPENAI_API_KEY` in your `.env` file or Streamlit Secrets.")

st.markdown("""
<div class="custom-footer">
    Project Gandalf • Built for Striver's A2Z DSA Course learners
</div>
""", unsafe_allow_html=True)