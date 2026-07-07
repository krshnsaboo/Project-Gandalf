import time

import streamlit as st

from rag_pipeline import RAGPipeline



st.set_page_config(
    page_title="Project Gandalf - Striver A2Z DSA Navigator",
    page_icon="🎯",
    layout="centered",
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
        index_path="lecture_embeddings/faiss_index.bin",
    )


pipeline = load_pipeline()


st.title("🎯 Striver A2Z DSA Navigator")
st.write("Find exactly where Striver explains a DSA topic.")

with st.form("search_form"):

    query = st.text_input(
        "Search",
        placeholder="Enter a DSA topic or problem (e.g., '4Sum Optimal Solution')",
    )

    submitted = st.form_submit_button(
        "Submit",
        use_container_width=True,
    )



def parse_response(response: str):
    recommendations = []

    blocks = response.strip().split("\n\n")

    for block in blocks:

        title = ""
        timestamp = ""
        url = ""
        description = []

        for line in block.splitlines():

            line = line.strip()

            if not line:
                continue

            if line.lower().startswith("timestamp"):
                timestamp = line.split(":", 1)[1].strip()

            elif line.startswith("http"):
                url = line

            elif title == "":
                title = line

            else:
                description.append(line)

        recommendations.append(
            {
                "title": title,
                "timestamp": timestamp,
                "url": url,
                "description": " ".join(description),
            }
        )

    return recommendations


if submitted:

    if not query.strip():
        st.warning("Please enter a query.")

    else:

        start = time.perf_counter()

        with st.spinner("Searching lectures..."):

            print("=" * 50)
            print("Search button clicked")
            print(time.strftime("%H:%M:%S"))
            response = pipeline.search(query)
            print("Search finished")
            print(time.strftime("%H:%M:%S"))

        latency = time.perf_counter() - start

        # st.success(f"Completed in {latency:.2f} seconds")

        recommendations = parse_response(response)

        if not recommendations:
            st.error("No recommendations found.")

        for rec in recommendations:

            with st.container(border=True):

                st.subheader(rec["title"])

                if rec["timestamp"]:
                    st.caption(f"🕒 {rec['timestamp']}")

                if rec["description"]:
                    st.write(rec["description"])

                if rec["url"]:
                    st.link_button(
                        "▶ Watch on YouTube",
                        rec["url"],
                        use_container_width=True,
                    )

st.markdown("""
<style>
.custom-footer{
    position:fixed;
    left:0;
    bottom:10px;
    width:100%;
    text-align:center;
    color:#888888;
    font-size:14px;
}
      </style>      
<div class="custom-footer">
Built with ❤️ for learners
</div>
""", unsafe_allow_html=True)
