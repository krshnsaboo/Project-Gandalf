import time
from retrieval import Retriever
from reranker import Reranker
from prompt_builder import PromptBuilder, format_timestamp
from llm import LLM
from logger import SearchLogger
from response_parser import ResponseParser
from query_normalizer import normalize_query
from config import FAISS_TOP_K, RERANK_TOP_K, USE_HYBRID_SEARCH


class RAGPipeline:

    def __init__(
        self,
        embeddings_path: str = "lecture_embeddings/all_lecture_embeddings.pkl",
        index_path: str = "lecture_embeddings/faiss_index.bin",
    ):
        self.retriever = Retriever(
            embeddings_path=embeddings_path,
            index_path=index_path,
        )
        self.reranker = Reranker()
        self.llm = LLM()
        self.logger = SearchLogger()

    def search_details(
        self,
        query: str,
        retrieval_k: int = FAISS_TOP_K,
        rerank_k: int = RERANK_TOP_K,
        hybrid: bool = USE_HYBRID_SEARCH,
    ) -> dict:
        """
        Executes full RAG search pipeline and returns rich structured results,
        including parsed recommendations, raw contexts, timings, and normalized query.
        """
        t0 = time.perf_counter()

        # Step 0: Normalize Query (e.g., LeetCode URLs)
        clean_query = normalize_query(query)

        # Step 1: Retrieve (Dense or Hybrid BM25+FAISS)
        candidates = self.retriever.search(
            query=clean_query,
            top_k=retrieval_k,
            hybrid=hybrid,
        )
        t1 = time.perf_counter()

        # Step 2: Rerank
        contexts = self.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_k=rerank_k,
        )
        t2 = time.perf_counter()

        # Step 3: Build Prompt
        system_prompt, user_prompt = PromptBuilder.build(
            query=clean_query,
            contexts=contexts,
        )
        t3 = time.perf_counter()

        # Step 4: LLM Generation (with graceful fallback if LLM fails)
        try:
            raw_response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"[Warning] LLM generation failed ({e}). Falling back to top reranked context.")
            if contexts:
                top = contexts[0]
                raw_response = (
                    f"Lecture: {top.get('lecture_title', 'DSA Lecture')}\n"
                    f"Timestamp: {format_timestamp(top.get('start', 0))}\n"
                    f"Watch: {top.get('timestamp_url', '')}"
                )
            else:
                raw_response = "No matching lecture found."

        t4 = time.perf_counter()

        # Parse recommendations
        parsed = ResponseParser.parse(raw_response)

        # Attach snippet context to parsed results if matching
        for i, rec in enumerate(parsed):
            if i < len(contexts):
                rec["text_snippet"] = contexts[i].get("text", "")
                rec["rerank_score"] = contexts[i].get("rerank_score", 0.0)
                rec["dense_score"] = contexts[i].get("dense_score", contexts[i].get("score", 0.0))

        # Step 5: Logging
        retrieval_time = t1 - t0
        rerank_time = t2 - t1
        llm_time = t4 - t3
        total_time = t4 - t0

        if contexts:
            self.logger.log(
                query=query,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                llm_time=llm_time,
                total_time=total_time,
                selected_result=contexts[0],
            )

        print("\n========== TIMINGS ==========")
        print(f"Retrieval      : {retrieval_time:.3f} sec")
        print(f"Reranking      : {rerank_time:.3f} sec")
        print(f"Prompt Builder : {t3 - t2:.3f} sec")
        print(f"LLM            : {llm_time:.3f} sec")
        print(f"Total          : {total_time:.3f} sec")
        print("=============================\n")

        return {
            "query": query,
            "normalized_query": clean_query,
            "response": raw_response,
            "recommendations": parsed,
            "contexts": contexts,
            "timings": {
                "retrieval": retrieval_time,
                "rerank": rerank_time,
                "llm": llm_time,
                "total": total_time,
            },
        }

    def search(
        self,
        query: str,
        retrieval_k: int = FAISS_TOP_K,
        rerank_k: int = RERANK_TOP_K,
    ) -> str:
        """
        Backwards-compatible search method returning the raw response string.
        """
        result = self.search_details(
            query=query,
            retrieval_k=retrieval_k,
            rerank_k=rerank_k,
        )
        return result["response"]