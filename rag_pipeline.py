import time
from retrieval import Retriever
from reranker import Reranker
from prompt_builder import PromptBuilder, format_timestamp
from llm import LLM
from logger import SearchLogger
from response_parser import ResponseParser
from query_normalizer import normalize_query, enhance_query
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
        t0 = time.perf_counter()

        clean_query = normalize_query(query)
        enhanced_query = enhance_query(query)

        candidates = self.retriever.search(
            query=clean_query,
            top_k=retrieval_k,
            hybrid=hybrid,
        )
        t1 = time.perf_counter()

        contexts = self.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_k=rerank_k,
        )
        t2 = time.perf_counter()

        system_prompt, user_prompt = PromptBuilder.build(
            query=enhanced_query,
            contexts=contexts,
        )
        t3 = time.perf_counter()

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

        parsed = ResponseParser.parse(raw_response)[:1]

        for i, rec in enumerate(parsed):
            if i < len(contexts):
                rec["text_snippet"] = contexts[i].get("text", "")
                rec["rerank_score"] = contexts[i].get("rerank_score", 0.0)
                rec["dense_score"] = contexts[i].get("dense_score", contexts[i].get("score", 0.0))

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
            "enhanced_query": enhanced_query,
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
        result = self.search_details(
            query=query,
            retrieval_k=retrieval_k,
            rerank_k=rerank_k,
        )
        return result["response"]