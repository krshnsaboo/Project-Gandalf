from retrieval import Retriever
from reranker import Reranker
from prompt_builder import PromptBuilder
from llm import LLM
import time
from logger import SearchLogger


class RAGPipeline:

    def __init__(
        self,
        embeddings_path: str,
        index_path: str,
    ):

        self.retriever = Retriever(
            embeddings_path=embeddings_path,
            index_path=index_path
        )

        self.reranker = Reranker()
        self.llm = LLM()
        self.logger = SearchLogger()

    def search(
        self,
        query: str,
        retrieval_k: int = 5,
        rerank_k: int = 3,
    ) -> str:
        
        t0 = time.perf_counter()

        # Step 1: Retrieve
        candidates = self.retriever.search(
            query=query,
            top_k=retrieval_k
        )

        t1 = time.perf_counter()

        # Step 2: Rerank
        contexts = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=rerank_k
        )
        t2 = time.perf_counter()

        # Step 3: Build Prompt
        system_prompt, user_prompt = PromptBuilder.build(
            query=query,
            contexts=contexts
        )
        t3 = time.perf_counter()

        # Step 4: LLM
        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        t4 = time.perf_counter()

        # --------------------------
        # Logging
        # --------------------------
        if contexts:
            self.logger.log(
                query=query,

                retrieval_time=t1 - t0,
                rerank_time=t2 - t1,
                llm_time=t4 - t3,
                total_time=t4 - t0,

                selected_result=contexts[0],
            )

        print("\n========== TIMINGS ==========")
        print(f"Retrieval      : {t1 - t0:.3f} sec")
        print(f"Reranking      : {t2 - t1:.3f} sec")
        print(f"Prompt Builder : {t3 - t2:.3f} sec")
        print(f"LLM            : {t4 - t3:.3f} sec")
        print(f"Total          : {t4 - t0:.3f} sec")
        print("=============================\n")
        print("\n===== CONFIDENCE =====")
        print("======================\n")

        return response