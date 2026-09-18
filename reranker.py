import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import RERANKER_MODEL, RERANK_MIN_SCORE


class Reranker:

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        min_score: float = RERANK_MIN_SCORE,
    ):
        print(f"Loading reranker model: {model_name}...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min_score = min_score
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
            ).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
            ).to(self.device)

        self.model.eval()
        print(f"Reranker loaded on device: {self.device}")

    @staticmethod
    def select_hierarchical(
        candidates: list[dict],
        reranked: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        seen_lectures = set()
        seen_chunk_ids = set()
        hierarchical_results = []

        for c in reranked:
            lid = c.get("lecture_id")
            if lid not in seen_lectures:
                seen_lectures.add(lid)
                best_c = next((item for item in candidates if item.get("lecture_id") == lid), c)
                best_c = best_c.copy()
                best_c["rerank_score"] = c.get("rerank_score", 0.0)
                hierarchical_results.append(best_c)
                seen_chunk_ids.add(best_c.get("chunk_id"))
                if len(hierarchical_results) == top_k:
                    break

        if len(hierarchical_results) < top_k:
            for c in reranked:
                cid = c.get("chunk_id")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    hierarchical_results.append(c)
                    if len(hierarchical_results) == top_k:
                        break

        return hierarchical_results

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        batch_size: int = 32,
        pre_prune_k: int = 20,
        hierarchical: bool = True,
    ) -> list[dict]:
        if not candidates:
            return []

        t0 = time.perf_counter()

        pruned_candidates = candidates[:pre_prune_k]

        scores = []

        with torch.no_grad():
            for start in range(0, len(pruned_candidates), batch_size):
                batch = pruned_candidates[start : start + batch_size]

                pairs = [
                    [
                        query,
                        f"Lecture: {c.get('lecture_title', '')}\n\n{c.get('text', '')}"
                    ]
                    for c in batch
                ]

                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                logits = self.model(**inputs).logits
                logits = logits.view(-1).float().cpu().tolist()
                scores.extend(logits)

        for candidate, score in zip(pruned_candidates, scores):
            candidate["rerank_score"] = float(score)

        pruned_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        filtered = [c for c in pruned_candidates if c["rerank_score"] >= self.min_score]
        if not filtered and pruned_candidates:
            filtered = [pruned_candidates[0]]

        if hierarchical:
            final_results = self.select_hierarchical(
                candidates=candidates,
                reranked=filtered,
                top_k=top_k,
            )
        else:
            final_results = filtered[:top_k]

        t1 = time.perf_counter()
        print(f"Reranking ({len(pruned_candidates)} candidates) : {t1 - t0:.3f} sec")

        return final_results