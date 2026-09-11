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

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        batch_size: int = 32,
        pre_prune_k: int = 20,
    ) -> list[dict]:
        """
        Reranks candidates using the cross-encoder model.
        Applies pre-pruning to limit CPU work and post-filtering based on min_score threshold.
        """
        if not candidates:
            return []

        t0 = time.perf_counter()

        # Pre-rerank pruning: take top candidate pool to save CPU cycles
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

        # Sort descending by rerank score
        pruned_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Score threshold filtering: prune candidates that fall below min_score
        filtered = [c for c in pruned_candidates if c["rerank_score"] >= self.min_score]
        # Always retain at least top 1 result if available
        if not filtered and pruned_candidates:
            filtered = [pruned_candidates[0]]

        t1 = time.perf_counter()
        print(f"Reranking ({len(pruned_candidates)} candidates) : {t1 - t0:.3f} sec")

        return filtered[:top_k]