# ============================
# reranker.py
# ============================

import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3"
    ):

        print("Loading reranker model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

    def rerank(
        self,
        query,
        candidates,
        top_k=5,
        batch_size=64
    ):

        if len(candidates) == 0:
            return []

        scores = []

        with torch.no_grad():

            for start in range(0, len(candidates), batch_size):

                batch = candidates[start:start + batch_size]

                pairs = [
                    [
                        query,
                        f"Lecture: {c['lecture_title']}\n\n{c['text']}"
                    ]
                    for c in batch
                ]

                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt"
                ).to(self.device)

                logits = self.model(**inputs).logits

                logits = logits.view(-1).float().cpu().tolist()

                scores.extend(logits)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return candidates[:top_k]