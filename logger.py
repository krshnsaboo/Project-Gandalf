import json
from pathlib import Path
from datetime import datetime


class SearchLogger:

    def __init__(self, log_dir="logs"):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "search_logs.jsonl"
        print(self.log_file.resolve())

    def log(
        self,
        query,
        retrieval_time,
        rerank_time,
        llm_time,
        total_time,
        selected_result,
    ):
        """
        Append one search record to logs/search_logs.jsonl
        """

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),

            "query": query,

            "latency": {
                "retrieval_sec": round(retrieval_time, 3),
                "reranking_sec": round(rerank_time, 3),
                "llm_sec": round(llm_time, 3),
                "total_sec": round(total_time, 3),
            },

            "selected_result": {
                "lecture_id": selected_result.get("lecture_id"),
                "lecture_title": selected_result.get("lecture_title"),
                "chunk_id": selected_result.get("chunk_id"),
                "timestamp_start": round(selected_result.get("start", 0), 2),
                "timestamp_end": round(selected_result.get("end", 0), 2),
                "youtube_url": selected_result.get("timestamp_url"),
                "faiss_score": round(selected_result.get("score", 0.0), 4),
                "rerank_score": round(selected_result.get("rerank_score", 0.0), 4),
            },
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")