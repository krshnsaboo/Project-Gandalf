import os
import json
import requests
import pandas as pd
import joblib

# ===============================
# Load video metadata
# ===============================
with open("videos_metadata.json", "r", encoding="utf-8") as f:
    video_meta = json.load(f)

# ===============================
# Embedding function (batched)
# ===============================
def create_embedding_batch(text_list):
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text_list},
            timeout=180
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    except Exception as e:
        print("Embedding failed:", e)
        return [None] * len(text_list)

# ===============================
# Process JSON files
# ===============================
all_rows = []
chunk_id = 0

json_files = sorted(os.listdir("new_jsons"))

for file_no, json_file in enumerate(json_files, start=1):

    print(f"\n[{file_no}/{len(json_files)}] Processing {json_file}")

    try:
        with open(f"new_jsons/{json_file}", "r", encoding="utf-8") as f:
            content = json.load(f)
    except:
        print("Skipping file (cannot read):", json_file)
        continue

    lecture_id = json_file.split("._")[0]   # e.g. L004

    if lecture_id not in video_meta:
        print("Skipping (no video metadata):", lecture_id)
        continue

    video_title = video_meta[lecture_id]["title"]
    video_url = video_meta[lecture_id]["url"]

    chunks = content["chunks"]

    texts = [c["text"] for c in chunks]

    # ---------- batch embeddings ----------
    batch_size = 16
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = create_embedding_batch(batch)
        embeddings.extend(emb)

    if len(embeddings) != len(texts):
        print("Skipping file (embedding mismatch):", json_file)
        continue

    # ---------- store rows ----------
    for c, emb in zip(chunks, embeddings):

        if emb is None:
            continue

        all_rows.append({
            "chunk_id": chunk_id,
            "lecture": lecture_id,
            "video_title": video_title,
            "video_url": video_url,
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "embedding": emb
        })

        chunk_id += 1

# ===============================
# Save dataframe
# ===============================
df = pd.DataFrame(all_rows)

print("\nTotal chunks stored:", len(df))

joblib.dump(df, "embeddings.joblib")

print("embeddings.joblib created successfully")