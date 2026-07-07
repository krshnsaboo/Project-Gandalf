import os
import json
from tqdm import tqdm

# =========================
# CONFIG
# =========================

INPUT_FOLDER = "new_json"
OUTPUT_FOLDER = "final_json"

TINY_LIMIT = 80
SAFE_MAX = 280

# =========================
# HELPERS
# =========================

def tokens(text):
    return len(text.split())

def ensure_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

# =========================
# ABSORB FUNCTION
# =========================

def absorb_small_chunks(chunks, lecture_id):

    result = []
    i = 0

    while i < len(chunks):

        current = chunks[i]
        cur_tokens = tokens(current["text"])

        # if not tiny → keep as it is
        if cur_tokens >= TINY_LIMIT:
            result.append(current)
            i += 1
            continue

        # --------------------------
        # Try merging with previous
        # --------------------------
        if result:
            prev = result[-1]
            merged_text = prev["text"] + " " + current["text"]

            if tokens(merged_text) <= SAFE_MAX:
                prev["text"] = merged_text
                prev["end"] = current["end"]
                i += 1
                continue

        # --------------------------
        # Try merging with next
        # --------------------------
        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            merged_text = current["text"] + " " + nxt["text"]

            if tokens(merged_text) <= SAFE_MAX:
                new_chunk = {
                    "chunk_id": f"{lecture_id}_C{i+1}",
                    "start": current["start"],
                    "end": nxt["end"],
                    "timestamp_url": current["timestamp_url"],
                    "text": merged_text
                }
                result.append(new_chunk)
                i += 2
                continue

        # If cannot merge → keep it
        result.append(current)
        i += 1

    # rebuild chunk IDs
    for idx, ch in enumerate(result, 1):
        ch["chunk_id"] = f"{lecture_id}_C{idx}"

    return result

# =========================
# MAIN
# =========================

def main():

    ensure_output_folder()

    files = sorted(os.listdir(INPUT_FOLDER))

    before = 0
    after = 0
    token_list = []

    for file in tqdm(files, desc="Absorbing Tiny Chunks"):

        if not file.endswith(".json"):
            continue

        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lecture_id = data["lecture_id"]
        chunks = data["chunks"]
        before += len(chunks)

        new_chunks = absorb_small_chunks(chunks, lecture_id)
        after += len(new_chunks)

        for c in new_chunks:
            token_list.append(tokens(c["text"]))

        new_data = {
            "lecture_id": data["lecture_id"],
            "lecture_number": data["lecture_number"],
            "lecture_title": data["lecture_title"],
            "youtube_url": data["youtube_url"],
            "chunks": new_chunks
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

    # =========================
    # FINAL STATS
    # =========================

    print("\n================= FINAL RESULTS =================")
    print(f"Chunks before: {before}")
    print(f"Chunks after: {after}")

    print("\nToken Stats:")
    print(f"Average tokens: {round(sum(token_list)/len(token_list),2)}")
    print(f"Min tokens: {min(token_list)}")
    print(f"Max tokens: {max(token_list)}")

if __name__ == "__main__":
    main()