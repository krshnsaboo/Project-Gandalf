import os
import json
from tqdm import tqdm

# =========================
# CONFIG
# =========================

INPUT_FOLDER = "jsons"
OUTPUT_FOLDER = "new_json"
METADATA_FILE = "videos_metadata.json"

MIN_TOKENS = 200
MAX_TOKENS = 250
HARD_TOKEN_CAP = 260       # absolute safety cap
MAX_DURATION = 80
OVERLAP_PERCENT = 0.10


# =========================
# HELPERS
# =========================

def estimate_tokens(text):
    return len(text.split())

def timestamp_url(base_url, start):
    return f"{base_url}&t={int(start)}s"

def ensure_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


# =========================
# CORE MERGE FUNCTION
# =========================

def merge_chunks(chunks, lecture_id, youtube_url):

    merged = []

    buffer_text = []
    buffer_start = None
    buffer_end = None
    buffer_tokens = 0

    chunk_counter = 1

    for ch in chunks:

        text = ch["text"].strip()
        if not text:
            continue

        tokens = estimate_tokens(text)

        if buffer_start is None:
            buffer_start = ch["start"]

        # simulate adding
        new_tokens = buffer_tokens + tokens
        new_duration = ch["end"] - buffer_start

        # ---------- HARD LIMIT CHECK ----------
        if (new_tokens > MAX_TOKENS) or (new_duration > MAX_DURATION):

            # finalize previous chunk first
            final_text = " ".join(buffer_text).strip()

            if final_text:
                merged.append({
                    "chunk_id": f"{lecture_id}_C{chunk_counter}",
                    "start": round(buffer_start, 2),
                    "end": round(buffer_end, 2),
                    "timestamp_url": timestamp_url(youtube_url, buffer_start),
                    "text": final_text
                })
                chunk_counter += 1

                # overlap
                words = final_text.split()
                overlap = int(len(words) * OVERLAP_PERCENT)
                overlap_words = words[-overlap:] if overlap > 0 else []

                buffer_text = [" ".join(overlap_words)] if overlap_words else []
                buffer_tokens = len(overlap_words)
                buffer_start = ch["start"]

            else:
                buffer_text = []
                buffer_tokens = 0
                buffer_start = ch["start"]

        # now add current chunk
        buffer_text.append(text)
        buffer_end = ch["end"]
        buffer_tokens = estimate_tokens(" ".join(buffer_text))

    # ---------- HANDLE LAST CHUNK ----------
    if buffer_text:

        final_text = " ".join(buffer_text).strip()
        tokens = estimate_tokens(final_text)

        if merged:
            prev_tokens = estimate_tokens(merged[-1]["text"])

            # attach only if safe
            if (tokens < MIN_TOKENS and 
                prev_tokens + tokens <= HARD_TOKEN_CAP and
                (buffer_end - merged[-1]["start"]) <= MAX_DURATION):

                merged[-1]["text"] += " " + final_text
                merged[-1]["end"] = round(buffer_end, 2)

            else:
                merged.append({
                    "chunk_id": f"{lecture_id}_C{chunk_counter}",
                    "start": round(buffer_start, 2),
                    "end": round(buffer_end, 2),
                    "timestamp_url": timestamp_url(youtube_url, buffer_start),
                    "text": final_text
                })

        else:
            merged.append({
                "chunk_id": f"{lecture_id}_C1",
                "start": round(buffer_start, 2),
                "end": round(buffer_end, 2),
                "timestamp_url": timestamp_url(youtube_url, buffer_start),
                "text": final_text
            })

    return merged


# =========================
# MAIN
# =========================

def main():

    ensure_output_folder()

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total_old = 0
    total_new = 0
    token_list = []
    duration_list = []

    files = sorted(os.listdir(INPUT_FOLDER))

    for file in tqdm(files, desc="Processing Lectures"):

        if not file.endswith(".json"):
            continue

        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data["chunks"]
        total_old += len(chunks)

        lecture_id = chunks[0]["number"]
        lecture_number = int(lecture_id[1:])
        meta = metadata.get(lecture_id, {})
        youtube_url = meta.get("url", "")
        lecture_title = meta.get("title", chunks[0]["title"])
        lecture_number = meta.get("lecture", lecture_number)

        merged_chunks = merge_chunks(chunks, lecture_id, youtube_url)
        total_new += len(merged_chunks)

        # stats
        for c in merged_chunks:
            token_list.append(estimate_tokens(c["text"]))
            duration_list.append(c["end"] - c["start"])

        new_data = {
            "lecture_id": lecture_id,
            "lecture_number": lecture_number,
            "lecture_title": lecture_title,
            "youtube_url": youtube_url,
            "chunks": merged_chunks
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

    # =========================
    # FINAL STATS
    # =========================

    print("\n================= RESULTS =================")
    print(f"Total lectures: 315, Old chunks: {total_old}")
    print(f"Total new chunks: {total_new}")

    print("\nToken Stats:")
    print(f"Average tokens per chunk: {round(sum(token_list)/len(token_list),2)}")
    print(f"Min tokens: {min(token_list)}")
    print(f"Max tokens: {max(token_list)}")

    print("\nDuration Stats (seconds):")
    print(f"Average duration: {round(sum(duration_list)/len(duration_list),2)}")
    print(f"Min duration: {round(min(duration_list),2)}")
    print(f"Max duration: {round(max(duration_list),2)}")


if __name__ == "__main__":
    main()