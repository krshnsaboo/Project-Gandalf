import os
import json
from tqdm import tqdm
import numpy as np

FOLDER = "final_jsons"

token_counts = []
chunk_counts = []
durations = []

very_small = 0
very_large = 0

files = [f for f in os.listdir(FOLDER) if f.endswith(".json")]

for file in tqdm(files):
    path = os.path.join(FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    chunk_counts.append(len(chunks))

    for c in chunks:
        text = c["text"]
        tokens = len(text.split())

        token_counts.append(tokens)

        duration = c["end"] - c["start"]
        durations.append(duration)

        if tokens < 120:
            very_small += 1

        if tokens > 300:
            very_large += 1


print("\n================= RESULTS =================\n")

print(f"Total lectures: {len(files)}")
print(f"Total chunks: {len(token_counts)}\n")

print("Token Stats:")
print(f"Average tokens per chunk: {round(np.mean(token_counts), 2)}")
print(f"Min tokens: {min(token_counts)}")
print(f"Max tokens: {max(token_counts)}\n")

print("Lecture Stats:")
print(f"Average chunks per lecture: {round(np.mean(chunk_counts), 2)}")
print(f"Min chunks in a lecture: {min(chunk_counts)}")
print(f"Max chunks in a lecture: {max(chunk_counts)}\n")

print("Duration Stats (seconds):")
print(f"Average duration: {round(np.mean(durations), 2)}")
print(f"Min duration: {round(min(durations), 2)}")
print(f"Max duration: {round(max(durations), 2)}\n")

print("Quality Check:")
print(f"Chunks smaller than 120 tokens: {very_small}")
print(f"Chunks larger than 300 tokens: {very_large}")