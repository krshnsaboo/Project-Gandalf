import json
import subprocess

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz"

def fetch_playlist_data(url):
    command = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        url
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    
    data = [json.loads(line) for line in lines if line.strip()]
    return data


def generate_metadata(data):
    metadata = {}

    for video in data:
        lecture_num = video.get("playlist_index")
        lecture_id = f"L{lecture_num:03d}"
        
        video_id = video.get("id")
        title = video.get("title")
        
        metadata[lecture_id] = {
            "lecture": lecture_num,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

    return metadata


def save_metadata(metadata, filename="videos_metadata.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Fetching playlist data...")
    playlist_data = fetch_playlist_data(PLAYLIST_URL)

    print("Generating metadata...")
    metadata = generate_metadata(playlist_data)

    print("Saving JSON file...")
    save_metadata(metadata)

    print("Done! videos_metadata.json created.")