import os
import sys
import json
import tempfile
import traceback

import requests
from openai import OpenAI

from video_captioner import VideoCaptioner
from caption_generator import generate_all_captions, STYLE_PROMPTS

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"

SCENE_MODEL = os.environ.get(
    "SCENE_MODEL",
    "accounts/fireworks/models/qwen3p7-plus"
)

CAPTION_MODEL = os.environ.get(
    "CAPTION_MODEL",
    "accounts/fireworks/models/qwen3p7-plus"
)

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))


def download_video(url: str, dest_dir: str) -> str:
    """Stream-download a video URL to a local temp file and return its path."""
    local_path = os.path.join(dest_dir, "clip.mp4")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local_path


def fallback_captions(styles):
    return {
        style: f"[Fallback] {style.replace('_', ' ').title()} caption for the described scene."
        for style in styles
        if style in STYLE_PROMPTS
    }


def main():
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("FATAL: FIREWORKS_API_KEY is not set inside the image.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    captioner = VideoCaptioner(
        api_key=api_key,
        model=SCENE_MODEL,
        max_retries=MAX_RETRIES,
    )

    results = []

    for task in tasks:
        task_id = task.get("task_id", "unknown")
        video_url = task.get("video_url")
        styles = task.get("styles", list(STYLE_PROMPTS.keys()))

        print(f"\n=== Task {task_id} ===")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                print(f"Downloading {video_url} ...")
                video_path = download_video(video_url, tmp_dir)
                print("Download complete.")

                scene_json = captioner.process(video_path)
                print(f"Scene JSON: {scene_json}")
                captions = generate_all_captions(
                    scene_json, client, CAPTION_MODEL, styles, max_retries=MAX_RETRIES
                )

                # Guarantee every requested style is present, even if one silently
                # dropped out for some reason.
                for style in styles:
                    if style not in captions and style in STYLE_PROMPTS:
                        captions[style] = fallback_captions([style])[style]

            results.append({"task_id": task_id, "captions": captions})
            print(f"Task {task_id} done.")

        except Exception as e:
            print(f"Task {task_id} FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
            # Never drop a task entirely -- still emit fallback captions so the
            # rest of the run isn't scored as zero because of one bad clip.
            results.append({"task_id": task_id, "captions": fallback_captions(styles)})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL error in main():", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
