import os
import sys
import json
import tempfile
import traceback
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Judge and refinement can be different models from the one that drafts the
# caption (e.g. a stronger/pickier model to grade and fix captions produced by
# a cheaper drafting model). Both default to CAPTION_MODEL if not set.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", CAPTION_MODEL)
REFINE_MODEL = os.environ.get("REFINE_MODEL", CAPTION_MODEL)

# Toggle the judge+refine loop on/off, and cap how many refine attempts a
# single caption can go through before we settle for its latest draft.
ENABLE_JUDGE = os.environ.get("ENABLE_JUDGE", "true").lower() in ("1", "true", "yes")
MAX_REFINE_ITERATIONS = int(os.environ.get("MAX_REFINE_ITERATIONS", "2"))

# If true, results.json includes judge scores/feedback per caption instead of
# just the caption string.
INCLUDE_JUDGE_METADATA = os.environ.get("INCLUDE_JUDGE_METADATA", "false").lower() in ("1", "true", "yes")

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

# How many tasks (video clips) to process concurrently. Each task does a
# network download + a scene-analysis call + N caption calls, so this is
# mostly I/O bound and benefits from threads even under the GIL. Keep this
# tunable via env var in case the grading environment rate-limits concurrent
# requests to the model API.
TASK_WORKERS = int(os.environ.get("TASK_WORKERS", "4"))


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
    if INCLUDE_JUDGE_METADATA:
        return {
            style: {
                "caption": f"[Fallback] {style.replace('_', ' ').title()} caption for the described scene.",
                "judged": False,
                "passed": None,
                "refine_iterations": 0,
                "attempts": [],
            }
            for style in styles
            if style in STYLE_PROMPTS
        }
    return {
        style: f"[Fallback] {style.replace('_', ' ').title()} caption for the described scene."
        for style in styles
        if style in STYLE_PROMPTS
    }


def process_task(task, client, captioner):
    """Run the full pipeline for a single task. Designed to be safe to call
    from multiple worker threads at once: it only touches its own local
    tempdir, and the shared `client`/`captioner` objects are read-only /
    stateless from the caller's perspective (the OpenAI SDK's HTTP client
    and whisper inference are both safe for concurrent use)."""
    start_time = time.perf_counter()
    task_id = task.get("task_id", "unknown")
    video_url = task.get("video_url")
    styles = task.get("styles", list(STYLE_PROMPTS.keys()))

    print(f"\n=== Task {task_id} start ===")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"[{task_id}] Downloading {video_url} ...")
            video_path = download_video(video_url, tmp_dir)
            print(f"[{task_id}] Download complete.")

            scene_json = captioner.process(video_path)
            print(f"[{task_id}] Scene JSON: {scene_json}")
            captions = generate_all_captions(
                scene_json,
                client,
                CAPTION_MODEL,
                styles,
                max_retries=MAX_RETRIES,
                # Judge and refine reuse the same Fireworks client (same API
                # key/base_url) but can point at different model strings.
                # Swap in separate OpenAI() clients here too if judge/refine
                # ever need to hit a different provider/endpoint.
                judge_client=client,
                judge_model=JUDGE_MODEL,
                refine_client=client,
                refine_model=REFINE_MODEL,
                enable_judge=ENABLE_JUDGE,
                max_refine_iterations=MAX_REFINE_ITERATIONS,
                include_judge_metadata=INCLUDE_JUDGE_METADATA,
            )

            # Guarantee every requested style is present, even if one silently
            # dropped out for some reason.
            for style in styles:
                if style not in captions and style in STYLE_PROMPTS:
                    captions[style] = fallback_captions([style])[style]

        end_time = time.perf_counter()
        print(f"[{task_id}] done in {end_time - start_time:.2f} seconds.")
        return {"task_id": task_id, "captions": captions}

    except Exception as e:
        print(f"Task {task_id} FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        # Never drop a task entirely -- still emit fallback captions so the
        # rest of the run isn't scored as zero because of one bad clip.
        return {"task_id": task_id, "captions": fallback_captions(styles)}


def main():
    st = time.perf_counter()
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    api_key = os.environ.get("FIREWORKS_API_KEY")
    # api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("FATAL: FIREWORKS_API_KEY is not set inside the image.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
        #base_url="https://router.huggingface.co/v1",
    )

    captioner = VideoCaptioner(
        api_key=api_key,
        model=SCENE_MODEL,
        max_retries=MAX_RETRIES,
    )

    # Process multiple tasks (video clips) concurrently. Each task is
    # dominated by network calls (video download + model API calls), so
    # threads give real wall-clock speedup despite the GIL. Results are
    # collected keyed by their original list position so output order
    # matches input order regardless of completion order.
    results_by_index = {}
    num_workers = min(TASK_WORKERS, len(tasks)) or 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(process_task, task, client, captioner): idx
            for idx, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results_by_index[idx] = future.result()
            except Exception as e:
                # Should not normally happen since process_task catches its
                # own exceptions, but guard against unexpected thread errors.
                task = tasks[idx]
                task_id = task.get("task_id", "unknown")
                styles = task.get("styles", list(STYLE_PROMPTS.keys()))
                print(f"Task {task_id} FAILED in executor: {e}", file=sys.stderr)
                traceback.print_exc()
                results_by_index[idx] = {"task_id": task_id, "captions": fallback_captions(styles)}

    results = [results_by_index[i] for i in range(len(tasks))]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    et = time.perf_counter()
    print(f"\nWrote {len(results)} results to {OUTPUT_PATH} in {et - st:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL error in main():", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
