# Video Captioner — Hackathon Submission

## What you actually submit
You do **not** submit code/zip files. You submit a **Docker image pushed to a
public registry** (Docker Hub is the easiest). The judging VM will:

1. `docker pull` your image (must have a `linux/amd64` manifest)
2. Mount a folder with `tasks.json` to `/input/tasks.json`
3. Mount an empty folder to `/output`
4. Run the container with no arguments and no env vars injected
5. Read `/output/results.json` after it exits (must exit code `0`, within 10 min)

**Input** (`/input/tasks.json`, already inside the container via mount):
```json
[
  {
    "task_id": "v1",
    "video_url": "https://.../clip1.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

**Output** (`/output/results.json`, written by your container before exit):
```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

Since no API key is injected at runtime, your OpenRouter (or other) key must
be **baked into the image** at build time (see `Dockerfile`, `ENV
OPENROUTER_API_KEY=...`). Don't push this image publicly if you're worried
about key exposure — anyone who pulls it can `docker run --entrypoint sh` and
read the env var. If that matters to you, use a key with spend limits, or a
free-tier model, or a proxy you control.

---

## 0. Install Docker (Mac M1, first time)
1. Download **Docker Desktop for Mac (Apple Silicon)**:
   https://www.docker.com/products/docker-desktop/
2. Install it, open it once, let it finish starting (whale icon in the menu
   bar goes steady).
3. Sign up for a free Docker Hub account: https://hub.docker.com/signup
4. In a terminal:
   ```bash
   docker login
   ```
   Enter your Docker Hub username/password.

## 1. Put your real API key in the Dockerfile
Edit `Dockerfile` and replace:
```
ENV OPENROUTER_API_KEY="REPLACE_WITH_YOUR_OPENROUTER_API_KEY"
```
Also double-check `OPENROUTER_SCENE_MODEL` — it must be a **vision-capable**
model on OpenRouter (it receives images), e.g. a Gemini/Llama-vision/Qwen-VL
model. `OPENROUTER_CAPTION_MODEL` just needs to handle text + JSON output.

## 2. Build for linux/amd64 (important on M1!)
The judging VM is `linux/amd64`; your Mac is `arm64`. Build explicitly for
the target platform:

```bash
cd video-captioner
docker buildx build --platform linux/amd64 -t YOUR_DOCKERHUB_USERNAME/video-captioner:latest --push .
```

`--push` builds *and* uploads in one step (buildx can't `docker run`
multi-arch images loaded locally the normal way, so pushing straight to the
registry is the simplest path). This can take a while the first time
(downloading/installing torch + whisper + opencv).

## 3. Test it locally first (recommended)
Build a local-runnable version (native arch, faster) before doing the
amd64 push:
```bash
docker build -t video-captioner-test .
mkdir -p output
docker run --rm \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  video-captioner-test
cat output/results.json
```
This uses the sample `input/tasks.json` already in this folder (2 of the
example clips). Fix any errors before pushing the amd64 image.

## 4. Verify the pushed image is really amd64
```bash
docker buildx imagetools inspect YOUR_DOCKERHUB_USERNAME/video-captioner:latest
```
Look for `linux/amd64` in the platform list.

## 5. Submit
Give the organizers the public image reference, e.g.:
```
docker.io/YOUR_DOCKERHUB_USERNAME/video-captioner:latest
```
Make sure the Docker Hub repo is **public** (not private) or the pull will
fail on their side.

---

## Notes on the pipeline itself
- `video_captioner.py`: downloads nothing itself — takes a local video path,
  pulls ~10 evenly-spaced frames + a Whisper transcript, and asks a vision
  model for a JSON scene description (`objects`, `actions`, `scene`, `mood`,
  `summary`). Retries the model call up to `MAX_RETRIES` (default 7) until it
  gets valid JSON back, then falls back to a safe default so one bad clip
  can't crash the whole run.
- `caption_generator.py`: takes that scene JSON and asks a text model to
  write a caption in each requested style, again requesting strict JSON
  (`{"caption": "..."}`) and retrying up to `MAX_RETRIES` (default 7) times.
  Style keys (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`)
  match exactly what shows up in `tasks.json`.
- `main.py`: reads `/input/tasks.json`, downloads each `video_url`, runs the
  two steps above per task, and always writes an entry per `task_id` — even
  on failure it writes fallback captions instead of dropping the task, since
  a missing style scores zero for that clip.

## Things worth tuning before the real submission
- **Runtime budget**: 10 minutes total for however many tasks are in
  `tasks.json`. Whisper `base` + 7 retries per style call can add up across
  ~12 hidden clips. If you're tight on time, consider: a smaller Whisper
  model (`tiny`), fewer frames (`max_frames`), or lowering `MAX_RETRIES` for
  the caption step (scene JSON retries matter more than caption retries).
- **Model choice**: `google/gemini-2.0-flash-exp:free` is a placeholder —
  confirm on https://openrouter.ai/models that your chosen model (a) exists,
  (b) accepts image inputs for the scene step, and (c) has enough free-tier
  quota for a full run of ~12 clips × up to 7 retries.
- **Generalization**: the scoring explicitly penalizes overfitting to the 3
  example clips — nothing in this pipeline is clip-specific, so it should
  generalize, but test it on a few clips you haven't seen (nature, sports,
  food, weather) before submitting.
