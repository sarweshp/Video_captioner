# 🎬 Stylized Video Captioning AI Agent

> **AMD Developer Hackathon: ACT II — Track 2 Submission**

An AI-powered video captioning agent that ingests raw video clips (30 seconds to 2 minutes) and generates **context-aware captions** in multiple styles. The pipeline combines **OpenAI Whisper** for high-quality speech transcription with **Fireworks AI Vision-Language Models** for visual understanding, producing captions in four distinct tones:

- 📘 Formal
- 😏 Sarcastic
- 💻 Humorous Tech
- 😂 Humorous Non-Tech

---

# 🚀 Features

- 🎥 Automatic video ingestion
- 🔊 Audio extraction using FFmpeg
- 📝 Local Whisper transcription
- 🖼️ Adaptive frame sampling
- 🤖 Vision-language understanding with Fireworks AI
- 🎭 Multi-style caption generation
- ✅ Strict JSON output validation
- 🔄 Automatic retry & fallback mechanisms
- ⚡ Optimized for the hackathon's **10-minute execution limit**

---

# 🏗️ Architecture

```text
                ┌─────────────────────────┐
                │  📥 Ingest tasks.json   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ 🌐 Async Video Download │
                └────────────┬────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
 ┌──────────────────────┐             ┌──────────────────────┐
 │ 🔊 Audio Extraction  │             │ 🖼️ Adaptive Frame    │
 │ (FFmpeg, 16kHz Mono) │             │    Downsampling      │
 └───────────┬──────────┘             └───────────┬──────────┘
             ▼                                    ▼
 ┌──────────────────────┐             ┌──────────────────────┐
 │ 📝 Whisper Base      │             │ 📐 Dynamic Resizing  │
 │ Local Transcription  │             │ (Aspect Ratio Safe)  │
 └───────────┬──────────┘             └───────────┬──────────┘
             └────────────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ 🧠 Fireworks Vision LLM │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ 📝 Caption Generator    │
                     │ JSON Validation Engine  │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ 📤 results.json         │
                     └─────────────────────────┘
```

---

# ⚙️ Pipeline Overview

## 1. Video Processing (`video_captioner.py`)

### 🔊 Audio Extraction

- Uses **FFmpeg** to extract clean 16kHz mono PCM audio.
- Runs inside an isolated subprocess for reliability.

### 📝 Whisper Transcription

- Uses **OpenAI Whisper Base** locally.
- Model weights are downloaded during Docker build.
- Eliminates runtime download latency.

### 🖼️ Adaptive Frame Sampling

- Samples up to **10 representative frames** evenly across the video.
- Automatically adapts to video duration and FPS.

### 📐 Dynamic Image Resizing

High-resolution frames (including 4K videos) are resized so that the longest edge is **512 pixels** while preserving aspect ratio.

Benefits:

- Smaller payloads
- Faster inference
- Lower memory usage
- No noticeable loss in visual context

---

## 2. Caption Generation (`caption_generator.py`)

### 🎭 Multi-Style Caption Engine

Generates captions in four styles:

- Formal
- Sarcastic
- Humorous Tech
- Humorous Non-Tech

### ✅ Strict JSON Enforcement

The model is constrained to output valid JSON.

Additional cleanup removes:

- Markdown code blocks
- Extra text
- Formatting artifacts

to guarantee parseable outputs.

### 🔄 Retry & Recovery

Robust retry strategy featuring:

- Exponential backoff
- 5–7 retry attempts
- Automatic fallback captions

This prevents a single failed inference from interrupting the evaluation pipeline.

---

# 🎭 Supported Caption Styles

| Style | Description |
|-------|-------------|
| **formal** | Professional, factual, objective, exactly one concise sentence |
| **sarcastic** | Dry, ironic, witty commentary |
| **humorous_tech** | Programming jokes, hardware metaphors, engineering humor |
| **humorous_non_tech** | Everyday relatable humor and playful observations |

---

# 📁 Repository Structure

```
.
├── Dockerfile
├── main.py
├── video_captioner.py
├── caption_generator.py
├── requirements.txt
└── README.md
```

### Dockerfile

- Python 3.11 Slim
- Installs OpenCV system dependencies
- Downloads Whisper model during build
- Creates `/input` and `/output` directories

### main.py

Responsible for:

- Reading `tasks.json`
- Downloading videos
- Calling processing pipeline
- Writing `results.json`

### video_captioner.py

Responsible for:

- Frame extraction
- Audio extraction
- Whisper transcription
- Vision model inputs

### caption_generator.py

Responsible for:

- Prompt engineering
- Style generation
- JSON validation
- Retry handling

---

# 🐳 Quick Start

## Prerequisites

- Docker
- Fireworks AI API Key

---

## 1. Build the Docker Image

```bash
docker build -t stylized-captioner-agent .
```

---

## 2. Prepare Input & Output Directories

```bash
mkdir -p input_dir output_dir
```

Create `input_dir/tasks.json`

```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "styles": [
      "formal",
      "sarcastic",
      "humorous_tech",
      "humorous_non_tech"
    ]
  }
]
```

---

## 3. Run the Container

```bash
docker run --rm \
  -v "$(pwd)/input_dir:/input" \
  -v "$(pwd)/output_dir:/output" \
  stylized-captioner-agent
```

Generated captions will be available in:

```
output_dir/results.json
```

---

# 📈 Performance Optimizations

## ⚡ Sub-10 Minute Runtime

Designed specifically for the AMD Hackathon evaluation environment.

Optimizations include:

- Build-time Whisper caching
- Adaptive frame sampling
- Image resizing
- Efficient API payloads

---

## 💾 Memory Optimization

Frames are resized to **512px maximum dimension**, reducing Base64 payload sizes by up to **75%**, resulting in:

- Lower RAM usage
- Faster inference
- Reduced network overhead
- Stable execution on constrained hardware

---

## 🛡️ Fault Tolerance

Every stage is protected with exception handling:

- Video download
- Audio extraction
- Whisper transcription
- Vision inference
- JSON parsing
- Caption generation

If any component fails, the pipeline logs the error and returns a safe fallback caption instead of terminating execution.

---

# 🧠 Technology Stack

- Python 3.11
- OpenAI Whisper
- Fireworks AI
- FFmpeg
- OpenCV
- Docker

---

# ✅ Output

The pipeline produces a `results.json` file containing captions for every requested style.

Example:

```json
{
  "task_id": "v1",
  "captions": {
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_non_tech": "..."
  }
}
```

---

# 📜 License

This project was developed as part of the **AMD Developer Hackathon – ACT II (Track 2)**.
