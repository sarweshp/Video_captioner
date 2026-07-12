
# 🎥 AI Video Captioning Agent

An AI-powered video captioning pipeline that generates high-quality captions in multiple styles using Large Language Models (LLMs). The system extracts visual and audio information from videos, produces a structured scene understanding, generates captions, evaluates them using an LLM-based judge, refines them when necessary, and outputs the best caption for each requested style.

The project is designed for the **AMD Developer Hackathon – Track 2: Video Captioning Agent** and is optimized for **parallel execution**, **robustness**, and **high caption quality**.

---

# Features

- 🎬 Automatic video scene understanding
- 🔊 Audio transcription using Whisper Large V3 Turbo
- 🖼️ Multi-frame visual analysis
- ✍️ Caption generation in four different styles
- ⚖️ LLM-based caption judging
- 🔄 Automatic caption refinement
- 🏆 Best-caption selection when refinement budget is exhausted
- ⚡ Parallel processing of videos
- ⚡ Parallel generation of captions for multiple styles
- 🛡️ Automatic retries and graceful fallbacks
- 🐳 Fully Dockerized

---

# Caption Styles

The agent supports the following styles:

| Style | Description |
|--------|-------------|
| formal | Professional, objective and factual |
| sarcastic | Dry irony and light mockery |
| humorous_tech | Technology/programming themed humor |
| humorous_non_tech | Everyday non-technical humor |

---

# 🚀 Live Demo

Try the project without setting up the repository:

**Hugging Face Space:**  
https://huggingface.co/spaces/sarweshp/Video_captioner-4_styles

### What you can do

- Upload or provide a **public video URL** (up to **2 minutes** long).
- The application automatically:
  - Extracts representative video frames
  - Transcribes the audio using Whisper
  - Builds a structured scene understanding
  - Generates captions in **four different styles**
  - Uses an LLM-based judge and refinement pipeline to improve caption quality

### Supported Caption Styles

- 📄 **Formal**
- 😏 **Sarcastic**
- 💻 **Humorous Tech**
- 😂 **Humorous Non-Tech**

The interface returns all four captions together, allowing users to compare different writing styles for the same video.

> **Note:** The demo currently supports videos up to **2 minutes** in duration. Longer videos may exceed the processing limits of the hosted demo.

# Pipeline Overview

```
                    Input Video
                         │
                         ▼
                Download Video
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
 Frame Extraction               Audio Extraction
         │                               │
         ▼                               ▼
 Frame Sampling                 Whisper Transcription
         └───────────────┬───────────────┘
                         ▼
               Scene Understanding LLM
                         │
                         ▼
             Structured Scene Description
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Caption Generation (Parallel)    │
        │                                  │
        │  Formal                          │
        │  Sarcastic                       │
        │  Humorous Tech                   │
        │  Humorous Non-Tech               │
        └──────────────────────────────────┘
                         │
                         ▼
                 LLM Caption Judge
                 PASS        FAIL
                  │            │
                  │            ▼
                  │      Caption Refiner
                  │            │
                  │            ▼
                  │       Judge Again
                  │            │
                  └──────PASS──┘
                               │
                               ▼
              Max Refinement Reached?
                      │
               No ────┘
                      │
                     Yes
                      │
                      ▼
             Best Caption Selector
                      │
                      ▼
                 Final Caption
```

---

# Project Structure

```
.
├── main.py
├── video_captioner.py
├── caption_generator.py
├── caption_judge.py
├── requirements.txt
├── Dockerfile
├── input
│   └── tasks.json
└── output
    └── results.json
```

---

# Pipeline Components

## 1. Video Processing

`video_captioner.py`

Responsible for:

- Downloading videos
- Extracting audio using FFmpeg
- Extracting representative video frames
- Running Whisper transcription
- Building multimodal prompts
- Producing structured scene JSON

Output example:

```json
{
  "objects": [...],
  "actions": [...],
  "scene": "...",
  "mood": "...",
  "summary": "...",
  "audio_transcript": "..."
}
```

---

## 2. Caption Generation

`caption_generator.py`

For every requested style:

- Builds style-specific prompts
- Generates captions
- Returns structured JSON captions

Caption generation for all styles happens **concurrently** using a thread pool.

---

## 3. Caption Judge

`caption_judge.py`

Each generated caption is evaluated for:

- factual correctness
- hallucinations
- scene grounding
- style compliance

The judge returns only two possible outcomes:

```
PASS
```

or

```
FAIL
```

along with concise feedback describing what needs to be improved.

---

## 4. Caption Refinement

If the caption fails:

- Previous caption
- Judge feedback
- Scene description

are sent to the refinement model.

The refiner generates an improved caption.

This loop continues until:

- Caption passes

or

- Maximum refinement iterations are exhausted.

---

## 5. Best Caption Selection

If no caption passes after all refinement attempts:

The selector LLM receives **every previous caption** together with their judge feedback and chooses the strongest candidate instead of blindly returning the last generated caption.

This prevents refinement from accidentally degrading caption quality.

---

# Parallelization

The system uses two levels of parallelism.

## Task Level

Multiple videos are processed simultaneously.

```
Video 1
Video 2
Video 3
Video 4
```

Each video has its own worker thread.

Controlled by

```
TASK_WORKERS
```

---

## Style Level

Within every video:

```
Formal
Sarcastic
Humorous Tech
Humorous Non-Tech
```

are generated simultaneously.

This significantly reduces total runtime.

---

# Retry Strategy

Every network-based component supports automatic retries.

Supported modules include:

- Scene generation
- Caption generation
- Judge
- Refiner
- Selector

If all retries fail, the pipeline falls back gracefully instead of crashing.

---

# Environment Variables

| Variable | Description | Default |
|-----------|-------------|----------|
| FIREWORKS_API_KEY | Fireworks API Key | Required |
| OPENROUTER_API_KEY | OpenRouter API Key (Whisper) | Required |
| SCENE_MODEL | Vision model | Fireworks Qwen |
| CAPTION_MODEL | Caption generation model | Fireworks Qwen |
| JUDGE_MODEL | Judge model | CAPTION_MODEL |
| REFINE_MODEL | Refinement model | CAPTION_MODEL |
| TASK_WORKERS | Number of concurrent videos | 4 |
| ENABLE_JUDGE | Enable judge pipeline | true |
| MAX_REFINE_ITERATIONS | Maximum refinement attempts | 2 |
| INCLUDE_JUDGE_METADATA | Include judge output in results | false |
| MAX_RETRIES | Retry attempts | 5 |

---

# Input Format

The project expects:

```
input/tasks.json
```

Example:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://...",
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

# Output Format

Results are written to

```
output/results.json
```

Example

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

---

# Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running Locally

Set environment variables

```bash
export FIREWORKS_API_KEY=YOUR_FIREWORKS_KEY
export OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
```

Place the input file

```
input/tasks.json
```

Run

```bash
python main.py
```

The generated captions will be available in

```
output/results.json
```

---

# Docker

Build the Docker image

```bash
docker build -t video-caption-agent .
```

Run

```bash
docker run \
    -e FIREWORKS_API_KEY=YOUR_FIREWORKS_KEY \
    -e OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    video-caption-agent
```

---

# Design Goals

- High factual accuracy
- Minimal hallucinations
- Strong style adherence
- Robust execution
- Parallel processing
- Automatic recovery from API failures
- Production-ready Docker deployment

---

# Technologies Used

- Python
- OpenAI SDK
- Fireworks AI
- OpenRouter
- Whisper Large V3 Turbo
- OpenCV
- FFmpeg
- ThreadPoolExecutor
- Docker

---

# Future Improvements

- Better adaptive frame sampling
- Temporal reasoning across frames
- Multi-caption ensemble generation
- Confidence-based caption selection
- Streaming video support
- Batch inference optimization
- Local speech recognition fallback
- Vision model caching

---

# License

This project was developed as part of the **AMD Developer Hackathon – Track 2: Video Captioning Agent**.

