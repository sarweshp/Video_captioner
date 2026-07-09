FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ffmpeg -> audio extraction / whisper. libgl1 + libglib2.0-0 -> opencv runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the whisper model at build time so runtime doesn't need network
# access for it and doesn't eat into the 10-minute run budget.
RUN python -c "import whisper; whisper.load_model('base')"

COPY main.py video_captioner.py caption_generator.py ./

# No API key is injected by the judging environment, so it must be baked in
# at build time. Replace with your real key before building.

# ENV OPENROUTER_SCENE_MODEL="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
# ENV OPENROUTER_CAPTION_MODEL="nvidia/nemotron-3-nano-30b-a3b"
ENV FIREWORKS_API_KEY=""
ENV SCENE_MODEL="accounts/fireworks/models/minimax-m3"
ENV CAPTION_MODEL="accounts/fireworks/models/minimax-m3"

ENV MAX_RETRIES="2"

RUN mkdir -p /input /output

CMD ["python", "main.py"]
