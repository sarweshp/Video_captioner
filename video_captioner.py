import os
import json
import base64
import tempfile
import subprocess
import time
from typing import List, Dict, Any, Optional

import cv2
import whisper
from openai import OpenAI


class VideoCaptioner:
    """
    Extracts audio + sampled frames from a video, sends them to a vision-capable
    model on OpenRouter, and returns a structured JSON scene description:
    {objects, actions, scene, mood, summary, audio_transcript}
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemma-4-31b-it:free",
        max_frames: int = 10,
        frame_interval: float = 1.0,
        adaptive_sampling: bool = True,
        max_retries: int = 7,
        whisper_model_size: str = "base",
    ):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("Fireworks API key must be provided or set as env var")
        self.model = model
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.adaptive_sampling = adaptive_sampling
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )

        # Loaded once per process; reused across all tasks/clips.
        self.whisper_model = whisper.load_model(whisper_model_size)

    # ------------------------------------------------------------------ #
    # Audio
    # ------------------------------------------------------------------ #
    def extract_audio(self, video_path: str) -> str:
        audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(audio_fd)

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            audio_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}") from e
        return audio_path

    def transcribe_audio(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()

    # ------------------------------------------------------------------ #
    # Frames
    # ------------------------------------------------------------------ #
    def extract_frames(self, video_path: str) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if self.adaptive_sampling:
            proposed_frames = int(duration / self.frame_interval) + 1
            if proposed_frames > self.max_frames:
                num_frames = self.max_frames
            else:
                num_frames = proposed_frames
        else:
            num_frames = min(self.max_frames, int(duration / self.frame_interval) + 1)

        if num_frames < 1:
            num_frames = 1

        frame_indices = []
        if self.adaptive_sampling and num_frames == self.max_frames and duration > 0:
            for i in range(num_frames):
                timestamp = duration / 2 if num_frames == 1 else i * (duration / (num_frames - 1))
                frame_idx = int(timestamp * fps)
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                frame_indices.append(frame_idx)
        else:
            for i in range(num_frames):
                timestamp = i * self.frame_interval
                frame_idx = int(timestamp * fps)
                if frame_idx >= total_frames:
                    break
                frame_indices.append(frame_idx)

        base64_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]
            new_width = 512
            new_height = int(height * (new_width / width))
            if new_height > 512:
                new_height = 512
                new_width = int(width * (512 / height))
            frame_resized = cv2.resize(frame, (new_width, new_height))

            _, buffer = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(b64_str)

        cap.release()
        return base64_frames

    # ------------------------------------------------------------------ #
    # Model call
    # ------------------------------------------------------------------ #
    def build_messages(self, transcript: str, frames_b64: List[str]) -> List[Dict[str, Any]]:
        system_msg = {
            "role": "system",
            "content": (
                "You are a video captioning assistant. Given a series of video frames and an "
                "audio transcript, produce a neutral JSON description with EXACTLY these keys: "
                "'objects' (list of strings), 'actions' (list of strings), 'scene' (string), "
                "'mood' (string), 'summary' (string, 1-2 sentences). "
                "Respond with ONLY the raw JSON object. No markdown, no code fences, no extra text."
            )
        }

        content_parts = []
        if transcript:
            content_parts.append({
                "type": "text",
                "text": f"Audio transcript (if any): {transcript}\n\nAnalyze the following frames and provide JSON:"
            })
        else:
            content_parts.append({
                "type": "text",
                "text": "The video doesn't have any dialogue/conversation in it. Analyze the following frames and provide JSON:"
            })

        for b64 in frames_b64:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        return [system_msg, {"role": "user", "content": content_parts}]

    def _call_model_once(self, messages: List[Dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1200,
            temperature=0.2,
            #extra_body={"reasoning": {"enabled": False}},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Empty response from model")
        return content

    @staticmethod
    def parse_response(raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group())

        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not an object")

        required = ["objects", "actions", "scene", "mood", "summary"]
        for key in required:
            if key not in data:
                data[key] = [] if key in ("objects", "actions") else ""
        return data

    def get_scene_json(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calls the model up to self.max_retries times until it returns valid,
        parseable JSON. Falls back to a minimal safe structure if every
        attempt fails, so the pipeline never crashes a whole task.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._call_model_once(messages)
                return self.parse_response(raw)
            except Exception as e:
                last_error = e
                print(f"[scene-json] attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(min(2 * attempt, 10))

        print(f"[scene-json] all {self.max_retries} attempts failed, using fallback. Last error: {last_error}")
        return {
            "objects": [],
            "actions": [],
            "scene": "unknown scene",
            "mood": "neutral",
            "summary": "Unable to analyze this video's visual content.",
        }

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def process(self, video_path: str) -> Dict[str, Any]:
        print(f"Processing video: {video_path}")

        audio_path = None
        transcript = ""
        try:
            audio_path = self.extract_audio(video_path)
            print("Audio extracted.")
        except Exception as e:
            print(f"Audio extraction failed (proceeding without audio): {e}")

        if audio_path and os.path.exists(audio_path):
            try:
                transcript = self.transcribe_audio(audio_path)
                print(f"Transcription: {transcript[:200]}...")
            except Exception as e:
                print(f"Transcription failed: {e}")
            finally:
                os.unlink(audio_path)

        frames_b64 = self.extract_frames(video_path)
        print(f"Extracted {len(frames_b64)} frames.")

        messages = self.build_messages(transcript, frames_b64)
        result = self.get_scene_json(messages)
        result["audio_transcript"] = transcript
        return result
