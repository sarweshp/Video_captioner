import json
import re
import time
from typing import Dict, List, Any

from openai import OpenAI

# Keys here MUST match the style strings that show up in tasks.json
# ("formal", "sarcastic", "humorous_tech", "humorous_non_tech").
STYLE_PROMPTS = {
    "formal": (
        "Based on the scene details below, write a formal, professional caption. "
        "Use objective language, avoid slang, and highlight the key activities and setting. "
        "Keep it to one concise sentence."
    ),
    "sarcastic": (
        "Using the objects and actions described, craft a sarcastic, witty caption. "
        "Use irony or dry humor to comment on the scene's activities. Keep it short and punchy."
    ),
    "humorous_tech": (
        "Create a playful, tech-flavored caption about this scene. Use tech metaphors, jargon, "
        "or analogies (e.g., buffering, algorithm, CPU, rendering) that connect to the objects "
        "and actions. Make it clever but still understandable."
    ),
    "humorous_non_tech": (
        "Write a funny, everyday caption using relatable analogies, puns, or observational humor. "
        "Reference the specific objects and actions in a way anyone would find amusing, even "
        "without technical knowledge."
    ),
}


def format_scene_data(data: Dict[str, Any]) -> str:
    """Format the scene JSON into a readable block for the prompt."""
    return (
        f"Scene Description: {data.get('scene', 'Not specified')}\n"
        f"Objects in Scene: {', '.join(data.get('objects', []) or [])}\n"
        f"Actions Occurring: {', '.join(data.get('actions', []) or [])}\n"
        f"Overall Mood: {data.get('mood', 'Not specified')}\n"
        f"Video Summary: {data.get('summary', 'Not specified')}\n"
        f"Audio/Transcript: {data.get('audio_transcript', 'Not specified')}"
    )


def _extract_caption_json(raw_text: str) -> str:
    """
    The model is asked to return {"caption": "..."} as JSON.
    This pulls the caption string out, tolerating code fences / stray text.
    """
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
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group())

    if not isinstance(data, dict) or "caption" not in data:
        raise ValueError("JSON response missing 'caption' key")

    caption = str(data["caption"]).strip()
    if not caption:
        raise ValueError("Empty caption in JSON response")
    return caption


def generate_caption(
    style: str,
    scene_text: str,
    client: OpenAI,
    model: str,
    max_retries: int = 7,
) -> str:
    """Generate a single caption in the requested style, requesting strict JSON output."""

    system_prompt = (
        "You are a creative video caption writer. Produce captions that match the requested "
        "tone while staying true to the given scene details. "
        'Respond with ONLY a raw JSON object of the form {"caption": "your caption here"}. '
        "No markdown, no code fences, no extra commentary."
    )

    user_prompt = f"""{STYLE_PROMPTS[style]}

Scene details:
{scene_text}

Respond with JSON: {{"caption": "..."}}"""

    temp = 0.4 if style == "formal" else 0.8

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp,
                max_tokens=1000,
                timeout=30,
            )

            if response and response.choices:
                message = response.choices[0].message
                content = getattr(message, "content", None) or getattr(message, "text", None)
                if content and content.strip():
                    return _extract_caption_json(content)
            print(response)
            print(f"[caption:{style}] attempt {attempt}/{max_retries}: empty/invalid response")

        except Exception as e:
            last_error = e
            print(f"[caption:{style}] attempt {attempt}/{max_retries} error: {e}")

        if attempt < max_retries:
            time.sleep(min(0.1 * attempt, 10))

    print(f"[caption:{style}] all {max_retries} attempts failed. Last error: {last_error}")
    return f"[Fallback] {style.replace('_', ' ').title()} caption for the described scene."


def generate_all_captions(
    json_data: Dict[str, Any],
    client: OpenAI,
    model: str,
    styles: List[str],
    max_retries: int = 7,
) -> Dict[str, str]:
    """Generate captions only for the styles requested for this task."""
    scene_text = format_scene_data(json_data)
    captions = {}
    for style in styles:
        if style not in STYLE_PROMPTS:
            print(f"[caption] unknown style '{style}', skipping")
            continue
        captions[style] = generate_caption(style, scene_text, client, model, max_retries=max_retries)
    return captions
