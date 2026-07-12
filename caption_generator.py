import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

from openai import OpenAI

from caption_judge import evaluate_caption, refine_caption, select_best_caption

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
        "Maintain the length of written caption between 25 to 60 words. "
        'Respond with ONLY a raw JSON object of the form {"caption": "your caption here"}. '
        "No markdown, no code fences, no extra commentary."
        
    )

    user_prompt = f"""{STYLE_PROMPTS[style]}

Scene details:
{scene_text}

Respond with JSON: {{"caption": "..."}}"""

    temp = 0.8 if style == "formal" else 0.7

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
                max_tokens=2000,
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


def generate_and_judge_caption(
    style: str,
    scene_text: str,
    gen_client: OpenAI,
    gen_model: str,
    judge_client: OpenAI,
    judge_model: str,
    refine_client: OpenAI,
    refine_model: str,
    max_retries: int = 7,
    enable_judge: bool = True,
    max_refine_iterations: int = 2,
) -> Dict[str, Any]:
    """
    Implements:

        Generate Caption -> Judge -> PASS -> Return
                               |
                              FAIL
                               |
                               v
                            Refiner (sees every prior attempt + feedback)
                               |
                               v
                          Judge again
                               |
                               v
                    iterations exhausted?
                       No -> Refine again
                       Yes -> Best Caption Selector -> Final Caption

    The judge gives a single PASS/FAIL verdict (see caption_judge.py for why: four
    separate numeric scores added variance without changing the decision). On FAIL,
    the refiner is shown the FULL history of attempts and their feedback -- not just
    the latest one -- so it can't oscillate between reintroducing the same couple of
    mistakes. If no attempt passes before the refinement budget runs out, a dedicated
    selector call picks the strongest candidate among ALL attempts made (the last
    attempt is not assumed to be the best one -- refining to fix one problem can
    introduce another).

    gen_client/gen_model, judge_client/judge_model, and refine_client/refine_model
    are independent so each stage can use a different model.

    Returns: {"caption": str, "judged": bool, "passed": bool | None,
              "refine_iterations": int, "attempts": [{"caption": str, "feedback": str}, ...]}
    """
    caption = generate_caption(style, scene_text, gen_client, gen_model, max_retries)

    result: Dict[str, Any] = {
        "caption": caption,
        "judged": False,
        "passed": None,
        "refine_iterations": 0,
        "attempts": [],
    }

    if not enable_judge:
        return result

    attempts: List[Dict[str, str]] = []
    current_caption = caption

    for iteration in range(max_refine_iterations + 1):
        verdict = evaluate_caption(style, scene_text, current_caption, judge_client, judge_model, max_retries)
        result["judged"] = True
        attempts.append({"caption": current_caption, "feedback": verdict["feedback"]})

        if verdict["passed"]:
            result["caption"] = current_caption
            result["passed"] = True
            result["refine_iterations"] = iteration
            result["attempts"] = attempts
            return result

        if iteration >= max_refine_iterations:
            break  # refinement budget exhausted, nothing has passed yet

        print(f"[judge:{style}] attempt {iteration + 1} FAILED, refining "
              f"({iteration + 1}/{max_refine_iterations}). feedback={verdict['feedback']!r}")
        current_caption = refine_caption(
            style,
            STYLE_PROMPTS[style],
            scene_text,
            attempts,
            refine_client,
            refine_model,
            max_retries,
        )

    # Budget exhausted with no passing attempt: don't just keep the last one --
    # ask the judge to pick the strongest candidate out of everything tried.
    print(f"[judge:{style}] exhausted refine budget with no pass, selecting best of "
          f"{len(attempts)} attempts.")
    best_caption = select_best_caption(style, scene_text, attempts, judge_client, judge_model, max_retries)
    result["caption"] = best_caption
    result["passed"] = False
    result["refine_iterations"] = max_refine_iterations
    result["attempts"] = attempts
    return result


def generate_all_captions(
    json_data: Dict[str, Any],
    client: OpenAI,
    model: str,
    styles: List[str],
    max_retries: int = 7,
    judge_client: OpenAI = None,
    judge_model: str = None,
    refine_client: OpenAI = None,
    refine_model: str = None,
    enable_judge: bool = False,
    max_refine_iterations: int = 2,
    include_judge_metadata: bool = False,
) -> Dict[str, Any]:
    """Generate captions for all requested styles concurrently, optionally passing
    each one through an LLM judge + refinement loop before it's accepted.

    Each style is an independent network call to the model, so they are
    fired off in parallel threads instead of sequentially. This turns
    (N styles * per-call latency) into roughly one call's worth of wall time.

    When enable_judge=True, judge_client/judge_model and refine_client/refine_model
    default to `client`/`model` (the caption-generation ones) if not supplied, but
    can be set independently to use different models for drafting vs. judging vs.
    refining.

    Returns {style: caption_str, ...} normally, or, if include_judge_metadata=True,
    {style: {"caption": str, "passed": bool, "refine_iterations": int, ...}, ...}.
    """
    scene_text = format_scene_data(json_data)
    valid_styles = []
    for style in styles:
        if style not in STYLE_PROMPTS:
            print(f"[caption] unknown style '{style}', skipping")
            continue
        valid_styles.append(style)

    captions: Dict[str, Any] = {}
    if not valid_styles:
        return captions

    j_client = judge_client or client
    j_model = judge_model or model
    r_client = refine_client or client
    r_model = refine_model or model

    with ThreadPoolExecutor(max_workers=len(valid_styles)) as executor:
        future_to_style = {
            executor.submit(
                generate_and_judge_caption,
                style,
                scene_text,
                client,
                model,
                j_client,
                j_model,
                r_client,
                r_model,
                max_retries,
                enable_judge,
                max_refine_iterations,
            ): style
            for style in valid_styles
        }
        for future in as_completed(future_to_style):
            style = future_to_style[future]
            try:
                outcome = future.result()
                captions[style] = outcome if include_judge_metadata else outcome["caption"]
            except Exception as e:
                # generate_caption/generate_and_judge_caption already retry
                # internally and return a fallback string on failure, so this
                # is a last-resort guard.
                print(f"[caption:{style}] unexpected error in thread: {e}")
                fallback = f"[Fallback] {style.replace('_', ' ').title()} caption for the described scene."
                captions[style] = (
                    {"caption": fallback, "judged": False, "passed": None, "refine_iterations": 0, "attempts": []}
                    if include_judge_metadata
                    else fallback
                )

    return captions
