import json
import re
import time
from typing import Dict, Any, List

from openai import OpenAI

# --------------------------------------------------------------------------- #
# Design note:
#
# A judge that hands back four separate 0-1 scores (accuracy, style_match,
# grounding, hallucination_severity) introduces a lot of call-to-call
# variance for very little payoff -- in practice all four move together, and
# the "pass" decision was just a threshold on top of them anyway. Simplifying
# to a single PASS/FAIL verdict removes that variance and makes the judge
# cheaper and more consistent, while still requiring it to justify itself
# with concrete feedback tied to the scene.
#
# The judge is also deliberately biased toward accuracy/style-fit over "is
# this joke funny enough". A caption that is funny but says something the
# scene doesn't support should FAIL; a caption that is accurate, on-style,
# and only mildly clever should PASS. The goal is captions that are hard for
# a strict grader to call inaccurate -- not captions optimized to be the
# funniest possible take on the scene.
# --------------------------------------------------------------------------- #

STYLE_CHECKLIST = {
    "formal": "clear, professional, objective, factual, concise. No jokes, no slang.",
    "sarcastic": "dry irony / light mockery, while remaining factually accurate about the scene.",
    "humorous_tech": (
        "tech/programming-flavored humor (buffering, algorithm, CPU, rendering, etc.) that "
        "arises naturally from what's actually visible -- not generic programmer jokes or "
        "buzzwords bolted onto an unrelated scene."
    ),
    "humorous_non_tech": (
        "everyday, non-technical humor or puns that relate directly to something visible in "
        "the scene. No programming/tech jargon."
    ),
}

JUDGE_SYSTEM_PROMPT = f"""You are a strict grader for AI-generated video captions. You are given
structured scene data extracted from a video, the style the caption was supposed to follow, and
the caption itself. Decide PASS or FAIL.

Style requirements:
- formal: {STYLE_CHECKLIST['formal']}
- sarcastic: {STYLE_CHECKLIST['sarcastic']}
- humorous_tech: {STYLE_CHECKLIST['humorous_tech']}
- humorous_non_tech: {STYLE_CHECKLIST['humorous_non_tech']}

FAIL the caption if ANY of the following are true:
- it mentions an object, action, or event that is not in the scene data (hallucination)
- it omits a major object or action that defines the scene
- it is generic enough to describe many unrelated scenes instead of this one
- it does not match the requested style's requirements above
- a joke/metaphor is used but is NOT grounded in something actually visible in the scene
  (e.g. "the ocean is rendering GPU shaders" invents a concept; "the waves crash like a failed
  deployment" is fine because it's a metaphor hung on a real, visible action)

Otherwise PASS. Being funnier, more clever, or more elaborate is NOT a reason to fail a caption
that is already accurate and on-style -- do not penalize a caption just because a punchier version
is imaginable. Prioritize catching anything a careful fact-checker could point to as wrong or
off-style over rewarding extra wit.

Respond with ONLY a raw JSON object of exactly this form, no markdown, no code fences, no extra
commentary:
{{"verdict": "PASS" or "FAIL", "feedback": "<if FAIL: one or two sentences naming the specific problem and what to fix; if PASS: empty string>"}}
"""


def _extract_json(raw_text: str) -> Dict[str, Any]:
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

    if not isinstance(data, dict):
        raise ValueError("Judge response is not a JSON object")
    return data


def build_judge_user_prompt(style: str, scene_text: str, caption: str) -> str:
    return f"""Requested style: {style}

Scene data:
{scene_text}

Caption to evaluate:
\"\"\"{caption}\"\"\"

Respond with the JSON verdict only.
{{"verdict": "PASS" or "FAIL", "feedback": "<if FAIL: feedback; if PASS: empty string>"}}
"""


def evaluate_caption(
    style: str,
    scene_text: str,
    caption: str,
    client: OpenAI,
    model: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Calls the judge model and returns {"passed": bool, "feedback": str}.
    On repeated judge failure, fails open (passed=True) so a flaky judge
    never blocks the pipeline -- it just skips refinement for that caption.
    """
    user_prompt = build_judge_user_prompt(style, scene_text, caption)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
                timeout=30,
            )
            if response and response.choices:
                content = response.choices[0].message.content
                if content and content.strip():
                    data = _extract_json(content)
                    verdict = str(data.get("verdict", "")).strip().upper()
                    feedback = str(data.get("feedback", "") or "").strip()
                    if verdict in ("PASS", "FAIL"):
                        return {"passed": verdict == "PASS", "feedback": feedback}
            print(f"[judge:{style}] attempt {attempt}/{max_retries}: empty/invalid response")

        except Exception as e:
            last_error = e
            print(f"[judge:{style}] attempt {attempt}/{max_retries} error: {e}")

        if attempt < max_retries:
            time.sleep(min(0.01 * attempt, 10))

    print(f"[judge:{style}] all {max_retries} attempts failed, failing open. Last error: {last_error}")
    return {"passed": True, "feedback": ""}


REFINE_SYSTEM_PROMPT = (
    "You are a video caption writer revising a caption based on judge feedback. You will be "
    "shown every previous attempt for this clip and exactly why each one failed. Produce a NEW "
    "caption that fixes all of those problems at once -- do not just patch the most recent "
    "attempt, and do not reintroduce a problem an earlier attempt already had. Stay strictly "
    "true to the given scene details and match the requested tone. "
    "Maintain the length of the caption between 25 and 60 words. "
    'Respond with ONLY a raw JSON object of the form {"caption": "your caption here"}. '
    "No markdown, no code fences, no extra commentary."
)


def _format_attempt_history(attempts: List[Dict[str, str]]) -> str:
    lines = []
    for i, a in enumerate(attempts, 1):
        lines.append(f'Attempt {i}: "{a["caption"]}"')
        lines.append(f"  -> judge feedback: {a['feedback'] or '(none)'}")
    return "\n".join(lines)


def build_refine_user_prompt(
    style: str, style_instruction: str, scene_text: str, attempts: List[Dict[str, str]]
) -> str:
    return f"""{style_instruction}

Scene details:
{scene_text}

Every previous attempt for this caption, in order, with the judge's feedback on each:
{_format_attempt_history(attempts)}

Write a new caption that addresses all of the feedback above without repeating any of the
previous attempts' mistakes. Respond with JSON: {{"caption": "..."}}"""


def refine_caption(
    style: str,
    style_instruction: str,
    scene_text: str,
    attempts: List[Dict[str, str]],
    client: OpenAI,
    model: str,
    max_retries: int = 5,
) -> str:
    """Calls the refinement model with the FULL history of prior attempts + their judge
    feedback (not just the latest one), so it can't oscillate between the same couple of
    mistakes. Falls back to the most recent caption if refinement fails entirely."""
    from caption_generator import _extract_caption_json  # local import avoids a cycle at module load

    user_prompt = build_refine_user_prompt(style, style_instruction, scene_text, attempts)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_tokens=2000,
                timeout=30,
            )
            if response and response.choices:
                content = response.choices[0].message.content
                if content and content.strip():
                    return _extract_caption_json(content)
            print(f"[refine:{style}] attempt {attempt}/{max_retries}: empty/invalid response")

        except Exception as e:
            last_error = e
            print(f"[refine:{style}] attempt {attempt}/{max_retries} error: {e}")

        if attempt < max_retries:
            time.sleep(min(0.01 * attempt, 10))

    print(f"[refine:{style}] all {max_retries} attempts failed, keeping prior caption. Last error: {last_error}")
    return attempts[-1]["caption"]


SELECTOR_SYSTEM_PROMPT = (
    "You are picking the best of several candidate video captions, none of which fully passed "
    "review. You will see each candidate and the judge feedback it received. Pick the one that "
    "is the most accurate and least hallucinated relative to the scene data, and is closest to "
    "matching the requested style -- accuracy and grounding matter more than being the most "
    "stylish. "
    'Respond with ONLY a raw JSON object of the form {"best_attempt": <integer, 1-based index>}. '
    "No markdown, no code fences, no extra commentary."
)


def build_selector_user_prompt(
    style: str, scene_text: str, attempts: List[Dict[str, str]]
) -> str:
    return f"""Requested style: {style}

Scene data:
{scene_text}

Candidates:
{_format_attempt_history(attempts)}

Respond with the JSON object naming the best attempt only."""


def select_best_caption(
    style: str,
    scene_text: str,
    attempts: List[Dict[str, str]],
    client: OpenAI,
    model: str,
    max_retries: int = 5,
) -> str:
    """
    Called when the refinement budget is exhausted and no attempt has passed.
    Rather than assuming the LAST attempt is the best one (refinement can make
    a caption worse on some axis while fixing another), asks the judge model
    to compare every attempt against the scene data and pick the strongest.
    Falls back to the first attempt (the original, unrefined draft) if the
    selector call fails, since that's the least likely to have drifted.
    """
    if len(attempts) == 1:
        return attempts[0]["caption"]

    user_prompt = build_selector_user_prompt(style, scene_text, attempts)

    last_error = None
    for attempt_num in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1500,
                timeout=30,
            )
            if response and response.choices:
                content = response.choices[0].message.content
                if content and content.strip():
                    data = _extract_json(content)
                    idx = int(data.get("best_attempt"))
                    if 1 <= idx <= len(attempts):
                        return attempts[idx - 1]["caption"]
            print(f"[selector:{style}] attempt {attempt_num}/{max_retries}: empty/invalid response")

        except Exception as e:
            last_error = e
            print(f"[selector:{style}] attempt {attempt_num}/{max_retries} error: {e}")

        if attempt_num < max_retries:
            time.sleep(min(0.01 * attempt_num, 10))

    print(f"[selector:{style}] all {max_retries} attempts failed, defaulting to first draft. "
          f"Last error: {last_error}")
    return attempts[0]["caption"]