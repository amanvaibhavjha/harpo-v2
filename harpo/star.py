"""
Preference readings: one sentence on what the seeker wants right now, written
by the instruct LLM from the dialogue alone (never the reply that names the
answer). CHARM reads the greedy reading appended to the dialogue; STAR's search
(star_search.py) proposes alternative, more specific readings and lets CHARM
judge them.
"""

import hashlib
import re
from typing import Dict, Optional

SYSTEM = "You are an expert movie recommender talking with a user."
_DOMAIN_TAG = re.compile(r"<\|domain:[a-z]+\|>")

READING_TASK = ("In one short sentence, describe what kind of movie the seeker wants right now: "
                "genres, era, mood, and movies they liked or disliked. Do not suggest a new title.")
REFINE_TASK = ("One reading of what the seeker wants is: \"{reading}\"\n"
               "Give a different, more specific reading in one short sentence -- another genre, "
               "mood or detail the conversation supports. Do not suggest a new title.")


def clean_dialogue(text: str, max_chars: int = 2000) -> str:
    """Drop training-format tags and keep the most recent turns."""
    text = _DOMAIN_TAG.sub("", text).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
        cut = text.find("\n")
        if 0 <= cut < 200:
            text = text[cut + 1:]
    return text


def dialogue_key(dialogue: str) -> str:
    """Stable key for a dialogue (reading files are keyed by it)."""
    return hashlib.md5(dialogue.encode()).hexdigest()


def reading_messages(dialogue: str, parent: Optional[str] = None):
    """Chat messages asking for a one-sentence reading of the seeker's wishes,
    or, given a ``parent`` reading, a different and more specific one."""
    task = READING_TASK if parent is None else REFINE_TASK.format(reading=parent)
    user = f"Conversation so far:\n{clean_dialogue(dialogue)}\n\n{task}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def clean_reading(text: str, max_words: int = 40) -> str:
    """First non-empty line, without quotes or labels, at most ``max_words`` words."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    line = lines[0] if lines else ""
    line = re.sub(r"^(reading|answer|summary)\s*:\s*", "", line, flags=re.I).strip(' "')
    return " ".join(line.split()[:max_words])


def load_readings(paths: Optional[str]) -> Dict[str, str]:
    """Merge comma-separated reading files (``{dialogue_key: reading}``)."""
    import json
    out: Dict[str, str] = {}
    for p in (paths or "").split(","):
        if p:
            with open(p) as f:
                out.update(json.load(f))
    return out
