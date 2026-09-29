"""
Satisfaction and engagement labels for CHARM's extra dimensions, from ReDial.

ReDial asked both workers, for every movie in a conversation, whether it was
suggested, whether the seeker had seen it and whether the seeker liked it
(0 no, 1 yes, 2 did not say). **Satisfaction** is the seeker's own "liked"
answer, or the recommender's answer about the seeker when the seeker gave none.

**Engagement** is read from the dialogue: after the recommender first suggests
a movie, does the seeker take it up before the next suggestion -- mention it
again, ask about it straight away, or say they will watch it -- without turning
it down? Both are training targets only; CHARM never sees them as inputs.
"""

import re
from typing import Dict, List, Optional

MENTION = re.compile(r"@(\d+)")
_ACCEPT = re.compile(
    r"\b(will (watch|check|try|see|look)|i'?ll (watch|check|try|see|look|add|give)|"
    r"check (it|that|them|those|this) out|sounds? (good|great|interesting|fun|awesome|cool|amazing)|"
    r"add (it|that|them|this)|have to (see|watch|check)|want to (see|watch)|"
    r"(great|good|nice) (idea|suggestion|choice|pick|one)|love (that|it|this)|"
    r"thanks?|thank you)\b", re.I)
_REJECT = re.compile(
    r"\b(not (really )?(interested|a fan|into|for me)|don'?t (like|want|care|enjoy)|"
    r"didn'?t (like|enjoy|care)|no thanks|not my (thing|type|style)|"
    r"anything else|something else|other (suggestions|ideas|options))\b", re.I)


def questionnaire(conv: Dict, movie_id: str, key: str) -> Optional[int]:
    """The seeker's 0/1 answer to ``key`` for a movie (recommender's as fallback)."""
    for field in ("initiatorQuestions", "respondentQuestions"):
        answers = conv.get(field)
        if isinstance(answers, dict):
            v = (answers.get(str(movie_id)) or {}).get(key)
            if v in (0, 1):
                return int(v)
    return None


def engaged_after(messages: List[Dict], i: int, movie_id: str, seeker) -> Optional[int]:
    """1/0 for the seeker taking up the movie suggested in message ``i``; None
    when the seeker says nothing before the next suggestion."""
    window = []
    for m in messages[i + 1:]:
        if m.get("senderWorkerId") != seeker:
            if any(x != movie_id for x in MENTION.findall(m["text"])):
                break                               # the recommender moved on
            continue
        window.append(m["text"])
    if not window:
        return None
    text = " ".join(window)
    if _REJECT.search(text):
        return 0
    if f"@{movie_id}" in text or "?" in window[0] or _ACCEPT.search(text):
        return 1
    return 0


def conversation_labels(conv: Dict) -> Dict[str, Dict[str, Optional[int]]]:
    """``{title: {"liked", "engaged"}}`` for every movie the recommender suggests."""
    seeker = conv.get("initiatorWorkerId")
    names = conv.get("movieMentions") if isinstance(conv.get("movieMentions"), dict) else {}
    messages = [m for m in conv.get("messages", [])
                if isinstance(m, dict) and str(m.get("text", "")).strip()]
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for i, m in enumerate(messages):
        if m.get("senderWorkerId") == seeker:
            continue
        for mid in MENTION.findall(m["text"]):
            title = names.get(mid) or names.get(str(mid))
            if not title or title in out:
                continue
            out[title] = {"liked": questionnaire(conv, mid, "liked"),
                          "engaged": engaged_after(messages, i, mid, seeker)}
    return out
