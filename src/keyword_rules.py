"""
Epic 3, Story 1: Text Preprocessing & Keyword Enhancement.

Deep models sometimes miss short, obvious emotional cues (e.g. "I hate this bug").
This module scores each emotion by counting explicit keyword matches, applies a
10x weight per match (since a keyword match is much stronger evidence than the
base model's per-class probability), adds that to the model's output, and then
renormalizes so probabilities sum back to 1.
"""
import re
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMOTION_LABELS

KEYWORD_LEXICON = {
    "Bored": ["bored", "boring", "dull", "tedious", "same thing", "dragging", "monotonous", "uninterested"],
    "Confident": ["i've got this", "i got this", "i understand", "confident", "ready", "i can do this",
                  "makes sense now", "i know this", "aced", "nailed it"],
    "Confused": ["confused", "don't understand", "dont understand", "lost", "no idea", "makes no sense",
                 "doesn't make sense", "unclear", "what does this mean", "stuck understanding"],
    "Curious": ["curious", "wonder", "interesting", "how does", "why does", "want to know more",
                 "intrigued", "fascinating", "what if"],
    "Frustrated": ["frustrated", "annoyed", "ugh", "so annoying", "keep failing", "still broken",
                    "losing my patience", "sick of this", "hate this bug", "doesn't work"],
}

KEYWORD_UNIT = 0.01    # base "evidence unit" per keyword match
KEYWORD_WEIGHT = 10    # explicit keywords count for 10x the base unit -> 0.10 boost per match


def _keyword_scores(text: str) -> dict:
    """Counts keyword matches per emotion (each distinct phrase in the lexicon can match once)."""
    text_lower = text.lower()
    scores = {emotion: 0 for emotion in EMOTION_LABELS}
    for emotion, keywords in KEYWORD_LEXICON.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                scores[emotion] += 1
    return scores


def apply_keyword_boost(text: str, probs: dict) -> dict:
    """
    probs: dict {emotion: probability}, must contain all EMOTION_LABELS.
    Returns an adjusted, renormalized probability dict after applying
    the 10x keyword-weighted boost.
    """
    scores = _keyword_scores(text)
    adjusted = dict(probs)

    for emotion, match_count in scores.items():
        if match_count > 0:
            boost = KEYWORD_WEIGHT * KEYWORD_UNIT * match_count
            adjusted[emotion] = adjusted.get(emotion, 0) + boost

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted


if __name__ == "__main__":
    sample_probs = {e: 1 / len(EMOTION_LABELS) for e in EMOTION_LABELS}
    print(apply_keyword_boost("I'm so lost, this makes no sense and I keep failing", sample_probs))
