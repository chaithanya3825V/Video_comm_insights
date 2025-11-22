import re
from typing import Dict


FILLER_WORDS = [
    "um", "uh", "like", "you know", "actually", "basically",
    "literally", "kinda", "sort of"
]


def clean_transcript(text: str) -> str:
    """
    Remove timestamps, brackets, and weird characters.
    Keep only basic punctuation and words.
    """
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ,.!?'\"\-\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_filler_stats(text: str) -> Dict[str, float]:
    """
    Count filler words and compute their density (per 100 words).
    """
    words = text.lower().split()
    total_words = len(words) if words else 1

    filler_count = 0
    for filler in FILLER_WORDS:
        if " " in filler:
            pattern = re.escape(filler)
            filler_count += len(re.findall(pattern, " ".join(words)))
        else:
            filler_count += sum(1 for w in words if w == filler)

    density = (filler_count / total_words) * 100.0
    return {
        "filler_count": filler_count,
        "filler_density_per_100_words": round(density, 2),
        "total_words": total_words,
    }


def compute_pace_wpm(total_words: int, duration_sec: float) -> float:
    """
    Approximate speaking pace in words per minute.
    """
    if duration_sec <= 0:
        return 0.0
    minutes = duration_sec / 60.0
    return round(total_words / minutes, 2)
