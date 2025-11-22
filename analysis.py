import json
import os
import re
from typing import Dict, Any

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_transcript(
    text: str,
    filler_stats: Dict[str, float],
    pace_wpm: float,
) -> Dict[str, Any]:
    """
    Use Groq LLM to analyze the transcript and generate:
    - clarity_score (0-100)
    - communication_focus (one sentence)
    - summary (3–4 sentences)
    - sentiment (string)
    - overall_tone (string)
    - coaching_tips (list of short sentences)
    """
    total_words = filler_stats.get("total_words", 0)

    if total_words < 10:
        # Not enough speech; safe default
        return {
            "clarity_score": 25,
            "communication_focus": "The audio contains too little speech to analyze.",
            "summary": "There was not enough spoken content to generate a detailed summary.",
            "sentiment": "unknown",
            "overall_tone": "unclear",
            "coaching_tips": [
                "Speak more so there is enough content to analyze.",
                "Ensure your microphone is working and background noise is minimal.",
            ],
        }

    prompt = f"""
You are an expert communication coach.

You are given:
1) A transcript of what was said.
2) Automatically computed stats:
   - total_words: {total_words}
   - filler_count: {filler_stats.get("filler_count")}
   - filler_density_per_100_words: {filler_stats.get("filler_density_per_100_words")}
   - pace_wpm: {pace_wpm}

TRANSCRIPT:
{text}

Analyze the speaker's communication and return STRICT JSON ONLY in this exact format:

{{
  "clarity_score": number (0-100),
  "communication_focus": "one concise sentence describing the main topic",
  "summary": "3-4 sentence summary of the speech",
  "sentiment": "positive" | "neutral" | "negative" | "mixed",
  "overall_tone": "one or two words (e.g. 'confident', 'nervous', 'enthusiastic')",
  "coaching_tips": [
    "short, specific suggestion 1",
    "short, specific suggestion 2",
    "short, specific suggestion 3"
  ]
}}

Rules:
- You MUST output valid JSON only. No extra commentary.
- Take into account fillers and pace: too many fillers and very high or very low pace should reduce clarity_score.
- If the transcript is messy or chaotic, still do your best to infer and return valid JSON.
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
    )

    raw = resp.choices[0].message.content.strip()

    # Try direct JSON parse
    try:
        return json.loads(raw)
    except Exception:
        # Try to extract JSON chunk
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    # Final fallback to avoid UI crash
    return {
        "clarity_score": 40,
        "communication_focus": "The transcript seems mixed or noisy, but a general theme is present.",
        "summary": "The speech appears somewhat disorganized or noisy, making detailed analysis difficult.",
        "sentiment": "mixed",
        "overall_tone": "unclear",
        "coaching_tips": [
            "Try to reduce background noise when speaking.",
            "Organize your thoughts into a clearer structure with a beginning, middle, and end.",
            "Focus on speaking at a steady pace and reducing filler words."
        ],
    }
