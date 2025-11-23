import os
import tempfile
import requests
from moviepy import VideoFileClip
from pydub import AudioSegment, effects
from groq import Groq, APIStatusError

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHUNK_DURATION_MS = 50_000


def download_video(url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "video.mp4")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return out_path


def extract_audio(video_path: str) -> tuple[str, float]:
    tmp_dir = tempfile.mkdtemp()
    raw_audio = os.path.join(tmp_dir, "raw.wav")
    clean_audio = os.path.join(tmp_dir, "clean.wav")

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(raw_audio, fps=16000, codec="pcm_s16le")
    duration_sec = video.duration

    audio = AudioSegment.from_wav(raw_audio)

    if audio.dBFS < -45:
        raise RuntimeError("Audio seems silent or extremely quiet. Please provide a video with clear speech.")

    normalized = effects.normalize(audio)
    boosted = normalized + 8  # +8 dB

    boosted.export(clean_audio, format="wav")
    return clean_audio, float(duration_sec)


def transcribe_audio(audio_path: str) -> str:

    audio = AudioSegment.from_wav(audio_path)
    transcript = ""

    for i in range(0, len(audio), CHUNK_DURATION_MS):
        chunk = audio[i:i + CHUNK_DURATION_MS]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk.export(tmp.name, format="wav")

            with open(tmp.name, "rb") as f:
                try:
                    part = client.audio.transcriptions.create(
                        model="whisper-large-v3-turbo",
                        file=f,
                        response_format="text",
                    )
                except APIStatusError as e:
                    raise RuntimeError(f"Transcription failed: {e}")

        transcript += " " + part

    return transcript.strip()
