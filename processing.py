# processing.py (updated)
import os
import tempfile
import requests
from moviepy import VideoFileClip
from pydub import AudioSegment, effects
from groq import Groq, APIStatusError

# new import
import yt_dlp

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 50 seconds per chunk => safe for Whisper request size
CHUNK_DURATION_MS = 50_000


def download_video(url: str) -> str:
    """
    Download a video from a public URL.
    - Primary: use yt-dlp which supports YouTube, Vimeo, Loom, many embeds and direct media.
    - Fallback: stream-download with requests for raw MP4 links.
    Returns the path to the downloaded MP4 file.
    """
    tmp_dir = tempfile.mkdtemp()
    # prefer mp4 merged output
    outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # avoid certificate issues in some environments
        "nocheckcertificate": True,
        # retry a few times
        "retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # prepare_filename usually returns the final filename (note: ext may be mp4)
            downloaded_path = ydl.prepare_filename(info)
            # If merge output format produced mp4 file, ensure extension is .mp4
            if not downloaded_path.lower().endswith(".mp4"):
                possible_mp4 = os.path.splitext(downloaded_path)[0] + ".mp4"
                if os.path.exists(possible_mp4):
                    downloaded_path = possible_mp4
            if not os.path.exists(downloaded_path):
                # sometimes yt-dlp writes a file with different id/title; scan tmp_dir for latest file
                files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
                if files:
                    # pick the largest file (likely the video)
                    downloaded_path = max(files, key=os.path.getsize)
            return downloaded_path

    except Exception as e:
        # If yt-dlp fails (rare), try a simple streaming download for direct MP4 links
        # This is useful for raw direct MP4 URLs.
        try:
            out_path = os.path.join(tmp_dir, "video.mp4")
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            # quick sanity check
            if os.path.getsize(out_path) < 1024:  # less than 1KB -> suspect failure
                raise RuntimeError("Downloaded file is too small; probably not a valid media file.")
            return out_path
        except Exception as e2:
            # combine errors for better debugging
            raise RuntimeError(f"Download failed: yt-dlp error: {e}; fallback error: {e2}")


def extract_audio(video_path: str) -> tuple[str, float]:
    """
    Extract audio from video into a boosted, normalized WAV.
    Returns (audio_path, duration_seconds).
    """
    tmp_dir = tempfile.mkdtemp()
    raw_audio = os.path.join(tmp_dir, "raw.wav")
    clean_audio = os.path.join(tmp_dir, "clean.wav")

    video = VideoFileClip(video_path)
    # write_audiofile default uses ffmpeg; ensure ffmpeg is installed on host
    video.audio.write_audiofile(raw_audio, fps=16000, codec="pcm_s16le")
    duration_sec = video.duration

    audio = AudioSegment.from_wav(raw_audio)

    # Detect very quiet / silent audio
    if audio.dBFS < -45:
        raise RuntimeError("Audio seems silent or extremely quiet. Please provide a video with clear speech.")

    # Normalize + lightly boost volume for clarity
    normalized = effects.normalize(audio)
    boosted = normalized + 8  # +8 dB

    boosted.export(clean_audio, format="wav")
    return clean_audio, float(duration_sec)


def transcribe_audio(audio_path: str) -> str:
    """
    Chunk audio and transcribe using Groq Whisper.
    This avoids 413 errors and works for long videos.
    """
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
