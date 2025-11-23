# processing.py
import os
import tempfile
import shutil
import subprocess
from typing import Tuple, Optional, List
import requests

from groq import Groq, APIStatusError

# Groq client will read API key from environment variable GROQ_API_KEY
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# seconds per chunk for transcription
CHUNK_DURATION_SEC = 50
CHUNK_FILENAME = "chunk_%03d.wav"


def _run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nexit={proc.returncode}\nstderr={stderr}")
    return proc


def _get_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    proc = _run_cmd(cmd)
    out = proc.stdout.decode().strip()
    try:
        return float(out)
    except Exception:
        raise RuntimeError(f"Could not determine duration for file: {path}")


def download_video(url: str, tmp_prefix: Optional[str] = "video_") -> Tuple[str, str]:
    """
    Download a direct MP4 URL into a temp directory.
    Returns (video_path, tmp_dir).
    """
    tmp_dir = tempfile.mkdtemp(prefix=tmp_prefix)
    out_path = os.path.join(tmp_dir, "video.mp4")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        with requests.get(url, stream=True, timeout=(10, 120), headers=headers) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()
            if content_type and ("text/html" in content_type or "application/json" in content_type):
                # Not a direct MP4
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(f"URL does not look like a direct MP4 (Content-Type: {content_type}).")
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to download video: {e}")

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Downloaded file missing or too small; ensure the URL is a direct MP4 link.")

    # quick validation via ffprobe
    try:
        _get_duration_seconds(out_path)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Downloaded file looks invalid to ffprobe: {e}")

    return out_path, tmp_dir


def extract_audio(video_path: str, tmp_prefix: Optional[str] = "audio_") -> Tuple[str, float, str]:
    """
    Extract 16kHz mono WAV from the MP4, normalize and lightly boost.
    Returns (clean_wav_path, duration_sec, tmp_dir).
    """
    tmp_dir = tempfile.mkdtemp(prefix=tmp_prefix)
    raw_wav = os.path.join(tmp_dir, "raw.wav")
    clean_wav = os.path.join(tmp_dir, "clean.wav")

    # extract audio
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        raw_wav
    ]
    _run_cmd(cmd_extract)

    duration_sec = _get_duration_seconds(raw_wav)

    # optional loudness check (using volumedetect)
    probe_cmd = ["ffmpeg", "-i", raw_wav, "-af", "volumedetect", "-f", "null", "-"]
    proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr.decode(errors="ignore")
    max_db = None
    for line in stderr.splitlines():
        if "max_volume:" in line:
            try:
                max_db = float(line.split("max_volume:")[1].strip().split(" ")[0])
            except Exception:
                max_db = None
            break

    if max_db is not None and max_db < -45:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Audio appears extremely quiet (max_volume < -45 dB). Provide a clearer video.")

    # normalize + small boost
    cmd_norm = [
        "ffmpeg", "-y",
        "-i", raw_wav,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,volume=6dB",
        clean_wav
    ]
    _run_cmd(cmd_norm)

    if not os.path.exists(clean_wav) or os.path.getsize(clean_wav) < 1024:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Audio processing failed; normalized file missing or too small.")

    return clean_wav, float(duration_sec), tmp_dir


def transcribe_audio(audio_path: str) -> str:
    """
    Split WAV into chunks and transcribe each chunk via Groq Whisper.
    Returns concatenated transcript string.
    """
    audio_tmp_dir = tempfile.mkdtemp(prefix="chunks_")
    segment_pattern = os.path.join(audio_tmp_dir, CHUNK_FILENAME)

    cmd_segment = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(CHUNK_DURATION_SEC),
        "-ar", "16000",
        "-ac", "1",
        "-c", "pcm_s16le",
        segment_pattern
    ]
    _run_cmd(cmd_segment)

    chunks = sorted([os.path.join(audio_tmp_dir, f) for f in os.listdir(audio_tmp_dir) if f.endswith(".wav")])
    if not chunks:
        shutil.rmtree(audio_tmp_dir, ignore_errors=True)
        raise RuntimeError("No audio chunks were produced by ffmpeg segmentation.")

    transcript_parts = []
    for chunk_path in chunks:
        try:
            with open(chunk_path, "rb") as f:
                part = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="text",
                )
        except APIStatusError as e:
            shutil.rmtree(audio_tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Transcription failed for chunk {chunk_path}: {e}")

        transcript_parts.append(str(part).strip())
        try:
            os.remove(chunk_path)
        except Exception:
            pass

    shutil.rmtree(audio_tmp_dir, ignore_errors=True)
    return " ".join(p for p in transcript_parts if p).strip()


def cleanup_dirs(*dirs: Optional[str]) -> None:
    """Remove provided temporary directories."""
    for d in dirs:
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
