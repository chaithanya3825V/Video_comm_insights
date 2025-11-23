# processing.py
import os
import tempfile
import shutil
import subprocess
import time
from typing import Tuple, Optional, List
import requests

# try yt-dlp (preferred) but fall back gracefully if missing
try:
    import yt_dlp as ytdlp
except Exception:
    ytdlp = None

from groq import Groq, APIStatusError

# Groq client (reads api key from env: GROQ_API_KEY)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# chunk length for transcription (seconds)
CHUNK_DURATION_SEC = 50
CHUNK_FILENAME = "chunk_%03d.wav"


# ---------- low-level helpers ----------
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
    except Exception as e:
        raise RuntimeError(f"Could not determine duration for file {path}: {e}")


def _is_valid_mp4(path: str) -> bool:
    try:
        _run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ])
        return True
    except Exception:
        return False


# ---------- downloader ----------
def download_video(url: str, tmp_prefix: Optional[str] = "video_") -> Tuple[str, str]:
    """
    Robust download:
    - Try yt-dlp when available (handles YouTube, Vimeo, embeds).
    - Else use HEAD + streaming requests with retries for direct MP4 links.
    Returns (video_path, tmp_dir)
    """
    tmp_dir = tempfile.mkdtemp(prefix=tmp_prefix)

    # 1) prefer yt-dlp if available
    if ytdlp:
        outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "nocheckcertificate": True,
            "retries": 3,
        }
        try:
            with ytdlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_path = ydl.prepare_filename(info)
                # prefer .mp4 if merge created mp4
                if not downloaded_path.lower().endswith(".mp4"):
                    alt = os.path.splitext(downloaded_path)[0] + ".mp4"
                    if os.path.exists(alt):
                        downloaded_path = alt
                if not os.path.exists(downloaded_path):
                    files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
                    if files:
                        downloaded_path = max(files, key=os.path.getsize)
                if _is_valid_mp4(downloaded_path):
                    return downloaded_path, tmp_dir
                # else fall through to requests fallback
        except Exception:
            # fall back to requests if yt-dlp fails
            pass

    # 2) HEAD check for direct-MP4 like links
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        head = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
    except Exception:
        head = None

    # 3) streaming download with retries
    out_path = os.path.join(tmp_dir, "video.mp4")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(10, 120), headers=headers, allow_redirects=True) as r:
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "").lower()
                # if server returns HTML, this likely isn't a direct MP4
                if content_type and ("text/html" in content_type or "application/json" in content_type):
                    raise RuntimeError(f"URL does not appear to be a direct media file (Content-Type: {content_type}).")
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                raise RuntimeError("Downloaded file too small or missing after streaming.")

            if _is_valid_mp4(out_path):
                return out_path, tmp_dir
            else:
                # truncated or missing moov atom — remove and retry
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                if attempt < max_attempts:
                    time.sleep(1)
                    continue
                break

        except Exception:
            if attempt < max_attempts:
                time.sleep(1)
                continue
            break

    # 4) final attempt: if yt-dlp exists, try it one more time
    if ytdlp:
        try:
            outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
            with ytdlp.YoutubeDL({"outtmpl": outtmpl, "format": "bestvideo+bestaudio/best", "merge_output_format": "mp4", "noplaylist": True, "quiet": True}) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_path = ydl.prepare_filename(info)
                if not downloaded_path.lower().endswith(".mp4"):
                    alt = os.path.splitext(downloaded_path)[0] + ".mp4"
                    if os.path.exists(alt):
                        downloaded_path = alt
                if os.path.exists(downloaded_path) and _is_valid_mp4(downloaded_path):
                    return downloaded_path, tmp_dir
        except Exception:
            pass

    # nothing worked
    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError(
        "Unable to download a valid MP4 from the provided URL. "
        "Possible causes: the URL points to a webpage (not a raw MP4), the file was truncated, "
        "or the host requires authentication. Provide a direct MP4 URL or a YouTube link."
    )


# ---------- audio extraction ----------
def extract_audio(video_path: str, tmp_prefix: Optional[str] = "audio_") -> Tuple[str, float, str]:
    """
    Extract a normalized WAV (16kHz mono) suitable for transcription.
    Returns (clean_wav_path, duration_seconds, tmp_dir)
    """
    tmp_dir = tempfile.mkdtemp(prefix=tmp_prefix)
    raw_wav = os.path.join(tmp_dir, "raw.wav")
    clean_wav = os.path.join(tmp_dir, "clean.wav")

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

    # check loudness (volumedetect)
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

    # normalize and boost slightly
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


# ---------- transcription ----------
def transcribe_audio(audio_path: str) -> str:
    """
    Split WAV into chunks and transcribe each chunk with Groq Whisper.
    Deletes chunk files and returns full transcript.
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


# ---------- cleanup helper ----------
def cleanup_dirs(*dirs: Optional[str]) -> None:
    """Remove temporary directories safely (ignore errors)."""
    for d in dirs:
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
