# processing.py
import os
import tempfile
import shutil
import subprocess
from typing import Tuple, Optional
import requests
import yt_dlp
from groq import Groq, APIStatusError

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# each chunk length for transcription (seconds)
CHUNK_DURATION_SEC = 50
# chunk filename template
CHUNK_FILENAME = "chunk_%03d.wav"


def _run_cmd(cmd: list, check=True):
    """Run a command and raise a helpful error if it fails."""
    try:
        subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nexit={e.returncode}\nstderr={e.stderr.decode(errors='ignore')}")


def _get_duration_seconds(path: str) -> float:
    """Return duration in seconds using ffprobe (part of ffmpeg)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode().strip()
    try:
        return float(out)
    except Exception:
        raise RuntimeError(f"Could not determine duration for file: {path}")


def download_video(url: str, max_duration_sec: Optional[int] = None) -> Tuple[str, str]:
    """
    Download video with yt-dlp (fallback to requests for direct mp4).
    Returns (video_path, tmp_dir).
    """
    tmp_dir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get("duration")
            if max_duration_sec is not None and duration is not None and duration > max_duration_sec:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(f"Video duration {duration}s exceeds limit of {max_duration_sec}s.")

            downloaded_path = ydl.prepare_filename(info)
            # ensure mp4 extension if merged
            if not downloaded_path.lower().endswith(".mp4"):
                alt = os.path.splitext(downloaded_path)[0] + ".mp4"
                if os.path.exists(alt):
                    downloaded_path = alt

            if not os.path.exists(downloaded_path):
                files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
                if not files:
                    raise RuntimeError("yt-dlp produced no files.")
                downloaded_path = max(files, key=os.path.getsize)

            return downloaded_path, tmp_dir

    except Exception as e_ydl:
        # fallback for raw mp4 urls
        try:
            out_path = os.path.join(tmp_dir, "video.mp4")
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            if os.path.getsize(out_path) < 1024:
                raise RuntimeError("Downloaded file is too small; probably not a valid media file.")
            return out_path, tmp_dir
        except Exception as e_requests:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Download failed. yt-dlp error: {e_ydl}; requests fallback error: {e_requests}")


def extract_audio(video_path: str) -> Tuple[str, float, str]:
    """
    Use ffmpeg to extract a 16kHz mono WAV, normalize and boost volume.
    Returns (clean_wav_path, duration_seconds, tmp_dir_for_audio).
    """
    tmp_dir = tempfile.mkdtemp()
    raw_wav = os.path.join(tmp_dir, "raw.wav")
    clean_wav = os.path.join(tmp_dir, "clean.wav")

    # Extract wav (16kHz mono, signed 16-bit)
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

    # get duration from ffprobe
    duration_sec = _get_duration_seconds(raw_wav)

    # quick silence check using ffmpeg's astats to get mean level OR use ffmpeg to compute max volume
    # here we use ffmpeg -i raw.wav -af volumedetect to get max_volume (safer for hosted env)
    probe_cmd = [
        "ffmpeg", "-i", raw_wav, "-af", "volumedetect",
        "-f", "null", "-"
    ]
    proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr.decode(errors="ignore")
    # look for "max_volume: -xx.xx dB"
    max_vol_line = None
    for line in stderr.splitlines():
        if "max_volume:" in line:
            max_vol_line = line.strip()
            break
    if max_vol_line:
        try:
            max_db = float(max_vol_line.split("max_volume:")[1].strip().split(" ")[0])
        except Exception:
            max_db = None
    else:
        max_db = None

    # if audio is extremely quiet (max_db very negative), raise
    if max_db is not None and max_db < -45:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Audio seems silent or extremely quiet. Please provide a video with clearer audio.")

    # Normalize + boost using ffmpeg's loudnorm & volume (loudnorm gives good normalization)
    # First apply loudnorm then apply small gain (+6 dB)
    cmd_norm = [
        "ffmpeg", "-y",
        "-i", raw_wav,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,volume=6dB",
        clean_wav
    ]
    _run_cmd(cmd_norm)

    return clean_wav, duration_sec, tmp_dir


def transcribe_audio(audio_path: str) -> str:
    """
    Split WAV into chunks using ffmpeg 'segment' and transcribe each chunk with Groq.
    Removes chunk files after processing.
    Returns concatenated transcript.
    """
    audio_tmp_dir = tempfile.mkdtemp()
    # build ffmpeg segment command
    # -force_key_frames isn't needed for WAV; segment muxer will split by time
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

    # get list of chunk files in sorted order
    chunks = sorted([os.path.join(audio_tmp_dir, f) for f in os.listdir(audio_tmp_dir) if f.endswith(".wav")])
    if not chunks:
        shutil.rmtree(audio_tmp_dir, ignore_errors=True)
        raise RuntimeError("No audio chunks created by ffmpeg.")

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
            # cleanup chunk dir and raise
            shutil.rmtree(audio_tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Transcription failed for chunk {chunk_path}: {e}")

        transcript_parts.append(str(part).strip())
        # delete chunk immediately
        try:
            os.remove(chunk_path)
        except Exception:
            pass

    # final cleanup for chunk directory
    shutil.rmtree(audio_tmp_dir, ignore_errors=True)

    return " ".join(p for p in transcript_parts if p)


def cleanup_dirs(*dirs: str) -> None:
    """Remove any temporary directories (ignore errors)."""
    for d in dirs:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
