# processing.py
import os
import tempfile
import shutil
import subprocess
from typing import Tuple
import requests
from groq import Groq, APIStatusError

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# chunk length used for transcription (seconds)
CHUNK_DURATION_SEC = 50
# output chunk filename template (ffmpeg segment will create chunk_000.wav, chunk_001.wav, ...)
CHUNK_FILENAME = "chunk_%03d.wav"


def _run_cmd(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run subprocess and raise helpful error on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nexit={proc.returncode}\nstderr={stderr}")
    return proc


def _get_duration_seconds(path: str) -> float:
    """Return duration using ffprobe."""
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


def download_video(url: str) -> str:
    """
    Download a direct MP4/video from a public URL to a temp file.
    Returns path to downloaded video (MP4).
    """
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "video.mp4")

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # basic sanity check
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Downloaded file is missing or too small; ensure the URL is a direct public MP4 link.")

    # return the full path (caller may want to keep tmp_dir if cleanup needed)
    return out_path, tmp_dir


def extract_audio(video_path: str) -> Tuple[str, float]:
    """
    Extract 16kHz mono WAV from video, run loudnorm + small boost, and return (clean_wav_path, duration_seconds).
    Uses ffmpeg/ffprobe; no pydub/moviepy required.
    """
    tmp_dir = tempfile.mkdtemp()
    raw_wav = os.path.join(tmp_dir, "raw.wav")
    clean_wav = os.path.join(tmp_dir, "clean.wav")

    # extract audio to WAV (16kHz mono s16)
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

    # determine duration (seconds)
    duration_sec = _get_duration_seconds(raw_wav)

    # detect max volume using ffmpeg volumedetect (parses stderr)
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

    # normalize and lightly boost using ffmpeg loudnorm and a small volume gain
    cmd_norm = [
        "ffmpeg", "-y",
        "-i", raw_wav,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,volume=6dB",
        clean_wav
    ]
    _run_cmd(cmd_norm)

    # simple check
    if not os.path.exists(clean_wav) or os.path.getsize(clean_wav) < 1024:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Audio processing failed; normalized file missing or too small.")

    # Return the clean audio path and duration (caller may choose to cleanup the temp dir)
    return clean_audio, float(duration_sec), tmp_dir


def transcribe_audio(audio_path: str) -> str:
    """
    Split WAV into CHUNK_DURATION_SEC segments (ffmpeg) and transcribe each chunk with Groq Whisper.
    Removes chunk files after processing and deletes chunk temp dir.
    Returns concatenated transcript string.
    """
    audio_tmp_dir = tempfile.mkdtemp()
    segment_pattern = os.path.join(audio_tmp_dir, CHUNK_FILENAME)

    # split into chunks; ensures 16000Hz mono pcm chunks
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

    # collect chunk files sorted
    chunks = sorted(
        [os.path.join(audio_tmp_dir, f) for f in os.listdir(audio_tmp_dir) if f.endswith(".wav")]
    )
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

        # delete chunk once transcribed
        try:
            os.remove(chunk_path)
        except Exception:
            pass

    # cleanup chunk folder
    shutil.rmtree(audio_tmp_dir, ignore_errors=True)

    # combine and return
    return " ".join(p for p in transcript_parts if p).strip()
    def cleanup_dirs(*dirs):
    import shutil
    for d in dirs:
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
