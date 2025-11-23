# app.py
import os
import traceback
import streamlit as st
from processing import (
    download_video,
    extract_audio,
    transcribe_audio,
    cleanup_dirs,
)
from text_features import clean_transcript, compute_filler_stats, compute_pace_wpm
from analysis import analyze_transcript

st.set_page_config(page_title="Video Communication Insights", layout="wide", page_icon="🎥")
st.title("🎥 Video Communication Insights")

# quick environment checks
def _check_env():
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY (set in Streamlit secrets or env)")
    if not shutil_which("ffmpeg"):
        missing.append("ffmpeg (system binary on PATH)")
    return missing

def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)

missing = _check_env()
if missing:
    for m in missing:
        st.warning(f"⚠️ Missing: {m}")
    st.write("Set the missing items and refresh. (Streamlit Secrets for API keys; packages.txt for ffmpeg in Streamlit Cloud)")

st.subheader("Enter Video URL (YouTube or direct MP4)")
url = st.text_input("Video URL:", placeholder="https://...")

analyze = st.button("🚀 Analyze Video")

def show_trace(e):
    with st.expander("Traceback"):
        st.code(traceback.format_exc())

if analyze:
    if not url or not url.strip():
        st.error("Please enter a video URL.")
    else:
        video_tmp_dir = None
        audio_tmp_dir = None
        try:
            with st.spinner("📥 Downloading video..."):
                video_path, video_tmp_dir = download_video(url)

            st.success("Downloaded video.")

            with st.spinner("🎧 Extracting & normalizing audio..."):
                audio_path, duration_sec, audio_tmp_dir = extract_audio(video_path)

            st.success("Audio ready.")

            with st.spinner("🔊 Transcribing audio..."):
                transcript = transcribe_audio(audio_path)

            if not transcript or len(transcript.split()) < 3:
                st.error("Transcription produced very little text. Try a clearer / shorter video.")
            else:
                cleaned = clean_transcript(transcript)
                filler = compute_filler_stats(cleaned)
                pace_wpm = compute_pace_wpm(filler["total_words"], duration_sec)

                with st.spinner("🧠 Running analysis..."):
                    results = analyze_transcript(cleaned, filler, pace_wpm)

                st.subheader("📊 Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Clarity Score", f"{results.get('clarity_score', 'N/A')}%")
                c2.metric("Speaking Pace", f"{round(pace_wpm,1)} WPM")
                c3.metric("Filler Density", f"{filler.get('filler_density_per_100_words','N/A')} per 100 words")

                c4, c5, c6 = st.columns(3)
                c4.metric("Total Words", f"{filler.get('total_words','N/A')}")
                c5.metric("Sentiment", results.get("sentiment","Unknown").capitalize())
                c6.metric("Audio Duration", f"{round(duration_sec,1)} sec")

                st.subheader("🎯 Communication Focus")
                st.write(results.get("communication_focus", "No focus generated."))

                st.subheader("📚 Summary")
                st.write(results.get("summary", "No summary generated."))

                with st.expander("📝 Raw Transcript"):
                    st.write(transcript)

                with st.expander("✨ Cleaned Transcript"):
                    st.write(cleaned)

        except Exception as e:
            st.error(f"Error: {e}")
            show_trace(e)

        finally:
            # cleanup temp directories created by processing functions
            try:
                cleanup_dirs(video_tmp_dir, audio_tmp_dir)
            except Exception:
                st.warning("Cleanup failed (non-fatal).")
