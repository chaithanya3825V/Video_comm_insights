# app.py
import streamlit as st
import traceback
from processing import (
    download_video,
    extract_audio,
    transcribe_audio,
    cleanup_dirs,
)
from text_features import clean_transcript, compute_filler_stats, compute_pace_wpm
from analysis import analyze_transcript

st.set_page_config(
    page_title="Video Communication Insights",
    layout="wide",
    page_icon="🎥"
)

st.title("🎥 Video Communication Insights")
st.write("Analyze communication clarity, focus, tone, and more from any public video URL (YouTube or direct MP4).")

st.subheader("Enter Video URL")
url = st.text_input("Video URL (YouTube or direct MP4):", placeholder="Paste public video URL here...")

analyze_button = st.button("Analyze Video")


def show_error(exc):
    st.error(f"{str(exc)}")
    with st.expander("Traceback (for debugging)"):
        st.code(traceback.format_exc())


if analyze_button:
    if not url or not url.strip():
        st.error("Please enter a valid video URL.")
    else:
        video_tmp_dir = None
        audio_tmp_dir = None
        try:
            with st.spinner("Downloading video..."):
                video_path, video_tmp_dir = download_video(url)

            st.success("Downloaded video.")

            with st.spinner("Extracting and normalizing audio..."):
                audio_path, duration_sec, audio_tmp_dir = extract_audio(video_path)

            st.success("Audio ready.")

            # Step 3 — Transcribe (Groq Whisper)
            with st.spinner("Transcribing audio... (this may take a while)"):
                raw_transcript = transcribe_audio(audio_path)

            if not raw_transcript or len(raw_transcript.split()) < 3:
                st.error("Transcription returned very little text. Try a clearer or shorter video.")
            else:
                cleaned_transcript = clean_transcript(raw_transcript)
                filler_stats = compute_filler_stats(cleaned_transcript)
                pace_wpm = compute_pace_wpm(filler_stats["total_words"], duration_sec)

                with st.spinner("Analyzing communication..."):
                    results = analyze_transcript(cleaned_transcript, filler_stats, pace_wpm)

                st.subheader("Communication Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Clarity Score", f"{results.get('clarity_score', 'N/A')}%")
                c2.metric("Speaking Pace", f"{round(pace_wpm, 1)} WPM")
                c3.metric("Filler Density", f"{filler_stats.get('filler_density_per_100_words', 'N/A')} per 100 words")

                c4, c5, c6 = st.columns(3)
                c4.metric("Total Words", f"{filler_stats.get('total_words', 'N/A')}")
                c5.metric("Sentiment", results.get("sentiment", "Unknown").capitalize())
                c6.metric("Audio Duration", f"{round(duration_sec, 1)} sec")

                st.subheader("Communication Focus")
                st.write(results.get("communication_focus", "No focus generated."))

                st.subheader("Summary")
                st.write(results.get("summary", "No summary generated."))

                with st.expander("Raw Transcript"):
                    st.write(raw_transcript)

                with st.expander("Cleaned Transcript"):
                    st.write(cleaned_transcript)

        except Exception as e:
            show_error(e)

        finally:
            try:
                cleanup_dirs(video_tmp_dir, audio_tmp_dir)
            except Exception:
                st.warning("Cleanup encountered an issue (non-fatal).")
