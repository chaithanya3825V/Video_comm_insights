# app.py
import os
import traceback
import streamlit as st
from processing import download_video, extract_audio, transcribe_audio, cleanup_dirs
from text_features import clean_transcript, compute_filler_stats, compute_pace_wpm
from analysis import analyze_transcript

st.set_page_config(page_title="Video Communication Insights", layout="centered", page_icon="🎥")
st.title("Video Communication Insights")

# small helper: allow one-time API key paste if not set in env
if not os.getenv("GROQ_API_KEY"):
    key = st.text_input("GROQ API Key (paste here for this session)", type="password")
    if key:
        os.environ["GROQ_API_KEY"] = key
        st.success("GROQ_API_KEY set for this session.")

st.write("Paste a direct MP4 URL (public). Example: https://storage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4")

url = st.text_input("MP4 URL:")
analyze = st.button("Analyze")

def _show_trace():
    with st.expander("Traceback (debug)"):
        st.code(traceback.format_exc())

if analyze:
    if not url or not url.strip():
        st.error("Please enter a direct MP4 URL.")
    else:
        video_tmp_dir = None
        audio_tmp_dir = None
        try:
            with st.spinner("Downloading video..."):
                video_path, video_tmp_dir = download_video(url)

            st.success("Video downloaded.")
            with st.spinner("Extracting audio..."):
                audio_path, duration_sec, audio_tmp_dir = extract_audio(video_path)

            st.success("Audio extracted.")
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path)

            if not transcript or len(transcript.split()) < 3:
                st.error("Transcription produced very little text. Try a clearer or shorter video.")
            else:
                cleaned = clean_transcript(transcript)
                filler = compute_filler_stats(cleaned)
                pace_wpm = compute_pace_wpm(filler["total_words"], duration_sec)

                results = analyze_transcript(cleaned, filler, pace_wpm)

                st.subheader("Communication Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Clarity Score", f"{results.get('clarity_score', 'N/A')}%")
                c2.metric("Speaking Pace", f"{round(pace_wpm,1)} WPM")
                c3.metric("Filler Density", f"{filler.get('filler_density_per_100_words','N/A')} per 100 words")

                st.subheader("Communication Focus")
                st.write(results.get("communication_focus", "No focus generated."))

                st.subheader("Summary")
                st.write(results.get("summary", "No summary generated."))

                with st.expander("Raw Transcript"):
                    st.write(transcript)

                with st.expander("Cleaned Transcript"):
                    st.write(cleaned)

        except Exception as e:
            st.error(f"Error: {e}")
            _show_trace()
        finally:
            try:
                cleanup_dirs(video_tmp_dir, audio_tmp_dir)
            except Exception:
                st.warning("Cleanup failed (non-fatal).")
