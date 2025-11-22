import streamlit as st
from processing import download_video, extract_audio, transcribe_audio
from text_features import clean_transcript, compute_filler_stats, compute_pace_wpm
from analysis import analyze_transcript

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Video Communication Insights",
    layout="wide",
    page_icon="🎥"
)

st.title("🎥 Video Communication Insights")
st.write("Analyze communication clarity, focus, tone, and more from any MP4 video URL.")


# -------------------------------
# Input Section (No Sidebar Now)
# -------------------------------
st.subheader("Enter Video URL")
url = st.text_input(
    "🔗 MP4 Video URL:",
    placeholder="Paste MP4 link here (must be a direct link)..."
)

analyze_button = st.button("🚀 Analyze Video")


# -------------------------------
# Processing Pipeline
# -------------------------------
if analyze_button:
    if not url.strip():
        st.error("❌ Please enter a valid MP4 video URL.")
    else:
        try:
            # Step 1 — Download video
            with st.spinner("📥 Downloading video..."):
                video_path = download_video(url)

            # Step 2 — Extract + clean audio
            with st.spinner("🎧 Extracting and enhancing audio..."):
                audio_path, duration_sec = extract_audio(video_path)

            # Step 3 — Transcribe audio
            with st.spinner("🔊 Transcribing audio (Whisper)..."):
                raw_transcript = transcribe_audio(audio_path)

            if not raw_transcript or len(raw_transcript.split()) < 3:
                st.error("❌ Transcription resulted in very little text. Please try a clearer video.")
            else:
                cleaned_transcript = clean_transcript(raw_transcript)
                filler_stats = compute_filler_stats(cleaned_transcript)
                pace_wpm = compute_pace_wpm(filler_stats["total_words"], duration_sec)

                # Step 4 — LLM Communication Analysis
                with st.spinner("🧠 Analyzing communication (LLaMA)..."):
                    results = analyze_transcript(cleaned_transcript, filler_stats, pace_wpm)

                # -------------------------------
                # Metrics Dashboard – Compact
                # -------------------------------
                st.subheader("📊 Communication Metrics")

                c1, c2, c3 = st.columns(3)
                c1.metric("Clarity Score", f"{results['clarity_score']}%")
                c2.metric("Speaking Pace", f"{pace_wpm} WPM")
                c3.metric("Filler Density", f"{filler_stats['filler_density_per_100_words']} per 100 words")

                c4, c5, c6 = st.columns(3)
                c4.metric("Total Words", f"{filler_stats['total_words']}")
                c5.metric("Sentiment", results.get("sentiment", "Unknown").capitalize())
                c6.metric("Audio Duration", f"{round(duration_sec, 1)} sec")

                # -------------------------------
                # Insights
                # -------------------------------
                st.subheader("🎯 Communication Focus")
                st.write(results["communication_focus"])

                st.subheader("📚 Summary")
                st.write(results["summary"])

                # -------------------------------
                # Transcript Sections
                # -------------------------------
                with st.expander("📝 Raw Transcript"):
                    st.write(raw_transcript)

                with st.expander("✨ Cleaned Transcript"):
                    st.write(cleaned_transcript)

        except Exception as e:
            st.error(f"❌ Error: {e}")
