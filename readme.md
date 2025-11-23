This project is a small Streamlit app that analyzes communication quality from a direct MP4 video link. It extracts audio, transcribes it using Whisper (Groq API), and generates three outputs: a clarity score, a one-sentence communication focus, and a short summary.
### Setup
1. Install Python 3.
2. Clone the project and open the folder.
3. Create and activate a virtual environment.
4. Install dependencies: pip install -r requirements.txt
5. Install ffmpeg on your system (required for audio extraction).
6. Set your API key: export GROQ_API_KEY="your_key"
7. Run the app: streamlit run app.py


### How it works

1. Downloads the video from a direct MP4 URL.
2. Extracts and normalizes audio using ffmpeg.
3. Splits and transcribes audio using Groq Whisper.
4. Analyzes the transcript to compute clarity, focus, and summary.

### Notes
1. The deployed Streamlit app supports only direct MP4 links, not YouTube, because Streamlit Cloud doesn’t support the libraries required for YouTube video extraction.
2. The UI is intentionally simple to meet the assessment requirements.
