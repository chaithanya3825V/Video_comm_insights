# Video Communication Insights

This app accepts a public video URL, extracts the audio, transcribes it, and uses an LLM to compute:

- Clarity Score (0–100%)
- Communication Focus (one-sentence summary)

## Setup

1. Clone the repo
2. Create and activate a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Set `OPENAI_API_KEY` as an environment variable
5. Run: `streamlit run app.py`

## How it works

1. Download video from URL (YouTube or direct MP4)
2. Extract audio using `moviepy`
3. Transcribe audio via OpenAI Whisper API
4. Analyze transcript with GPT to generate metrics