"""
ASR (Automatic Speech Recognition) module for live user queries.

Uses Sarvam Saaras V3's real-time speech-to-text endpoint. Unlike the
Batch API used in src/transcribe.py (designed for long files), this
synchronous endpoint is optimised for short audio clips (<30 seconds)
typical of conversational queries.

This module is imported by app.py and called whenever the user submits
voice input via the microphone.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

# === Configuration ===
MODEL = "saaras:v3"
DEFAULT_LANGUAGE = "unknown"  # Let Saaras auto-detect for live queries


# === Module-level cached client ===
_client = None


def _get_client() -> SarvamAI:
    """Lazily load the Sarvam API client."""
    global _client
    if _client is None:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY not found in .env.")
        _client = SarvamAI(api_subscription_key=api_key)
    return _client


def transcribe_audio(audio_path: str, language_code: str = DEFAULT_LANGUAGE) -> str:
    """
    Transcribe a short audio file (under 30 seconds) using Saaras V3 real-time API.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        language_code: BCP-47 language code, or "unknown" for auto-detect.
                       Common options: hi-IN, en-IN, ta-IN, te-IN

    Returns:
        Transcribed text string. Returns empty string on error.
    """
    if not audio_path or not Path(audio_path).exists():
        print(f"[ASR Error] Audio file not found: {audio_path}")
        return ""

    file_size_kb = Path(audio_path).stat().st_size / 1024
    print(f"[ASR] Transcribing {audio_path} ({file_size_kb:.1f} KB)...")

    client = _get_client()

    try:
        # Real-time STT endpoint — takes the file directly
        with open(audio_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model=MODEL,
                language_code=language_code,
            )

        transcript = response.transcript or ""
        detected_lang = getattr(response, "language_code", None) or "unknown"

        print(f"[ASR] Detected language: {detected_lang}")
        print(f"[ASR] Transcript: {transcript}")

        return transcript.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ASR Error] {e}")
        return ""


# === Standalone testing mode ===
def main():
    """Test by transcribing an existing audio file."""
    print("=" * 60)
    print("ASR TEST MODE")
    print(f"Model: {MODEL}")
    print("=" * 60)

    # Default to the lecture file or any small audio file
    test_files = [
        "data/test_query.wav",
        "data/lecture.mp3",  # fallback (will be slow + might fail since >30s)
    ]

    audio_path = None
    for candidate in test_files:
        if Path(candidate).exists():
            audio_path = candidate
            break

    if not audio_path:
        print("\nNo test audio found. To test:")
        print("1. Record a short audio query (5-10 seconds) using Windows Voice Recorder")
        print("2. Save it as data/test_query.wav")
        print("3. Re-run this script")
        return

    print(f"\nTesting with: {audio_path}")

    if "lecture.mp3" in audio_path:
        print("WARNING: lecture.mp3 is >30 seconds, real-time API may reject it.")
        print("Better to test with a short query you record yourself.")

    transcript = transcribe_audio(audio_path)

    print("\n" + "=" * 60)
    print("TRANSCRIPT:")
    print(transcript or "[empty]")
    print("=" * 60)


if __name__ == "__main__":
    main()