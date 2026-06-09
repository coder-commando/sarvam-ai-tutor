"""
TTS module: converts text responses to audio using Sarvam Bulbul v3.

Auto-detects whether text is Hindi (Devanagari) or English/Romanized and
selects the appropriate language code. Bulbul v3 handles code-mixed text
natively, so the language code is primarily for pre-TTS normalization.

Saves audio as WAV file. Returns the path so the Gradio UI can play it.
"""

import os
import re
import base64
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

# === Configuration ===
MODEL = "bulbul:v3"
SPEAKER = "neha"   # Warm female voice, v3 equivalent of Anushka.
                   # Other v3 options: priya, ritu, simran, kavya, ishita, shreya, suhani

# Bulbul v3 supports up to 2500 chars per request; safety limit
MAX_TEXT_CHARS = 2400


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


def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Hindi (Devanagari) or English/Latin.

    Returns a BCP-47 language code string for Bulbul.

    Heuristic: if Devanagari characters make up >20% of letters, it's Hindi.
    The 20% threshold handles code-mixed text where Hindi dominates the
    grammar even when English keywords are sprinkled in.
    """
    if not text:
        return "en-IN"

    # Devanagari Unicode range: U+0900 to U+097F
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total = devanagari_chars + latin_chars

    if total == 0:
        return "en-IN"

    devanagari_ratio = devanagari_chars / total

    if devanagari_ratio > 0.20:
        return "hi-IN"
    return "en-IN"


def _truncate_for_tts(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """Truncate text to fit Bulbul's per-request limit, ending at a sentence boundary."""
    if len(text) <= limit:
        return text

    truncated = text[:limit]
    # Try to end at last sentence boundary (Devanagari danda or Latin punctuation)
    last_boundary = max(
        truncated.rfind("।"),
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_boundary > limit * 0.5:  # Only use boundary if it's reasonably late
        return truncated[: last_boundary + 1]
    return truncated


def text_to_speech(text: str, output_path: str = None) -> str:
    """
    Convert text to speech audio file.

    Args:
        text: The response text to vocalize
        output_path: Optional path for the output WAV. If None, uses a temp file.

    Returns:
        Path to the saved WAV file. Returns None on error.
    """
    if not text or not text.strip():
        return None

    text = _truncate_for_tts(text.strip())
    language_code = detect_language(text)

    client = _get_client()

    try:
        response = client.text_to_speech.convert(
            text=text,
            target_language_code=language_code,
            model=MODEL,
            speaker=SPEAKER,
        )

        # Sarvam returns base64-encoded audio data in response.audios (a list)
        # First audio is what we want for single-text requests
        audio_b64 = response.audios[0]
        audio_bytes = base64.b64decode(audio_b64)

        # Determine output path
        if output_path is None:
            # Use a temp file that persists during the session
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="tts_response_"
            )
            output_path = tmp.name
            tmp.close()

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the WAV file
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[TTS Error] {e}")
        return None


# === Standalone testing mode ===
def main():
    """Quick test: type some text, get an audio file path back."""
    print("=" * 60)
    print("TTS TEST MODE")
    print(f"Model: {MODEL}, Speaker: {SPEAKER}")
    print("Type text. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print()
        text = input("Text> ").strip()
        if text.lower() in ("quit", "exit", "q", ""):
            break

        lang = detect_language(text)
        print(f"[Detected language: {lang}]")
        print("[Calling Bulbul...]")

        output_path = text_to_speech(text, output_path=f"data/tts_test.wav")

        if output_path:
            print(f"[OK] Audio saved to: {output_path}")
            print("Play it with your default audio player, or open in File Explorer.")
        else:
            print("[FAIL] TTS did not produce audio.")


if __name__ == "__main__":
    main()