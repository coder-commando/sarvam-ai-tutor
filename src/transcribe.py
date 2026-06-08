"""
Transcribe a long audio file using Sarvam's Saaras V3 Batch API.
Saves the transcript as JSON for downstream processing.

Usage:
    python src/transcribe.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

# === Configuration ===
AUDIO_PATH = "data/lecture.mp3"
OUTPUT_TRANSCRIPT_PATH = "data/transcript.json"
LANGUAGE_CODE = "hi-IN"   # Change as needed: en-IN, ta-IN, te-IN, "unknown" for auto-detect
MODE = "transcribe"        # transcribe = same language; translate = to English
MODEL = "saaras:v3"
JOB_OUTPUT_DIR = "data/batch_output"


def get_client() -> SarvamAI:
    """Initialize and return a Sarvam API client."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: SARVAM_API_KEY not found in .env")
    return SarvamAI(api_subscription_key=api_key)


def transcribe_with_batch_api(audio_path: str) -> dict:
    """Submit audio to Sarvam Batch API, wait for completion, return parsed results."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
    print(f"Audio file: {audio_path} ({file_size_mb:.2f} MB)")

    client = get_client()

    # Step 1: Create a batch job with flat keyword args (not nested under job_parameters)
    print("\n[1/5] Creating batch job...")
    job = client.speech_to_text_job.create_job(
        model=MODEL,
        mode=MODE,
        language_code=LANGUAGE_CODE,
        with_timestamps=True,
        with_diarization=False,  # Single speaker (lecture)
    )
    print(f"      Job created. ID: {job.job_id}")

    # Step 2: Upload the audio file
    print("\n[2/5] Uploading audio file...")
    job.upload_files(file_paths=[audio_path])
    print("      Upload complete.")

    # Step 3: Start the job
    print("\n[3/5] Starting job...")
    job.start()
    print("      Job started.")

    # Step 4: Wait for completion (SDK handles polling)
    print("\n[4/5] Waiting for job to complete (this may take several minutes)...")
    job.wait_until_complete()
    print("      Job complete.")

    # Check file-level results before downloading
    file_results = job.get_file_results()
    print(f"\n      Successful: {len(file_results['successful'])}")
    print(f"      Failed: {len(file_results['failed'])}")

    if file_results["failed"]:
        for f in file_results["failed"]:
            print(f"      ERROR on {f['file_name']}: {f['error_message']}")
        raise RuntimeError("One or more files failed to process.")

    # Step 5: Download outputs
    print("\n[5/5] Downloading results...")
    Path(JOB_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    job.download_outputs(output_dir=JOB_OUTPUT_DIR)
    print(f"      Results saved to: {JOB_OUTPUT_DIR}")

    return parse_batch_output(JOB_OUTPUT_DIR)


def parse_batch_output(output_dir: str) -> dict:
    """Find and load the JSON output produced by the batch job."""
    output_dir = Path(output_dir)
    json_files = list(output_dir.glob("**/*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON output found in {output_dir}")

    print(f"      Found {len(json_files)} result file(s)")
    # If multiple files, take the first; we only sent one
    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def save_transcript(result: dict, output_path: str):
    """Save the parsed result to a clean JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nTranscript saved to: {output_path}")


def preview_transcript(result: dict):
    """Print a preview to verify quality. Handles various response field names."""
    print("\n" + "=" * 60)
    print("TRANSCRIPT PREVIEW")
    print("=" * 60)

    transcript = result.get("transcript") or result.get("text") or ""

    if not transcript:
        print("\nWARNING: No transcript text found in expected fields.")
        print(f"Available top-level keys: {list(result.keys())}")
        print(f"\nFirst 500 chars of raw result:\n{str(result)[:500]}")
        return

    print(f"\nTotal length: {len(transcript)} characters")
    print(f"\nFirst 500 characters:\n{transcript[:500]}")
    print(f"\nLast 200 characters:\n{transcript[-200:]}")


def main():
    print(f"Starting transcription of {AUDIO_PATH}")
    print(f"Language: {LANGUAGE_CODE}, Mode: {MODE}, Model: {MODEL}")
    print("-" * 60)

    result = transcribe_with_batch_api(AUDIO_PATH)
    save_transcript(result, OUTPUT_TRANSCRIPT_PATH)
    preview_transcript(result)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()