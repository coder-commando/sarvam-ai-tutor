"""
Minimal sanity check that Sarvam API is reachable and your key works.
"""

import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("ERROR: SARVAM_API_KEY not found in .env")

print(f"API key loaded (first 8 chars: {api_key[:8]}...)")
print(f"Length: {len(api_key)} characters")

# Initialize client
try:
    client = SarvamAI(api_subscription_key=api_key)
    print("[OK] SarvamAI client initialized successfully")
except Exception as e:
    print(f"[FAIL] Error initializing client: {e}")
    raise SystemExit(1)

# Quick API test - try a translation (cheapest operation to test connectivity)
try:
    response = client.text.translate(
        input="Hello, how are you?",
        source_language_code="en-IN",
        target_language_code="hi-IN",
    )
    print(f"[OK] API connectivity confirmed")
    print(f"Test translation: 'Hello, how are you?' -> '{response.translated_text}'")
except Exception as e:
    print(f"[FAIL] API call failed: {e}")
    raise SystemExit(1)

print("\nSarvam API is ready to use.")