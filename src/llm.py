"""
LLM module: takes a user query + retrieved lecture chunks, produces a
grounded response from Sarvam-105B.

Handles Sarvam's hybrid thinking mode: extracts the final answer from
the reasoning trace when the model's 'content' field is empty.
"""

import os
import re
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

# === Configuration ===
MODEL = "sarvam-105b"
TEMPERATURE = 0.2
MAX_TOKENS = 1500   # Generous budget: thinking trace + final answer


# === Short, direct system prompt — long prompts trigger more thinking ===
SYSTEM_PROMPT = """You are a friendly tutor explaining a lecture about the history of the Kohinoor diamond.

RULES (must follow ALL):
1. Answer in 2-4 sentences. No more.
2. Use ONLY the lecture excerpts in the user message. Do not use general knowledge.
3. Match the language of the student's question — Hindi question gets a Hindi (Devanagari) answer; English question gets an English answer.
4. If the excerpts don't answer the question, briefly say so and suggest 2-3 of these covered topics: diamond origins in India, Mughal era, Nadir Shah, Sikh Empire, British acquisition, 1947 claims.
5. Write conversationally. No analysis steps, no bullet points, no labels.

Speak directly to the student. Begin your answer immediately."""


# === Module-level cached client ===
_client = None


def _get_client() -> SarvamAI:
    global _client
    if _client is None:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY not found in .env.")
        _client = SarvamAI(api_subscription_key=api_key)
    return _client


def format_context(retrieved_chunks: list[dict]) -> str:
    """Format the retrieved chunks as a clean context block."""
    if not retrieved_chunks:
        return "[No relevant lecture excerpts found.]"
    return "\n\n".join(chunk["text"] for chunk in retrieved_chunks)


def extract_final_answer(text: str) -> str:
    """
    When the model produces a reasoning trace (numbered steps, bullet points,
    analytical structure), try to extract just the final user-facing answer.

    Heuristic: the last substantial paragraph that doesn't start with structural
    markers (numbers, asterisks, bullets) is usually the actual answer.
    """
    if not text:
        return text

    # Strip leading/trailing whitespace
    text = text.strip()

    # Check if the response looks like structured reasoning (has numbered steps)
    has_reasoning_structure = bool(
        re.search(r'\n\s*\d+\.\s+\*\*', text) or
        re.search(r'\*\*(Analyze|Scan|Identify|Locate|Synthesize|Draft|Combine)', text, re.IGNORECASE)
    )

    if not has_reasoning_structure:
        # Already clean - return as is
        return text

    # Strategy: find the last "Combine" or "Draft" section, then extract
    # the longest quoted/un-bulleted paragraph from it
    # Try several extraction patterns
    patterns = [
        # Pattern 1: text after "Combine and refine for flow."
        r'Combine and refine[^\n]*\n\s*\*?\s*"?([^"]+?)"?\s*(?:\n|$)',
        # Pattern 2: text after "Draft the Response" followed by content
        r'Draft the Response[^\n]*\n(?:[^\n]+\n)*?\s*\*?\s*"?([^"]+?)"?\s*(?:\n|$)',
        # Pattern 3: last quoted sentence in the text
        r'"([^"]{50,})"(?!.*"[^"]{50,}")',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # Clean any leftover markdown
            extracted = re.sub(r'\*+', '', extracted).strip()
            if len(extracted) > 30:  # Sanity check: must be substantial
                return extracted

    # Fallback: return the last paragraph that's not a list item
    paragraphs = text.split('\n\n')
    for para in reversed(paragraphs):
        para = para.strip()
        # Skip empty, structural, or short paragraphs
        if not para or len(para) < 50:
            continue
        if para.startswith(('*', '-', '1.', '2.', '3.', '4.', '5.')):
            continue
        if re.match(r'^\d+\.\s+\*\*', para):
            continue
        return re.sub(r'\*+', '', para).strip()

    # Final fallback: return original (cleaned of markdown)
    return re.sub(r'\*+', '', text).strip()


def generate_response(query: str, retrieved_chunks: list[dict]) -> str:
    """Generate a grounded response to the user's query using Sarvam-105B."""
    if not query or not query.strip():
        return "Please ask a question about the lecture."

    client = _get_client()
    context = format_context(retrieved_chunks)

    # Put the context in the user message, not system, to reduce prompt complexity
    user_message = f"""Lecture excerpts:
---
{context}
---

Student's question: {query}

Give your answer in 2-4 sentences in the same language as the question. Begin immediately, no preamble."""

    try:
        response = client.chat.completions(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_effort=None,
        )

        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        reasoning_content = getattr(msg, "reasoning_content", None)

        # Prefer content if it has substantial output
        if content and len(content.strip()) > 30:
            return content.strip()

        # If content is empty, extract from reasoning_content
        if reasoning_content:
            return extract_final_answer(reasoning_content)

        # Neither has anything
        return (
            f"[No response text returned. "
            f"content={repr(content)}, "
            f"reasoning_content={repr(reasoning_content)[:200]}]"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[Error calling Sarvam LLM: {e}]"


# === Standalone testing mode ===
def main():
    from retrieve import retrieve_chunks

    print("=" * 60)
    print("LLM TEST MODE")
    print("Type a question. Type 'quit' to exit.")
    print("=" * 60)

    print("\nWarming up...")
    _ = retrieve_chunks("test", top_k=1)
    print("Ready.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break

        chunks = retrieve_chunks(query, top_k=4)
        print(f"\n[Top distance: {chunks[0]['distance']:.3f}]")
        print("[Calling Sarvam LLM...]\n")

        response = generate_response(query, chunks)

        print("-" * 60)
        print("RESPONSE:")
        print(response)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()