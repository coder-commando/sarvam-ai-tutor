"""
Gradio web interface for the Sarvam AI Tutor.

Wires together:
  - asr.py     for voice input transcription
  - retrieve.py for finding relevant lecture chunks
  - llm.py     for generating grounded responses
  - tts.py     for converting responses to audio

Run with:
  python src/app.py

Then open the URL it prints (typically http://127.0.0.1:7860)
"""

import gradio as gr

# Local module imports
# These trigger the lazy-loaded model loads, so the first query
# is slightly slower. After that, everything is in memory.
from asr import transcribe_audio
from retrieve import retrieve_chunks
from llm import generate_response
from tts import text_to_speech


# === Configuration ===
TOP_K_CHUNKS = 4
APP_TITLE = "Sarvam AI Tutor"
APP_DESCRIPTION = """
An AI tutor that answers questions about a Hindi lecture on the history 
of the Kohinoor diamond. Ask questions in Hindi, English, or any other 
Indian language — the tutor responds in the same language.

Try questions like:
- "When was the Kohinoor first discovered?"
- "कोहिनूर पर किन देशों ने अपना दावा किया?"
- "Who took the Kohinoor to Britain?"
"""


def process_query(text_input: str, audio_input: str) -> tuple:
    """
    The main query pipeline. Takes text or voice input, returns text + audio responses.

    Args:
        text_input: Text from the textbox (may be empty)
        audio_input: Path to recorded audio file (may be None)

    Returns:
        Tuple of (response_text, response_audio_path, retrieved_chunks_display)
    """
    # === Step 1: Determine the query ===
    # Voice input takes priority if both are present
    if audio_input:
        print(f"\n[Pipeline] Voice input received: {audio_input}")
        query = transcribe_audio(audio_input, language_code="unknown")
        if not query:
            return (
                "Sorry, I couldn't understand the audio. Please try again or type your question.",
                None,
                "",
            )
    elif text_input and text_input.strip():
        query = text_input.strip()
        print(f"\n[Pipeline] Text input received: {query}")
    else:
        return (
            "Please type a question or record audio to get started.",
            None,
            "",
        )

    # === Step 2: Retrieve relevant chunks ===
    print(f"[Pipeline] Retrieving top-{TOP_K_CHUNKS} chunks...")
    chunks = retrieve_chunks(query, top_k=TOP_K_CHUNKS)

    if not chunks:
        return (
            "I couldn't find any relevant information. Please try a different question.",
            None,
            "",
        )

    # Format chunks for the expandable display panel
    chunks_display = format_chunks_for_display(chunks)

    # === Step 3: Generate grounded response via LLM ===
    print(f"[Pipeline] Generating response...")
    response_text = generate_response(query, chunks)
    print(f"[Pipeline] Response: {response_text[:100]}...")

    # === Step 4: Convert response to audio ===
    print(f"[Pipeline] Converting to speech...")
    audio_path = text_to_speech(response_text)

    return response_text, audio_path, chunks_display


def format_chunks_for_display(chunks: list[dict]) -> str:
    """Format retrieved chunks as markdown for the expandable display panel."""
    lines = ["### Retrieved lecture chunks (used to ground the answer)\n"]
    for i, chunk in enumerate(chunks, 1):
        similarity = 1 - chunk["distance"]
        lines.append(
            f"**Chunk {i}** — *similarity: {similarity:.2f}*\n\n"
            f"{chunk['text']}\n\n"
            f"---\n"
        )
    return "\n".join(lines)


def build_ui():
    """Construct the Gradio interface."""

    with gr.Blocks(title=APP_TITLE) as app:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row():
            # === Left column: input ===
            with gr.Column(scale=1):
                gr.Markdown("### Ask your question")

                text_input = gr.Textbox(
                    label="Type your question",
                    placeholder="e.g., कोहिनूर सबसे पहले कब मिला था?",
                    lines=3,
                )

                gr.Markdown("**OR**")

                audio_input = gr.Audio(
                    label="Record your question",
                    sources=["microphone"],
                    type="filepath",
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

            # === Right column: output ===
            with gr.Column(scale=1):
                gr.Markdown("### Response")

                response_text = gr.Textbox(
                    label="Tutor's answer",
                    lines=6,
                )

                response_audio = gr.Audio(
                    label="Listen to the answer",
                    type="filepath",
                    autoplay=False,
                )

        # === Expandable source chunks (for transparency) ===
        with gr.Accordion("Show retrieved lecture sources", open=False):
            chunks_display = gr.Markdown("Retrieved chunks will appear here after a query.")

        # === Wire up events ===
        submit_btn.click(
            fn=process_query,
            inputs=[text_input, audio_input],
            outputs=[response_text, response_audio, chunks_display],
        )

        # Clear button resets everything
        def clear_all():
            return "", None, "", None, "Retrieved chunks will appear here after a query."

        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, audio_input, response_text, response_audio, chunks_display],
        )

    return app


def main():
    print("=" * 60)
    print(f"Starting {APP_TITLE}")
    print("=" * 60)
    print("\nWarming up models (one-time setup)...")
    # Trigger lazy loads so the first user query isn't slow
    _ = retrieve_chunks("warmup", top_k=1)
    print("Ready.\n")

    app = build_ui()

    # launch() options:
    #   share=True creates a public URL via Gradio's tunnel (not needed locally)
    #   server_name="0.0.0.0" allows access from other devices on your network
    #   inbrowser=True auto-opens your default browser
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=True,
    )


if __name__ == "__main__":
    main()