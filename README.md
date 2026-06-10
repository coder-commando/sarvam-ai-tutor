Sarvam AI Tutor
A RAG-based AI tutor that answers questions about an educational video lecture. Supports voice and text input in 22 Indian languages, responding in the user's chosen language using Sarvam AI's full speech and language stack.

Built as a take-home assignment for Sarvam AI's GTM & Strategy role.

What it does
You ask a question — by typing or speaking — about a Hindi lecture on the history of the Kohinoor diamond. The tutor responds in the same language with a grounded answer drawn only from the lecture content. If your question isn't covered by the lecture, the tutor refuses politely and suggests related topics that are.

Try queries like:

"When was the Kohinoor first discovered?"
"कोहिनूर पर किन देशों ने अपना दावा किया?"
"Who took the Kohinoor to Britain?"
"What is the price of gold today?" (out-of-scope — should refuse with suggestions)
Architecture

                ┌─────────────────────────────────────────────┐
                │              GRADIO WEB UI                  │
                │      (text input, mic input, audio out)     │
                └────────────────────┬────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        ┌──────────┐         ┌──────────────┐        ┌────────────┐
        │ ASR      │         │ RETRIEVAL    │        │ TTS        │
        │ Saaras V3│         │ BGE-M3 +     │        │ Bulbul V3  │
        │          │         │ ChromaDB     │        │ (Neha voice)│
        └──────────┘         └──────┬───────┘        └────────────┘
                                    │
                                    ▼
                            ┌────────────────┐
                            │ LLM            │
                            │ Sarvam-105B    │
                            │ (RAG grounded) │
                            └────────────────┘
Pipeline
Offline (one-time indexing):

Download audio from YouTube (yt-dlp)
Transcribe with Sarvam Saaras V3 Batch API
Chunk transcript on sentence boundaries (~900 chars, 180 char overlap)
Embed chunks with BGE-M3 (multilingual, 1024 dims)
Store in ChromaDB with cosine similarity
Online (per query):

(If voice) Transcribe user query with Saaras V3 real-time API
Embed query with BGE-M3 → semantic search ChromaDB → retrieve top-4 chunks
Send query + chunks to Sarvam-105B with strict grounding system prompt
Convert response to audio with Bulbul V3 (Neha voice)
Return text + audio to UI
Setup
Prerequisites
Python 3.10+ (tested on 3.13)
FFmpeg (for audio download/conversion)
~3GB free disk space (for BGE-M3 embedding model cache)
A Sarvam API key (free tier sufficient for testing) — get one at https://dashboard.sarvam.ai

Installation

bash
git clone git@github.com:coder-commando/sarvam-ai-tutor.git
cd sarvam-ai-tutor

*Important:* You'll need to activate the virtual environment every time you open a new terminal session for this project. If you see ModuleNotFoundError, that's the most likely cause — re-run the activation command above. When the venv is active, your shell prompt will show (venv) at the start

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Install FFmpeg if you don't have it
# Windows: winget install Gyan.FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

cp .env.example .env
# Then edit .env and paste your Sarvam API key
One-time data preparation

bash
yt-dlp -x --audio-format mp3 -o "data/lecture.%(ext)s" "https://www.youtube.com/watch?v=2Nzyv6vGiic"
python src/transcribe.py
python src/ingest.py
Run the app

bash
python src/app.py
The UI opens automatically at http://127.0.0.1:7860.

Project structure

sarvam-ai-tutor/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── transcribe.py    # Offline: Saaras Batch API for the lecture audio
│   ├── ingest.py        # Offline: chunking + embedding + ChromaDB indexing
│   ├── retrieve.py      # Online: semantic search over the lecture
│   ├── llm.py           # Online: Sarvam-105B with strict grounding prompt
│   ├── tts.py           # Online: Bulbul V3 for spoken responses
│   ├── asr.py           # Online: Saaras V3 real-time for voice queries
│   └── app.py           # Gradio UI tying it all together
├── data/                # (gitignored) audio, transcripts, ChromaDB

Each module is runnable standalone for testing. E.g. python src/retrieve.py drops you into an interactive query loop against the index.

Key architectural decisions
Why BGE-M3 for embeddings (not OpenAI / not Sarvam)
Sarvam doesn't currently expose an embeddings API. BGE-M3 is the strongest open-source multilingual embedding model with native support for all 22 Indian languages. It runs locally — no API costs, no network latency, no privacy concerns. The ~2.3GB one-time download is a small price for a model that handles cross-lingual retrieval (English query → Hindi chunk match) cleanly.

Why ChromaDB (not Pinecone / Postgres + pgvector)
For a single-user demo over a small corpus (~14 chunks), file-based ChromaDB is the right choice — zero infrastructure, fast cold-start, embedded in the Python process. Production at scale would warrant pgvector or Pinecone, but here ChromaDB is the lowest-friction option that demonstrates the same retrieval semantics.

Why Sarvam-105B for the LLM (not Sarvam-30B / Sarvam-M)
Sarvam-M is deprecated as of the current Sarvam API
I tried out Sarvam-30B, but it has noticeably weaker instruction-following on prompts with multiple constraints (grounding + language matching + format + tone)
Sarvam-105B's stronger reasoning was worth the slightly higher per-call cost for a demo where prompt adherence matters
Why reasoning_effort=None
Sarvam's hybrid thinking models default to thinking mode, which buries the final answer in reasoning_content and often hits the max_tokens limit before producing the user-facing answer. Setting reasoning_effort=None forces direct response mode — faster, cleaner output, content always lands in the content field where you'd expect.

Why chunks ≈900 chars with 180 char overlap
For Hindi text (which is denser per token than English), 900 chars maps to roughly 250-300 tokens — large enough to contain a complete "thought" from the lecture but small enough that retrieval can be specific. The 180-char overlap (~20%) ensures facts that span chunk boundaries are still retrievable. Smaller chunks (~500 chars) hurt recall on multi-sentence facts; larger (~1500 chars) dilute relevance.

Why "refuse + suggest topics" instead of just refuse
A reviewer testing the bot will probe its boundaries with out-of-scope queries. A flat "I don't know" feels unhelpful. The bot's refusal now includes 2-3 specific topics the lecture does cover, redirecting the user productively. This shows the bot has metacognition about its own scope — a small UX touch that signals product thinking.

Why match query language, not chunk language
A Hindi question gets a Hindi answer; an English question gets an English answer. The system prompt explicitly directs the model to look at the script of the question rather than the language of the retrieved chunks. Without this, the model tends to drift toward the language of the dense Hindi context.

Limitations and future work
Known limitations:

Code-mixed bias in responses. For queries about Hindi-language content, responses naturally lean Hindi-dominant with English keywords. This matches educated Indian speech patterns but means an English query may receive a Hindi-dominant answer. Documented as a deliberate UX choice — forced translation would strip cultural nuance.

Audio gaps in transcript. The source video has ~3 minutes of non-speech content (music, English news clips). Saaras V3 in transcribe mode skips these. A codemix mode re-run would recover the English clips at the cost of more API credits.

Future work I'd prioritise:

Re-rank retrieval results with a small cross-encoder for sharper top-K
Stream the LLM response and start TTS on completed sentences
Auto-detect and suggest a different language if response quality is low

Sample interactions
In-scope, Hindi query:

User: कोहिनूर सबसे पहले कब मिला था?

Tutor: "कोहिनूर के origin के बारे में कई तरह की कहानियां प्रचलित हैं। कई दावों के हिसाब से कोहिनूर की कहानी 13th century में शुरू होती है जब आज के आंध्र प्रदेश में काकतीय डायनेस्टी का राज था। खैर सच्चाई जो भी हो लेकिन अब तक इस हीरे की origin के बारे में accurate और valid information जुटाई नहीं जा सकी है।"

In-scope, English query (multi-chunk synthesis):

User: Tell me about the Sikh Empire's connection to the Kohinoor

Tutor: "Britishers ने 1845 में Punjab पर attack किया और Sikh Empire को हरा दिया। इसके बाद उन्होंने 1849 में फिर से हमला किया और Sikh Empire को पूरी तरह से हड़प लिया। इस जीत के बाद, Governor General Lord Dalhousie खुद Lahore Fort आए और Kohinoor diamond को हासिल किया, जिसे बाद में 1850 में Britain भेज दिया गया।"

Out-of-scope:

User: What is the price of gold today?

Tutor: "That topic isn't covered in this lecture — it focuses on the history of the Kohinoor diamond. You might find these covered topics interesting instead: Lord Dalhousie's acquisition of the diamond, Britain's refusal to return it after 1947, or the Sikh Empire under Ranjit Singh."

Tech stack
Layer	Technology
Embedding	BGE-M3 (BAAI)
Vector DB	ChromaDB
LLM	Sarvam-105B
ASR	Sarvam Saaras V3 (Batch + Real-time)
TTS	Sarvam Bulbul V3 (Neha voice)
UI	Gradio
Audio	FFmpeg + yt-dlp
Built in Python 3.13.

Author
Sriram Ragunathan