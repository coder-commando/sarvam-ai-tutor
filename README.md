# Sarvam AI Tutor

A RAG-based AI tutor that answers questions about an educational video lecture. Supports voice and text input in 22 Indian languages. Built as a take-home assignment for Sarvam AI's GTM & Strategy role.

## Architecture

- Audio transcription via Sarvam Saaras
- Multilingual embeddings via BGE-M3
- Vector storage in ChromaDB
- LLM responses via Sarvam-M
- Voice output via Sarvam Bulbul
- Interface via Gradio

## Setup

bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # Then fill in your Sarvam API key


## Running

bash
python src/app.py