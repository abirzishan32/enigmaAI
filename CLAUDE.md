# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EnigmaAI** is a privacy-preserving AI thesis project demonstrating Fully Homomorphic Encryption (FHE) with three interactive demos:
1. **Digit Recognition** — MNIST classification over FHE (TenSEAL CKKS)
2. **Private Chatbot** — ALSA (Advanced Language Sanitization Architecture) pipeline for word-level privacy scoring before sending prompts to Gemini
3. **Sentiment Analysis** — TF-IDF + MLP classifier with optional FHE inference

## Commands

### Backend
```bash
cd backend
source venv/bin/activate          # activate Python venv
uvicorn app:app --reload          # dev server at http://localhost:8000
python test_api.py                # basic API smoke test
python digit_recognition/train.py # train MNIST model (15 epochs, ~10 min)
python sentiment_analysis/train.py # train sentiment model (20 epochs)
```

### Frontend
```bash
cd frontend
npm run dev    # dev server at http://localhost:3000
npm run build  # production build
npm run lint   # ESLint
```

### First-time setup
```bash
# Backend
cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## Architecture

### Backend (`backend/`)

**Entry point**: `app.py` — FastAPI server that wires together three routers and initializes TenSEAL context and digit model on startup via an async lifespan manager.

**NLP Phase system**: `app.py` defines `NLPRegistry` with stubs for BERT, spaCy, NLTK, and sklearn models. Phase 1 (current) mocks these at startup; Phase 2+ replaces stubs with real model loading. Don't remove the phase comments.

**Key routers**:
- `routers/transformer.py` (~77 KB) — core ALSA pipeline. The main entry points are `alsa_pipeline()` and `compute_scores()`. Each word gets three scores:
  - **PLRS** (Privacy Leakage Risk Score) — BERT embeddings + spaCy NER
  - **CIIS** (Contextual Information Importance Score) — word position, POS tags
  - **TRS** (Task Relevance Score) — WordNet synonyms + Llama-3 inference
  - Action thresholds are constants at the top: `PLRS_HIGH_THRESHOLD`, `CIIS_HIGH_THRESHOLD`, `TRS_HIGH_THRESHOLD` (all 0.55)
- `routers/chat.py` — `/api/chat` endpoint; calls ALSA pipeline then forwards sanitized prompt to Gemini 2.5 Flash
- `routers/sentiment.py` — sentiment endpoints; has its own TenSEAL context (poly degree 8192)

**FHE (digit recognition)**: TenSEAL CKKS context configured in `app.py` with poly_modulus_degree=16384. The two-layer MLP (`digit_recognition/model.py`) uses square activation (FHE-compatible). Weights are extracted post-training and used directly for encrypted matrix-vector multiply.

**Generated artifacts** (not in git, created on first train):
- `backend/mnist_model.pth`
- `backend/sentiment_model.pth`, `sentiment_vectorizer.pkl`, `sentiment_labels.pkl`
- `backend/model_cache/` — BERT/transformers local cache

### Frontend (`frontend/`)

**Framework**: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4.

**Page routes**:
- `/` — landing page composed from `components/sections/` components
- `/digit-recog` — canvas drawing + FHE classification workflow
- `/chatbot` — ALSA chat with word-level privacy inspector
- `/sentiment-analysis` — sentiment demo

**State logic lives in hooks**, not components:
- `hooks/use-fhe.ts` — manages the full FHE encrypt → send → decrypt state machine for digit recognition
- `hooks/use-canvas.ts` — canvas drawing, 28×28 downsampling
- `hooks/use-sentiment-fhe.ts` — sentiment FHE workflow

**Backend URL**: configured in `frontend/.env.local` as `NEXT_PUBLIC_API_URL=http://localhost:8000`.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | FHE digit inference (accepts serialized TenSEAL ciphertext) |
| `/api/chat` | POST | ALSA pipeline + Gemini LLM response |
| `/sentiment/status` | GET | Check if sentiment models are loaded |
| `/sentiment/get-vectorizer-params` | POST | TF-IDF vocabulary for client-side preprocessing |
| `/sentiment/predict` | POST | Plaintext sentiment prediction |
| `/sentiment/predict-encrypted` | POST | FHE sentiment prediction |

## Environment Variables

- `backend/.env` — `GEMINI_API_KEY`, `HUGGING_FACE_TOKEN`, `LLAMA_MODEL_NAME`
- `frontend/.env.local` — `NEXT_PUBLIC_API_URL`
