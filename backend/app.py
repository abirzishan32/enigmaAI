"""
FastAPI Application Entry Point
================================
Serves:
  - ALSA Chat API  (/api/sanitize, /api/chat)        → routers/chat.py
  - Sentiment Analysis API                            → routers/sentiment.py
  - FHE Digit Recognition API (/classify)             → inline below

ALSA NLP Library Initialization (lifespan):
  On startup, the lifespan context manager mock-initializes the heavy NLP
  libraries that Phase 2+ will rely on (BERT, spaCy, NLTK, sklearn).
  In Phase 1 these are stubs; the actual model loading is replaced with
  import checks and log messages so startup remains fast.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import tenseal as ts
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from digit_recognition.model import ConvNet
from routers import chat, sentiment

load_dotenv()


# ============================================================
# SECTION 1: NLP LIBRARY STUB INITIALIZER (Phase 1)
# ============================================================

class NLPRegistry:
    """
    Holds references to the NLP tools used by the ALSA scoring pipeline.
    Phase 1: all stubs (None). Phase 2+: real model objects.
    """
    bert_feature_extractor = None   # transformers.pipeline("feature-extraction", ...)
    spacy_nlp              = None   # spacy.load("en_core_web_sm")
    nltk_ready             = False  # nltk.corpus.wordnet loaded
    isolation_forest       = None   # sklearn IsolationForest (anomaly → PLRS)
    kmeans                 = None   # sklearn KMeans (word cluster → action)


nlp_registry = NLPRegistry()


def _init_nlp_stubs() -> None:
    """
    Called during app lifespan startup.
    Checks each library is importable and logs its status.
    In Phase 2+ replace each block with real model loading.
    """
    print("=" * 60)
    print("🚀  ALSA Phase 1 — NLP Library Initialization")
    print("=" * 60)

    # ── 1. transformers (BERT embeddings) ──────────────────────
    try:
        import transformers  # noqa: F401
        print(f"✅  transformers {transformers.__version__} available "
              "(Phase 1: embedding pipeline not loaded — stub)")
        # Phase 2: nlp_registry.bert_feature_extractor = pipeline(
        #     "feature-extraction", model="bert-base-uncased", truncation=True
        # )
    except ImportError:
        print("⚠️  transformers not installed — run: pip install transformers")

    # ── 2. spaCy (POS tagging → CIIS) ──────────────────────────
    try:
        import spacy  # noqa: F401
        print(f"✅  spaCy {spacy.__version__} available "
              "(Phase 1: en_core_web_sm not loaded — stub)")
        # Phase 2: nlp_registry.spacy_nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print("⚠️  spaCy not installed — run: pip install spacy && python -m spacy download en_core_web_sm")

    # ── 3. NLTK / WordNet (synonym graphs → TRS) ───────────────
    try:
        import nltk  # noqa: F401
        print(f"✅  NLTK {nltk.__version__} available "
              "(Phase 1: WordNet corpus not loaded — stub)")
        # Phase 2:
        #   nltk.download("wordnet", quiet=True)
        #   from nltk.corpus import wordnet as wn  # noqa: F401
        #   nlp_registry.nltk_ready = True
        nlp_registry.nltk_ready = False  # Phase 1 stub
    except ImportError:
        print("⚠️  NLTK not installed — run: pip install nltk")

    # ── 4. scikit-learn (IsolationForest → PLRS, K-Means → action clusters) ──
    try:
        from sklearn.ensemble import IsolationForest      # noqa: F401
        from sklearn.cluster  import KMeans               # noqa: F401
        import sklearn
        print(f"✅  scikit-learn {sklearn.__version__} available "
              "(Phase 1: IsolationForest + KMeans not fitted — stubs)")
        # Phase 2:
        #   nlp_registry.isolation_forest = IsolationForest(contamination=0.1)
        #   nlp_registry.kmeans = KMeans(n_clusters=4, random_state=42)
    except ImportError:
        print("⚠️  scikit-learn not installed — run: pip install scikit-learn")

    print("=" * 60)
    print("✅  Phase 1 stub initialization complete.")
    print("    Scores (PLRS/CIIS/TRS) and actions are mocked in Phase 1.")
    print("=" * 60)


# ============================================================
# SECTION 2: APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: startup → yield → shutdown."""
    global digit_model, tenseal_context

    # ── NLP stubs (ALSA) ───────────────────────────────────────
    _init_nlp_stubs()

    # ── FHE digit recognition model ────────────────────────────
    print("🔢  Loading FHE digit recognition model...")
    device = torch.device("cpu")
    digit_model = ConvNet().to(device)
    model_path = "mnist_model.pth"
    if os.path.exists(model_path):
        digit_model.load_state_dict(torch.load(model_path, map_location=device))
        digit_model.eval()
        print("✅  Digit model loaded.")
    else:
        print("⚠️  mnist_model.pth not found. Run train.py first.")

    # ── TenSEAL context ────────────────────────────────────────
    print("🔒  Setting up TenSEAL CKKS context...")
    tenseal_context = _setup_tenseal_context()
    print("✅  TenSEAL context ready.")

    # ── Sentiment model ────────────────────────────────────────
    print("💬  Loading sentiment analysis model...")
    sentiment.load_sentiment_model()
    print("✅  Sentiment model loaded.")

    print("\n🟢  All systems ready. ALSA Phase 1 API is live.\n")

    yield  # ← application runs

    # Shutdown (nothing to clean up in Phase 1)
    print("🔴  Shutting down ALSA API.")


# ============================================================
# SECTION 3: APP INSTANCE & MIDDLEWARE
# ============================================================

app = FastAPI(
    title="ALSA Privacy-Preserving LLM API",
    description=(
        "Phase 1 foundation: word-level ALSA tokenizer, mocked PLRS/CIIS/TRS scores, "
        "Gemini-2.5-flash integration, and FHE digit recognition."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ALSA chat router + existing routers
app.include_router(chat.router)
app.include_router(sentiment.router)


# ============================================================
# SECTION 4: FHE DIGIT RECOGNITION ENDPOINT (unchanged)
# ============================================================

digit_model    = None
tenseal_context = None


def _setup_tenseal_context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


class ImageInput(BaseModel):
    image: list  # flat list of 784 float pixel values


@app.post("/classify", tags=["FHE Digit Recognition"])
async def classify_digit(payload: ImageInput):
    if digit_model is None or tenseal_context is None:
        raise HTTPException(status_code=503, detail="Model or Context not initialized")

    try:
        input_data = np.array(payload.image, dtype=np.float32).flatten()
        if len(input_data) != 784:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 784 pixels, got {len(input_data)}",
            )

        enc_input = ts.ckks_vector(tenseal_context, input_data)

        fc1_weight = digit_model.fc1.weight.data.numpy().T
        fc1_bias   = digit_model.fc1.bias.data.numpy()
        fc2_weight = digit_model.fc2.weight.data.numpy().T
        fc2_bias   = digit_model.fc2.bias.data.numpy()

        enc_hidden = enc_input.matmul(fc1_weight) + fc1_bias
        enc_hidden.square_()
        enc_output = enc_hidden.matmul(fc2_weight) + fc2_bias

        output_vec = enc_output.decrypt()
        prediction = int(np.argmax(output_vec))
        confidence = float(np.max(output_vec))

        return {"prediction": prediction, "confidence": confidence, "encrypted": True}

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
