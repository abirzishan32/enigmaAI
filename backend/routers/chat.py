"""
ALSA (Adaptive Linguistic Sanitization and Anonymization) — Phase 1
======================================================================
Core Architecture:
  - Tokenizes each prompt word-by-word (no NER, no Regex, no Faker).
  - Assigns per-word scores: PLRS, CIIS, TRS  [mocked in Phase 1].
  - Assigns an AnonymizationAction per word    [mocked in Phase 1].
  - Builds a sanitized prompt from the actions.
  - Passes sanitized prompt to Gemini-2.5-flash.

Phase 1 NOTE: All score computations are intentionally mocked with
random floats (0.0–1.0) and randomly chosen actions. The actual
mathematical models (BERT embeddings, spaCy POS, WordNet,
IsolationForest, K-Means) will be wired in Phase 2+.
"""

from __future__ import annotations

import os
import random
import traceback
from enum import Enum
from typing import List, Optional

import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["ALSA Chat"])


# ============================================================
# SECTION 1: DATA MODELS
# ============================================================

class AnonymizationAction(str, Enum):
    """
    The four possible actions the ALSA framework can assign to a word.

    Retain  — word is safe and kept verbatim in the sanitized prompt.
    Replace — word is substituted with a generic placeholder [WORD_N].
    Encrypt — word is treated as sensitive and replaced with [ENC_N].
    Delete  — word is considered irreversibly privacy-leaking and omitted.
    """
    RETAIN  = "Retain"
    REPLACE = "Replace"
    ENCRYPT = "Encrypt"
    DELETE  = "Delete"


class WordToken(BaseModel):
    """
    Represents a single word/token from the user's prompt together
    with its three ALSA evaluation scores and the resulting action.

    Scores (all 0.0–1.0):
      plrs — Privacy Leakage Risk Score:
             How likely is this word to leak personal/sensitive information?
             Higher → more privacy risk.
      ciis — Contextual Information Importance Score:
             How important is this word for preserving the semantic
             meaning / context of the prompt?
             Higher → more important to keep.
      trs  — Task Relevance Score:
             How relevant is this word to the task the LLM must perform?
             Higher → more task-critical.

    action — The final anonymization decision derived from the three scores.
    """
    word:   str               = Field(..., description="Original word text")
    index:  int               = Field(..., description="Zero-based position in the prompt")
    plrs:   float             = Field(..., ge=0.0, le=1.0, description="Privacy Leakage Risk Score")
    ciis:   float             = Field(..., ge=0.0, le=1.0, description="Contextual Information Importance Score")
    trs:    float             = Field(..., ge=0.0, le=1.0, description="Task Relevance Score")
    action: AnonymizationAction = Field(..., description="Assigned anonymization action")


# ── Request / Response models ─────────────────────────────────

class SanitizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw user prompt to analyze")


class SanitizeResponse(BaseModel):
    original_prompt:  str             = Field(..., description="The prompt exactly as the user typed it")
    sanitized_prompt: str             = Field(..., description="Prompt with anonymization actions applied")
    word_tokens:      List[WordToken] = Field(..., description="Per-word analysis results")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the chatbot")


class ChatResponse(BaseModel):
    original_prompt:  str             = Field(..., description="User message as received")
    sanitized_prompt: str             = Field(..., description="Prompt sent to the LLM after sanitization")
    word_tokens:      List[WordToken] = Field(..., description="Per-word ALSA analysis")
    llm_response:     str             = Field(..., description="Response from Gemini")


# ============================================================
# SECTION 2: WORD-LEVEL TOKENIZER & SCORER  (Phase 1 — Mocked)
# ============================================================

# Action weight distribution for a realistic-looking mix in Phase 1:
# Most words are retained; a smaller fraction gets the other actions.
_ACTION_WEIGHTS = {
    AnonymizationAction.RETAIN:  0.55,
    AnonymizationAction.REPLACE: 0.25,
    AnonymizationAction.ENCRYPT: 0.12,
    AnonymizationAction.DELETE:  0.08,
}
_ACTIONS       = list(_ACTION_WEIGHTS.keys())
_ACTION_WVALUES = list(_ACTION_WEIGHTS.values())


def _mock_score() -> float:
    """Return a random float in [0.0, 1.0] rounded to 4 decimal places."""
    return round(random.uniform(0.0, 1.0), 4)


def tokenize_and_score(prompt: str) -> List[WordToken]:
    """
    Phase 1 tokenizer: split prompt on whitespace and assign mocked scores.

    In Phase 2+ this function will be replaced with a proper pipeline:
      1. BERT embeddings → compute semantic substitutability vectors
      2. spaCy POS tags  → inform CIIS (grammatical role awareness)
      3. WordNet synsets → inform TRS  (synonym-based task relevance)
      4. IsolationForest → anomaly score feeds PLRS
      5. K-Means clusters → group words for contextual action decisions

    For Phase 1, all three scores are random floats and the action is
    sampled from a weighted distribution so the UI shows variety.
    """
    words = prompt.split()
    tokens: List[WordToken] = []

    for idx, word in enumerate(words):
        action = random.choices(_ACTIONS, weights=_ACTION_WVALUES, k=1)[0]
        tokens.append(
            WordToken(
                word=word,
                index=idx,
                plrs=_mock_score(),
                ciis=_mock_score(),
                trs=_mock_score(),
                action=action,
            )
        )

    return tokens


def build_sanitized_prompt(tokens: List[WordToken]) -> str:
    """
    Reconstruct the prompt after applying each word's anonymization action.

    Rules:
      Retain  → keep the original word unchanged.
      Replace → substitute with [WORD_{index}]   (generic placeholder).
      Encrypt → substitute with [ENC_{index}]    (signals sensitive content).
      Delete  → omit the word entirely.
    """
    parts: List[str] = []
    for token in tokens:
        if token.action == AnonymizationAction.RETAIN:
            parts.append(token.word)
        elif token.action == AnonymizationAction.REPLACE:
            parts.append(f"[WORD_{token.index}]")
        elif token.action == AnonymizationAction.ENCRYPT:
            parts.append(f"[ENC_{token.index}]")
        elif token.action == AnonymizationAction.DELETE:
            pass  # word is omitted

    return " ".join(parts)


# ============================================================
# SECTION 3: GEMINI INTEGRATION
# ============================================================

async def call_gemini(sanitized_prompt: str) -> str:
    """
    Send the sanitized prompt to Gemini-2.5-flash and return the response text.

    The API key is read from the GOOGLE_GENERATIVE_AI_API_KEY environment
    variable (set in backend/.env).  If the key is absent or the SDK call
    fails, a descriptive HTTPException is raised.
    """
    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini API key not found. "
                "Please set GOOGLE_GENERATIVE_AI_API_KEY in backend/.env"
            ),
        )

    try:
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-2.5-flash")
        result = model.generate_content(sanitized_prompt)
        return result.text
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API call failed: {exc}",
        )


# ============================================================
# SECTION 4: API ENDPOINTS
# ============================================================

@router.post(
    "/sanitize",
    response_model=SanitizeResponse,
    summary="Tokenize & score a prompt (no LLM call)",
    description=(
        "Splits the prompt word-by-word, assigns mocked PLRS / CIIS / TRS scores "
        "and an AnonymizationAction to each word, then returns the sanitized prompt. "
        "Useful for debugging the word-level inspector without consuming LLM quota."
    ),
)
async def sanitize_prompt(payload: SanitizeRequest) -> SanitizeResponse:
    """
    Phase 1 endpoint: tokenize + score only (no Gemini call).

    Returns:
      - original_prompt  : the raw user input.
      - sanitized_prompt : prompt with ALSA actions applied.
      - word_tokens      : per-word breakdown with PLRS, CIIS, TRS, action.
    """
    tokens           = tokenize_and_score(payload.prompt)
    sanitized        = build_sanitized_prompt(tokens)

    return SanitizeResponse(
        original_prompt=payload.prompt,
        sanitized_prompt=sanitized,
        word_tokens=tokens,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Privacy-preserving chat (ALSA → Gemini)",
    description=(
        "Runs the full Phase 1 pipeline: tokenize the user message, score each word "
        "with ALSA (mocked in Phase 1), build a sanitized prompt, pass it to "
        "Gemini-2.5-flash, and return both the LLM response and the full word-level "
        "analysis for the inspector panel."
    ),
)
async def alsa_chat(payload: ChatRequest) -> ChatResponse:
    """
    Full pipeline endpoint:
      1. Tokenize & score each word.
      2. Build sanitized prompt via action rules.
      3. Call Gemini-2.5-flash with the sanitized prompt.
      4. Return LLM response + full word-token analysis.
    """
    tokens    = tokenize_and_score(payload.message)
    sanitized = build_sanitized_prompt(tokens)

    llm_response = await call_gemini(sanitized)

    return ChatResponse(
        original_prompt=payload.message,
        sanitized_prompt=sanitized,
        word_tokens=tokens,
        llm_response=llm_response,
    )
