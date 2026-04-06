

# ── Standard Library ─────────────────────────────────────────────────────────
import asyncio
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import spacy
import nltk
from nltk.corpus import wordnet as wn

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import rbf_kernel

from scipy.spatial.distance import mahalanobis

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("privacy_pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────
BERT_MODEL_NAME   = "bert-base-uncased"
GPT2_MODEL_NAME   = "gpt2"
SPACY_MODEL_NAME  = "en_core_web_sm"
CACHE_DIR         = Path("./model_cache")
BERT_CACHE_FILE   = CACHE_DIR / "bert_common_word_vectors.pt"
N_COMMON_WORDS    = 10_000
MAX_BERT_TOKENS   = 512
KMEANS_N_CLUSTERS = 8
MASK_TOKEN        = "[MASK]"

# ── Llama-3 Backbone (TRS + Encrypt Self-Recommendation) ─────────────
LLAMA_MODEL_NAME          = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
TRS_NUM_ITERS             = 5           # k in TRS(w_i) = (1/k) Σ LLM_j(w_i|P)
TRS_FALLBACK_SCORE        = 0.5         # graceful fallback if LLM fails
TRS_MAX_TOKENS            = 16          # response is a single float

# Encrypt self-recommendation weights:
#   Score(P'_i) = α·Fluency + β·Coherence + γ·TaskConsistency
ENCRYPT_NUM_CANDIDATES    = 3           # m candidate completions
ENCRYPT_CANDIDATE_TOKENS  = 256         # max tokens for candidate generation
ENCRYPT_ALPHA             = 0.33        # fluency weight
ENCRYPT_BETA              = 0.33        # coherence weight
ENCRYPT_GAMMA             = 0.34        # task-consistency weight


# POS → coherence weight
POS_WEIGHT: Dict[str, float] = {
    "NOUN": 1.0, "PROPN": 1.0, "ADJ": 1.0, "ADV": 0.8, "VERB": 0.9,
    "ADP":  0.3, "DET":   0.3, "CONJ": 0.3, "CCONJ": 0.3,
    "SCONJ":0.3, "PART":  0.3, "PRON": 0.5, "NUM": 0.7,
    "PUNCT":0.1, "SYM":   0.2, "X":   0.5, "INTJ": 0.6,
}

# ── PII Regex Patterns ────────────────────────────────────────────────────────
# Each entry: (compiled_pattern, override_plrs_score)
# A score of 1.0 means "always Encrypt regardless of anything else"
PII_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'),           1.0),  # SSN
    (re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),        0.95), # Phone
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), 0.95), # Email
    (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), 0.95), # Credit card
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),  0.90), # IP address
    (re.compile(r'\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b'),     0.80), # ISO date
    (re.compile(r'\$\s*\d[\d,]*(?:\.\d{2})?\b'),               0.75), # Dollar amounts
    (re.compile(r'\b\d+\s*(?:million|billion|thousand)\b', re.I), 0.70), # Large sums
]

# SpaCy NER labels that indicate high-sensitivity named entities.
# ONLY truly identifying labels — CARDINAL/DATE/TIME/PERCENT/QUANTITY
# are excluded because they tag non-identifying values (ages, stages,
# percentages) that should NOT be auto-escalated.
SENSITIVE_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "MONEY"}


# ── Global Model Registry ──────────────────────────────────────────────────────
class ModelRegistry:
    device: torch.device
    bert_tokenizer: AutoTokenizer
    bert_model: AutoModel
    gpt2_tokenizer: AutoTokenizer
    gpt2_model: AutoModelForCausalLM
    spacy_nlp: spacy.Language
    bert_common_vectors: torch.Tensor  # (N_COMMON_WORDS, 768)
    llama_model: object               # mlx_lm model handle
    llama_tokenizer: object           # mlx_lm tokenizer handle


registry = ModelRegistry()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Environment & Global Initialisation
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> torch.device:
    """MPS → CUDA → CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("Device: Apple MPS"); return torch.device("mps")
    if torch.cuda.is_available():
        log.info("Device: CUDA — %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    log.info("Device: CPU"); return torch.device("cpu")


def load_bert(device: torch.device):
    log.info("Loading BERT …")
    tok = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    mdl = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()
    return tok, mdl


def load_gpt2(device: torch.device):
    log.info("Loading GPT-2 …")
    tok = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME).to(device).eval()
    return tok, mdl


def load_llama():
    """
    Load Llama-3-8B-Instruct-4bit via mlx-lm (Apple Silicon only).

    Uses the mlx-community pre-quantised model optimised for Apple's
    Metal GPU via the MLX framework.  Requires ~5–6 GB unified memory.
    """
    from mlx_lm import load as mlx_load
    log.info("Loading Llama-3 backbone (%s) …", LLAMA_MODEL_NAME)
    model, tokenizer = mlx_load(LLAMA_MODEL_NAME)
    log.info("Llama-3 loaded on MLX (Apple Silicon)")
    return model, tokenizer


def _llama_generate(prompt_text: str, max_tokens: int = 16) -> str:
    """
    Synchronous single-call Llama-3 generation via mlx-lm.

    Applies the Llama-3 Instruct chat template for proper formatting,
    then generates up to *max_tokens* of output.

    This function is SYNCHRONOUS and runs on the MLX Metal backend.
    It should be called from async code via ``run_in_executor``.
    """
    from mlx_lm import generate as mlx_generate
    messages = [{"role": "user", "content": prompt_text}]
    formatted = registry.llama_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(
        registry.llama_model, registry.llama_tokenizer,
        prompt=formatted, max_tokens=max_tokens, verbose=False,
    )


def _build_bert_common_vectors(tokenizer, model, device) -> torch.Tensor:
    """
    Offline: embed N_COMMON_WORDS from the Brown corpus in ISOLATION
    (single word → [CLS] vector). This is the 'normal' reference
    distribution for the Isolation Forest.

    CRITICAL: words are embedded one-at-a-time (no context), so the
    Isolation Forest sees the same kind of vector as prompt words will
    be scored with (isolated embeddings, not contextual).
    """
    from nltk.corpus import brown
    nltk.download("brown", quiet=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    freq: Dict[str, int] = {}
    for w in brown.words():
        lw = w.lower()
        if lw.isalpha():
            freq[lw] = freq.get(lw, 0) + 1

    common_words = sorted(freq, key=freq.get, reverse=True)[:N_COMMON_WORDS]  # type: ignore
    log.info("  Embedding %d common words …", len(common_words))

    vectors: List[torch.Tensor] = []
    bs = 64
    for i in range(0, len(common_words), bs):
        batch = common_words[i : i + bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=16).to(device)
        with torch.no_grad():
            out = model(**enc)
        vectors.append(out.last_hidden_state[:, 0, :].cpu())

    all_vecs = torch.cat(vectors, dim=0)
    torch.save(all_vecs, BERT_CACHE_FILE)
    log.info("  Cached → %s  shape=%s", BERT_CACHE_FILE, all_vecs.shape)
    return all_vecs


def load_bert_common_vectors(tokenizer, model, device) -> torch.Tensor:
    if BERT_CACHE_FILE.exists():
        log.info("Loading cached BERT reference vectors …")
        vecs = torch.load(BERT_CACHE_FILE, map_location="cpu")
        log.info("  shape: %s", vecs.shape)
        return vecs
    return _build_bert_common_vectors(tokenizer, model, device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.device = detect_device()
    registry.bert_tokenizer, registry.bert_model = load_bert(registry.device)
    registry.gpt2_tokenizer, registry.gpt2_model = load_gpt2(registry.device)
    log.info("Loading SpaCy …")
    registry.spacy_nlp = spacy.load(SPACY_MODEL_NAME)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    registry.bert_common_vectors = load_bert_common_vectors(
        registry.bert_tokenizer, registry.bert_model, registry.device
    )
    registry.llama_model, registry.llama_tokenizer = load_llama()
    log.info("=== All models loaded. Server ready. ===")
    yield
    log.info("=== Shutting down. ===")


app = FastAPI(
    title="Privacy-Preserving NLP Pipeline",
    description="Word-level anonymisation via PLRS · CIIS · TRS",
    version="2.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def strip_punctuation(word: str) -> str:
    """
    Remove leading/trailing punctuation from a surface token so BERT and
    GPT-2 receive clean lexical forms.

    E.g.  "Smith,"  →  "Smith"
          "8765."   →  "8765"
          "'hello'" →  "hello"

    We keep internal punctuation (hyphens in "432-91-8765") intact.
    """
    return re.sub(r'^[^\w]+|[^\w]+$', '', word) or word


def get_word_char_spans(words: List[str], text: str) -> List[Tuple[int, int]]:
    """
    Return (start, end) character indices for each surface word within `text`.
    Uses a running cursor so repeated words are correctly disambiguated.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for w in words:
        idx = text.find(w, cursor)
        if idx == -1:          # defensive: shouldn't happen with split()
            idx = cursor
        spans.append((idx, idx + len(w)))
        cursor = idx + len(w)
    return spans


def _robust_scale_1d(arr: np.ndarray) -> np.ndarray:
    """
    Percentile-clipped Min-Max normalisation.

    Pure min-max is fragile: a single outlier (or an all-identical array)
    collapses everything to 0. We clip to [p5, p95] first to remove
    extreme outliers, then scale to [0, 1].

    For small arrays (< 10 elements) percentile clipping is skipped to
    avoid distorting results.
    """
    if len(arr) >= 10:
        lo  = float(np.percentile(arr, 5))
        hi  = float(np.percentile(arr, 95))
        arr = np.clip(arr, lo, hi)
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-9)).astype(np.float32)


# ── BERT embedding helpers ────────────────────────────────────────────────────

@torch.no_grad()
def embed_sentences_bert(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Return [CLS] embedding for each sentence. Shape: (N, 768)."""
    all_cls: List[torch.Tensor] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_BERT_TOKENS).to(device)
        out = model(**enc)
        all_cls.append(out.last_hidden_state[:, 0, :].cpu())
    return torch.cat(all_cls, dim=0)


@torch.no_grad()
def embed_words_isolated_bert(
    words: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """
    Embed each word IN ISOLATION (no surrounding context) using [CLS].

    WHY: The BERT common-word reference corpus was built the same way
    (each word alone). Using contextual embeddings here would compare
    apples to oranges in the Isolation Forest, because the reference
    distribution is context-free.

    Returns shape (n_words, 768).
    """
    clean_words = [strip_punctuation(w) for w in words]
    return embed_sentences_bert(clean_words, tokenizer, model, device)


@torch.no_grad()
def embed_words_in_context_bert(
    words: List[str],
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """
    Embed words using their CONTEXTUAL BERT representation.

    FIX over v1:
    - Use offset_mapping to find which sub-tokens correspond to which
      surface word by CHARACTER OVERLAP, not greedy string matching.
    - Average ALL sub-tokens that belong to the word (not just the first).
    - Fall back to [CLS] only when no sub-token overlaps (very rare).

    Returns shape (n_words, 768).
    """
    # Remove offset_mapping from inputs before passing to model
    enc_raw = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_BERT_TOKENS,
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc_raw.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc_raw.items()}

    out   = model(**enc)
    hidden = out.last_hidden_state[0].cpu()  # (seq_len, 768)

    word_spans = get_word_char_spans(words, prompt)

    word_vecs: List[torch.Tensor] = []
    for (ws, we) in word_spans:
        sub_vecs: List[torch.Tensor] = []
        for t_idx, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:      # [CLS] / [SEP] / padding
                continue
            if ts < we and te > ws:      # character overlap condition
                sub_vecs.append(hidden[t_idx])
        if sub_vecs:
            word_vecs.append(torch.stack(sub_vecs).mean(0))
        else:
            word_vecs.append(hidden[0])  # [CLS] as final fallback

    return torch.stack(word_vecs, dim=0)  # (n_words, 768)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Privacy Leakage Risk Score (PLRS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pii_override_scores(words: List[str], prompt: str) -> np.ndarray:
    """
    Hard PII detection layer using regex + SpaCy NER.

    This runs BEFORE the statistical pipeline and produces a per-word
    override score.  Words matched by PII patterns receive a score ≥ 0.7;
    unmatched words receive 0.0 and proceed normally.

    The override score is later MAX-merged with the statistical PLRS so
    PII can never be silently under-scored.

    Returns: np.ndarray (n_words,) in [0, 1]
    """
    override = np.zeros(len(words), dtype=np.float32)
    word_spans = get_word_char_spans(words, prompt)

    # ── Regex patterns ───────────────────────────────────────────────────
    for pattern, score in PII_PATTERNS:
        for m in pattern.finditer(prompt):
            ms, me = m.start(), m.end()
            for i, (ws, we) in enumerate(word_spans):
                # A word is "inside" the match if its span overlaps
                if ws < me and we > ms:
                    override[i] = max(override[i], score)

    return override


def compute_ner_sensitivity_boost(
    words: List[str],
    prompt: str,
    spacy_nlp: spacy.Language,
) -> np.ndarray:
    """
    SpaCy Named Entity Recognition (NER) boost.

    Proper nouns and named entities (PERSON, ORG, GPE, MONEY) are
    inherently sensitive even when they don't match a rigid regex.
    E.g. "John Smith" will be tagged PERSON by SpaCy.

    Returns a boost array (n_words,) in [0, 1]:
        SENSITIVE_NER_LABELS  → 0.4 boost  (reduced from 0.6 to prevent
                                            over-amplification of PLRS)
        PROPN (proper noun)   → 0.3 boost
        Others                → 0.0

    NOTE: The boost is used ADDITIVELY (not as an override) when fused
    into PLRS. This preserves the statistical signal from the Isolation
    Forest and Exposure Risk components.
    """
    doc = spacy_nlp(prompt)
    word_spans = get_word_char_spans(words, prompt)
    boost = np.zeros(len(words), dtype=np.float32)

    # ── NER entity labels ────────────────────────────────────────────────
    for ent in doc.ents:
        if ent.label_ in SENSITIVE_NER_LABELS:
            for i, (ws, we) in enumerate(word_spans):
                if ws < ent.end_char and we > ent.start_char:
                    boost[i] = max(boost[i], 0.4)

    # ── PROPN (proper noun) — catches names SpaCy didn't tag as entities ─
    for tok in doc:
        if tok.pos_ == "PROPN" and not tok.is_space:
            for i, w in enumerate(words):
                if tok.text.lower() == strip_punctuation(w).lower():
                    boost[i] = max(boost[i], 0.3)

    return boost


def compute_intrinsic_sensitivity(
    prompt_word_vecs_isolated: np.ndarray,
    common_vecs: np.ndarray,
) -> np.ndarray:
    """
    Task 2.1 — Intrinsic Sensitivity via Isolation Forest.

    Fit the forest ONLY on the common-word reference distribution,
    then SCORE the prompt words as unseen test points.

    CALIBRATION FIX (Bug 6):
    Instead of normalising anomaly scores within the prompt (which
    makes the least-common common word score 1.0 regardless of
    actual sensitivity), we calibrate against the reference
    distribution itself:

        IS(w_i) = percentile_rank(score(w_i), ref_scores)

    This gives each prompt word its true anomaly percentile relative
    to ordinary English. Common words like "quick" get ~0.15, while
    genuinely rare words like "adenocarcinoma" get ~0.95.

    Uses contamination='auto' for a theoretically sound decision
    boundary instead of an arbitrary 5% contamination rate.

    Args:
        prompt_word_vecs_isolated : np.ndarray (n_words, 768)
        common_vecs : np.ndarray (N_COMMON, 768)

    Returns: np.ndarray (n_words,) in [0, 1]
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",   # theoretically sound decision boundary
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(common_vecs)                        # FIT on reference only

    # Score prompt words (lower = more anomalous → flip sign)
    prompt_scores = -iso.score_samples(prompt_word_vecs_isolated)

    # Score a reference sample for calibration (subsample for speed)
    ref_sample = common_vecs[:2000]
    ref_scores = -iso.score_samples(ref_sample)

    # Percentile rank: fraction of reference words less anomalous
    anomaly = np.array(
        [float(np.mean(ref_scores < ps)) for ps in prompt_scores],
        dtype=np.float32,
    )
    return anomaly  # already in [0, 1] — no _robust_scale_1d needed


@torch.no_grad()
def compute_exposure_risk(
    prompt: str,
    words: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> np.ndarray:
    """
    Task 2.2 — Exposure Risk via GPT-2 Negative Log-Likelihood (FIXED).

    FIX: Replace fragile greedy string-reconstruction with OFFSET MAPPING.

    The tokenizer returns (char_start, char_end) for every BPE token.
    We group tokens by which surface word's character span they fall in,
    then average the NLL values within each group.

    This correctly handles:
        • Hyphenated tokens:  "432-91-8765"  (multiple BPE tokens)
        • Punctuation:        "Smith,"        (BPE splits at comma)
        • Contractions:       "don't"         (split at apostrophe)

    NLL formula:
        NLL_t = -log₂ P(w_t | w_1 … w_{t-1})

    The first token has no preceding context, so we skip it (its NLL
    is undefined in a purely auto-regressive sense). Instead, words
    whose first BPE token is at position 0 receive the mean NLL.

    Returns: np.ndarray (n_words,) in [0, 1]
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Tokenise with character offsets ──────────────────────────────────
    enc_raw = tokenizer(
        prompt, return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc_raw.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc_raw.items()}
    input_ids = enc["input_ids"]  # (1, seq_len)

    out = model(**enc)
    logits = out.logits[0]  # (seq_len, vocab_size)

    # Shift: logits[t] predicts the probability distribution for token[t+1]
    # So nll_tokens[t] = NLL of input_ids[t+1] given all prior tokens
    shift_logits = logits[:-1]                              # (seq_len-1, vocab)
    shift_ids    = input_ids[0, 1:]                         # (seq_len-1,)
    probs        = F.softmax(shift_logits, dim=-1)          # (seq_len-1, vocab)
    token_probs  = probs[range(len(shift_ids)), shift_ids]  # (seq_len-1,)
    nll_tokens   = -torch.log2(token_probs + 1e-9).cpu().numpy()  # (seq_len-1,)

    mean_nll = float(nll_tokens.mean()) if len(nll_tokens) else 5.0

    # ── Map BPE tokens → surface words via character overlap ─────────────
    # offsets[t] = (char_start, char_end) of token t in the original string.
    # nll_tokens[t] = NLL of token t+1, so token t+1 corresponds to
    # offset_mapping[t+1].  We iterate t from 1 (the second token onward)
    # since we have NLL values for tokens[1..seq_len-1].
    word_spans = get_word_char_spans(words, prompt)

    # Build a list: for each token index t (1-indexed), its NLL
    # nll_tokens has len = seq_len - 1.  nll_tokens[i] is NLL of token[i+1].
    # So token at offset_mapping index `j` has NLL = nll_tokens[j-1].
    token_nll_map: List[Optional[float]] = [None] * len(offsets)
    for j in range(1, len(offsets)):  # skip token 0 (no prior context)
        nll_idx = j - 1
        if nll_idx < len(nll_tokens):
            token_nll_map[j] = float(nll_tokens[nll_idx])

    word_nlls: List[float] = []
    for (ws, we) in word_spans:
        tok_nlls: List[float] = []
        for j, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:   # special tokens
                continue
            if ts < we and te > ws:   # overlap
                nll_val = token_nll_map[j]
                if nll_val is not None:
                    tok_nlls.append(nll_val)
        word_nlls.append(float(np.mean(tok_nlls)) if tok_nlls else mean_nll)

    return _robust_scale_1d(np.array(word_nlls, dtype=np.float32))


def compute_plrs(
    prompt_word_vecs_isolated: np.ndarray,
    common_vecs: np.ndarray,
    prompt: str,
    words: List[str],
    spacy_nlp: spacy.Language,
    gpt2_tokenizer: AutoTokenizer,
    gpt2_model: AutoModelForCausalLM,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 2.3 — Final PLRS for each word (FIXED).

    FIXED FORMULA: Weighted-additive fusion instead of multiplicative.

        PLRS_raw_i = 0.5 × Intrinsic_i + 0.5 × Exposure_i

    Then merged with PII/NER override:

        PLRS_i = max( PLRS_raw_i, PII_override_i, NER_boost_i )

    WHY additive? The multiplicative formula `A × B` collapses to 0 if
    either A or B is 0 (which happens after min-max normalisation when
    one metric has very low variance). Additive fusion preserves signal
    from either component independently.

    Returns:
        plrs          : np.ndarray (n_words,) in [0, 1]
        pii_override  : np.ndarray (n_words,) in [0, 1]  (for hard overrides)
    """
    intrinsic = compute_intrinsic_sensitivity(prompt_word_vecs_isolated, common_vecs)
    exposure  = compute_exposure_risk(prompt, words, gpt2_tokenizer, gpt2_model, device)

    plrs_raw = 0.5 * intrinsic + 0.5 * exposure  # additive fusion

    pii_override = compute_pii_override_scores(words, prompt)
    ner_boost    = compute_ner_sensitivity_boost(words, prompt, spacy_nlp)

    # NER boost is ADDITIVE (not override) — preserves statistical ranking.
    # PII regex patterns always override (they are hard-confirmed).
    plrs = np.clip(plrs_raw + 0.5 * ner_boost, 0.0, 1.0)
    plrs = np.maximum(plrs, pii_override).astype(np.float32)

    return plrs, pii_override


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Contextual Information Importance Score (CIIS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_contextual_coherence(
    words: List[str],
    prompt: str,
    spacy_nlp: spacy.Language,
    bert_word_vecs: np.ndarray,
) -> np.ndarray:
    """
    Task 3.1 — Contextual Coherence via Mahalanobis distance.

    A word that is semantically DISTANT from the POS-weighted centroid
    of the sentence carries unique, non-redundant meaning — deleting it
    would structurally damage the sentence.

    Mahalanobis:  d_M(x, μ) = √[(x−μ)ᵀ · Σ⁻¹ · (x−μ)]

    We regularise Σ with λI (λ=0.1) to handle the n << p case
    (sentence length << 768 BERT dimensions).
    """
    doc = spacy_nlp(prompt)
    spacy_toks = [t for t in doc if not t.is_space]

    pos_weights = np.ones(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        clean = strip_punctuation(w).lower()
        for st in spacy_toks:
            if st.text.lower() == clean:
                pos_weights[i] = POS_WEIGHT.get(st.pos_, 0.5)
                break

    V   = bert_word_vecs                               # (n, 768)
    w_col = pos_weights.reshape(-1, 1)                 # (n, 1)
    centroid = (w_col * V).sum(0) / (w_col.sum() + 1e-9)  # (768,)

    if V.shape[0] > 1:
        cov     = np.cov(V.T)                          # (768, 768)
        cov_reg = cov + 0.1 * np.eye(cov.shape[0])    # regularise
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(cov_reg.shape[0])
    else:
        cov_inv = np.eye(V.shape[1])

    distances = np.array(
        [mahalanobis(V[i], centroid, cov_inv) for i in range(len(words))],
        dtype=np.float32,
    )
    return _robust_scale_1d(distances)


def get_wordnet_synonyms(word: str, n: int = 5) -> List[str]:
    """WordNet synonyms — single-token only, excluding the word itself."""
    syns: List[str] = []
    for synset in wn.synsets(strip_punctuation(word)):
        for lemma in synset.lemmas():
            cand = lemma.name().replace("_", " ")
            if cand.lower() != word.lower() and " " not in cand:
                syns.append(cand)
    seen, unique = set(), []
    for s in syns:
        if s.lower() not in seen:
            seen.add(s.lower()); unique.append(s)
    return unique[:n]


def compute_semantic_distinctiveness(
    words: List[str],
    prompt: str,
    bert_tokenizer: AutoTokenizer,
    bert_model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    """
    Task 3.2 — Semantic Distinctiveness via Maximum Mean Discrepancy (MMD).

    If swapping a word with synonyms drastically shifts the sentence's [CLS]
    embedding, that word is semantically critical.

    MMD² = k(X,X) − 2·k(X,Y) + k(Y,Y)    where k = RBF kernel.

    Words with no synonyms receive distinctiveness = 0.5 (moderate).
    Using 1.0 would over-inflate CIIS for technical terms (e.g.
    "backpropagation", "adenocarcinoma") that have no WordNet entries,
    causing them to be over-encrypted instead of retained.
    """
    orig_cls = embed_sentences_bert([prompt], bert_tokenizer, bert_model, device)  # (1, 768)
    gamma    = 1.0 / orig_cls.shape[1]   # γ = 1/768

    mmd_scores: List[float] = []
    for word in words:
        synonyms = get_wordnet_synonyms(word, n=5)
        if not synonyms:
            mmd_scores.append(0.5)   # moderate: irreplaceable ≠ sensitive
            continue

        mutants = []
        clean_w = strip_punctuation(word)
        for syn in synonyms:
            mutant = re.sub(r'\b' + re.escape(clean_w) + r'\b', syn,
                            prompt, flags=re.IGNORECASE)
            mutants.append(mutant)

        mutant_cls = embed_sentences_bert(mutants, bert_tokenizer, bert_model, device)

        kXX = float(rbf_kernel(orig_cls.numpy(),   orig_cls.numpy(),   gamma=gamma).mean())
        kXY = float(rbf_kernel(orig_cls.numpy(),   mutant_cls.numpy(), gamma=gamma).mean())
        kYY = float(rbf_kernel(mutant_cls.numpy(), mutant_cls.numpy(), gamma=gamma).mean())

        mmd2 = kXX - 2 * kXY + kYY
        mmd_scores.append(max(mmd2, 0.0))

    return _robust_scale_1d(np.array(mmd_scores, dtype=np.float32))


def compute_ciis(
    words: List[str],
    prompt: str,
    spacy_nlp: spacy.Language,
    bert_word_vecs: np.ndarray,
    bert_tokenizer: AutoTokenizer,
    bert_model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    """
    Task 3.3 — CIIS = 0.4 × Coherence + 0.6 × Distinctiveness
    """
    coh  = compute_contextual_coherence(words, prompt, spacy_nlp, bert_word_vecs)
    dist = compute_semantic_distinctiveness(words, prompt, bert_tokenizer, bert_model, device)
    return (0.4 * coh + 0.6 * dist).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Integration, Clustering, and Action Assignment
# ══════════════════════════════════════════════════════════════════════════════

async def get_trs(prompt: str, word: str) -> float:
    """
    Task 4.1 — Task Relevance Score (TRS) via Llama-3 backbone.

    Queries the LLM:
        "How relevant is word w_i to the task described by prompt P?"

    The model returns a float ∈ [0, 1].  If the response cannot be
    parsed as a valid number, falls back to TRS_FALLBACK_SCORE (0.5).

    Formula (single iteration):
        LLM_j(w_i | P) = parse_float(Llama3(relevance_query))
    """
    clean_word = strip_punctuation(word)
    query = (
        f"On a scale of 0.0 to 1.0, how relevant is the word "
        f"'{clean_word}' to completing this task: "
        f"'{prompt}'? Answer with ONLY a number."
    )
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, _llama_generate, query, TRS_MAX_TOKENS
        )
        match = re.search(r'(\d+\.?\d*)', response)
        if match is None:
            log.warning("TRS: no float in LLM response for '%s': %r", word, response)
            return TRS_FALLBACK_SCORE
        score = float(match.group(1))
        return float(np.clip(score, 0.0, 1.0))
    except Exception as exc:
        log.warning("TRS fallback for '%s': %s", word, exc)
        return TRS_FALLBACK_SCORE


async def compute_trs_batch(prompt: str, words: List[str]) -> np.ndarray:
    """
    Task 4.1 — Batch TRS computation with per-word caching.

    TRS(w_i) = (1/k) × Σ_{j=1}^{k} LLM_j(w_i | P)

    where:
        k         = TRS_NUM_ITERS (default 5)
        LLM_j     = j-th Llama-3 inference for word w_i given prompt P
        Output    ∈ [0.0, 1.0]

    Duplicate words (case-insensitive, punctuation-stripped) share
    cached scores to avoid redundant LLM calls.

    NOTE: Execution is sequential because mlx-lm uses a single
    GPU-bound model that is not thread-safe for concurrent access.
    """
    cache: Dict[str, float] = {}
    results: List[float] = []

    for word_idx, word in enumerate(words):
        clean = strip_punctuation(word).lower()
        if clean in cache:
            log.debug("TRS cache hit: '%s' → %.3f", word, cache[clean])
            results.append(cache[clean])
            continue

        scores: List[float] = []
        for j in range(TRS_NUM_ITERS):
            s = await get_trs(prompt, word)
            scores.append(s)
            log.debug("TRS iter %d/%d for '%s': %.3f", j + 1, TRS_NUM_ITERS, word, s)

        avg = float(np.mean(scores))
        cache[clean] = avg
        log.info("TRS[%d] '%s': %.3f  (k=%d)", word_idx, word, avg, TRS_NUM_ITERS)
        results.append(avg)

    return np.array(results, dtype=np.float32)


def assign_actions_kmeans(
    plrs: np.ndarray,
    ciis: np.ndarray,
    trs: np.ndarray,
    pii_override: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 4.2 — Two-stage Action Assignment (ALSA Paper Table 1).

    STAGE 1 — PII Hard Override (before K-Means):
        If a word matched a PII regex (pii_override ≥ 0.9) → force Encrypt.
        This guarantees SSNs, emails, phone numbers are ALWAYS masked.

    STAGE 2 — K-Means in [PLRS, CIIS, TRS] space:
        For all remaining words, cluster and apply the FULL 8-case
        decision table from ALSA Paper Table 1:

        ┌──────┬──────┬──────┬──────────┐
        │ PLRS │ CIIS │ TRS  │ Action   │
        ├──────┼──────┼──────┼──────────┤
        │ High │ High │ High │ Encrypt  │
        │ High │ High │ Low  │ Encrypt  │
        │ High │ Low  │ High │ Replace  │
        │ High │ Low  │ Low  │ Delete   │
        │ Low  │ High │ High │ Retain   │
        │ Low  │ High │ Low  │ Retain   │
        │ Low  │ Low  │ High │ Retain   │
        │ Low  │ Low  │ Low  │ Delete   │
        └──────┴──────┴──────┴──────────┘

    Returns:
        cluster_ids : np.ndarray (n_words,)
        actions     : np.ndarray (n_words, dtype=object)
    """
    n = len(plrs)
    actions     = np.empty(n, dtype=object)
    cluster_ids = np.zeros(n, dtype=int)

    # ── Stage 1: PII hard override ──────────────────────────────────────
    pii_mask = pii_override >= 0.9  # regex-confirmed PII
    actions[pii_mask] = "Encrypt"

    # ── Stage 2: K-Means for the rest ──────────────────────────────────
    remaining_idx = np.where(~pii_mask)[0]
    if len(remaining_idx) == 0:
        return cluster_ids, actions

    X = np.stack([plrs, ciis, trs], axis=1).astype(np.float64)  # (n, 3)
    k = min(KMEANS_N_CLUSTERS, len(remaining_idx), n)
    km = KMeans(n_clusters=k, n_init=15, random_state=42)
    all_cluster_ids = km.fit_predict(X)   # fit on all n points for consistency
    cluster_ids[:] = all_cluster_ids
    centroids = km.cluster_centers_       # (k, 3)

    mean_plrs = plrs.mean()
    mean_ciis = ciis.mean()
    mean_trs  = trs.mean()

    # Cluster → action mapping (ALSA Paper Table 1 — full 8-case table)
    cluster_action: Dict[int, str] = {}
    for c in range(k):
        cp, cc, ct = centroids[c]
        p_high = cp > mean_plrs
        c_high = cc > mean_ciis
        t_high = ct > mean_trs

        if p_high and c_high:                  # (H, H, *) → Encrypt
            cluster_action[c] = "Encrypt"
        elif p_high and not c_high and t_high: # (H, L, H) → Replace
            cluster_action[c] = "Replace"
        elif p_high and not c_high and not t_high:  # (H, L, L) → Delete
            cluster_action[c] = "Delete"
        elif not p_high and c_high:            # (L, H, *) → Retain
            cluster_action[c] = "Retain"
        elif not p_high and not c_high and t_high:  # (L, L, H) → Retain
            cluster_action[c] = "Retain"
        else:                                  # (L, L, L) → Delete
            cluster_action[c] = "Delete"

    for i in remaining_idx:
        actions[i] = cluster_action.get(int(cluster_ids[i]), "Retain")

    return cluster_ids, actions


def _build_masked_prompt(words: List[str], actions: np.ndarray) -> str:
    """
    Build the intermediate masked version of the prompt.
    Encrypt words become [MASK], Replace uses WordNet synonyms,
    Delete is omitted, Retain is kept verbatim.
    """
    out: List[str] = []
    for word, action in zip(words, actions):
        if action == "Retain":
            out.append(word)
        elif action == "Delete":
            pass
        elif action == "Replace":
            syns = get_wordnet_synonyms(word, n=3)
            out.append(syns[0] if syns else word)
        elif action == "Encrypt":
            out.append(MASK_TOKEN)
    return " ".join(out)


async def _generate_encrypt_candidates(
    masked_prompt: str,
    original_prompt: str,
    n_candidates: int = ENCRYPT_NUM_CANDIDATES,
) -> List[str]:
    """
    Generate m candidate prompts by asking Llama-3 to fill [MASK]
    tokens in the masked prompt.

    Each candidate is a complete prompt with [MASK] replaced by
    FICTIONAL, NON-IDENTIFYING substitutes that preserve sentence
    structure but do NOT leak real PII.  Per ALSA Appendix A:
    "words requiring encryption are entirely replaced with distinct
    alternative expressions that preserve semantic integrity."
    """
    query = (
        f"Below is a prompt with [MASK] tokens replacing sensitive information.\n\n"
        f"Masked prompt: \"{masked_prompt}\"\n\n"
        f"Generate {n_candidates} different versions of this prompt where each [MASK] "
        f"is replaced with a FICTIONAL, NON-IDENTIFYING substitute. Rules:\n"
        f"- Names must be clearly fictional (e.g. 'Person A', 'Entity X')\n"
        f"- Numbers must be obviously fake placeholders (e.g. 'XXX-XX-XXXX')\n"
        f"- Medical/legal terms should be replaced with generic categories "
        f"(e.g. 'a medical condition', 'a medication')\n"
        f"- Preserve the sentence structure and grammatical correctness\n"
        f"- Do NOT use real-sounding names, addresses, or identifying details\n\n"
        f"Format your response as:\n"
        + "".join(f"{i+1}. <completed prompt>\n" for i in range(n_candidates))
        + "\nProvide ONLY the numbered prompts, nothing else."
    )
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, _llama_generate, query, ENCRYPT_CANDIDATE_TOKENS
        )
        candidates: List[str] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                candidate = match.group(1).strip().strip('"').strip("'")
                if candidate:
                    candidates.append(candidate)

        if not candidates:
            log.warning("No candidates parsed from LLM response, falling back")
            return [masked_prompt]

        return candidates[:n_candidates]
    except Exception as exc:
        log.warning("Encrypt candidate generation failed: %s", exc)
        return [masked_prompt]


async def _score_candidates(
    original_prompt: str,
    candidates: List[str],
) -> int:
    """
    Encrypt Self-Recommendation: select the best candidate prompt.

    Score(P'_i) = α × Fluency(P'_i)
               + β × Coherence(P'_i, P)
               + γ × TaskConsistency(P'_i, P)

    where α = ENCRYPT_ALPHA, β = ENCRYPT_BETA, γ = ENCRYPT_GAMMA.

    Uses a single LLM call with a self-recommendation prompt that
    asks the model to holistically evaluate all three criteria.
    Returns the 0-indexed index of the best candidate.
    """
    if len(candidates) == 1:
        return 0

    candidate_list = "\n".join(
        f'{i+1}. "{c}"' for i, c in enumerate(candidates)
    )
    query = (
        f"You are an expert in language understanding and evaluation.\n\n"
        f"Reference prompt: \"{original_prompt}\"\n\n"
        f"Candidate prompts:\n{candidate_list}\n\n"
        f"Instruction: Select the prompt that is most fluent, coherent, "
        f"and best achieves the same task result as the reference prompt. "
        f"Consider:\n"
        f"- Fluency (weight: {ENCRYPT_ALPHA}): grammatical correctness and naturalness\n"
        f"- Coherence (weight: {ENCRYPT_BETA}): logical consistency with the reference\n"
        f"- Task Consistency (weight: {ENCRYPT_GAMMA}): achieves the same goal\n\n"
        f"Provide ONLY the number."
    )
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, _llama_generate, query, 8
        )
        match = re.search(r'(\d+)', response)
        if match:
            idx = int(match.group(1)) - 1  # 1-indexed → 0-indexed
            if 0 <= idx < len(candidates):
                return idx
        log.warning("Could not parse candidate index from: %r", response)
        return 0
    except Exception as exc:
        log.warning("Self-recommendation scoring failed: %s", exc)
        return 0


async def reconstruct_prompt_with_llm(
    words: List[str],
    actions: np.ndarray,
    original_prompt: str,
) -> Tuple[str, str, List[str]]:
    """
    Task 4.3 — Apply per-word actions with LLM-powered Encrypt.

    For non-Encrypt actions, behaviour is identical to the simple version.
    For Encrypt actions:
        1. Build masked prompt (word → [MASK])
        2. Generate m candidates via Llama-3
        3. Score candidates via self-recommendation
        4. Select: P*_encrypt = argmax_i Score(P'_i)

    Returns:
        sanitised        : str        — final prompt with LLM fills
        masked_prompt    : str        — intermediate [MASK] version
        encrypt_candidates : List[str] — all candidates considered
    """
    masked_prompt = _build_masked_prompt(words, actions)

    # If no [MASK] tokens, no self-recommendation needed
    if MASK_TOKEN not in masked_prompt:
        return masked_prompt, masked_prompt, []

    log.info("Encrypt self-recommendation: generating candidates …")
    candidates = await _generate_encrypt_candidates(
        masked_prompt, original_prompt
    )
    log.info("Generated %d candidates", len(candidates))
    for i, c in enumerate(candidates):
        log.info("  Candidate %d: %s", i + 1, c)

    best_idx = await _score_candidates(original_prompt, candidates)
    log.info("Self-recommendation selected candidate %d", best_idx + 1)

    sanitised = candidates[best_idx]
    return sanitised, masked_prompt, candidates


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(prompt: str) -> Dict:
    """Full end-to-end pipeline. Returns a structured result dict."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    words = prompt.split()
    log.info("Pipeline start | %d words", len(words))

    # ── BERT embeddings (two flavours) ─────────────────────────────────────
    # (a) Isolated — for Isolation Forest (same as reference cache method)
    bert_vecs_isolated = embed_words_isolated_bert(
        words, registry.bert_tokenizer, registry.bert_model, registry.device
    ).numpy()

    # (b) Contextual — for CIIS (richer, sentence-aware representation)
    bert_vecs_contextual = embed_words_in_context_bert(
        words, prompt,
        registry.bert_tokenizer, registry.bert_model, registry.device
    ).numpy()

    # ── Phase 2: PLRS ────────────────────────────────────────────────────
    common_vecs = registry.bert_common_vectors.numpy()
    plrs, pii_override = compute_plrs(
        bert_vecs_isolated, common_vecs, prompt, words,
        registry.spacy_nlp,
        registry.gpt2_tokenizer, registry.gpt2_model, registry.device,
    )
    log.info("PLRS: %s", np.round(plrs, 3))

    # ── Phase 3: CIIS ────────────────────────────────────────────────────
    ciis = compute_ciis(
        words, prompt, registry.spacy_nlp,
        bert_vecs_contextual,
        registry.bert_tokenizer, registry.bert_model, registry.device,
    )
    log.info("CIIS: %s", np.round(ciis, 3))

    # ── Phase 4a: TRS ────────────────────────────────────────────────────
    trs = await compute_trs_batch(prompt, words)
    log.info("TRS : %s", np.round(trs, 3))

    # ── Phase 4b: K-Means + actions ──────────────────────────────────────
    cluster_ids, actions = assign_actions_kmeans(plrs, ciis, trs, pii_override)
    log.info("Actions: %s", actions)

    # ── Phase 4c: Reconstruct (with LLM self-recommendation for Encrypt) ──
    sanitised, masked_prompt, encrypt_candidates = await reconstruct_prompt_with_llm(
        words, actions, prompt
    )
    log.info("Masked:    %s", masked_prompt)
    log.info("Sanitised: %s", sanitised)

    return {
        "words":              words,
        "plrs":               plrs.tolist(),
        "ciis":               ciis.tolist(),
        "trs":                trs.tolist(),
        "pii_override":       pii_override.tolist(),
        "cluster_ids":        cluster_ids.tolist(),
        "actions":            actions.tolist(),
        "sanitised":          sanitised,
        "masked_prompt":      masked_prompt,
        "encrypt_candidates": encrypt_candidates,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI — Request / Response Models
# ══════════════════════════════════════════════════════════════════════════════

class PromptRequest(BaseModel):
    prompt: str


class WordAnalysis(BaseModel):
    word:         str
    plrs:         float
    ciis:         float
    trs:          float
    pii_override: float
    cluster_id:   int
    action:       str


class PipelineResponse(BaseModel):
    original_prompt:    str
    word_analyses:      List[WordAnalysis]
    sanitised_prompt:   str
    masked_prompt:      Optional[str] = None
    encrypt_candidates: Optional[List[str]] = None


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI — Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Meta"])
async def health_check():
    return {
        "status": "ok",
        "device": str(registry.device),
        "models": [BERT_MODEL_NAME, GPT2_MODEL_NAME, SPACY_MODEL_NAME, LLAMA_MODEL_NAME],
    }


@app.post("/analyse", response_model=PipelineResponse, tags=["Pipeline"])
async def analyse_prompt(body: PromptRequest):
    try:
        result = await run_pipeline(body.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    word_analyses = [
        WordAnalysis(
            word         = result["words"][i],
            plrs         = round(result["plrs"][i], 4),
            ciis         = round(result["ciis"][i], 4),
            trs          = round(result["trs"][i], 4),
            pii_override = round(result["pii_override"][i], 4),
            cluster_id   = result["cluster_ids"][i],
            action       = result["actions"][i],
        )
        for i in range(len(result["words"]))
    ]
    return PipelineResponse(
        original_prompt    = body.prompt,
        word_analyses      = word_analyses,
        sanitised_prompt   = result["sanitised"],
        masked_prompt      = result.get("masked_prompt"),
        encrypt_candidates = result.get("encrypt_candidates"),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL TESTING  —  if __name__ == "__main__"
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # 1. Standard everyday sentence
    "The quick brown fox jumps over the lazy dog near the river.",

    # 2. Heavy PII — personal information + SSN
    "My name is John Smith and my social security number is 432-91-8765.",

    # 3. Highly technical jargon
    "The convolutional neural network uses backpropagation and gradient "
    "descent to optimise the cross-entropy loss.",

    # 4. Medical context with sensitive data
    "Patient Alice Johnson aged 54 has been diagnosed with stage-3 "
    "pancreatic adenocarcinoma and is prescribed gemcitabine.",

    # 5. Legal / financial context
    "The defendant transferred 2.4 million dollars to an offshore account "
    "in the Cayman Islands on March 15th.",
]


# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m";  BOLD  = "\033[1m"
    CYAN    = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED     = "\033[91m"; MAGENTA = "\033[95m"
    BLUE    = "\033[94m"; GREY  = "\033[90m"; WHITE  = "\033[97m"

ACTION_COLOUR = {
    "Retain":  C.GREEN,
    "Delete":  C.RED,
    "Replace": C.YELLOW,
    "Encrypt": C.MAGENTA,
}


def _sep(char: str = "═", w: int = 100) -> None:
    print(C.GREY + char * w + C.RESET)


def _header(text: str, w: int = 100) -> None:
    pad = (w - len(text) - 2) // 2
    print(C.BOLD + C.CYAN + "║" + " " * pad + text +
          " " * (w - pad - len(text) - 2) + "║" + C.RESET)


def _score(v: float) -> str:
    col = C.GREEN if v < 0.33 else (C.YELLOW if v < 0.66 else C.RED)
    return f"{col}{v:.3f}{C.RESET}"


def _print_table(result: Dict) -> None:
    words = result["words"]
    W = 16
    print(f"  {C.BOLD}{'Word':<{W}}{'PLRS':>7}{'CIIS':>7}{'TRS':>7}"
          f"{'PII-OVR':>9}{'Clust':>7}{'Action':>10}{C.RESET}")
    print("  " + C.GREY + "─" * 68 + C.RESET)
    for i, word in enumerate(words):
        acol = ACTION_COLOUR.get(result["actions"][i], C.WHITE)
        pii_flag = (C.RED + "  ★" + C.RESET) if result["pii_override"][i] >= 0.9 else ""
        print(
            f"  {C.WHITE}{word:<{W}}{C.RESET}"
            f"  {_score(result['plrs'][i])}"
            f"  {_score(result['ciis'][i])}"
            f"  {_score(result['trs'][i])}"
            f"  {_score(result['pii_override'][i])}"
            f"  {C.BLUE}{result['cluster_ids'][i]:>4}{C.RESET}"
            f"  {acol}{result['actions'][i]:>9}{C.RESET}"
            f"{pii_flag}"
        )


# ── Chart directory (next to this file) ─────────────────────────────────────
_CHART_DIR = Path(__file__).parent / "charts"


def generate_score_chart(result: Dict, test_idx: int) -> None:
    """
    Save a grouped bar chart of PLRS / CIIS / TRS / PII-Override for every
    word in *result*.  The chart is written to::

        <script_dir>/charts/test_<idx>_scores.png

    Action colour bands appear behind each word group:
        Retain  → green   Delete  → red
        Replace → gold    Encrypt → magenta
    """
    _CHART_DIR.mkdir(parents=True, exist_ok=True)

    words    = result["words"]
    plrs     = result["plrs"]
    ciis     = result["ciis"]
    trs      = result["trs"]
    pii_ovr  = result["pii_override"]
    actions  = result["actions"]

    n = len(words)
    x = np.arange(n)
    bar_w = 0.18

    ACTION_FACECOLOR = {
        "Retain":  "#2ecc71",   # green
        "Delete":  "#e74c3c",   # red
        "Replace": "#f1c40f",   # gold
        "Encrypt": "#9b59b6",   # purple
    }
    BAR_COLORS = {
        "PLRS":      "#e74c3c",
        "CIIS":      "#3498db",
        "TRS":       "#2ecc71",
        "PII-Ovr":   "#e67e22",
    }

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # ── Action background bands ──────────────────────────────────────────
    for i, action in enumerate(actions):
        band_col = ACTION_FACECOLOR.get(action, "#888888")
        ax.axvspan(i - 0.45, i + 0.45, color=band_col, alpha=0.12, zorder=0)

    # ── Grouped bars ─────────────────────────────────────────────────────
    offsets    = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]
    score_keys = [("PLRS", plrs), ("CIIS", ciis), ("TRS", trs), ("PII-Ovr", pii_ovr)]

    for (label, scores), offset in zip(score_keys, offsets):
        bars = ax.bar(
            x + offset, scores, bar_w,
            label=label,
            color=BAR_COLORS[label],
            alpha=0.88,
            zorder=2,
            linewidth=0.5,
            edgecolor="#ffffff30",
        )
        # value labels on top of bars
        for bar, val in zip(bars, scores):
            if val > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#dddddd",
                )

    # ── Mean reference lines ─────────────────────────────────────────────
    for (label, scores), ls in zip(
        [("PLRS", plrs), ("CIIS", ciis), ("TRS", trs)],
        ["--", "-.", ":"],
    ):
        ax.axhline(
            np.mean(scores), color=BAR_COLORS[label],
            linewidth=1.0, linestyle=ls, alpha=0.55,
            zorder=1,
        )

    # ── Action labels below x-axis ───────────────────────────────────────
    action_labels = []
    for i, (word, action) in enumerate(zip(words, actions)):
        col = ACTION_FACECOLOR.get(action, "#aaaaaa")
        action_labels.append(
            ax.text(
                i, -0.08, action[0],  # first letter: R / D / P / E
                ha="center", va="top",
                fontsize=8, fontweight="bold",
                color=col,
                transform=ax.get_xaxis_transform(),
            )
        )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        [w if len(w) <= 12 else w[:11] + "…" for w in words],
        rotation=35, ha="right", fontsize=9, color="#cccccc",
    )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score  [0 – 1]", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.set_title(
        f"Test {test_idx} — Word-Level Scores\n"
        f"(dashed lines = per-metric means)",
        color="#e0e0ff", fontsize=12, fontweight="bold", pad=12,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    bar_legend = [mpatches.Patch(color=c, label=l) for l, c in BAR_COLORS.items()]
    action_legend = [
        mpatches.Patch(color=c, alpha=0.55, label=f"{a} (bg)")
        for a, c in ACTION_FACECOLOR.items()
    ]
    ax.legend(
        handles=bar_legend + action_legend,
        loc="upper right",
        fontsize=8,
        framealpha=0.25,
        labelcolor="#dddddd",
        facecolor="#1a1a2e",
        edgecolor="#555577",
    )

    fig.tight_layout()
    out_path = _CHART_DIR / f"test_{test_idx:02d}_scores.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Chart saved → %s", out_path)
    print(f"\n  {C.CYAN}📊 Chart saved → {out_path}{C.RESET}\n")


async def run_tests() -> None:
    print("\n")
    _sep()
    _header("Privacy-Preserving NLP Pipeline v3 — Llama-3 TRS + Self-Recommendation")
    _sep()
    print()

    # Bootstrap registry (mirrors lifespan logic)
    registry.device = detect_device()
    registry.bert_tokenizer, registry.bert_model = load_bert(registry.device)
    registry.gpt2_tokenizer, registry.gpt2_model = load_gpt2(registry.device)
    log.info("Loading SpaCy …")
    registry.spacy_nlp = spacy.load(SPACY_MODEL_NAME)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    registry.bert_common_vectors = load_bert_common_vectors(
        registry.bert_tokenizer, registry.bert_model, registry.device
    )
    registry.llama_model, registry.llama_tokenizer = load_llama()

    log.info("All models ready — running %d test cases …\n", len(TEST_CASES))

    for idx, prompt in enumerate(TEST_CASES, 1):
        _sep("─")
        print(f"\n  {C.BOLD}{C.CYAN}TEST CASE {idx} / {len(TEST_CASES)}{C.RESET}\n")
        print(f"  {C.BOLD}Original Prompt:{C.RESET}")
        print(f"  {C.WHITE}» {prompt}{C.RESET}\n")

        t0 = time.perf_counter()
        try:
            result = await run_pipeline(prompt)
        except Exception as exc:
            print(f"  {C.RED}ERROR: {exc}{C.RESET}\n")
            continue
        elapsed = time.perf_counter() - t0

        print(f"  {C.BOLD}Per-Word Analysis:{C.RESET}  "
              f"(★ = PII regex hard-override)")
        _print_table(result)
        generate_score_chart(result, idx)

        print()
        print(f"  Legend: {C.GREEN}■ Retain{C.RESET}  {C.RED}■ Delete{C.RESET}"
              f"  {C.YELLOW}■ Replace{C.RESET}  {C.MAGENTA}■ Encrypt{C.RESET}")

        # ── Masked intermediate version ──────────────────────────────────
        if result.get("masked_prompt"):
            print(f"\n  {C.BOLD}Masked Prompt:{C.RESET}")
            print(f"  {C.YELLOW}» {result['masked_prompt']}{C.RESET}")

        # ── Encrypt self-recommendation candidates ───────────────────────
        if result.get("encrypt_candidates"):
            print(f"\n  {C.BOLD}Encrypt Self-Recommendation Candidates:{C.RESET}")
            for ci, cand in enumerate(result["encrypt_candidates"], 1):
                marker = f"{C.GREEN}✓{C.RESET}" if cand == result["sanitised"] else " "
                print(f"    {marker} {C.BLUE}{ci}.{C.RESET} {cand}")

        print(f"\n  {C.BOLD}Sanitised Output:{C.RESET}")
        print(f"  {C.GREEN}» {result['sanitised']}{C.RESET}")

        print(f"\n  {C.GREY}⏱  {elapsed:.1f}s{C.RESET}\n")

    _sep()
    _header("All tests complete.")
    _sep()
    print()


if __name__ == "__main__":
    """
    python app.py            → run local test suite
    uvicorn app:app --reload → run as API server
    """
    asyncio.run(run_tests())