"""
================================================================================
  Privacy-Preserving NLP Pipeline  —  app.py
  Author  : Senior ML Engineer / Backend Architect
  Purpose : Thesis-grade, production-ready FastAPI service that evaluates every
            word in a prompt across three metrics (PLRS, CIIS, TRS) and assigns
            one of four anonymisation actions: Retain | Delete | Replace | Encrypt
================================================================================

High-Level Flow
───────────────
  Incoming Prompt
      │
      ▼
  Phase 2 ──► PLRS  (Privacy Leakage Risk Score)
      │            • Isolation-Forest anomaly score   (BERT embeddings)
      │            • Negative Log-Likelihood          (GPT-2 perplexity)
      │
      ▼
  Phase 3 ──► CIIS  (Contextual Information Importance Score)
      │            • Contextual Coherence             (SpaCy POS + Mahalanobis)
      │            • Semantic Distinctiveness         (WordNet mutants + MMD)
      │
      ▼
  Phase 4 ──► TRS   (Task Relevance Score)            [mocked async LLM call]
      │        K-Means clustering in [PLRS, CIIS, TRS] space
      │        Centroid-vs-mean heuristic → Action per cluster
      │        String reconstruction
      ▼
  Sanitised Prompt
"""

# ── Standard Library ────────────────────────────────────────────────────────
import asyncio
import logging
import math
import os
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

# Transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# SpaCy
import spacy

# NLTK
import nltk
from nltk.corpus import wordnet as wn

# Scikit-Learn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# SciPy
from scipy.spatial.distance import mahalanobis

# RBF kernel from sklearn
from sklearn.metrics.pairwise import rbf_kernel

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("privacy_pipeline")

# ── Constants ────────────────────────────────────────────────────────────────
BERT_MODEL_NAME   = "bert-base-uncased"
GPT2_MODEL_NAME   = "gpt2"
SPACY_MODEL_NAME  = "en_core_web_sm"
CACHE_DIR         = Path("./model_cache")
BERT_CACHE_FILE   = CACHE_DIR / "bert_common_word_vectors.pt"
N_COMMON_WORDS    = 10_000       # size of the offline reference distribution
MAX_BERT_TOKENS   = 512          # BERT hard limit
KMEANS_N_CLUSTERS = 8            # number of clusters in 3-D action space
MASK_TOKEN        = "[MASK]"     # replacement for Encrypted words
TRS_MOCK_ITERS    = 5            # iterations to average for mocked TRS call

# POS tags → coherence weight (1.0 = content word; 0.3 = function word)
POS_WEIGHT: Dict[str, float] = {
    "NOUN":  1.0, "PROPN": 1.0,
    "ADJ":   1.0, "ADV":   0.8,
    "VERB":  0.9,
    "ADP":   0.3, "DET":   0.3,
    "CONJ":  0.3, "CCONJ": 0.3,
    "SCONJ": 0.3, "PART":  0.3,
    "PRON":  0.5, "NUM":   0.7,
    "PUNCT": 0.1, "SYM":   0.2,
    "X":     0.5, "INTJ":  0.6,
}

# ── Global Model Registry ─────────────────────────────────────────────────────
# All heavy objects live here so they are loaded ONCE during startup and
# referenced cheaply by every request handler.
class ModelRegistry:
    device: torch.device
    bert_tokenizer: AutoTokenizer
    bert_model: AutoModel
    gpt2_tokenizer: AutoTokenizer
    gpt2_model: AutoModelForCausalLM
    spacy_nlp: spacy.Language
    bert_common_vectors: torch.Tensor   # shape (N_COMMON_WORDS, 768)

registry = ModelRegistry()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Environment & Global Initialisation
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> torch.device:
    """
    Prefer Apple Silicon MPS → CUDA → CPU.
    MPS gives ~5-10× speed-up on M-series Macs for inference.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("Device: Apple MPS (Metal Performance Shaders)")
        return torch.device("mps")
    if torch.cuda.is_available():
        log.info("Device: CUDA GPU — %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    log.info("Device: CPU")
    return torch.device("cpu")


def load_bert(device: torch.device):
    """Load BERT tokeniser + model in evaluation mode."""
    log.info("Loading BERT (%s) …", BERT_MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    mdl = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()
    log.info("BERT loaded — %d parameters", sum(p.numel() for p in mdl.parameters()))
    return tok, mdl


def load_gpt2(device: torch.device):
    """Load GPT-2 tokeniser + causal-LM model in evaluation mode."""
    log.info("Loading GPT-2 (%s) …", GPT2_MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME).to(device).eval()
    log.info("GPT-2 loaded — %d parameters", sum(p.numel() for p in mdl.parameters()))
    return tok, mdl


def _build_bert_common_vectors(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """
    Offline preparation (Task 2.1):
    Pull N_COMMON_WORDS words from NLTK's Brown corpus (most common tokens),
    embed each with BERT [CLS] representation, and cache to disk as a .pt file.

    Why [CLS]? For a single-word input the [CLS] vector encodes the semantic
    context of that word within BERT's training distribution — a useful proxy
    for 'how ordinary' the word is.
    """
    from nltk.corpus import brown
    nltk.download("brown", quiet=True)

    log.info("Building BERT common-word vector cache …")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect most-frequent lowercase alphabetic tokens
    freq: Dict[str, int] = {}
    for word in brown.words():
        w = word.lower()
        if w.isalpha():
            freq[w] = freq.get(w, 0) + 1

    common_words = sorted(freq, key=freq.get, reverse=True)[:N_COMMON_WORDS]  # type: ignore[arg-type]
    log.info("  Collected %d common words; embedding …", len(common_words))

    vectors: List[torch.Tensor] = []
    batch_size = 64
    for i in range(0, len(common_words), batch_size):
        batch = common_words[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Take [CLS] token (index 0) from last_hidden_state
        cls_vecs = out.last_hidden_state[:, 0, :].cpu()  # (B, 768)
        vectors.append(cls_vecs)

    all_vecs = torch.cat(vectors, dim=0)  # (N_COMMON_WORDS, 768)
    torch.save(all_vecs, BERT_CACHE_FILE)
    log.info("  Cached %s vectors → %s", all_vecs.shape, BERT_CACHE_FILE)
    return all_vecs


def load_bert_common_vectors(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """Load cached BERT common-word vectors or build them on first run."""
    if BERT_CACHE_FILE.exists():
        log.info("Loading cached BERT vectors from %s …", BERT_CACHE_FILE)
        vecs = torch.load(BERT_CACHE_FILE, map_location="cpu")
        log.info("  Loaded shape: %s", vecs.shape)
        return vecs
    return _build_bert_common_vectors(tokenizer, model, device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    All heavy model loading happens here BEFORE the server accepts requests.
    This ensures zero cold-start latency per request.
    """
    # ── Device ────────────────────────────────────────────────────────────
    registry.device = detect_device()

    # ── BERT ──────────────────────────────────────────────────────────────
    registry.bert_tokenizer, registry.bert_model = load_bert(registry.device)

    # ── GPT-2 ─────────────────────────────────────────────────────────────
    registry.gpt2_tokenizer, registry.gpt2_model = load_gpt2(registry.device)

    # ── SpaCy ─────────────────────────────────────────────────────────────
    log.info("Loading SpaCy (%s) …", SPACY_MODEL_NAME)
    registry.spacy_nlp = spacy.load(SPACY_MODEL_NAME)

    # ── NLTK WordNet ──────────────────────────────────────────────────────
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # ── BERT Common-Word Vectors (offline cache) ──────────────────────────
    registry.bert_common_vectors = load_bert_common_vectors(
        registry.bert_tokenizer, registry.bert_model, registry.device
    )

    log.info("=== All models loaded. Server ready. ===")
    yield  # ← application runs here
    log.info("=== Shutting down. Releasing models. ===")


# ── FastAPI App Instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="Privacy-Preserving NLP Pipeline",
    description="Word-level anonymisation via PLRS · CIIS · TRS metrics",
    version="1.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY — BERT Embedding Helper
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def embed_sentences_bert(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Embed a list of sentences with BERT and return the [CLS] vector for each.

    Args:
        sentences : list of raw strings
        tokenizer : BERT tokeniser (loaded in registry)
        model     : BERT model (loaded in registry)
        device    : compute device
        batch_size: micro-batch size to avoid OOM

    Returns:
        Tensor of shape (len(sentences), 768)
    """
    all_cls: List[torch.Tensor] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_BERT_TOKENS,
        ).to(device)
        out = model(**enc)
        # last_hidden_state: (B, seq_len, 768)
        cls = out.last_hidden_state[:, 0, :].cpu()  # (B, 768)
        all_cls.append(cls)
    return torch.cat(all_cls, dim=0)  # (N, 768)


@torch.no_grad()
def embed_words_in_context_bert(
    words: List[str],
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """
    Embed individual words using their contextualised BERT representation.

    Strategy:
        1. Tokenise the full prompt.
        2. Run BERT once; obtain last_hidden_state (seq_len, 768).
        3. For each surface word, find the first sub-token position that
           corresponds to it and use that sub-token's hidden state.

    This gives a richer, context-sensitive embedding vs. embedding each
    word in isolation.

    Returns:
        Tensor of shape (len(words), 768)
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_BERT_TOKENS,
    ).to(device)

    out = model(**enc)
    hidden = out.last_hidden_state[0].cpu()  # (seq_len, 768)

    # Build a token-id → word-index mapping via BERT's offset_mapping
    enc_with_offsets = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_BERT_TOKENS,
        return_offsets_mapping=True,
    )
    offsets = enc_with_offsets["offset_mapping"][0].tolist()  # [(start, end), …]

    # Tokenise to surface words (simple whitespace split keeps punctuation attached)
    surface_tokens = prompt.split()

    # For each surface word find its character-span in the original string
    char_pos = 0
    word_spans: List[Tuple[int, int]] = []
    for w in surface_tokens:
        start = prompt.find(w, char_pos)
        end = start + len(w)
        word_spans.append((start, end))
        char_pos = end

    word_vecs: List[torch.Tensor] = []
    for (ws, we) in word_spans:
        # Find the first sub-token whose span overlaps the word
        best_idx = 1  # fallback: first non-[CLS] token
        for tok_i, (ts, te) in enumerate(offsets):
            if ts >= ws and te <= we and ts < te:
                best_idx = tok_i
                break
        word_vecs.append(hidden[best_idx])  # (768,)

    return torch.stack(word_vecs, dim=0)  # (len(words), 768)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Privacy Leakage Risk Score (PLRS)
# ══════════════════════════════════════════════════════════════════════════════

def _minmax_scale_1d(arr: np.ndarray) -> np.ndarray:
    """
    Min-Max normalisation to [0, 1].

        x_scaled = (x - x_min) / (x_max - x_min + ε)

    The small ε prevents division-by-zero when all values are identical
    (e.g., single-word prompt or all tokens receive the same score).
    """
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def compute_intrinsic_sensitivity(
    prompt_word_vecs: np.ndarray,
    common_vecs: np.ndarray,
) -> np.ndarray:
    """
    Task 2.1 — Intrinsic Sensitivity via Isolation Forest.

    Isolation Forest is an unsupervised anomaly-detection algorithm.
    It works by randomly partitioning the feature space; anomalous points
    (short average path lengths to isolation) receive negative scores.

    Here the 'normal' distribution is the embedding space of common English
    words. Prompt words that lie far from this distribution (rare,
    domain-specific, or PII-like terms) receive HIGH anomaly scores —
    indicative of HIGH sensitivity.

    Args:
        prompt_word_vecs : np.ndarray (n_words, 768)   — prompt embeddings
        common_vecs      : np.ndarray (N_COMMON, 768)  — reference distribution

    Returns:
        anomaly_scores_scaled : np.ndarray (n_words,)  in [0, 1]
                                (0 = totally ordinary, 1 = highly anomalous)
    """
    # Combine reference + prompt vectors so the forest sees both distributions
    combined = np.vstack([common_vecs, prompt_word_vecs])  # (N+n, 768)

    # Isolation Forest hyper-params:
    #   contamination='auto' lets sklearn estimate the anomaly fraction.
    #   n_estimators=100 balances accuracy vs. speed.
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(combined)

    # score_samples returns negative anomaly scores (lower = more anomalous).
    # We slice off only the prompt-word portion.
    raw_scores = iso.score_samples(prompt_word_vecs)  # (n_words,)

    # Negate so that HIGH value = HIGH anomaly (= HIGH sensitivity)
    anomaly = -raw_scores  # now: higher ⟹ more anomalous

    return _minmax_scale_1d(anomaly)


@torch.no_grad()
def compute_exposure_risk(
    prompt: str,
    words: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> np.ndarray:
    """
    Task 2.2 — Exposure Risk via Negative Log-Likelihood (NLL).

    Intuition:
        If GPT-2 assigns HIGH probability to a word given the preceding
        context, the word is PREDICTABLE — low surprise, low exposure risk.
        If GPT-2 assigns LOW probability (high NLL), the word is UNEXPECTED —
        it may be rare, specific, or PII-like, making it risky to expose.

    Formula:
        P(w_t | w_1 … w_{t-1}) extracted from GPT-2 softmax logits.
        NLL_t = -log₂( P(w_t | context) )

    The result is Min-Max scaled to [0, 1].

    Args:
        prompt    : raw prompt string
        words     : surface words (space-split)
        tokenizer : GPT-2 tokeniser
        model     : GPT-2 causal LM
        device    : compute device

    Returns:
        nll_scaled : np.ndarray (n_words,) in [0, 1]
    """
    # GPT-2 uses its own vocabulary; set padding token = eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]  # (1, seq_len)

    # Forward pass; logits shape: (1, seq_len, vocab_size)
    out = model(**enc)
    logits = out.logits[0]  # (seq_len, vocab_size)

    # Shift: logits[i] predicts token[i+1], so we align accordingly.
    # logits[:-1] → probabilities for tokens[1:]
    shift_logits = logits[:-1, :]  # (seq_len-1, vocab_size)
    shift_ids    = input_ids[0, 1:]  # (seq_len-1,)

    # Softmax over vocabulary dimension
    probs = F.softmax(shift_logits, dim=-1)  # (seq_len-1, vocab_size)

    # Probability assigned to each actual token
    token_probs = probs[range(len(shift_ids)), shift_ids]  # (seq_len-1,)

    # NLL in bits: -log₂(p).  Add ε to avoid log(0).
    nll_tokens = -torch.log2(token_probs + 1e-9).cpu().numpy()  # (seq_len-1,)

    # ── Map GPT-2 token NLLs → surface words ────────────────────────────
    # GPT-2 uses Byte-Pair Encoding (BPE), so one surface word may span
    # multiple GPT-2 tokens. We attribute a word's NLL as the mean of its
    # constituent tokens' NLLs.
    gpt2_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    # gpt2_tokens[0] is the first real token (GPT-2 has no [CLS])
    # For alignment, we work with tokens index 1…end matching nll_tokens

    word_nlls: List[float] = []
    surface_lower = [w.lower() for w in words]

    # Reconstruct surface strings by decoding BPE tokens progressively
    # and greedily matching them to surface words.
    reconstructed = ""
    tok_buf: List[float] = []
    word_idx = 0
    current_word_target = surface_lower[word_idx] if surface_lower else ""

    # We iterate over (token_id, nll) pairs starting from position 1
    for t_idx in range(len(nll_tokens)):
        gpt2_tok_str = tokenizer.decode([input_ids[0, t_idx + 1].item()]).strip()
        tok_buf.append(nll_tokens[t_idx])
        reconstructed = (reconstructed + gpt2_tok_str).lower().strip()

        # Check if we have accumulated enough tokens to form the next surface word
        if reconstructed == current_word_target:
            word_nlls.append(float(np.mean(tok_buf)))
            tok_buf = []
            reconstructed = ""
            word_idx += 1
            if word_idx < len(surface_lower):
                current_word_target = surface_lower[word_idx]
            else:
                break

    # Pad / trim to match n_words (in case BPE alignment diverges slightly)
    while len(word_nlls) < len(words):
        word_nlls.append(float(np.mean(nll_tokens)) if len(nll_tokens) else 0.5)
    word_nlls = word_nlls[: len(words)]

    return _minmax_scale_1d(np.array(word_nlls, dtype=np.float32))


def compute_plrs(
    prompt_word_vecs: np.ndarray,
    common_vecs: np.ndarray,
    prompt: str,
    words: List[str],
    gpt2_tokenizer: AutoTokenizer,
    gpt2_model: AutoModelForCausalLM,
    device: torch.device,
) -> np.ndarray:
    """
    Task 2.3 — Final PLRS for each word.

    PLRS_i = Intrinsic_Sensitivity_i × Exposure_Risk_i

    The multiplicative combination ensures that a word scores HIGH only when
    it is BOTH semantically anomalous AND statistically surprising to the
    language model — a conservative but precise privacy-risk signal.

    Returns:
        plrs : np.ndarray (n_words,) in [0, 1]
    """
    intrinsic  = compute_intrinsic_sensitivity(prompt_word_vecs, common_vecs)
    exposure   = compute_exposure_risk(prompt, words, gpt2_tokenizer, gpt2_model, device)
    plrs       = intrinsic * exposure  # element-wise product
    return plrs.astype(np.float32)


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

    Intuition:
        A word that is semantically DISTANT from the rest of the sentence
        (in a POS-weighted embedding space) is structurally IMPORTANT — it
        carries unique, non-redundant information. If we deleted it, the
        sentence would lose coherence.

    Mahalanobis distance:
        d_M(x, μ) = √[ (x - μ)ᵀ · Σ⁻¹ · (x - μ) ]

    where Σ is the covariance matrix of all word vectors in the sentence,
    and μ is the POS-weighted centroid.

    We weight each word's embedding by its POS-importance coefficient
    before computing the centroid, so function words (articles, prepositions)
    have reduced influence on what the 'centre' of the sentence looks like.

    Args:
        words         : surface word list
        prompt        : original prompt string (used by SpaCy)
        spacy_nlp     : loaded SpaCy pipeline
        bert_word_vecs: np.ndarray (n_words, 768)

    Returns:
        coherence_scaled : np.ndarray (n_words,) in [0, 1]
    """
    doc = spacy_nlp(prompt)

    # Build per-word POS weight array
    # SpaCy tokenises differently than str.split(); we align greedily by text.
    spacy_tokens = [tok for tok in doc if not tok.is_space]
    pos_weights = np.ones(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        for st in spacy_tokens:
            if st.text.lower() == w.lower():
                pos_weights[i] = POS_WEIGHT.get(st.pos_, 0.5)
                break

    V = bert_word_vecs  # (n, 768)

    # POS-weighted centroid: μ = Σ w_i * v_i / Σ w_i
    w_col = pos_weights.reshape(-1, 1)          # (n, 1)
    centroid = (w_col * V).sum(axis=0) / (w_col.sum() + 1e-9)  # (768,)

    # Covariance matrix of word vectors (needed for Mahalanobis)
    # np.cov expects features in rows, observations in columns.
    if V.shape[0] > 1:
        cov = np.cov(V.T)  # (768, 768)
        # Regularise: add λI to avoid singular covariance matrix
        # (sentence has far fewer samples than dimensions)
        lam = 1e-3
        cov_reg = cov + lam * np.eye(cov.shape[0])
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(cov_reg.shape[0])
    else:
        cov_inv = np.eye(V.shape[1])

    # Mahalanobis distance from each word to the POS-weighted centroid
    distances = np.array(
        [mahalanobis(V[i], centroid, cov_inv) for i in range(len(words))],
        dtype=np.float32,
    )

    return _minmax_scale_1d(distances)


def get_wordnet_synonyms(word: str, n: int = 5) -> List[str]:
    """
    Retrieve up to `n` WordNet synonyms for `word`.

    WordNet organises words into synsets (synonym sets). We flatten all
    lemmas across all synsets, de-duplicate, and exclude the word itself.
    Multi-word lemmas (e.g., 'happy_go_lucky') are filtered out because
    they cannot cleanly replace a single word.

    Returns empty list if no synonyms are found.
    """
    syns: List[str] = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and " " not in candidate:
                syns.append(candidate)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in syns:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
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

    Intuition:
        If swapping a word with its synonyms drastically changes the sentence's
        BERT [CLS] embedding, that word is SEMANTICALLY CRITICAL — its exact
        choice carries non-substitutable meaning.

    MMD (simplified, single-kernel):
        MMD²(X, Y) ≈ mean_k( k(xᵢ, xⱼ) ) - 2·mean_k( k(xᵢ, yⱼ) ) + mean_k( k(yᵢ, yⱼ) )

        where k is the RBF (Gaussian) kernel  k(x, y) = exp(-γ‖x - y‖²).

    Here X = {[CLS] of original sentence}, Y = {[CLS] of mutant sentences}.
    A large MMD means the mutant sentences are semantically distant from the
    original, so the swapped word mattered a lot.

    Args:
        words         : surface word list
        prompt        : original prompt string
        bert_tokenizer, bert_model, device : BERT components

    Returns:
        distinctiveness_scaled : np.ndarray (n_words,) in [0, 1]
    """
    # Embed original sentence once
    orig_cls = embed_sentences_bert([prompt], bert_tokenizer, bert_model, device)  # (1, 768)

    mmd_scores: List[float] = []
    for word in words:
        synonyms = get_wordnet_synonyms(word, n=5)
        if not synonyms:
            # No synonyms → assume word is highly specific/rare → high distinctiveness
            mmd_scores.append(1.0)
            continue

        # Create mutant sentences by substituting the word with each synonym
        mutants: List[str] = []
        for syn in synonyms:
            # Case-preserving substitution using regex word boundaries
            mutant = re.sub(r"\b" + re.escape(word) + r"\b", syn, prompt, flags=re.IGNORECASE)
            mutants.append(mutant)

        mutant_cls = embed_sentences_bert(mutants, bert_tokenizer, bert_model, device)  # (m, 768)

        # Compute MMD² using RBF kernel (sklearn default γ = 1/n_features)
        gamma = 1.0 / orig_cls.shape[1]  # 1/768

        # k(X, X): kernel of original with itself → scalar since X has 1 row
        kXX = float(rbf_kernel(orig_cls.numpy(), orig_cls.numpy(), gamma=gamma).mean())
        # k(X, Y): cross-kernel between original and mutants
        kXY = float(rbf_kernel(orig_cls.numpy(), mutant_cls.numpy(), gamma=gamma).mean())
        # k(Y, Y): kernel of mutants with themselves
        kYY = float(rbf_kernel(mutant_cls.numpy(), mutant_cls.numpy(), gamma=gamma).mean())

        mmd2 = kXX - 2 * kXY + kYY  # MMD² (can be slightly negative due to estimation)
        mmd_scores.append(max(mmd2, 0.0))  # clamp to non-negative

    return _minmax_scale_1d(np.array(mmd_scores, dtype=np.float32))


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
    Task 3.3 — Final CIIS for each word.

    CIIS_i = 0.4 × Contextual_Coherence_i + 0.6 × Semantic_Distinctiveness_i

    Weighting rationale:
        • Semantic Distinctiveness (0.6) — a word's irreplaceability is the
          stronger signal for importance; synonym-sensitivity directly captures
          meaning-carrying capacity.
        • Contextual Coherence (0.3)    — structural role (POS) matters, but
          structural words (articles, preps) should still be deletable even
          when Mahalanobis says they're central.

    Returns:
        ciis : np.ndarray (n_words,) in [0, 1]
    """
    coh  = compute_contextual_coherence(words, prompt, spacy_nlp, bert_word_vecs)
    dist = compute_semantic_distinctiveness(words, prompt, bert_tokenizer, bert_model, device)
    ciis = 0.4 * coh + 0.6 * dist
    return ciis.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Integration, Clustering, and Action Assignment
# ══════════════════════════════════════════════════════════════════════════════

async def get_trs(prompt: str, word: str) -> float:
    """
    Task 4.1 — Task Relevance Score (TRS) — mocked async LLM call.

    In production this would call an external LLM API:
        POST /v1/chat/completions
        { "messages": [ {"role":"user", "content": f"Rate the task-relevance
                          of '{word}' in this prompt on a scale 0-1: {prompt}"} ] }

    Here we mock the call with:
        • A simulated network delay (0–50 ms per iteration)
        • A reproducible but word-specific pseudo-random score based on
          the hash of (word, prompt), averaged over TRS_MOCK_ITERS iterations
          to mimic real LLM stochasticity.

    Returns:
        trs : float in [0, 1]
    """
    scores: List[float] = []
    for iteration in range(TRS_MOCK_ITERS):
        # Simulate network I/O without blocking the event loop
        await asyncio.sleep(random.uniform(0.001, 0.010))

        # Deterministic pseudo-random score seeded on (word, prompt, iteration)
        seed = hash((word.lower(), prompt[:50], iteration)) % (2**32)
        rng  = np.random.RandomState(seed)

        # Heuristic: longer, less common words tend to be more task-relevant
        base_score = min(1.0, len(word) / 12)  # length proxy
        noise      = rng.uniform(-0.15, 0.15)
        scores.append(float(np.clip(base_score + noise, 0.0, 1.0)))

    return float(np.mean(scores))


async def compute_trs_batch(prompt: str, words: List[str]) -> np.ndarray:
    """
    Run TRS computation for all words concurrently via asyncio.gather.
    This is crucial for throughput — we don't want O(n) sequential LLM calls.
    """
    tasks = [get_trs(prompt, w) for w in words]
    results = await asyncio.gather(*tasks)
    return np.array(results, dtype=np.float32)


def assign_actions_kmeans(
    plrs: np.ndarray,
    ciis: np.ndarray,
    trs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 4.2 — K-Means Action Assignment in 3-D [PLRS, CIIS, TRS] space.

    Approach:
        1. Stack scores into a matrix X of shape (n_words, 3).
        2. Fit K-Means with KMEANS_N_CLUSTERS clusters.
        3. Compute per-dimension means across the entire prompt.
        4. For each cluster centroid, compare its coordinates against the
           prompt means to determine the cluster's anonymisation action.

    Decision Heuristic (centroid vs. prompt mean):
    ┌─────────────────────────────────────────────────────┬───────────┐
    │  PLRS > mean_PLRS  AND  CIIS > mean_CIIS            │  Encrypt  │
    │  PLRS > mean_PLRS  AND  CIIS ≤ mean_CIIS            │  Delete   │
    │  PLRS ≤ mean_PLRS  AND  TRS  > mean_TRS             │  Retain   │
    │  PLRS ≤ mean_PLRS  AND  TRS  ≤ mean_TRS (CIIS high) │  Replace  │
    │  default (fallback)                                  │  Retain   │
    └─────────────────────────────────────────────────────┴───────────┘

    Rationale:
        • Encrypt  → word is BOTH sensitive AND structurally important;
                     we can't delete it without breaking coherence, so we mask.
        • Delete   → word is sensitive but expendable; removing it is safe.
        • Retain   → word is low-risk AND task-critical; keep it.
        • Replace  → word is low-risk, low task-relevance, replaceable by synonym;
                     swap to reduce PII surface while preserving rough meaning.

    Returns:
        cluster_ids : np.ndarray (n_words,)  — which K-Means cluster each word belongs to
        actions     : np.ndarray (n_words,)  — string action per word
    """
    X = np.stack([plrs, ciis, trs], axis=1).astype(np.float64)  # (n, 3)

    # Use min(KMEANS_N_CLUSTERS, n_words) in case prompt has very few words
    k = min(KMEANS_N_CLUSTERS, len(plrs))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(X)          # (n_words,)
    centroids   = km.cluster_centers_        # (k, 3)  — [PLRS, CIIS, TRS]

    mean_plrs = plrs.mean()
    mean_ciis = ciis.mean()
    mean_trs  = trs.mean()

    # Map each cluster to an action
    cluster_action: Dict[int, str] = {}
    for c in range(k):
        cp, cc, ct = centroids[c]  # centroid PLRS, CIIS, TRS

        if cp > mean_plrs and cc > mean_ciis:
            action = "Encrypt"   # high sensitivity + high importance → mask
        elif cp > mean_plrs and cc <= mean_ciis:
            action = "Delete"    # high sensitivity + low importance → drop
        elif cp <= mean_plrs and ct > mean_trs:
            action = "Retain"    # low sensitivity + high relevance → keep
        elif cp <= mean_plrs and ct <= mean_trs and cc >= mean_ciis:
            action = "Replace"   # low sensitivity + low relevance but coherent → synonym
        else:
            action = "Retain"    # safe default

        cluster_action[c] = action

    actions = np.array([cluster_action[cid] for cid in cluster_ids])
    return cluster_ids, actions


def reconstruct_prompt(
    words: List[str],
    actions: np.ndarray,
) -> str:
    """
    Task 4.3 — Apply actions to reconstruct the sanitised prompt.

    Actions:
        Retain  → keep word verbatim
        Delete  → omit word entirely
        Replace → swap with first available WordNet synonym; fall back to Retain
                  if no synonyms exist (to preserve readability)
        Encrypt → replace with MASK_TOKEN ('[MASK]')

    Returns the sanitised prompt as a single joined string.
    """
    output_tokens: List[str] = []
    for word, action in zip(words, actions):
        if action == "Retain":
            output_tokens.append(word)

        elif action == "Delete":
            pass  # simply omit

        elif action == "Replace":
            synonyms = get_wordnet_synonyms(word, n=3)
            if synonyms:
                output_tokens.append(synonyms[0])  # use best synonym
            else:
                output_tokens.append(word)          # fallback: retain

        elif action == "Encrypt":
            output_tokens.append(MASK_TOKEN)

    return " ".join(output_tokens)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(prompt: str) -> Dict:
    """
    Full end-to-end pipeline for a single prompt.

    Returns a structured dict containing:
        - words          : list of surface words
        - plrs           : PLRS scores
        - ciis           : CIIS scores
        - trs            : TRS scores
        - cluster_ids    : K-Means cluster assignment per word
        - actions        : anonymisation action per word
        - sanitised      : the final reconstructed prompt
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    # ── Tokenise into surface words ────────────────────────────────────────
    words = prompt.split()
    n     = len(words)
    log.info("Pipeline start | %d words", n)

    # ── BERT contextual embeddings for all words in this prompt ───────────
    bert_word_vecs_t = embed_words_in_context_bert(
        words, prompt,
        registry.bert_tokenizer, registry.bert_model, registry.device
    )
    bert_word_vecs = bert_word_vecs_t.numpy()  # (n, 768)

    # ── Phase 2: PLRS ──────────────────────────────────────────────────────
    common_vecs = registry.bert_common_vectors.numpy()  # (N_COMMON, 768)
    plrs = compute_plrs(
        bert_word_vecs, common_vecs,
        prompt, words,
        registry.gpt2_tokenizer, registry.gpt2_model, registry.device
    )
    log.info("PLRS computed: %s", np.round(plrs, 3))

    # ── Phase 3: CIIS ──────────────────────────────────────────────────────
    ciis = compute_ciis(
        words, prompt,
        registry.spacy_nlp,
        bert_word_vecs,
        registry.bert_tokenizer, registry.bert_model, registry.device
    )
    log.info("CIIS computed: %s", np.round(ciis, 3))

    # ── Phase 4a: TRS (async) ─────────────────────────────────────────────
    trs = await compute_trs_batch(prompt, words)
    log.info("TRS computed:  %s", np.round(trs, 3))

    # ── Phase 4b: K-Means clustering + action assignment ──────────────────
    cluster_ids, actions = assign_actions_kmeans(plrs, ciis, trs)
    log.info("Actions: %s", actions)

    # ── Phase 4c: String reconstruction ──────────────────────────────────
    sanitised = reconstruct_prompt(words, actions)
    log.info("Sanitised: %s", sanitised)

    return {
        "words":       words,
        "plrs":        plrs.tolist(),
        "ciis":        ciis.tolist(),
        "trs":         trs.tolist(),
        "cluster_ids": cluster_ids.tolist(),
        "actions":     actions.tolist(),
        "sanitised":   sanitised,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI — Request/Response Models
# ══════════════════════════════════════════════════════════════════════════════

class PromptRequest(BaseModel):
    prompt: str


class WordAnalysis(BaseModel):
    word:       str
    plrs:       float
    ciis:       float
    trs:        float
    cluster_id: int
    action:     str


class PipelineResponse(BaseModel):
    original_prompt: str
    word_analyses:   List[WordAnalysis]
    sanitised_prompt: str


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI — Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Meta"])
async def health_check():
    """Liveness probe — confirm models are loaded and device is ready."""
    return {
        "status":  "ok",
        "device":  str(registry.device),
        "models":  ["bert-base-uncased", "gpt2", "en_core_web_sm"],
    }


@app.post("/analyse", response_model=PipelineResponse, tags=["Pipeline"])
async def analyse_prompt(body: PromptRequest):
    """
    Main endpoint: run the full privacy-preserving pipeline on a prompt.

    Returns per-word scores (PLRS, CIIS, TRS), cluster ID, action, and the
    final sanitised (anonymised) string.
    """
    try:
        result = await run_pipeline(body.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    word_analyses = [
        WordAnalysis(
            word       = result["words"][i],
            plrs       = round(result["plrs"][i], 4),
            ciis       = round(result["ciis"][i], 4),
            trs        = round(result["trs"][i], 4),
            cluster_id = result["cluster_ids"][i],
            action     = result["actions"][i],
        )
        for i in range(len(result["words"]))
    ]

    return PipelineResponse(
        original_prompt  = body.prompt,
        word_analyses    = word_analyses,
        sanitised_prompt = result["sanitised"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL TESTING  —  if __name__ == "__main__"
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # 1. Standard everyday sentence
    "The quick brown fox jumps over the lazy dog near the river.",

    # 2. Heavy PII — personal information
    "My name is John Smith and my social security number is 432-91-8765.",

    # 3. Highly technical jargon
    "The convolutional neural network uses backpropagation and gradient descent to optimise the cross-entropy loss.",

    # 4. Medical context with sensitive data
    "Patient Alice Johnson, aged 54, has been diagnosed with stage-3 pancreatic adenocarcinoma and is prescribed gemcitabine.",

    # 5. Legal / financial context
    "The defendant transferred 2.4 million dollars to an offshore account in the Cayman Islands on March 15th.",
]

# ANSI colour codes for pretty terminal output
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"

ACTION_COLOUR = {
    "Retain":  C.GREEN,
    "Delete":  C.RED,
    "Replace": C.YELLOW,
    "Encrypt": C.MAGENTA,
}


def print_separator(char: str = "═", width: int = 90) -> None:
    print(C.GREY + char * width + C.RESET)


def print_header(text: str, width: int = 90) -> None:
    pad = (width - len(text) - 2) // 2
    print(C.BOLD + C.CYAN + "║" + " " * pad + text + " " * (width - pad - len(text) - 2) + "║" + C.RESET)


def format_score(val: float) -> str:
    """Colour-code a 0-1 score: green (low) → yellow (mid) → red (high)."""
    if val < 0.33:
        col = C.GREEN
    elif val < 0.66:
        col = C.YELLOW
    else:
        col = C.RED
    return f"{col}{val:.3f}{C.RESET}"


def print_word_table(result: Dict) -> None:
    """Print a formatted table of per-word metrics."""
    words = result["words"]
    plrs  = result["plrs"]
    ciis  = result["ciis"]
    trs   = result["trs"]
    cids  = result["cluster_ids"]
    acts  = result["actions"]

    # Header row
    col_w = 14
    print(
        f"  {C.BOLD}{'Word':<{col_w}}{'PLRS':>8}{'CIIS':>8}{'TRS':>8}"
        f"{'Cluster':>9}{'Action':>10}{C.RESET}"
    )
    print("  " + C.GREY + "─" * 60 + C.RESET)

    for i, word in enumerate(words):
        act_col = ACTION_COLOUR.get(acts[i], C.WHITE)
        print(
            f"  {C.WHITE}{word:<{col_w}}{C.RESET}"
            f"  {format_score(plrs[i])}"
            f"  {format_score(ciis[i])}"
            f"  {format_score(trs[i])}"
            f"  {C.BLUE}{cids[i]:>5}{C.RESET}"
            f"  {act_col}{acts[i]:>9}{C.RESET}"
        )


async def run_tests() -> None:
    """
    Bootstrap the model registry exactly as the lifespan does, then run
    all test cases, printing richly formatted output to the terminal.
    """
    print("\n")
    print_separator("═")
    print_header("Privacy-Preserving NLP Pipeline — Local Test Suite")
    print_separator("═")
    print()

    # ── Load models (same logic as lifespan, but inline for __main__) ─────
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

    log.info("All models ready — running %d test cases …\n", len(TEST_CASES))

    for case_num, prompt in enumerate(TEST_CASES, start=1):
        print_separator("─")
        print(f"\n  {C.BOLD}{C.CYAN}TEST CASE {case_num} / {len(TEST_CASES)}{C.RESET}\n")
        print(f"  {C.BOLD}Original Prompt:{C.RESET}")
        print(f"  {C.WHITE}» {prompt}{C.RESET}\n")

        try:
            result = await run_pipeline(prompt)
        except Exception as exc:
            print(f"  {C.RED}ERROR: {exc}{C.RESET}\n")
            continue

        print(f"  {C.BOLD}Per-Word Analysis:{C.RESET}")
        print_word_table(result)

        # Legend
        print()
        print(
            f"  Legend:  "
            f"{C.GREEN}■ Retain{C.RESET}  "
            f"{C.RED}■ Delete{C.RESET}  "
            f"{C.YELLOW}■ Replace{C.RESET}  "
            f"{C.MAGENTA}■ Encrypt{C.RESET}"
        )

        print(f"\n  {C.BOLD}Sanitised Output:{C.RESET}")
        print(f"  {C.GREEN}» {result['sanitised']}{C.RESET}\n")

    print_separator("═")
    print_header("All tests complete.")
    print_separator("═")
    print()


if __name__ == "__main__":
    """
    Run the full local test suite.
    Usage:
        python app.py
    Or to start the FastAPI server instead:
        uvicorn app:app --reload --port 8000
    """
    asyncio.run(run_tests())