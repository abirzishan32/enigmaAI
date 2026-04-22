# ── Standard Library ─────────────────────────────────────────────────────────
import asyncio
import json
import logging
import os
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
MASK_TOKEN        = "[MASK]"

# ALSA contextual coherence controls (paper-faithful defaults).
ALSA_COHERENCE_ALPHA = 0.8
ALSA_POSITION_BETA = 0.5
ALSA_COHERENCE_EPS = 1e-9

# Function words are down-weighted in pairwise coherence interactions.
FUNCTION_POS_TAGS = {
    "ADP", "DET", "CCONJ", "SCONJ", "PART", "PRON", "AUX", "PUNCT", "SYM"
}

# Replace-action quality controls.
REPLACE_MIN_FREQUENCY = 2
REPLACE_MIN_SEMANTIC_SIM = 0.90
REPLACE_MAX_CANDIDATES = 10

# ── Llama-3 Backbone (TRS + Encrypt Self-Recommendation) ─────────────
LLAMA_MODEL_NAME          = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
TRS_NUM_ITERS             = 5           # k in TRS(w_i) = (1/k) Σ LLM_j(w_i|P)
TRS_FALLBACK_SCORE        = 0.5         # graceful fallback if LLM fails
TRS_MAX_TOKENS            = 16          # response is a single float

# Encrypt self-recommendation weights:
#   Score(P'_i) = α·Fluency + β·Coherence + γ·TaskConsistency
ENCRYPT_NUM_CANDIDATES    = 3           # m candidate completions (Appendix A)
ENCRYPT_CANDIDATE_TOKENS  = 500         # max_new_tokens per paper experiment setup (§4)
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

# spaCy POS to WordNet POS mapping for POS-safe synonym retrieval.
_SPACY_TO_WN_POS: Dict[str, str] = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

# Lazy Brown-corpus frequency cache used to avoid obscure replacements.
_BROWN_FREQ_CACHE: Optional[Dict[str, int]] = None


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

    from mlx_lm import load as mlx_load
    log.info("Loading Llama-3 backbone (%s) …", LLAMA_MODEL_NAME)
    model, tokenizer = mlx_load(LLAMA_MODEL_NAME)
    log.info("Llama-3 loaded on MLX (Apple Silicon)")
    return model, tokenizer


def _llama_generate(
    prompt_text: str,
    max_tokens: int = 16,
    temp: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> str:
    """
    Single-call Llama-3 inference via mlx_lm ≥ 0.31.

    mlx_lm 0.31+ removed the old temp/top_p kwargs from generate_step.
    Temperature and nucleus/top-k sampling are now configured through a
    sampler callable produced by make_sampler().

    Args:
        temp  : Sampling temperature.  0.0 → greedy (default for TRS).
        top_p : Top-p nucleus threshold.  0.0 → disabled.
        top_k : Top-k threshold.  0 → disabled.
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "user", "content": prompt_text}]
    formatted = registry.llama_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    return mlx_generate(
        registry.llama_model, registry.llama_tokenizer,
        prompt=formatted, max_tokens=max_tokens, verbose=False,
        sampler=sampler,
    )


def _build_bert_common_vectors(tokenizer, model, device) -> torch.Tensor:

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

    return re.sub(r'^[^\w]+|[^\w]+$', '', word) or word


def get_word_char_spans(words: List[str], text: str) -> List[Tuple[int, int]]:

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for w in words:
        idx = text.find(w, cursor)
        if idx == -1:          # defensive: shouldn't happen with split()
            idx = cursor
        spans.append((idx, idx + len(w)))
        cursor = idx + len(w)
    return spans


def _get_brown_frequency_map() -> Dict[str, int]:
    """Load Brown word frequencies once for replacement naturalness scoring."""
    global _BROWN_FREQ_CACHE
    if _BROWN_FREQ_CACHE is None:
        from nltk.corpus import brown
        nltk.download("brown", quiet=True)
        freq: Dict[str, int] = {}
        for w in brown.words():
            lw = w.lower()
            if lw.isalpha():
                freq[lw] = freq.get(lw, 0) + 1
        _BROWN_FREQ_CACHE = freq
    return _BROWN_FREQ_CACHE


def _word_frequency(word: str) -> int:
    return int(_get_brown_frequency_map().get(word.lower(), 0))


def _frequency_score(word: str) -> float:
    """Frequency proxy in [0, 1] for lexical naturalness."""
    freq = _word_frequency(strip_punctuation(word))
    return float(np.clip(np.log1p(freq) / np.log1p(5000.0), 0.0, 1.0))


def _apply_surface_form(source_word: str, replacement: str) -> str:
    """Preserve source casing and edge punctuation for a replacement token."""
    m = re.match(r"^([^\w]*)([\w'-]+)([^\w]*)$", source_word)
    if not m:
        return replacement

    prefix, core, suffix = m.groups()
    rep = replacement
    if core.isupper():
        rep = rep.upper()
    elif core.istitle():
        rep = rep.title()
    return f"{prefix}{rep}{suffix}"


def _get_word_pos_tags(words: List[str], prompt: str, spacy_nlp: spacy.Language) -> List[str]:
    """Map each split-word to the best-overlap spaCy POS tag."""
    doc = spacy_nlp(prompt)
    spans = get_word_char_spans(words, prompt)
    tags: List[str] = []

    for ws, we in spans:
        best_overlap = 0
        best_pos = "X"
        for tok in doc:
            if tok.is_space:
                continue
            ts, te = tok.idx, tok.idx + len(tok.text)
            overlap = max(0, min(we, te) - max(ws, ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = tok.pos_
        tags.append(best_pos)

    return tags


def _robust_scale_1d(arr: np.ndarray) -> np.ndarray:

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


def compute_intrinsic_sensitivity(
    prompt_word_vecs_isolated: np.ndarray,
    common_vecs: np.ndarray,
) -> np.ndarray:
    """
    Paper §3.3.1 + Eq. 1–2.

    Isolation Forest is fit on the common-word reference distribution;
    each prompt word is scored as an outlier in that space.

        OS(w_i)  = 2^(-E[depth(w_i)] / c(n))   ≈ -score_samples()
        IS(w_i)  = (OS(w_i) - min_{v∈V*} OS(v))
                 / (max_{v∈V*} OS(v) - min_{v∈V*} OS(v))   [Eq. 2]

    where V* is the set of words in the input prompt (normalised over
    prompt words only, exactly as stated in the paper).

    Paper hyperparameter: T = 8 decision trees (experiment section:
    "we employ eight decision trees").
    """
    iso = IsolationForest(
        n_estimators=8,          # T=8 as per paper experiment setup
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(common_vecs)                        # FIT on reference only

    # OS(w_i): higher = more anomalous (rare/out-of-distribution)
    prompt_os = -iso.score_samples(prompt_word_vecs_isolated)

    # Eq. 2: min-max normalisation over V* (prompt words only)
    lo = float(np.min(prompt_os))
    hi = float(np.max(prompt_os))
    if hi - lo < 1e-9:
        return np.full(len(prompt_os), 0.5, dtype=np.float32)
    return ((prompt_os - lo) / (hi - lo)).astype(np.float32)


@torch.no_grad()
def compute_exposure_risk(
    prompt: str,
    words: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> np.ndarray:

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


    shift_logits = logits[:-1]                              # (seq_len-1, vocab)
    shift_ids    = input_ids[0, 1:]                         # (seq_len-1,)
    probs        = F.softmax(shift_logits, dim=-1)          # (seq_len-1, vocab)
    token_probs  = probs[range(len(shift_ids)), shift_ids]  # (seq_len-1,)
    nll_tokens   = -torch.log2(token_probs + 1e-9).cpu().numpy()  # (seq_len-1,)

    mean_nll = float(nll_tokens.mean()) if len(nll_tokens) else 5.0


    word_spans = get_word_char_spans(words, prompt)


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
        # Paper Eq. 3: NLL_wi = Σ_{j∈S_wi} nll_tj  (SUM, not mean)
        word_nlls.append(float(np.sum(tok_nlls)) if tok_nlls else mean_nll)

    nll_arr = np.array(word_nlls, dtype=np.float32)
    lo = float(np.min(nll_arr)) if nll_arr.size else 0.0
    hi = float(np.max(nll_arr)) if nll_arr.size else 1.0
    if hi - lo < 1e-9:
        return np.full_like(nll_arr, 0.5, dtype=np.float32)
    return ((nll_arr - lo) / (hi - lo + 1e-9)).astype(np.float32)


def compute_plrs(
    prompt_word_vecs_isolated: np.ndarray,
    common_vecs: np.ndarray,
    prompt: str,
    words: List[str],
    gpt2_tokenizer: AutoTokenizer,
    gpt2_model: AutoModelForCausalLM,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper §3.3 + Eq. 5: PLRS(w_i) = IS(w_i) × E(w_i)

    IS — Intrinsic Sensitivity (Isolation Forest on Wikipedia frequency
         space, §3.3.1 / Eq. 1–2)
    E  — Exposure Risk (autoregressive token-level NLL, §3.3.2 / Eq. 3–4)

    pii_override is computed separately and returned so that
    assign_actions_kmeans can hard-force Encrypt on regex-confirmed PII
    (action-assignment stage, not part of the PLRS score itself).
    """
    intrinsic = compute_intrinsic_sensitivity(prompt_word_vecs_isolated, common_vecs)
    exposure  = compute_exposure_risk(prompt, words, gpt2_tokenizer, gpt2_model, device)

    # Eq. 5: PLRS(w_i) = IS(w_i) × E(w_i)
    plrs = (intrinsic * exposure).astype(np.float32)

    # Regex-confirmed PII override — used only in action assignment, not
    # folded into PLRS so the score stays faithful to the paper formula.
    pii_override = compute_pii_override_scores(words, prompt)

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
    Paper §3.4.2 + Eq. 8–10.

    For each pair (i, j):

        T_ij   = sqrt((v_i - v_j)^T Q_ij (v_i - v_j))       [Eq. 8]
        Q_ij   = I + α * D_ij,   D_ij = d_ij * I,  d_ij = ρ_i * ρ_j
        R_ij   = |q_i - q_j|                                  [Eq. 9]
        r*_ij  = β * R_ij + T_ij
        CC(w_i) = 1 / Σ_{j≠i} (r*_ij + ε)                   [Eq. 10]

    where ρ = 1.0 for content words, 0.3 for function words (Appendix C).

    The paper formula is the INVERSE OF THE SUM of distances, not the
    sum/mean of inverse distances.  A word far from even one other word
    gets a large denominator → small CC (not well-integrated contextually).

    Returns a robustly scaled coherence vector in [0, 1].
    """
    n = len(words)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.array([0.5], dtype=np.float32)

    pos_tags = _get_word_pos_tags(words, prompt, spacy_nlp)

    # ρ = 1.0 for content words, 0.3 for function words (Appendix C, Table 6)
    rho = np.array(
        [0.3 if p in FUNCTION_POS_TAGS else 1.0 for p in pos_tags],
        dtype=np.float64,
    )

    V = np.asarray(bert_word_vecs, dtype=np.float64)
    dim = V.shape[1]
    cc = np.zeros(n, dtype=np.float64)

    for i in range(n):
        dist_sum = 0.0                       # Σ_{j≠i} (r*_ij + ε)
        for j in range(n):
            if i == j:
                continue

            delta = V[i] - V[j]
            d_ij = float(rho[i] * rho[j])

            # Q_ij = I + α * D_ij; diagonal entries = (1 + α * d_ij)
            q_diag = np.full(dim, 1.0 + ALSA_COHERENCE_ALPHA * d_ij, dtype=np.float64)
            quadratic = float(np.sum((delta * delta) * q_diag))
            t_ij = float(np.sqrt(max(quadratic, 0.0)))

            r_ij = float(abs(i - j))         # positional distance [Eq. 9]
            r_star = ALSA_POSITION_BETA * r_ij + t_ij
            dist_sum += r_star + ALSA_COHERENCE_EPS

        # Eq. 10: CC(w_i) = 1 / Σ_{j≠i}(r*_ij + ε)
        cc[i] = 1.0 / dist_sum if dist_sum > 0.0 else 0.0

    return _robust_scale_1d(cc.astype(np.float32))


def get_wordnet_synonyms(word: str, n: int = 5) -> List[str]:
    """
    WordNet synonyms with optional POS filtering and rarity suppression.

    `target_pos` should be a spaCy POS tag (NOUN/VERB/ADJ/ADV).
    """
    return get_wordnet_synonyms_with_pos(word, n=n, target_pos=None)


def get_wordnet_synonyms_with_pos(
    word: str,
    n: int = 5,
    target_pos: Optional[str] = None,
) -> List[str]:
    """POS-safe WordNet synonyms ranked by corpus frequency."""
    clean = strip_punctuation(word)
    wn_pos = _SPACY_TO_WN_POS.get(target_pos or "")
    synsets = wn.synsets(clean, pos=wn_pos) if wn_pos else wn.synsets(clean)

    syns: List[str] = []
    for synset in synsets:
        for lemma in synset.lemmas():
            cand = lemma.name().replace("_", " ")
            if cand.lower() != clean.lower() and " " not in cand and cand.isalpha():
                syns.append(cand)

    seen, unique = set(), []
    for s in syns:
        if s.lower() not in seen:
            seen.add(s.lower()); unique.append(s)

    if not unique:
        return []

    natural = [s for s in unique if _word_frequency(s) >= REPLACE_MIN_FREQUENCY]
    ranked = natural if natural else unique
    ranked.sort(key=lambda s: (_frequency_score(s), -abs(len(s) - len(clean))), reverse=True)
    return ranked[:n]


@torch.no_grad()
def _sentence_semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of BERT [CLS] sentence embeddings."""
    emb = embed_sentences_bert(
        [a, b],
        registry.bert_tokenizer,
        registry.bert_model,
        registry.device,
    ).numpy()
    va, vb = emb[0], emb[1]
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)


@torch.no_grad()
def _sentence_fluency_score(text: str) -> float:
    """Higher is better; computed as negative mean GPT-2 token NLL."""
    tok = registry.gpt2_tokenizer
    mdl = registry.gpt2_model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(registry.device) for k, v in enc.items()}
    out = mdl(**enc)

    logits = out.logits[0, :-1, :]
    target = enc["input_ids"][0, 1:]
    if len(target) == 0:
        return -5.0

    probs = F.softmax(logits, dim=-1)
    token_probs = probs[range(len(target)), target]
    mean_nll = float((-torch.log2(token_probs + 1e-9)).mean().item())
    return -mean_nll


def _llm_suggest_contextual_replacement(
    source_word: str,
    source_pos: str,
    sentence: str,
) -> Optional[str]:
    """Optional one-token replacement from LLM with strict constraints."""
    query = (
        "You are performing lexical substitution in one sentence.\n"
        "Return one COMMON English replacement token for the target word.\n"
        "Rules:\n"
        "- Keep the same part-of-speech.\n"
        "- Preserve sentence meaning.\n"
        "- One token only (letters only).\n"
        "- Avoid rare/archaic/technical words.\n"
        "- Return JSON only: {\"replacement\":\"word\"}.\n\n"
        f"Sentence: \"{sentence}\"\n"
        f"Target word: \"{strip_punctuation(source_word)}\"\n"
        f"Target POS: {source_pos}\n"
    )

    try:
        response = _llama_generate(query, max_tokens=24)
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            parsed = json.loads(match.group(0))
            replacement = str(parsed.get("replacement", "")).strip()
        else:
            fallback = re.search(r"([A-Za-z]+)", response)
            replacement = fallback.group(1) if fallback else ""

        if not replacement or not replacement.isalpha():
            return None
        if _contains_sensitive_patterns(replacement):
            return None
        return replacement.lower()
    except Exception:
        return None


def _select_replacement_word(
    working_words: List[str],
    word_idx: int,
    original_prompt: str,
    source_pos: str,
) -> str:
    """Context-aware lexical substitution for Replace action."""
    source_word = working_words[word_idx]
    if source_pos not in {"NOUN", "VERB", "ADJ", "ADV"}:
        return source_word

    candidate_pool = get_wordnet_synonyms_with_pos(
        source_word,
        n=REPLACE_MAX_CANDIDATES,
        target_pos=source_pos,
    )

    llm_candidate = _llm_suggest_contextual_replacement(
        source_word,
        source_pos,
        " ".join(working_words),
    )
    if llm_candidate and llm_candidate not in candidate_pool:
        candidate_pool.append(llm_candidate)

    if not candidate_pool:
        return source_word

    evaluated: List[Tuple[str, float, float, float]] = []
    for cand in candidate_pool:
        surface = _apply_surface_form(source_word, cand)
        if strip_punctuation(surface).lower() == strip_punctuation(source_word).lower():
            continue

        trial_words = working_words.copy()
        trial_words[word_idx] = surface
        trial_prompt = " ".join(trial_words)

        trial_pos = _get_word_pos_tags(trial_words, trial_prompt, registry.spacy_nlp)[word_idx]
        if trial_pos != source_pos:
            continue

        sim = _sentence_semantic_similarity(original_prompt, trial_prompt)
        if sim < REPLACE_MIN_SEMANTIC_SIM:
            continue

        fluency = _sentence_fluency_score(trial_prompt)
        freq = _frequency_score(surface)
        evaluated.append((surface, sim, fluency, freq))

    if not evaluated:
        return source_word

    if len(evaluated) == 1:
        return evaluated[0][0]

    fluencies = np.array([x[2] for x in evaluated], dtype=np.float32)
    f_min, f_max = float(fluencies.min()), float(fluencies.max())
    f_span = f_max - f_min + 1e-9

    best_word = source_word
    best_score = -1e9
    for word, sim, fluency, freq in evaluated:
        fluency_norm = (fluency - f_min) / f_span
        score = 0.55 * sim + 0.30 * float(fluency_norm) + 0.15 * freq
        if score > best_score:
            best_score = score
            best_word = word

    return best_word


def compute_semantic_distinctiveness(
    words: List[str],
    prompt: str,
    bert_tokenizer: AutoTokenizer,
    bert_model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    """
    Paper §3.4.1 + Eq. 6–7.

    For each word w_i, construct S_{w_i→w'} by replacing only the token at
    position i (positional substitution, not global string replace).

        Δ(S, S_{w_i→w'}) = MMD(Φ(S), Φ(S_{w_i→w'}))          [Eq. 6]
        SD(w_i) = (1/|C(w_i)|) Σ_{w'∈C} Δ(S, S_{w_i→w'})     [Eq. 7]

    For a SINGLE-SAMPLE MMD (one embedding per sentence):
        k(x,x) = 1  and  k(y,y) = 1  (RBF self-kernel)
        MMD²(x, y) = k(x,x) − 2·k(x,y) + k(y,y) = 2(1 − k(x,y))

    This means each Δ is computed INDEPENDENTLY per synonym, then averaged.
    The previous implementation computed one MMD between the original and
    the entire synonym set, which introduced cross-synonym kYY terms not
    present in the paper and therefore produced systematically different
    (lower) SD scores for words with many synonyms.

    BERT sentence embeddings are batched across all synonyms for efficiency;
    only the final aggregation changes.

    Words with no WordNet synonyms receive SD = 0.5 (moderate default).
    """
    orig_cls = embed_sentences_bert([prompt], bert_tokenizer, bert_model, device)  # (1, 768)
    orig_np  = orig_cls.numpy()
    gamma    = 1.0 / orig_cls.shape[1]   # γ = 1/d  (d = 768)

    mmd_scores: List[float] = []
    for word_idx, word in enumerate(words):
        synonyms = get_wordnet_synonyms(word, n=5)
        if not synonyms:
            mmd_scores.append(0.5)   # moderate: irreplaceable ≠ sensitive
            continue

        # Build one S_{w_i→w'} per synonym (positional substitution only)
        mutants: List[str] = []
        for syn in synonyms:
            words_copy = words.copy()
            words_copy[word_idx] = _apply_surface_form(word, syn)
            mutants.append(" ".join(words_copy))

        # Batch-encode all synonym sentences in one BERT forward pass
        mutant_cls = embed_sentences_bert(mutants, bert_tokenizer, bert_model, device)
        mutant_np  = mutant_cls.numpy()  # (|C|, 768)

        # Eq. 6: Δ(S, S_{w→w'}) = MMD²(orig, syn_i) = 2(1 − k(orig, syn_i))
        # Eq. 7: SD = mean over individual per-synonym MMD² values
        individual_mmds: List[float] = []
        for i in range(len(synonyms)):
            k_xy = float(rbf_kernel(orig_np, mutant_np[[i]], gamma=gamma)[0][0])
            individual_mmds.append(max(2.0 * (1.0 - k_xy), 0.0))

        mmd_scores.append(float(np.mean(individual_mmds)))

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

    n = len(plrs)
    actions     = np.empty(n, dtype=object)
    cluster_ids = np.full(n, -1, dtype=int)

    if n == 0:
        return cluster_ids, actions

    # ── Stage 2: K-Means over all prompt words ─────────────────────────
    X = np.stack([plrs, ciis, trs], axis=1).astype(np.float64)  # (n, 3)
    k = min(8, n)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(X).astype(int)
    centroids = km.cluster_centers_  # (k, 3)

    # Prompt-level means used for relative High/Low decisions.
    mean_plrs = float(np.mean(plrs))
    mean_ciis = float(np.mean(ciis))
    mean_trs  = float(np.mean(trs))

    cluster_action: Dict[int, str] = {}
    for c in range(k):
        mu_plrs, mu_ciis, mu_trs = centroids[c]
        p_high = mu_plrs > mean_plrs
        c_high = mu_ciis > mean_ciis
        t_high = mu_trs > mean_trs

        # ALSA Table 1 exact mapping.
        if p_high and c_high and t_high:
            cluster_action[c] = "Encrypt"
        elif p_high and c_high and not t_high:
            cluster_action[c] = "Encrypt"
        elif p_high and not c_high and t_high:
            cluster_action[c] = "Replace"
        elif p_high and not c_high and not t_high:
            cluster_action[c] = "Delete"
        elif not p_high and c_high and t_high:
            cluster_action[c] = "Retain"
        elif not p_high and c_high and not t_high:
            cluster_action[c] = "Retain"
        elif not p_high and not c_high and t_high:
            cluster_action[c] = "Retain"
        else:  # (L, L, L)
            cluster_action[c] = "Delete"

    for i, cid in enumerate(cluster_ids):
        actions[i] = cluster_action.get(int(cid), "Retain")

    # ── Stage 1 override (applied last to guarantee Encrypt) ───────────
    pii_mask = pii_override >= 0.9  # regex-confirmed PII
    actions[pii_mask] = "Encrypt"

    return cluster_ids, actions


def _build_masked_prompt(words: List[str], actions: np.ndarray) -> str:
    """
    Build the intermediate masked version of the prompt.
    Encrypt words become [MASK], Replace uses WordNet synonyms,
    Delete is omitted, Retain is kept verbatim.
    """
    return _build_masked_prompt_contextual(words, actions, " ".join(words))


def _build_masked_prompt_contextual(
    words: List[str],
    actions: np.ndarray,
    original_prompt: str,
) -> str:
    """Action application with context-aware lexical replacement."""
    out_words = words.copy()
    pos_tags = _get_word_pos_tags(words, original_prompt, registry.spacy_nlp)

    for i, action in enumerate(actions):
        if action == "Delete":
            out_words[i] = ""
        elif action == "Encrypt":
            out_words[i] = MASK_TOKEN
        elif action == "Replace":
            out_words[i] = _select_replacement_word(
                out_words,
                i,
                original_prompt,
                pos_tags[i],
            )

    return " ".join(w for w in out_words if w)


def _contains_sensitive_patterns(text: str) -> bool:
    """Conservative regex check to prevent regenerated PII in candidates."""
    for pattern, _ in PII_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _safe_mask_fallback(masked_prompt: str) -> str:
    """
    Deterministic fallback when no safe LLM candidate is available.
    Replaces each [MASK] with synthetic non-identifying placeholders.
    """
    counter = {"n": 0}

    def repl(_: re.Match) -> str:
        counter["n"] += 1
        return f"REDACTED_TOKEN_{counter['n']}"

    return re.sub(re.escape(MASK_TOKEN), repl, masked_prompt)


def _parse_single_candidate(response: str, masked_prompt: str) -> str:
    """
    Extract a single completed sentence from a one-shot LLM response.
    Strip preamble lines (e.g. "Sure! Here is…"), quotes, and leading
    whitespace, returning the first non-empty content line.
    """
    for line in response.strip().split("\n"):
        line = line.strip().strip('"').strip("'")
        # Skip empty lines and meta-commentary lines
        if not line:
            continue
        if re.match(r'^(sure|here|certainly|of course|okay|ok)[,!:\s]', line, re.I):
            continue
        if MASK_TOKEN in line:
            continue
        return line
    # If nothing usable, return the whole response cleaned up
    return response.strip().strip('"').strip("'")


async def _generate_encrypt_candidates(
    masked_prompt: str,
    original_prompt: str,
    n_candidates: int = ENCRYPT_NUM_CANDIDATES,
) -> List[str]:
    """
    Generate m candidate prompts by running the masked prompt through Llama-3
    m times separately (each with temperature=0.7, top_p=0.9) to obtain
    diverse completions.  Per ALSA Appendix A, the LLM is run on P_M multiple
    times; the self-recommendation step then selects the best.

    Each call asks for ONE completed sentence with [MASK] replaced by
    contextually plausible, non-identifying alternatives.
    """
    # Single-call prompt: ask for one completed sentence only.
    def _build_query() -> str:
        return (
            "Complete the following sentence by replacing every [MASK] token "
            "with a plausible, contextually appropriate word or short phrase. "
            "Do NOT use real personal names, account numbers, medical record IDs, "
            "or any other identifying information. "
            "Output the completed sentence ONLY — no explanations, no lists.\n\n"
            f"Sentence: {masked_prompt}"
        )

    loop = asyncio.get_running_loop()
    candidates: List[str] = []
    seen: set = set()
    max_attempts = n_candidates * 3  # allow retries for failures / duplicates

    for _ in range(max_attempts):
        if len(candidates) >= n_candidates:
            break
        try:
            response = await loop.run_in_executor(
                None,
                lambda: _llama_generate(
                    _build_query(),
                    max_tokens=ENCRYPT_CANDIDATE_TOKENS,
                    temp=0.7,   # paper §4 experiment setup
                    top_p=0.9,  # paper §4 experiment setup
                    top_k=50,   # paper §4 experiment setup
                ),
            )
            candidate = _parse_single_candidate(response, masked_prompt)
            if (
                candidate
                and MASK_TOKEN not in candidate
                and candidate != original_prompt
                and not _contains_sensitive_patterns(candidate)
                and candidate not in seen
            ):
                seen.add(candidate)
                candidates.append(candidate)
        except Exception as exc:
            log.warning("Encrypt candidate generation attempt failed: %s", exc)

    if not candidates:
        log.warning("All candidate generation attempts failed; using deterministic fallback")
        return [_safe_mask_fallback(masked_prompt)]

    return candidates


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

    # Build candidate list exactly as shown in ALSA paper Figure 5.
    candidate_list = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(candidates)
    )
    # ALSA Figure 5 — Self-Recommendation Prompt (verbatim structure):
    #   Role    : expert evaluator
    #   Task    : Here is the task …
    #   Reference: Here is the reference prompt …
    #   Candidates: numbered list
    #   Instruction: evaluate fluency / coherence / semantic consistency
    query = (
        f"You are an expert in language understanding and evaluation.\n\n"
        f"Here is the task: {original_prompt}\n\n"
        f"Here is the reference prompt: {original_prompt}\n\n"
        f"{candidate_list}\n\n"
        f"Please evaluate each prompt above based on fluency, coherence, and "
        f"semantic consistency with the reference prompt. "
        f"Select the prompt that is the most fluent, coherent, and best achieves "
        f"the same task result as the reference prompt. "
        f"Provide only the number of the selected prompt."
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
    masked_prompt = _build_masked_prompt_contextual(words, actions, original_prompt)

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
    "A quick brown fox jumps over the lazy dog near the river.",

    # 2. Heavy PII — personal information + SSN
    # "My name is John Smith and my social security number is 432-91-8765.",

    # "You can reach Dr. Emily Carter at random34241@gmail.com or call her private cell at 555-019-8372",

    # "Patient Sarah Jenkins, born on 11/14/1985, recently tested positive for the BRCA1 genetic mutation",

    # 3. Highly technical jargon
    # "The convolutional neural network uses backpropagation and gradient "
    # "descent to optimise the cross-entropy loss.",

    # 4. Medical context with sensitive data
    # "Patient Alice Johnson aged 54 has been diagnosed with stage-3 "
    # "pancreatic adenocarcinoma and is prescribed gemcitabine.",

    # 5. Legal / financial context
    # "The defendant transferred 2.4 million dollars to an offshore account "
    # "in the Cayman Islands on March 15th.",
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