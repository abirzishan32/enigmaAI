
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          Privacy-Preserving NLP Pipeline  — Google Colab Version           ║
# ║                                                                              ║
# ║  Paste every cell (delimited by  # ── CELL ──)  into separate Colab cells. ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
# Run this cell first. Colab will restart the runtime once — that is normal.
#
# !pip install -q \
#     transformers==4.40.2 \
#     torch \
#     spacy \
#     nltk \
#     scikit-learn \
#     scipy \
#     matplotlib \
#     nest_asyncio \
#     accelerate \
#     sentencepiece
#
# !python -m spacy download en_core_web_sm -q


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Imports & NLTK downloads
# ══════════════════════════════════════════════════════════════════════════════

# ── Standard Library ──────────────────────────────────────────────────────────
import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Colab async compatibility ──────────────────────────────────────────────────
import nest_asyncio
nest_asyncio.apply()          # lets asyncio.run() work inside Jupyter/Colab

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in notebooks too
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from IPython.display import Image as IPImage, display  # inline chart display

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,                   # used for the TRS LLM backbone
)

import spacy
import nltk

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import rbf_kernel

from scipy.spatial.distance import mahalanobis

# ── NLTK data ─────────────────────────────────────────────────────────────────
# Colab Fix: download all needed corpora explicitly at import-time.
# The original code downloaded these in the FastAPI lifespan; here we do it
# upfront so wordnet is available when the module-level _SPACY_TO_WN_POS dict
# is built (which references wn.NOUN / wn.VERB etc.).
for _corpus in ("wordnet", "omw-1.4", "brown"):
    nltk.download(_corpus, quiet=True)

from nltk.corpus import wordnet as wn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("privacy_pipeline")

print("✅ Imports OK")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Constants
# ══════════════════════════════════════════════════════════════════════════════

BERT_MODEL_NAME   = "bert-base-uncased"
GPT2_MODEL_NAME   = "gpt2"
SPACY_MODEL_NAME  = "en_core_web_sm"
CACHE_DIR         = Path("./model_cache")
BERT_CACHE_FILE   = CACHE_DIR / "bert_common_word_vectors.pt"
N_COMMON_WORDS    = 10_000
MAX_BERT_TOKENS   = 512
KMEANS_N_CLUSTERS = 8
MASK_TOKEN        = "[MASK]"

# ── Colab LLM backbone (replaces Apple-only MLX / Llama-3) ───────────────────
# google/flan-t5-large fits in ~3 GB VRAM and runs on Colab T4 GPUs.
# To use a larger model (e.g. mistralai/Mistral-7B-Instruct-v0.2) simply
# change this constant — the rest of the code is model-agnostic.
COLAB_LLM_MODEL   = "google/flan-t5-large"

TRS_NUM_ITERS             = 3           # reduced from 5 for Colab speed
TRS_FALLBACK_SCORE        = 0.5
TRS_MAX_NEW_TOKENS        = 16

ENCRYPT_NUM_CANDIDATES    = 3
ENCRYPT_CANDIDATE_TOKENS  = 200
ENCRYPT_ALPHA             = 0.33
ENCRYPT_BETA              = 0.33
ENCRYPT_GAMMA             = 0.34

# POS → coherence weight
POS_WEIGHT: Dict[str, float] = {
    "NOUN": 1.0, "PROPN": 1.0, "ADJ": 1.0, "ADV": 0.8, "VERB": 0.9,
    "ADP":  0.3, "DET":   0.3, "CONJ": 0.3, "CCONJ": 0.3,
    "SCONJ":0.3, "PART":  0.3, "PRON": 0.5, "NUM": 0.7,
    "PUNCT":0.1, "SYM":   0.2, "X":   0.5, "INTJ": 0.6,
}

# ── PII Regex Patterns ────────────────────────────────────────────────────────
PII_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'),           1.0),   # SSN
    (re.compile(r'\b\d{3}[-.\\s]?\d{3}[-.\\s]?\d{4}\b'),     0.95),  # Phone
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), 0.95),  # Email
    (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), 0.95),  # Credit card
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),  0.90),  # IP address
    (re.compile(r'\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b'),     0.80),  # ISO date
    (re.compile(r'\$\s*\d[\d,]*(?:\.\d{2})?\b'),               0.75),  # Dollar amounts
    (re.compile(r'\b\d+\s*(?:million|billion|thousand)\b', re.I), 0.70),  # Large sums
]

SENSITIVE_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "MONEY"}

print("✅ Constants set")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Model Registry & Loader
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    device: torch.device
    bert_tokenizer: AutoTokenizer
    bert_model: AutoModel
    gpt2_tokenizer: AutoTokenizer
    gpt2_model: AutoModelForCausalLM
    spacy_nlp: spacy.Language
    bert_common_vectors: torch.Tensor       # (N_COMMON_WORDS, 768)
    llm_pipe: object                        # HuggingFace text2text pipeline


registry = ModelRegistry()


def detect_device() -> torch.device:
    """CUDA → CPU (MPS is Apple-only, not available on Colab)."""
    if torch.cuda.is_available():
        log.info("Device: CUDA — %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    log.info("Device: CPU")
    return torch.device("cpu")


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


def load_llm_pipeline(device: torch.device):
    """
    Load a seq2seq (text-to-text) pipeline as the TRS / Encrypt backbone.

    Colab Change:  The original code used MLX + Llama-3-8B-Instruct which
    requires Apple Silicon.  Here we use google/flan-t5-large which:
      • Runs on any Colab GPU (T4 / A100) or CPU
      • Supports instruction-style prompts natively
      • Fits in ~3 GB VRAM

    To switch to a larger causal LM (e.g. Mistral-7B), change
    COLAB_LLM_MODEL and set task="text-generation" instead of
    "text2text-generation".
    """
    log.info("Loading LLM pipeline (%s) …", COLAB_LLM_MODEL)
    pipe = pipeline(
        "text2text-generation",
        model=COLAB_LLM_MODEL,
        device=0 if torch.cuda.is_available() else -1,   # 0 = first GPU, -1 = CPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    log.info("LLM pipeline ready")
    return pipe


def _llm_generate(prompt_text: str, max_new_tokens: int = TRS_MAX_NEW_TOKENS) -> str:
    """
    Synchronous LLM call via HuggingFace pipeline.

    Colab Change:  Replaces the MLX-specific _llama_generate() function.
    The interface is identical (str in → str out) so all callers are unchanged.

    Flan-T5 does not produce chat-template artefacts like <|eot_id|>, so the
    post-processing strip in the original is not needed.
    """
    outputs = registry.llm_pipe(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=False,        # greedy decoding for consistency
    )
    return outputs[0]["generated_text"].strip()


def _build_bert_common_vectors(tokenizer, model, device) -> torch.Tensor:
    """
    Embed N_COMMON_WORDS from the Brown corpus in isolation.
    Identical to the original — Brown corpus downloaded at top of Cell 2.
    """
    from nltk.corpus import brown
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


def bootstrap_registry():
    """
    Colab replacement for the FastAPI lifespan context manager.
    Call this once at the start of your Colab session.
    """
    registry.device = detect_device()
    registry.bert_tokenizer, registry.bert_model = load_bert(registry.device)
    registry.gpt2_tokenizer, registry.gpt2_model = load_gpt2(registry.device)
    log.info("Loading SpaCy …")
    registry.spacy_nlp = spacy.load(SPACY_MODEL_NAME)
    registry.bert_common_vectors = load_bert_common_vectors(
        registry.bert_tokenizer, registry.bert_model, registry.device
    )
    registry.llm_pipe = load_llm_pipeline(registry.device)
    log.info("=== All models loaded. Ready. ===")

print("✅ Registry helpers defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Utility Helpers
# ══════════════════════════════════════════════════════════════════════════════

def strip_punctuation(word: str) -> str:
    return re.sub(r'^[^\w]+|[^\w]+$', '', word) or word


def get_word_char_spans(words: List[str], text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for w in words:
        idx = text.find(w, cursor)
        if idx == -1:
            idx = cursor
        spans.append((idx, idx + len(w)))
        cursor = idx + len(w)
    return spans


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
    enc_raw = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_BERT_TOKENS,
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc_raw.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc_raw.items()}

    out    = model(**enc)
    hidden = out.last_hidden_state[0].cpu()  # (seq_len, 768)

    word_spans = get_word_char_spans(words, prompt)

    word_vecs: List[torch.Tensor] = []
    for (ws, we) in word_spans:
        sub_vecs: List[torch.Tensor] = []
        for t_idx, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:
                continue
            if ts < we and te > ws:
                sub_vecs.append(hidden[t_idx])
        if sub_vecs:
            word_vecs.append(torch.stack(sub_vecs).mean(0))
        else:
            word_vecs.append(hidden[0])

    return torch.stack(word_vecs, dim=0)

print("✅ Utility helpers defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Phase 2: PLRS
# ══════════════════════════════════════════════════════════════════════════════

def compute_pii_override_scores(words: List[str], prompt: str) -> np.ndarray:
    override = np.zeros(len(words), dtype=np.float32)
    word_spans = get_word_char_spans(words, prompt)
    for pattern, score in PII_PATTERNS:
        for m in pattern.finditer(prompt):
            ms, me = m.start(), m.end()
            for i, (ws, we) in enumerate(word_spans):
                if ws < me and we > ms:
                    override[i] = max(override[i], score)
    return override


def compute_ner_sensitivity_boost(
    words: List[str],
    prompt: str,
    spacy_nlp: spacy.Language,
) -> np.ndarray:
    doc = spacy_nlp(prompt)
    word_spans = get_word_char_spans(words, prompt)
    boost = np.zeros(len(words), dtype=np.float32)

    for ent in doc.ents:
        if ent.label_ in SENSITIVE_NER_LABELS:
            for i, (ws, we) in enumerate(word_spans):
                if ws < ent.end_char and we > ent.start_char:
                    boost[i] = max(boost[i], 0.4)

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
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(common_vecs)

    prompt_scores = -iso.score_samples(prompt_word_vecs_isolated)
    ref_sample    = common_vecs[:2000]
    ref_scores    = -iso.score_samples(ref_sample)

    anomaly = np.array(
        [float(np.mean(ref_scores < ps)) for ps in prompt_scores],
        dtype=np.float32,
    )
    return anomaly


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

    enc_raw = tokenizer(
        prompt, return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc_raw.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc_raw.items()}
    input_ids = enc["input_ids"]

    out = model(**enc)
    logits = out.logits[0]

    shift_logits = logits[:-1]
    shift_ids    = input_ids[0, 1:]
    probs        = F.softmax(shift_logits, dim=-1)
    token_probs  = probs[range(len(shift_ids)), shift_ids]
    nll_tokens   = -torch.log2(token_probs + 1e-9).cpu().numpy()

    mean_nll = float(nll_tokens.mean()) if len(nll_tokens) else 5.0

    word_spans = get_word_char_spans(words, prompt)
    token_nll_map: List[Optional[float]] = [None] * len(offsets)
    for j in range(1, len(offsets)):
        nll_idx = j - 1
        if nll_idx < len(nll_tokens):
            token_nll_map[j] = float(nll_tokens[nll_idx])

    word_nlls: List[float] = []
    for (ws, we) in word_spans:
        tok_nlls: List[float] = []
        for j, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:
                continue
            if ts < we and te > ws:
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
    intrinsic = compute_intrinsic_sensitivity(prompt_word_vecs_isolated, common_vecs)
    exposure  = compute_exposure_risk(prompt, words, gpt2_tokenizer, gpt2_model, device)

    plrs_raw = 0.5 * intrinsic + 0.5 * exposure

    pii_override = compute_pii_override_scores(words, prompt)
    ner_boost    = compute_ner_sensitivity_boost(words, prompt, spacy_nlp)

    plrs = np.clip(plrs_raw + 0.5 * ner_boost, 0.0, 1.0)
    plrs = np.maximum(plrs, pii_override).astype(np.float32)

    return plrs, pii_override

print("✅ PLRS functions defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Phase 3: CIIS
# ══════════════════════════════════════════════════════════════════════════════

# SpaCy POS → WordNet POS mapping
_SPACY_TO_WN_POS: Dict[str, Optional[str]] = {
    "NOUN": wn.NOUN, "PROPN": None,
    "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV,
}

FUNCTION_WORD_POS = {"ADP", "DET", "CCONJ", "SCONJ", "PART", "PUNCT"}


def compute_contextual_coherence(
    words: List[str],
    prompt: str,
    spacy_nlp: spacy.Language,
    bert_word_vecs: np.ndarray,
) -> np.ndarray:
    doc = spacy_nlp(prompt)
    spacy_toks = [t for t in doc if not t.is_space]

    pos_weights = np.ones(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        clean = strip_punctuation(w).lower()
        for st in spacy_toks:
            if st.text.lower() == clean:
                pos_weights[i] = POS_WEIGHT.get(st.pos_, 0.5)
                break

    V      = bert_word_vecs
    w_col  = pos_weights.reshape(-1, 1)
    centroid = (w_col * V).sum(0) / (w_col.sum() + 1e-9)

    if V.shape[0] > 1:
        cov     = np.cov(V.T)
        cov_reg = cov + 0.1 * np.eye(cov.shape[0])
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


def get_wordnet_synonyms(
    word: str, n: int = 5, pos_tag: Optional[str] = None,
) -> List[str]:
    clean = strip_punctuation(word)
    if not clean:
        return []
    if pos_tag == "PROPN":
        return []

    wn_pos = _SPACY_TO_WN_POS.get(pos_tag) if pos_tag else None

    syns: List[str] = []
    synsets = wn.synsets(clean, pos=wn_pos) if wn_pos else wn.synsets(clean)
    for synset in synsets:
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
    orig_cls = embed_sentences_bert([prompt], bert_tokenizer, bert_model, device)
    gamma    = 1.0 / orig_cls.shape[1]

    mmd_scores: List[float] = []
    for word in words:
        synonyms = get_wordnet_synonyms(word, n=5)
        if not synonyms:
            mmd_scores.append(0.5)
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
    coh  = compute_contextual_coherence(words, prompt, spacy_nlp, bert_word_vecs)
    dist = compute_semantic_distinctiveness(words, prompt, bert_tokenizer, bert_model, device)
    return (0.4 * coh + 0.6 * dist).astype(np.float32)

print("✅ CIIS functions defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Phase 4: TRS, Actions, Reconstruction
# ══════════════════════════════════════════════════════════════════════════════

async def get_trs(prompt: str, word: str) -> float:
    """
    TRS via HuggingFace pipeline (replaces MLX Llama-3).

    Colab Change: _llm_generate() is synchronous, so we call it directly
    inside run_in_executor with the default ThreadPoolExecutor to avoid
    blocking the event loop while the model generates.
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
            None, _llm_generate, query, TRS_MAX_NEW_TOKENS
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
    cache: Dict[str, float] = {}
    results: List[float] = []

    for word_idx, word in enumerate(words):
        clean = strip_punctuation(word).lower()
        if clean in cache:
            results.append(cache[clean])
            continue

        scores: List[float] = []
        for j in range(TRS_NUM_ITERS):
            s = await get_trs(prompt, word)
            scores.append(s)

        avg = float(np.mean(scores))
        cache[clean] = avg
        log.info("TRS[%d] '%s': %.3f  (k=%d)", word_idx, word, avg, TRS_NUM_ITERS)
        results.append(avg)

    return np.array(results, dtype=np.float32)


def _action_from_triple(p_high: bool, c_high: bool, t_high: bool) -> str:
    if p_high and c_high:                    return "Encrypt"
    if p_high and not c_high and t_high:     return "Replace"
    if p_high and not c_high:               return "Delete"
    if not p_high and c_high:               return "Retain"
    if not p_high and not c_high and t_high: return "Retain"
    return "Delete"


def assign_actions_kmeans(
    plrs: np.ndarray,
    ciis: np.ndarray,
    trs: np.ndarray,
    pii_override: np.ndarray,
    words: List[str],
    spacy_nlp: spacy.Language,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n = len(plrs)
    actions     = np.empty(n, dtype=object)
    cluster_ids = np.zeros(n, dtype=int)

    doc = spacy_nlp(" ".join(words))
    spacy_toks = [t for t in doc if not t.is_space]
    pos_tags: List[str] = []
    for w in words:
        clean = strip_punctuation(w).lower()
        matched_pos = "X"
        for st in spacy_toks:
            if st.text.lower() == clean:
                matched_pos = st.pos_
                break
        pos_tags.append(matched_pos)

    pii_mask = pii_override >= 0.9
    actions[pii_mask] = "Encrypt"

    plrs = plrs.copy()
    mean_plrs_raw = plrs.mean()
    for i in range(n):
        if pos_tags[i] in FUNCTION_WORD_POS and pii_override[i] < 0.9:
            plrs[i] = min(plrs[i], mean_plrs_raw - 0.01)

    remaining_idx = np.where(~pii_mask)[0]
    if len(remaining_idx) == 0:
        return cluster_ids, actions, pos_tags

    X = np.stack([plrs, ciis, trs], axis=1).astype(np.float64)
    k = min(KMEANS_N_CLUSTERS, len(remaining_idx), n)
    km = KMeans(n_clusters=k, n_init=15, random_state=42)
    all_cluster_ids = km.fit_predict(X)
    cluster_ids[:] = all_cluster_ids
    centroids = km.cluster_centers_

    mean_plrs = plrs.mean()
    mean_ciis = ciis.mean()
    mean_trs  = trs.mean()

    cluster_action: Dict[int, str] = {}
    for c in range(k):
        cp, cc, ct = centroids[c]
        cluster_action[c] = _action_from_triple(
            cp > mean_plrs, cc > mean_ciis, ct > mean_trs
        )

    for i in remaining_idx:
        cluster_act = cluster_action.get(int(cluster_ids[i]), "Retain")
        word_act = _action_from_triple(
            plrs[i] > mean_plrs, ciis[i] > mean_ciis, trs[i] > mean_trs
        )
        actions[i] = word_act if cluster_act != word_act else cluster_act

    return cluster_ids, actions, pos_tags


def _build_masked_prompt(
    words: List[str], actions: np.ndarray, pos_tags: List[str],
) -> str:
    out: List[str] = []
    for word, action, pos in zip(words, actions, pos_tags):
        if action == "Retain":
            out.append(word)
        elif action == "Delete":
            pass
        elif action == "Replace":
            syns = get_wordnet_synonyms(word, n=3, pos_tag=pos)
            out.append(syns[0] if syns else word)
        elif action == "Encrypt":
            out.append(MASK_TOKEN)
    result = " ".join(out)
    result = re.sub(r'\s{2,}', ' ', result).strip()
    return result


async def _generate_encrypt_candidates(
    masked_prompt: str,
    original_prompt: str,
    n_candidates: int = ENCRYPT_NUM_CANDIDATES,
) -> List[str]:
    """
    Generate candidate prompts to fill [MASK] tokens.

    Colab Change: uses _llm_generate() (HuggingFace pipeline) instead of
    the Llama-3 MLX backend.  Flan-T5 handles this instruction well.
    """
    query = (
        f"Below is a prompt with [MASK] tokens replacing sensitive information.\n\n"
        f"Masked prompt: \"{masked_prompt}\"\n\n"
        f"Generate {n_candidates} different versions of this prompt where each [MASK] "
        f"is replaced with a FICTIONAL, NON-IDENTIFYING substitute. Rules:\n"
        f"- Names must be clearly fictional (e.g. 'Person A', 'Entity X')\n"
        f"- For SSN-format placeholders, use ONLY literal text like "
        f"'[REDACTED-SSN]' or 'XXX-XX-XXXX'. NEVER use digit sequences "
        f"in NNN-NN-NNNN format (e.g. 123-45-6789 is FORBIDDEN)\n"
        f"- Other numbers must be obviously fake (e.g. '[REDACTED]')\n"
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
            None, _llm_generate, query, ENCRYPT_CANDIDATE_TOKENS
        )
        candidates: List[str] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                candidate = match.group(1).strip().strip('"').strip("'")
                if candidate:
                    candidates.append(candidate)

        return candidates[:n_candidates] if candidates else [masked_prompt]
    except Exception as exc:
        log.warning("Encrypt candidate generation failed: %s", exc)
        return [masked_prompt]


async def _score_candidates(
    original_prompt: str,
    candidates: List[str],
) -> int:
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
            None, _llm_generate, query, 8
        )
        match = re.search(r'(\d+)', response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return idx
        return 0
    except Exception as exc:
        log.warning("Self-recommendation scoring failed: %s", exc)
        return 0


async def reconstruct_prompt_with_llm(
    words: List[str],
    actions: np.ndarray,
    original_prompt: str,
    pos_tags: List[str],
) -> Tuple[str, str, List[str]]:
    masked_prompt = _build_masked_prompt(words, actions, pos_tags)

    if MASK_TOKEN not in masked_prompt:
        return masked_prompt, masked_prompt, []

    log.info("Encrypt self-recommendation: generating candidates …")
    candidates = await _generate_encrypt_candidates(masked_prompt, original_prompt)
    log.info("Generated %d candidates", len(candidates))

    best_idx = await _score_candidates(original_prompt, candidates)
    log.info("Self-recommendation selected candidate %d", best_idx + 1)

    sanitised = candidates[best_idx]
    return sanitised, masked_prompt, candidates

print("✅ Phase 4 functions defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(prompt: str) -> Dict:
    """Full end-to-end pipeline. Returns a structured result dict."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    words = prompt.split()
    log.info("Pipeline start | %d words", len(words))

    bert_vecs_isolated = embed_words_isolated_bert(
        words, registry.bert_tokenizer, registry.bert_model, registry.device
    ).numpy()

    bert_vecs_contextual = embed_words_in_context_bert(
        words, prompt,
        registry.bert_tokenizer, registry.bert_model, registry.device
    ).numpy()

    common_vecs = registry.bert_common_vectors.numpy()
    plrs, pii_override = compute_plrs(
        bert_vecs_isolated, common_vecs, prompt, words,
        registry.spacy_nlp,
        registry.gpt2_tokenizer, registry.gpt2_model, registry.device,
    )
    log.info("PLRS: %s", np.round(plrs, 3))

    ciis = compute_ciis(
        words, prompt, registry.spacy_nlp,
        bert_vecs_contextual,
        registry.bert_tokenizer, registry.bert_model, registry.device,
    )
    log.info("CIIS: %s", np.round(ciis, 3))

    trs = await compute_trs_batch(prompt, words)
    log.info("TRS : %s", np.round(trs, 3))

    cluster_ids, actions, pos_tags = assign_actions_kmeans(
        plrs, ciis, trs, pii_override, words, registry.spacy_nlp,
    )
    log.info("Actions: %s", actions)

    sanitised, masked_prompt, encrypt_candidates = await reconstruct_prompt_with_llm(
        words, actions, prompt, pos_tags
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

print("✅ Pipeline orchestrator defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Chart generation (Colab inline display)
# ══════════════════════════════════════════════════════════════════════════════

_CHART_DIR = Path("./charts")

ACTION_FACECOLOR = {
    "Retain":  "#2ecc71",
    "Delete":  "#e74c3c",
    "Replace": "#f1c40f",
    "Encrypt": "#9b59b6",
}
BAR_COLORS = {
    "PLRS":    "#e74c3c",
    "CIIS":    "#3498db",
    "TRS":     "#2ecc71",
    "PII-Ovr": "#e67e22",
}


def generate_score_chart(result: Dict, test_idx: int, show_inline: bool = True) -> Path:
    """
    Save a grouped bar chart and (optionally) display it inline in Colab.

    Colab Change: added `show_inline=True` parameter.  When True, the chart
    is displayed using IPython.display after saving so it appears directly
    in the notebook cell output.
    """
    _CHART_DIR.mkdir(parents=True, exist_ok=True)

    words   = result["words"]
    plrs    = result["plrs"]
    ciis    = result["ciis"]
    trs     = result["trs"]
    pii_ovr = result["pii_override"]
    actions = result["actions"]

    n = len(words)
    x = np.arange(n)
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, action in enumerate(actions):
        band_col = ACTION_FACECOLOR.get(action, "#888888")
        ax.axvspan(i - 0.45, i + 0.45, color=band_col, alpha=0.12, zorder=0)

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
        for bar, val in zip(bars, scores):
            if val > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#dddddd",
                )

    for (label, scores), ls in zip(
        [("PLRS", plrs), ("CIIS", ciis), ("TRS", trs)],
        ["--", "-.", ":"],
    ):
        ax.axhline(
            np.mean(scores), color=BAR_COLORS[label],
            linewidth=1.0, linestyle=ls, alpha=0.55,
            zorder=1,
        )

    for i, (word, action) in enumerate(zip(words, actions)):
        col = ACTION_FACECOLOR.get(action, "#aaaaaa")
        ax.text(
            i, -0.08, action[0],
            ha="center", va="top",
            fontsize=8, fontweight="bold",
            color=col,
            transform=ax.get_xaxis_transform(),
        )

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
        f"Test {test_idx} — Word-Level Scores\n(dashed lines = per-metric means)",
        color="#e0e0ff", fontsize=12, fontweight="bold", pad=12,
    )

    bar_legend    = [mpatches.Patch(color=c, label=l) for l, c in BAR_COLORS.items()]
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

    # ── Colab inline display ──────────────────────────────────────────────────
    if show_inline:
        display(IPImage(str(out_path)))

    plt.close(fig)
    log.info("Chart saved → %s", out_path)
    return out_path

print("✅ Chart helpers defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Pretty-print helpers (Colab-safe, no ANSI colours by default)
# ══════════════════════════════════════════════════════════════════════════════

# Colab terminals DO support ANSI on some backends, but it is safer to
# disable colours by default.  Set USE_ANSI = True to re-enable them.
USE_ANSI = False


class C:
    """ANSI colour codes — no-ops when USE_ANSI is False."""
    RESET = BOLD = CYAN = GREEN = YELLOW = RED = MAGENTA = BLUE = GREY = WHITE = ""
    if USE_ANSI:
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
    print(char * w)


def _header(text: str, w: int = 100) -> None:
    pad = (w - len(text) - 2) // 2
    print("║" + " " * pad + text + " " * (w - pad - len(text) - 2) + "║")


def _score_str(v: float) -> str:
    return f"{v:.3f}"


def _print_table(result: Dict) -> None:
    words = result["words"]
    W = 16
    print(f"  {'Word':<{W}}{'PLRS':>7}{'CIIS':>7}{'TRS':>7}"
          f"{'PII-OVR':>9}{'Clust':>7}{'Action':>10}")
    print("  " + "─" * 68)
    for i, word in enumerate(words):
        pii_flag = "  ★" if result["pii_override"][i] >= 0.9 else ""
        print(
            f"  {word:<{W}}"
            f"  {_score_str(result['plrs'][i])}"
            f"  {_score_str(result['ciis'][i])}"
            f"  {_score_str(result['trs'][i])}"
            f"  {_score_str(result['pii_override'][i])}"
            f"  {result['cluster_ids'][i]:>4}"
            f"  {result['actions'][i]:>9}"
            f"{pii_flag}"
        )

print("✅ Print helpers defined")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Test Cases & Run
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # 1. Heavy PII — personal information + SSN
    "My name is John Smith and my social security number is 432-91-8765.",

    # 2. Standard everyday sentence
    # "The quick brown fox jumps over the lazy dog near the river.",

    # 3. Medical context with sensitive data
    # "Patient Alice Johnson aged 54 has been diagnosed with stage-3 "
    # "pancreatic adenocarcinoma and is prescribed gemcitabine.",
]


async def run_tests() -> None:
    """
    Main test runner — mirrors the original __main__ block.

    Colab Change:
      • Charts are displayed inline via IPython.display (see generate_score_chart)
      • ANSI colour codes replaced with plain text (see USE_ANSI constant)
      • bootstrap_registry() is called here instead of the FastAPI lifespan
    """
    print("\n")
    _sep()
    _header("Privacy-Preserving NLP Pipeline — Colab Version")
    _sep()
    print()

    # Load all models
    bootstrap_registry()
    log.info("All models ready — running %d test cases …\n", len(TEST_CASES))

    for idx, prompt in enumerate(TEST_CASES, 1):
        _sep("─")
        print(f"\n  TEST CASE {idx} / {len(TEST_CASES)}\n")
        print(f"  Original Prompt:\n  » {prompt}\n")

        t0 = time.perf_counter()
        try:
            result = await run_pipeline(prompt)
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            continue
        elapsed = time.perf_counter() - t0

        print(f"  Per-Word Analysis:  (★ = PII regex hard-override)")
        _print_table(result)
        generate_score_chart(result, idx, show_inline=True)

        print("\n  Legend: ■ Retain  ■ Delete  ■ Replace  ■ Encrypt")

        if result.get("masked_prompt"):
            print(f"\n  Masked Prompt:\n  » {result['masked_prompt']}")

        if result.get("encrypt_candidates"):
            print("\n  Encrypt Self-Recommendation Candidates:")
            for ci, cand in enumerate(result["encrypt_candidates"], 1):
                marker = "✓" if cand == result["sanitised"] else " "
                print(f"    {marker} {ci}. {cand}")

        print(f"\n  Sanitised Output:\n  » {result['sanitised']}")
        print(f"\n  ⏱  {elapsed:.1f}s\n")

    _sep()
    _header("All tests complete.")
    _sep()
    print()


# ── Entry point (works in both Colab notebook and plain Python) ───────────────
# In Colab, nest_asyncio.apply() (Cell 2) allows asyncio.run() to work inside
# the already-running Jupyter event loop.
asyncio.run(run_tests())
