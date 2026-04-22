# =============================================================================
#   ALSA — Paper-Faithful Implementation for Google Colab
#   "ALSA: Context-Sensitive Prompt Privacy Preservation in Large Language
#    Models" (Ma et al., KDD '25)
#
#   This file is structured as a sequence of Colab cells, each separated
#   by the marker `# ═══ CELL N — <name> ═══`. Copy each cell into its own
#   Colab cell in order. Run them sequentially.
#
#   All equations referenced (Eq. 1 .. Eq. 15) are from the ALSA paper.
#
#   Fidelity notes vs. the original buggy codebase:
#       • PLRS uses MULTIPLICATIVE fusion (Eq. 5) — IS × E
#       • Isolation Forest uses T = 8 decision trees (§4 Setup)
#       • IS is min-max-normalised over V* (prompt words only) (Eq. 2)
#       • Exposure Risk is per-prompt min-max on SUMMED sub-token NLLs (Eq. 3, 4)
#       • Contextual Coherence is PAIRWISE (n×n), not centroid-based (Eq. 8–10)
#       • POS weights are binary: content = 1.0, function = 0.3 (Table 6)
#       • CIIS: λ1 = 0.4, λ2 = 0.6; CC: α = 0.8, β = 0.5 (§4 Setup)
#       • K-Means (k = 8) on (PLRS, CIIS, TRS); centroid-vs-global-mean rule
#       • Backbone LLM is Llama-2-7B-chat (4-bit via bitsandbytes)
#       • Reference corpus is Wikipedia top-10k (via HuggingFace dataset)
#       • Replace action = uniform random WordNet synonym (§3.6)
#       • Encrypt uses Self-Recommendation Prompt (Appendix A / Figure 5)
#
#   Hardware target: Google Colab Pro (T4 / V100 / A100). On T4 (16 GB),
#   total VRAM is ~5 GB after all models are loaded in 4-bit.
# =============================================================================


# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — ENVIRONMENT SETUP
#   Mount Google Drive (for caching + figure output) and install deps.
#   Run this first. Re-run after a Colab runtime restart.
# ═══════════════════════════════════════════════════════════════════════

# -- Mount Google Drive so we can cache heavy artifacts and save figures
from google.colab import drive
drive.mount('/content/drive')

# -- Install packages. We pin transformers + bitsandbytes to versions
#    that are known to load Llama-2-7B cleanly in 4-bit on Colab's CUDA
#    12.x runtime. spacy's en_core_web_sm is downloaded too.
#    Note: we use `--quiet` to keep the output clean; remove if debugging.
!pip install --quiet --upgrade \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    bitsandbytes==0.43.3 \
    sentencepiece \
    datasets==2.21.0 \
    scikit-learn \
    scipy \
    spacy \
    nltk \
    matplotlib

# spaCy English pipeline (required for POS tagging in CC & SD)
!python -m spacy download en_core_web_sm --quiet


# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS & GLOBAL CONFIGURATION
#   All hyperparameters from ALSA §4 "Setup" are centralised here so
#   they are easy to audit against the paper.
# ═══════════════════════════════════════════════════════════════════════

import os
import re
import json
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import spacy
import nltk
from nltk.corpus import wordnet as wn

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ALSA")

# ── Reproducibility ────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths on Google Drive ─────────────────────────────────────────────
# All heavy artifacts are cached under /content/drive/MyDrive/ALSA/ so that
# they survive runtime restarts. The figure output goes here too.
DRIVE_ROOT       = Path("/content/drive/MyDrive/ALSA")
CACHE_DIR        = DRIVE_ROOT / "cache"
FIGURE_DIR       = DRIVE_ROOT / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

WIKI_FREQ_FILE   = CACHE_DIR / "wiki_top10k.json"        # word -> frequency
BERT_REFVEC_FILE = CACHE_DIR / "bert_wiki_top10k.pt"     # (10000, 768) tensor

# ── Paper §4 Setup: models & constants ─────────────────────────────────
# These model names are EXACTLY what the paper specifies.
BERT_MODEL_NAME   = "bert-base-uncased"
GPT2_MODEL_NAME   = "gpt2"
# Paper uses Llama-2-7B as the default backbone (§4). Requires HF approval.
LLAMA_MODEL_NAME  = "meta-llama/Llama-2-7b-chat-hf"
SPACY_MODEL_NAME  = "en_core_web_sm"

# Paper §3.3.1 / §4: top-K most frequent Wikipedia words. K = 10,000.
N_WIKI_REFERENCE_WORDS = 10_000

# Paper Eq. 1 / §4: T = 8 decision trees in the Isolation Forest.
IFOREST_N_TREES = 8

# Paper §3.6: K-Means clustering with k = 8.
KMEANS_K = 8

# Paper Eq. 10 / §4 Setup: α = 0.8, β = 0.5 for Contextual Coherence.
CC_ALPHA   = 0.8   # α — scales the POS-dependency contribution to Q_ij
CC_BETA    = 0.5   # β — decay weight for positional distance R_ij
CC_EPSILON = 1e-6  # ε — numerical-stability constant in the denominator

# Paper Eq. 11 / §4 Setup: CIIS weights.
CIIS_LAMBDA1 = 0.4   # λ1 — weight of Contextual Coherence
CIIS_LAMBDA2 = 0.6   # λ2 — weight of Semantic Distinctiveness

# Paper Eq. 12 / §4 Setup: k = 5 iterations for Task Relevance.
TRS_NUM_ITERS = 5

# Paper Appendix C: POS → ρ(w_i) mapping (binary).
# Content words get 1.0, function words get γ = 0.3.
GAMMA_FUNCTION = 0.3
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}  # Table 6

# Paper Appendix A: number of candidates in the Self-Recommendation loop.
# The paper doesn't pin this exactly ("More candidates ..." in Figure 5);
# we default to 3 which matches the figure's 3-item list.
ENCRYPT_NUM_CANDIDATES = 3

# Paper §4: backbone LLM generation hyperparameters.
LLM_MAX_NEW_TOKENS = 500
LLM_TEMPERATURE    = 0.7
LLM_TOP_P          = 0.9
LLM_TOP_K          = 50

# BERT input cap (standard).
MAX_BERT_TOKENS = 512


# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — HUGGINGFACE AUTHENTICATION & DEVICE
#   Llama-2-7B is gated. You must:
#     1. Accept the license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#     2. Create a token at https://huggingface.co/settings/tokens
#     3. Paste it into the login prompt below
#   After approval is granted (usually minutes to hours), this cell unlocks
#   Llama-2-7B downloading in Cell 4.
# ═══════════════════════════════════════════════════════════════════════

from huggingface_hub import notebook_login
notebook_login()  # interactive token prompt — paste your HF token

# -- Device detection. Colab Pro usually gives T4 / V100 / A100.
def detect_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info(f"GPU detected: {name}")
        return torch.device("cuda")
    log.warning("No GPU detected — falling back to CPU. Pipeline will be slow.")
    return torch.device("cpu")

DEVICE = detect_device()

# Helpful GPU memory snapshot
if DEVICE.type == "cuda":
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Total GPU memory: {total_mem:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — LOAD ALL MODELS
#   BERT + GPT-2 in fp32 on GPU (small).
#   Llama-2-7B loaded in 4-bit via bitsandbytes NF4 → ~4 GB VRAM.
#   SpaCy pipeline on CPU.
#   NLTK WordNet data for synonyms.
# ═══════════════════════════════════════════════════════════════════════

# ---- BERT (semantic encoder for CIIS + IS reference embeddings) -------
log.info("Loading BERT …")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model     = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE).eval()

# ---- GPT-2 (autoregressive LM for Exposure Risk NLL, Eq. 3) -----------
# Paper §3.3.2 specifies "an autoregressive LLM" — we use GPT-2 as the
# canonical small autoregressive baseline. Any GPT-2 variant works; the
# absolute NLL values are then normalised per-prompt (Eq. 4) so the exact
# model choice doesn't bias across prompts.
log.info("Loading GPT-2 …")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME).to(DEVICE).eval()

# ---- Llama-2-7B-chat, 4-bit NF4 quantised (backbone for TRS + Encrypt) ---
# Paper §4: Llama-2-7B (temperature=0.7, top_p=0.9, top_k=50, max_new_tokens=500).
# We use 4-bit NF4 quantization so the ~13 GB fp16 model fits in a T4's 16 GB
# alongside BERT + GPT-2 + our intermediate tensors.
log.info("Loading Llama-2-7B-chat (4-bit NF4) …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
llama_model     = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",          # auto-distribute across available GPUs
    torch_dtype=torch.float16,
)
llama_model.eval()
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

# ---- SpaCy (POS tagging for Contextual Coherence Q_ij matrix) ---------
log.info("Loading SpaCy …")
spacy_nlp = spacy.load(SPACY_MODEL_NAME)

# ---- NLTK WordNet (for Semantic Distinctiveness + Replace synonyms) ----
log.info("Loading NLTK WordNet …")
nltk.download("wordnet",  quiet=True)
nltk.download("omw-1.4",  quiet=True)

# Free CUDA fragmentation after large model loads.
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    used = torch.cuda.memory_allocated() / 1e9
    log.info(f"GPU memory after model loads: {used:.2f} GB")


# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — BUILD WIKIPEDIA TOP-10k REFERENCE CORPUS
#   Paper §3.3.1 / §4 Setup:
#     "we extract the top 10,000 most frequent words from the
#      Wikipedia corpus"
#
#   We stream the HuggingFace `wikimedia/wikipedia` English snapshot,
#   count word frequencies, and save the top-10k list to Drive.
#
#   This cell runs ONCE (5–15 min) and then uses the Drive cache.
#   Re-run only if you delete the cache file.
# ═══════════════════════════════════════════════════════════════════════

def build_wiki_top_k_frequencies(
    k: int = N_WIKI_REFERENCE_WORDS,
    n_articles: int = 50_000,
) -> Dict[str, int]:
    """Stream a slice of English Wikipedia, count word frequencies, return
    the top-k alphabetic words.

    50k articles is ample for a stable top-10k frequency ranking — the
    Zipf distribution makes the top of the list converge very fast.
    """
    if WIKI_FREQ_FILE.exists():
        log.info(f"Using cached Wikipedia frequencies from {WIKI_FREQ_FILE}")
        with open(WIKI_FREQ_FILE, "r") as f:
            return json.load(f)

    from datasets import load_dataset

    log.info("Streaming wikimedia/wikipedia (20231101.en) …")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    # Word-frequency counter over lowercased, alphabetic tokens only.
    freq: Dict[str, int] = {}
    token_re = re.compile(r"[A-Za-z]+")
    processed = 0

    for article in ds:
        text = article.get("text", "")
        for match in token_re.finditer(text):
            tok = match.group(0).lower()
            if len(tok) >= 2:        # skip 1-letter tokens (a, i, …)
                freq[tok] = freq.get(tok, 0) + 1
        processed += 1
        if processed % 5_000 == 0:
            log.info(f"  processed {processed:,} articles, {len(freq):,} unique words")
        if processed >= n_articles:
            break

    # Sort by frequency desc and keep the top-k.
    top_k = dict(sorted(freq.items(), key=lambda x: -x[1])[:k])
    log.info(f"Top-{k} Wikipedia words extracted from {processed:,} articles")

    # Persist to Drive so we never recompute.
    with open(WIKI_FREQ_FILE, "w") as f:
        json.dump(top_k, f)
    log.info(f"Cached to {WIKI_FREQ_FILE}")

    return top_k

WIKI_FREQUENCIES = build_wiki_top_k_frequencies()
WIKI_WORDS       = list(WIKI_FREQUENCIES.keys())
log.info(f"Loaded {len(WIKI_WORDS)} reference words. "
         f"Most frequent: {WIKI_WORDS[:10]}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — PRE-COMPUTE BERT EMBEDDINGS FOR REFERENCE CORPUS
#   Each word is embedded in ISOLATION (as the single input sentence),
#   because the Isolation Forest in Cell 8 will score prompt words that
#   are also embedded in isolation. Comparing contextual vs. context-free
#   vectors would compare apples to oranges.
#
#   Cached to Drive (~30 MB). Runs once (~2 min on T4).
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def embed_words_isolated(words: List[str], batch_size: int = 64) -> torch.Tensor:
    """Return a (len(words), 768) tensor of [CLS] vectors.

    Each word is embedded on its own — no surrounding context. This matches
    the way we will score prompt words for Intrinsic Sensitivity, so the
    distributions are directly comparable.
    """
    all_cls: List[torch.Tensor] = []
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        enc = bert_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,
        ).to(DEVICE)
        out = bert_model(**enc)
        # [CLS] token vector (index 0) per sequence.
        all_cls.append(out.last_hidden_state[:, 0, :].cpu())
    return torch.cat(all_cls, dim=0)


def load_or_build_reference_embeddings() -> torch.Tensor:
    if BERT_REFVEC_FILE.exists():
        log.info(f"Loading cached reference embeddings from {BERT_REFVEC_FILE}")
        return torch.load(BERT_REFVEC_FILE, map_location="cpu")
    log.info("Embedding 10k reference words with BERT (isolated) …")
    vecs = embed_words_isolated(WIKI_WORDS)
    torch.save(vecs, BERT_REFVEC_FILE)
    log.info(f"Reference embeddings cached → {BERT_REFVEC_FILE}  shape={tuple(vecs.shape)}")
    return vecs

REF_VECTORS = load_or_build_reference_embeddings()           # torch.Tensor
REF_VECTORS_NP = REF_VECTORS.numpy().astype(np.float32)       # numpy copy for IF

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — CORE UTILITIES
#   Shared helpers used by multiple modules:
#     • character-span alignment between whitespace-split words and
#       sub-word tokenizers (BERT / GPT-2)
#     • POS lookup for every prompt word via SpaCy
#     • Llama-2 chat-format generation helper
#     • Min-max normalisation (plain and safe)
# ═══════════════════════════════════════════════════════════════════════

def strip_punctuation(word: str) -> str:
    """Remove leading / trailing non-word chars. Keep the core lexical
    form. Used for WordNet lookup and frequency comparisons."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", word) or word


def get_word_char_spans(words: List[str], text: str) -> List[Tuple[int, int]]:
    """Return (start, end) character indices for each surface word in text.

    Uses a forward-only cursor so repeated words are disambiguated
    correctly (first occurrence matched first)."""
    spans, cursor = [], 0
    for w in words:
        idx = text.find(w, cursor)
        if idx == -1:                 # defensive; shouldn't trigger with split()
            idx = cursor
        spans.append((idx, idx + len(w)))
        cursor = idx + len(w)
    return spans


def get_word_pos_tags(words: List[str], prompt: str) -> List[str]:
    """Map each surface word in `words` to a SpaCy POS tag via
    max-character-overlap with tokens from `spacy_nlp(prompt)`."""
    doc = spacy_nlp(prompt)
    spans = get_word_char_spans(words, prompt)
    tags: List[str] = []
    for ws, we in spans:
        best_overlap, best_pos = 0, "X"
        for tok in doc:
            if tok.is_space:
                continue
            ts, te = tok.idx, tok.idx + len(tok.text)
            overlap = max(0, min(we, te) - max(ws, ts))
            if overlap > best_overlap:
                best_overlap, best_pos = overlap, tok.pos_
        tags.append(best_pos)
    return tags


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Paper Eq. 2 / 4 style min-max to [0, 1]. Safe against zero-range
    vectors (returns zeros)."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).astype(np.float32)


# ----- Llama-2 chat generation ----------------------------------------

@torch.no_grad()
def llama_generate(user_message: str,
                   max_new_tokens: int = LLM_MAX_NEW_TOKENS,
                   do_sample: bool = True) -> str:
    """Generate a single response from Llama-2-7B-chat using the exact
    hyperparameters from ALSA §4 Setup.

    We use the tokenizer's chat template so the `<s>[INST] … [/INST]`
    format is applied correctly. For deterministic scoring queries (TRS
    and Self-Recommendation) the caller can set do_sample=False.
    """
    messages = [{"role": "user", "content": user_message}]
    input_ids = llama_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(llama_model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=llama_tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
        )
    else:
        gen_kwargs.update(do_sample=False)

    out_ids = llama_model.generate(input_ids, **gen_kwargs)
    # Strip the prompt portion; decode only the newly generated tokens.
    new_tokens = out_ids[0, input_ids.shape[-1]:]
    return llama_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — PRIVACY LEAKAGE RISK SCORE  (PLRS)
#   Paper §3.3 — Eqs. 1 – 5.
#
#   PLRS(w_i) = IS(w_i)  ×  E(w_i)                   (Eq. 5)
#
#   where:
#     IS  = Intrinsic Sensitivity via Isolation Forest (Eqs. 1–2)
#     E   = Exposure Risk via GPT-2 NLL              (Eqs. 3–4)
# ═══════════════════════════════════════════════════════════════════════

# --- 8.1  Intrinsic Sensitivity (Eqs. 1–2) ----------------------------

@torch.no_grad()
def embed_prompt_words_isolated(words: List[str]) -> np.ndarray:
    """Same recipe as the reference corpus: each prompt word is embedded
    alone so the IF sees vectors from the SAME distribution family it
    was fit on. Returns (n_words, 768) numpy array."""
    clean = [strip_punctuation(w) for w in words]
    # Replace empty strings (punctuation-only tokens) with a space so
    # tokenizer doesn't fail; downstream IS will score them normally.
    clean = [w if w else " " for w in clean]
    vecs = embed_words_isolated(clean)
    return vecs.numpy().astype(np.float32)


def compute_intrinsic_sensitivity(words: List[str]) -> np.ndarray:
    """ Eq. 1 — OS(w_i) = 2^(-(1/T) * Σ DEP_t(w_i)) via sklearn's IF
        Eq. 2 — min-max normalise OS over V* (prompt words only)

    sklearn's IsolationForest.score_samples() returns a value that is a
    monotonic decreasing function of the mean path length; taking its
    negative gives a score that rises with "outlier-ness", which is
    directly proportional to 2^(-E[h(x)]/c(n)) up to constants. Since we
    min-max normalise anyway (Eq. 2), the absolute scale is irrelevant."""

    prompt_vecs = embed_prompt_words_isolated(words)

    # Fit IF on the Wikipedia reference distribution (the "normal" class).
    # Paper §4 Setup: T = 8 decision trees.
    iso = IsolationForest(
        n_estimators=IFOREST_N_TREES,
        contamination=0.1,          # sklearn default; not overridden by paper
        random_state=SEED,
        n_jobs=-1,
    )
    iso.fit(REF_VECTORS_NP)

    # Score prompt words. Negate so that higher = more anomalous.
    raw_os = -iso.score_samples(prompt_vecs)

    # Eq. 2 — min-max over the prompt words ONLY.
    return min_max_normalize(raw_os)


# --- 8.2  Exposure Risk (Eqs. 3–4) ------------------------------------

@torch.no_grad()
def compute_exposure_risk(prompt: str, words: List[str]) -> np.ndarray:
    """Eq. 3: NLL_wi = Σ_{j∈S_wi}  -log2 P(t_j | t_1..t_{j-1})
                      (SUM of token-level NLLs belonging to w_i)
       Eq. 4: E(w_i) = min-max normalised version of NLL_wi over V*.

    Working through Eq. 4 algebraically: defining
        NLL_scale_i = max_k NLL_wk  -  NLL_wi
    gives
        E(w_i) = (max NLL_scale - NLL_scale_i) / (max NLL_scale - min NLL_scale)
               = (NLL_wi - min_k NLL_wk) / (max_k NLL_wk - min_k NLL_wk)
    which is exactly plain min-max of raw NLL values. Higher NLL → more
    surprising → higher exposure risk (E closer to 1)."""

    # Tokenise WITH char offsets so we can align BPE tokens back to words.
    enc_raw = gpt2_tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc_raw.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc_raw.items()}
    input_ids = enc["input_ids"]           # (1, seq_len)

    out = gpt2_model(**enc)
    logits = out.logits[0]                 # (seq_len, vocab)

    # Causal shift: logits[t] predicts token[t+1].
    shift_logits = logits[:-1]
    shift_ids    = input_ids[0, 1:]
    probs        = F.softmax(shift_logits, dim=-1)
    token_probs  = probs[range(len(shift_ids)), shift_ids]
    # Eq. 3 uses log base 2.
    nll_per_tok  = (-torch.log2(token_probs + 1e-12)).cpu().numpy()   # (seq_len-1,)

    # nll_per_tok[i] corresponds to token index (i+1) in the original seq.
    token_nll = np.full(len(offsets), np.nan, dtype=np.float32)
    for j in range(1, len(offsets)):       # token 0 has no causal context
        nll_idx = j - 1
        if nll_idx < len(nll_per_tok):
            token_nll[j] = float(nll_per_tok[nll_idx])

    # Now group by word via character-span overlap and SUM (Eq. 3).
    word_spans = get_word_char_spans(words, prompt)
    mean_nll_fallback = float(np.nanmean(token_nll)) if np.any(~np.isnan(token_nll)) else 5.0

    word_nll = np.zeros(len(words), dtype=np.float32)
    for i, (ws, we) in enumerate(word_spans):
        total, found = 0.0, False
        for j, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:        # special tokens
                continue
            if ts < we and te > ws:        # char overlap
                if not np.isnan(token_nll[j]):
                    total += float(token_nll[j])
                    found = True
        word_nll[i] = total if found else mean_nll_fallback

    # Eq. 4 — per-prompt min-max over V*.
    return min_max_normalize(word_nll)


# --- 8.3  PLRS fusion (Eq. 5) -----------------------------------------

def compute_plrs(prompt: str, words: List[str]) -> np.ndarray:
    """Eq. 5 — PLRS(w_i) = IS(w_i) × E(w_i)  (element-wise product)."""
    IS = compute_intrinsic_sensitivity(words)
    E  = compute_exposure_risk(prompt, words)
    PLRS = (IS * E).astype(np.float32)
    log.info(f"  IS   : {np.round(IS, 3)}")
    log.info(f"  E    : {np.round(E, 3)}")
    log.info(f"  PLRS : {np.round(PLRS, 3)}")
    return PLRS


# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — CONTEXTUAL INFORMATION IMPORTANCE SCORE  (CIIS)
#   Paper §3.4 — Eqs. 6 – 11, 13 – 15.
#
#   CIIS(w_i) = λ1 · CC(w_i) + λ2 · SD(w_i)           (Eq. 11)
#
#   CC = Contextual Coherence (pairwise Mahalanobis + position, Eqs. 8–10)
#   SD = Semantic Distinctiveness (WordNet + MMD, Eqs. 6–7)
# ═══════════════════════════════════════════════════════════════════════

# --- 9.1  Contextual Coherence (Eqs. 8–10, 13–15) ---------------------

@torch.no_grad()
def embed_words_in_context(words: List[str], prompt: str) -> np.ndarray:
    """Contextual BERT embedding for each word.

    We tokenize the entire prompt once, then for every surface word we
    average the sub-token vectors whose character span overlaps that
    word. This yields sentence-aware vectors (Eq. 8 uses v_i := this
    embedding)."""
    enc_raw = bert_tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_BERT_TOKENS,
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc_raw.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc_raw.items()}
    out = bert_model(**enc)
    hidden = out.last_hidden_state[0].cpu()          # (seq_len, 768)

    word_spans = get_word_char_spans(words, prompt)
    word_vecs: List[torch.Tensor] = []
    for (ws, we) in word_spans:
        sub = []
        for t, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:                  # [CLS]/[SEP]/pad
                continue
            if ts < we and te > ws:                  # char overlap
                sub.append(hidden[t])
        if sub:
            word_vecs.append(torch.stack(sub).mean(0))
        else:
            word_vecs.append(hidden[0])              # [CLS] fallback
    return torch.stack(word_vecs).numpy().astype(np.float32)


def compute_contextual_coherence(words: List[str], prompt: str) -> np.ndarray:
    """PAIRWISE formulation exactly as in Eqs. 8-10 and Appendix C.

    For every pair (i, j), i ≠ j:
        ρ(w)   = 1.0 if w is content, γ=0.3 if function           (Eq. 13)
        d_ij   = ρ(w_i) · ρ(w_j)                                   (Eq. 14)
        D_ij   = d_ij · I                                          (Eq. 15)
        Q_ij   = I + α · D_ij = (1 + α·d_ij) · I                  (Eq. 8)
        T_ij   = sqrt((v_i - v_j)ᵀ Q_ij (v_i - v_j))
               = sqrt(1 + α·d_ij) · || v_i - v_j ||               (simplified)
        R_ij   = | q_i - q_j |                                     (Eq. 9)
        r*_ij  = β · R_ij + T_ij
        CC_i   = 1 / Σ_{j ≠ i} (r*_ij + ε)                         (Eq. 10)

    The simplification follows because Q_ij is always a scalar multiple
    of the identity matrix, collapsing the quadratic form to a scaled
    Euclidean distance. This is both mathematically exact AND fast.

    After raw CC computation, min-max normalise over V* for fusion."""

    n = len(words)
    if n == 1:
        return np.array([0.5], dtype=np.float32)

    V = embed_words_in_context(words, prompt)                # (n, 768)
    pos_tags = get_word_pos_tags(words, prompt)

    # ρ(w_i) from Eq. 13 / Table 6
    rho = np.array(
        [1.0 if p in CONTENT_POS else GAMMA_FUNCTION for p in pos_tags],
        dtype=np.float32,
    )

    # Pairwise Euclidean distance matrix ||v_i - v_j||   (n, n)
    diff = V[:, None, :] - V[None, :, :]                     # (n, n, d)
    eucl = np.linalg.norm(diff, axis=-1).astype(np.float32)  # (n, n)

    # d_ij = ρ_i · ρ_j   (outer product)
    D = np.outer(rho, rho).astype(np.float32)                # (n, n)

    # Scalar Mahalanobis: T_ij = sqrt(1 + α·d_ij) · eucl_ij
    scale = np.sqrt(1.0 + CC_ALPHA * D).astype(np.float32)
    T = scale * eucl

    # Positional distance R_ij = |q_i - q_j|
    idx = np.arange(n, dtype=np.float32)
    R = np.abs(idx[:, None] - idx[None, :])

    # r*_ij = β · R_ij + T_ij
    r_star = CC_BETA * R + T

    # Sum over j ≠ i (diagonal is 0 by construction; adds ε only)
    # We add ε once per summand to guard against tiny distances collapsing
    # the sum; this follows the paper's literal Σ (r*_ij + ε).
    denom = (r_star + CC_EPSILON).sum(axis=1)                # (n,)
    denom -= CC_EPSILON                                      # remove j=i contribution
    denom = np.maximum(denom, CC_EPSILON)                    # safety

    CC_raw = 1.0 / denom
    return min_max_normalize(CC_raw)


# --- 9.2  Semantic Distinctiveness (Eqs. 6–7) -------------------------

def _spacy_pos_to_wordnet_pos(pos: str) -> Optional[str]:
    """Map SpaCy POS tags to WordNet POS constants."""
    return {
        "NOUN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ":  wn.ADJ,
        "ADV":  wn.ADV,
        "PROPN": wn.NOUN,      # proper nouns map to noun synsets
    }.get(pos)


def get_wordnet_synonyms(word: str, target_pos: Optional[str] = None,
                         max_syns: int = 5) -> List[str]:
    """Return up to `max_syns` distinct single-token synonyms from
    WordNet for `word`, optionally filtered to the given spaCy POS."""
    clean = strip_punctuation(word)
    wn_pos = _spacy_pos_to_wordnet_pos(target_pos or "")
    synsets = wn.synsets(clean, pos=wn_pos) if wn_pos else wn.synsets(clean)

    syns, seen = [], set()
    for ss in synsets:
        for lemma in ss.lemmas():
            cand = lemma.name().replace("_", " ")
            if " " in cand or not cand.isalpha():
                continue
            if cand.lower() == clean.lower():
                continue
            if cand.lower() in seen:
                continue
            seen.add(cand.lower())
            syns.append(cand)
            if len(syns) >= max_syns:
                return syns
    return syns


@torch.no_grad()
def bert_cls_embed(sentence: str) -> np.ndarray:
    """Single-sentence [CLS] vector. Shape: (1, 768)."""
    enc = bert_tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=MAX_BERT_TOKENS,
    ).to(DEVICE)
    out = bert_model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)


def compute_semantic_distinctiveness(words: List[str], prompt: str) -> np.ndarray:
    """Paper Eqs. 6–7.

    For each word w_i:
      1. Extract WordNet synonyms C(w_i)                          (§3.4.1)
      2. For each w' ∈ C(w_i), form S_{w_i→w'} by substitution
      3. Δ(S, S_{w_i→w'}) = MMD( Φ(S), Φ(S_{w_i→w'}) )             (Eq. 6)
         with Φ = BERT [CLS] and MMD using the RBF kernel (§4 Setup).
      4. SD(w_i) = mean over all synonyms                          (Eq. 7)

    For single-sample MMD with an RBF kernel, the standard unbiased
    estimator simplifies to
        MMD²(x, y) = k(x,x) − 2·k(x,y) + k(y,y)
                   = 2·(1 − exp(−γ·||x−y||²))
    We use sklearn's `rbf_kernel` with γ = 1/d (d = 768). This matches
    Gretton et al. (2012), §3 — a standard choice when only one sample
    per distribution is available.

    Words with no WordNet synonyms receive SD = 0.5 (neutral): they
    cannot be scored by substitution but shouldn't dominate via 1.0 (over-
    encrypt) or disappear via 0.0 (under-encrypt)."""

    pos_tags = get_word_pos_tags(words, prompt)
    orig_cls = bert_cls_embed(prompt)                # (1, 768)
    gamma = 1.0 / orig_cls.shape[1]                  # RBF γ = 1/d

    raw_sd = np.full(len(words), 0.5, dtype=np.float32)   # neutral default

    for i, w in enumerate(words):
        synonyms = get_wordnet_synonyms(w, target_pos=pos_tags[i])
        if not synonyms:
            continue

        # Construct substituted sentences: replace ONLY the i-th surface
        # word (word-boundary regex to avoid matching substrings).
        clean_w = strip_punctuation(w)
        if not clean_w:
            continue
        pat = re.compile(r"\b" + re.escape(clean_w) + r"\b", re.IGNORECASE)

        mmd_vals: List[float] = []
        for syn in synonyms:
            mutated = pat.sub(syn, prompt, count=1)
            if mutated == prompt:                     # substitution failed
                continue
            mut_cls = bert_cls_embed(mutated)
            # Eq. 6 — MMD² with RBF kernel on single samples.
            kxx = float(rbf_kernel(orig_cls, orig_cls, gamma=gamma)[0, 0])
            kxy = float(rbf_kernel(orig_cls, mut_cls, gamma=gamma)[0, 0])
            kyy = float(rbf_kernel(mut_cls,  mut_cls, gamma=gamma)[0, 0])
            mmd2 = max(kxx - 2.0 * kxy + kyy, 0.0)
            mmd_vals.append(mmd2)

        if mmd_vals:
            raw_sd[i] = float(np.mean(mmd_vals))      # Eq. 7

    # Normalise over V* for consistent fusion with CC.
    return min_max_normalize(raw_sd)


# --- 9.3  CIIS fusion (Eq. 11) ----------------------------------------

def compute_ciis(words: List[str], prompt: str) -> np.ndarray:
    """Eq. 11 — CIIS(w_i) = λ1 · CC(w_i) + λ2 · SD(w_i)."""
    CC = compute_contextual_coherence(words, prompt)
    SD = compute_semantic_distinctiveness(words, prompt)
    CIIS = (CIIS_LAMBDA1 * CC + CIIS_LAMBDA2 * SD).astype(np.float32)
    log.info(f"  CC   : {np.round(CC, 3)}")
    log.info(f"  SD   : {np.round(SD, 3)}")
    log.info(f"  CIIS : {np.round(CIIS, 3)}")
    return CIIS


# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — TASK RELEVANCE SCORE  (TRS)
#   Paper §3.5 — Eq. 12.
#
#   TRS(w_i) = (1/k) * Σ_{j=1..k}  LLM_j(w_i | P)
#
#   We query the backbone LLM (Llama-2-7B-chat) k = 5 times per word and
#   average the parsed relevance floats. Repeated inference stabilises
#   the score against sampling randomness (temperature = 0.7).
#
#   To avoid k·n LLM calls for prompts with duplicate words, we cache
#   by case-insensitive stripped form.
# ═══════════════════════════════════════════════════════════════════════

TRS_FALLBACK = 0.5      # if the LLM response isn't parseable

def _parse_trs_float(text: str) -> Optional[float]:
    """Extract the first float in [0, 1] from a Llama response."""
    if not text:
        return None
    # Prefer a decimal; fall back to integers.
    m = re.search(r"\b(0?\.\d+|1(?:\.0+)?|0|1)\b", text)
    if m is None:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m is None:
        return None
    try:
        val = float(m.group(1))
        return max(0.0, min(1.0, val))
    except ValueError:
        return None


def _trs_query_once(prompt: str, word: str) -> float:
    """One LLM query. Uses do_sample=True so repeated calls differ."""
    clean = strip_punctuation(word)
    query = (
        f"On a scale of 0.0 to 1.0, how relevant is the word '{clean}' "
        f"to completing the following task?\n"
        f"Task: \"{prompt}\"\n"
        f"Answer with ONLY a single number between 0.0 and 1.0."
    )
    try:
        # TRS is a scoring call — keep generations short.
        response = llama_generate(query, max_new_tokens=16, do_sample=True)
        val = _parse_trs_float(response)
        return val if val is not None else TRS_FALLBACK
    except Exception as exc:
        log.warning(f"TRS query failed for '{word}': {exc}")
        return TRS_FALLBACK


def compute_trs(prompt: str, words: List[str]) -> np.ndarray:
    """Eq. 12 — average k = TRS_NUM_ITERS LLM relevance estimates per word.

    Cache by lowercased clean form so repeated words cost O(k) instead of
    O(k · frequency). Returns (n_words,) float32 in [0, 1]."""
    cache: Dict[str, float] = {}
    out = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        key = strip_punctuation(w).lower()
        if key in cache:
            out[i] = cache[key]
            continue
        trials = [_trs_query_once(prompt, w) for _ in range(TRS_NUM_ITERS)]
        avg = float(np.mean(trials))
        cache[key] = avg
        out[i] = avg
        log.info(f"  TRS[{i}] '{w}' → {avg:.3f}   (iters={trials})")
    return out


# ═══════════════════════════════════════════════════════════════════════
# CELL 11 — CLUSTERING + ACTION ASSIGNMENT
#   Paper §3.6 — K-Means (k = 8) on (PLRS, CIIS, TRS) triplets.
#
#   For each cluster c, centroid μ_c = (μ_c^PLRS, μ_c^CIIS, μ_c^TRS).
#   Global mean x̄^(d) = mean of dimension d over ALL n words in the prompt.
#
#   A word's dimension d is classified HIGH iff  μ_c^(d) > x̄^(d).
#
#   Paper Table 1 (full 2×2×2 → action map):
#       (L, H, L)  (L, L, H)  (L, H, H)  → Retain
#       (H, L, H)                         → Replace
#       (H, H, L)  (H, H, H)              → Encrypt
#       (L, L, L)  (H, L, L)              → Delete
# ═══════════════════════════════════════════════════════════════════════

# Lookup table: (PLRS_high, CIIS_high, TRS_high) → action
_ACTION_TABLE: Dict[Tuple[bool, bool, bool], str] = {
    (False, False, False): "Delete",
    (False, False, True ): "Retain",
    (False, True,  False): "Retain",
    (False, True,  True ): "Retain",
    (True,  False, False): "Delete",
    (True,  False, True ): "Replace",
    (True,  True,  False): "Encrypt",
    (True,  True,  True ): "Encrypt",
}


def assign_actions_via_kmeans(
    plrs: np.ndarray,
    ciis: np.ndarray,
    trs:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Cluster and assign an action per word exactly per §3.6.

    Returns:
      actions      : np.ndarray of strings, shape (n,)
      cluster_ids  : cluster index each word belongs to, shape (n,)
      debug_info   : dict with centroids, global means, per-cluster H/L flags
    """
    n = len(plrs)
    features = np.stack([plrs, ciis, trs], axis=1).astype(np.float32)  # (n, 3)

    # With fewer words than k, cap k to n so KMeans doesn't fail.
    k = min(KMEANS_K, n)

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=SEED,
    ).fit(features)

    centroids   = kmeans.cluster_centers_          # (k, 3)
    cluster_ids = kmeans.labels_                   # (n,)

    # Global means per dimension over ALL prompt words.
    global_mean = features.mean(axis=0)            # (3,)

    # Per-cluster classification (applied identically to all its members).
    cluster_hl = np.zeros((k, 3), dtype=bool)
    for c in range(k):
        cluster_hl[c] = centroids[c] > global_mean

    # Per-word action by looking up each word's cluster H/L triple.
    actions = np.empty(n, dtype=object)
    for i in range(n):
        p_h, c_h, t_h = cluster_hl[cluster_ids[i]]
        actions[i] = _ACTION_TABLE[(bool(p_h), bool(c_h), bool(t_h))]

    debug = {
        "centroids":        centroids,
        "global_mean":      global_mean,
        "cluster_hl_flags": cluster_hl,
        "k_used":           k,
    }
    return actions, cluster_ids, debug


# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — ENCRYPT: SELF-RECOMMENDATION PROMPT
#   Paper Appendix A / Figure 5.
#
#   Pipeline for Encrypt action:
#     1. Apply Retain / Replace / Delete to produce an intermediate P^M
#        where every word marked for Encrypt is swapped with [MASK].
#     2. Backbone LLM fills the masks n' times → candidate prompts
#        P^M' = {P^M'_1, P^M'_2, …, P^M'_{n'}}.
#     3. The SAME backbone LLM scores those candidates via the
#        Self-Recommendation Prompt (Figure 5) and returns the best index.
#
#   For Replace: random WordNet synonym (Paper §3.6 specifies this).
# ═══════════════════════════════════════════════════════════════════════

MASK_TOKEN = "[MASK]"


def apply_non_encrypt_actions(
    words: List[str],
    actions: np.ndarray,
    prompt: str,
) -> List[str]:
    """Produce the intermediate masked word list: Retain kept as-is,
    Delete → empty string, Replace → uniform-random WordNet synonym
    of the same POS, Encrypt → [MASK]. Returns the new word list (same
    length)."""
    pos_tags = get_word_pos_tags(words, prompt)
    out = list(words)

    for i, (w, a) in enumerate(zip(words, actions)):
        if a == "Retain":
            continue
        if a == "Delete":
            out[i] = ""
            continue
        if a == "Encrypt":
            out[i] = MASK_TOKEN
            continue
        if a == "Replace":
            # Paper §3.6: "random synonym substitution using WordNet"
            syns = get_wordnet_synonyms(w, target_pos=pos_tags[i])
            if syns:
                out[i] = random.choice(syns)
            else:
                out[i] = w          # no synonym → leave unchanged
            continue

    return out


def build_masked_prompt(masked_words: List[str]) -> str:
    """Join non-empty words with single spaces to form the masked prompt P^M."""
    return " ".join(w for w in masked_words if w).strip()


# ---- Step (2): generate n' candidate fills for [MASK] tokens ----------

def generate_encrypt_candidates(masked_prompt: str,
                                original_prompt: str,
                                n_candidates: int = ENCRYPT_NUM_CANDIDATES
                                ) -> List[str]:
    """Ask Llama-2 to fill every [MASK] with a semantically coherent
    REPLACEMENT that differs from the original. Appendix A:
      'words requiring encryption are entirely replaced with distinct
       alternative expressions that preserve semantic integrity.'

    We request multiple candidates in one call for efficiency. Sampling
    at the default temperature (0.7) provides the n' different instances
    P^M'_1..n'.
    """
    if MASK_TOKEN not in masked_prompt:
        return [masked_prompt]

    query = (
        "You are rewriting a sentence for privacy. Every [MASK] must be "
        "replaced with a COMPLETELY DIFFERENT word or short phrase that "
        "is contextually fluent but does NOT reveal the original content.\n"
        "Rules:\n"
        "1. Do NOT include any real personal names, phone numbers, "
        "SSNs, emails, addresses, or credit card numbers.\n"
        "2. Preserve grammar, tense, and overall sentence intent.\n"
        "3. Each candidate must be a complete rewritten sentence.\n"
        "4. No [MASK] tokens may remain.\n"
        f"5. Produce exactly {n_candidates} alternatives, numbered 1. 2. 3.\n\n"
        f"Masked sentence to rewrite: \"{masked_prompt}\"\n\n"
        f"Output format (and ONLY this):\n"
        + "\n".join(f"{i+1}. <rewritten sentence {i+1}>"
                    for i in range(n_candidates))
    )

    response = llama_generate(query, max_new_tokens=256, do_sample=True)

    # Parse numbered lines.
    candidates: List[str] = []
    for line in response.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+?)\s*$", line)
        if m:
            cand = m.group(2).strip().strip('"').strip("'")
            if cand and MASK_TOKEN not in cand and cand != original_prompt:
                candidates.append(cand)

    # Dedup & cap.
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c); unique.append(c)
        if len(unique) >= n_candidates:
            break

    # Fallback: if parsing failed, emit a deterministic placeholder
    # version where every [MASK] is replaced by REDACTED_k.
    if not unique:
        counter = {"n": 0}
        def _repl(_):
            counter["n"] += 1
            return f"REDACTED_{counter['n']}"
        fallback = re.sub(re.escape(MASK_TOKEN), _repl, masked_prompt)
        return [fallback]

    return unique


# ---- Step (3): Self-Recommendation Prompt (Figure 5) ------------------

def self_recommend_candidate(original_prompt: str,
                             candidates: List[str]) -> int:
    """Implements the prompt template from Figure 5 (Appendix A).

    Role definition → Task specification → Reference prompt → Candidate list
    → Instruction. The LLM returns ONLY the number of the selected
    candidate. Returns a 0-indexed integer into `candidates`.
    """
    if len(candidates) == 1:
        return 0

    candidate_block = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(candidates)
    )

    # Figure 5 template, adapted to an inline chat message.
    query = (
        "Role definition:\n"
        "You are an expert in language understanding and evaluation.\n\n"
        "Task specification:\n"
        "Here is the task: evaluate candidate rewrites of a reference "
        "prompt and select the best one.\n\n"
        "Reference prompt:\n"
        f"\"{original_prompt}\"\n\n"
        "Candidate list:\n"
        f"{candidate_block}\n\n"
        "Instruction:\n"
        "Please evaluate each prompt above based on fluency, coherence, "
        "and semantic consistency with the reference prompt. Select the "
        "prompt that is the most fluent, coherent, and best achieves the "
        "same task result as the reference prompt. "
        "Provide only the number of the selected prompt.\n\n"
        "Answer:"
    )

    try:
        # Deterministic scoring → do_sample=False so the same prompt
        # always selects the same candidate.
        response = llama_generate(query, max_new_tokens=8, do_sample=False)
        m = re.search(r"\b(\d+)\b", response)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(candidates):
                return idx
    except Exception as exc:
        log.warning(f"Self-recommendation failed: {exc}")

    return 0                # default: first candidate


# ═══════════════════════════════════════════════════════════════════════
# CELL 13 — PIPELINE ORCHESTRATOR
#   End-to-end flow:
#     (a) tokenise into whitespace-separated words
#     (b) PLRS, CIIS, TRS
#     (c) K-Means clustering → per-word action
#     (d) apply Retain / Replace / Delete, mask Encrypts
#     (e) generate & self-recommend the best Encrypt candidate
# ═══════════════════════════════════════════════════════════════════════

def run_alsa(prompt: str) -> Dict:
    """Full ALSA pipeline on one prompt. Returns a structured dict of
    per-word scores, actions, and the final sanitised prompt."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be non-empty.")

    log.info("=" * 72)
    log.info(f"Prompt: {prompt!r}")
    words = prompt.split()                          # whitespace-tokenised
    n = len(words)
    log.info(f"Number of words: {n}")

    # ── Scores ────────────────────────────────────────────────────────
    log.info("-- PLRS --")
    plrs = compute_plrs(prompt, words)

    log.info("-- CIIS --")
    ciis = compute_ciis(words, prompt)

    log.info("-- TRS --")
    trs = compute_trs(prompt, words)

    # ── Cluster & assign action ────────────────────────────────────────
    log.info("-- K-Means clustering & action assignment --")
    actions, cluster_ids, cluster_debug = assign_actions_via_kmeans(
        plrs, ciis, trs
    )
    for i, (w, a) in enumerate(zip(words, actions)):
        log.info(f"  [{i:02d}] {w:<20s} → {a:<8s}  "
                 f"PLRS={plrs[i]:.3f}  CIIS={ciis[i]:.3f}  TRS={trs[i]:.3f}  "
                 f"cluster={cluster_ids[i]}")

    # ── Apply non-encrypt actions ─────────────────────────────────────
    intermediate_words = apply_non_encrypt_actions(words, actions, prompt)
    masked_prompt = build_masked_prompt(intermediate_words)
    log.info(f"Masked prompt: {masked_prompt!r}")

    # ── Encrypt: generate candidates + self-recommendation ─────────────
    if MASK_TOKEN in masked_prompt:
        log.info("-- Encrypt: generating candidates --")
        candidates = generate_encrypt_candidates(masked_prompt, prompt)
        log.info(f"  {len(candidates)} candidate(s) generated:")
        for j, c in enumerate(candidates):
            log.info(f"   [{j+1}] {c}")
        best = self_recommend_candidate(prompt, candidates)
        log.info(f"Self-recommendation selected candidate #{best+1}")
        sanitised = candidates[best]
    else:
        candidates = []
        sanitised = masked_prompt

    log.info(f"Sanitised prompt: {sanitised!r}")

    # Free any transient CUDA blocks.
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "prompt":          prompt,
        "words":           words,
        "plrs":            plrs,
        "ciis":            ciis,
        "trs":             trs,
        "cluster_ids":     cluster_ids,
        "cluster_debug":   cluster_debug,
        "actions":         actions,
        "masked_prompt":   masked_prompt,
        "candidates":      candidates,
        "sanitised":       sanitised,
    }


# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — VISUALISATION
#   Two figures are saved to Google Drive under /MyDrive/ALSA/figures/:
#     (a) per-word grouped bar chart of PLRS / CIIS / TRS + action colour
#     (b) 3-D scatter of (PLRS, CIIS, TRS) with cluster colours
#         (reproduces Figure 2 of the paper)
# ═══════════════════════════════════════════════════════════════════════

from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (enables 3-D projection)

ACTION_COLOR = {
    "Retain":  "#2ecc71",
    "Replace": "#f1c40f",
    "Encrypt": "#9b59b6",
    "Delete":  "#e74c3c",
}


def plot_word_scores(result: Dict, test_idx: int,
                     save_dir: Path = FIGURE_DIR) -> Path:
    """Save a per-word grouped bar chart.

    Bars: PLRS, CIIS, TRS. Background band colour encodes the final action.
    Dashed horizontal lines show the per-metric global means (used by
    the K-Means high/low rule).
    """
    words   = result["words"]
    plrs    = result["plrs"]
    ciis    = result["ciis"]
    trs     = result["trs"]
    actions = result["actions"]
    global_mean = result["cluster_debug"]["global_mean"]

    n = len(words)
    x = np.arange(n)
    bar_w = 0.26

    fig, ax = plt.subplots(figsize=(max(9, n * 0.8), 5.5))

    # Action background bands
    for i, a in enumerate(actions):
        ax.axvspan(i - 0.5, i + 0.5, color=ACTION_COLOR[a], alpha=0.12, zorder=0)

    ax.bar(x - bar_w, plrs, bar_w, label="PLRS", color="#e74c3c", zorder=2)
    ax.bar(x,         ciis, bar_w, label="CIIS", color="#3498db", zorder=2)
    ax.bar(x + bar_w, trs,  bar_w, label="TRS",  color="#2ecc71", zorder=2)

    # Paper's K-Means rule uses the global means as thresholds — show them.
    ax.axhline(global_mean[0], ls="--", color="#e74c3c", alpha=0.55, lw=1)
    ax.axhline(global_mean[1], ls=":",  color="#3498db", alpha=0.55, lw=1)
    ax.axhline(global_mean[2], ls="-.", color="#2ecc71", alpha=0.55, lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [w if len(w) <= 14 else w[:13] + "…" for w in words],
        rotation=35, ha="right",
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Test {test_idx} — Per-word ALSA Scores\n"
        f"(dashed lines = global means; band colour = assigned action)"
    )
    ax.legend(loc="upper right")

    # Add an action-colour legend below the bar legend.
    from matplotlib.patches import Patch
    action_handles = [Patch(color=c, alpha=0.5, label=a)
                      for a, c in ACTION_COLOR.items()]
    ax.legend(handles=action_handles +
              [Patch(color="#e74c3c", label="PLRS"),
               Patch(color="#3498db", label="CIIS"),
               Patch(color="#2ecc71", label="TRS")],
              loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    out = save_dir / f"test_{test_idx:02d}_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {out}")
    return out


def plot_cluster_scatter(result: Dict, test_idx: int,
                         save_dir: Path = FIGURE_DIR) -> Path:
    """3-D scatter plot reproducing ALSA's Figure 2.

    Points are (PLRS, CIIS, TRS) triplets; colour encodes cluster id.
    Each point is annotated with its word."""
    words       = result["words"]
    plrs        = result["plrs"]
    ciis        = result["ciis"]
    trs         = result["trs"]
    cluster_ids = result["cluster_ids"]
    k_used      = result["cluster_debug"]["k_used"]

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10", max(k_used, 1))
    for c in range(k_used):
        mask = cluster_ids == c
        if not mask.any():
            continue
        ax.scatter(
            plrs[mask], ciis[mask], trs[mask],
            s=60, color=cmap(c), label=f"Cluster {c+1}",
            depthshade=True,
        )

    # Annotate each point with its word
    for i, w in enumerate(words):
        ax.text(plrs[i], ciis[i], trs[i], f"  {w}", fontsize=8)

    ax.set_xlabel("PLRS"); ax.set_ylabel("CIIS"); ax.set_zlabel("TRS")
    ax.set_xlim(0, 1);    ax.set_ylim(0, 1);    ax.set_zlim(0, 1)
    ax.set_title(
        f"Test {test_idx} — K-Means Clustering of Word Triplets\n"
        f"(PLRS, CIIS, TRS) — k={k_used}"
    )
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    out = save_dir / f"test_{test_idx:02d}_cluster3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# CELL 15 — TEST CASES & RUNNER
#   Edit TEST_CASES to suit your thesis experiments. Each prompt runs
#   through the full pipeline and produces two PNGs under
#   /content/drive/MyDrive/ALSA/figures/.
# ═══════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # Classic PII-heavy example
    "My name is John Smith and my SSN is 432-91-8765.",

    # Semantic-ambiguity example from the paper's introduction
    "During lunch I ate an apple and later attended an Apple product launch.",

    # Medical / sensitive domain
    "Patient Sarah Connor, aged 34, was diagnosed with Type-2 diabetes last Tuesday.",

    # Neutral non-sensitive
    "The quick brown fox jumps over the lazy dog.",

    # Corporate confidential pattern
    "Our Q3 revenue hit 4.2 million dollars, driven by the Project Atlas launch.",
]


def run_all_tests():
    """Run every case in TEST_CASES, save both figures per case,
    and print a compact summary table."""
    print("\n" + "=" * 72)
    print(" ALSA — Running all test cases")
    print("=" * 72 + "\n")

    for idx, prompt in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        try:
            result = run_alsa(prompt)
        except Exception as exc:
            log.error(f"Test {idx} failed: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        # Per-test summary table
        print(f"\n── Test {idx}  ({elapsed:.1f}s) ─────────────────────────────")
        print(f"Original : {prompt}")
        print(f"Sanitised: {result['sanitised']}")
        print(f"{'Word':<20}{'PLRS':>8}{'CIIS':>8}{'TRS':>8}  {'Action':<8}")
        for i, w in enumerate(result["words"]):
            print(f"{w[:19]:<20}"
                  f"{result['plrs'][i]:>8.3f}"
                  f"{result['ciis'][i]:>8.3f}"
                  f"{result['trs'][i]:>8.3f}  "
                  f"{result['actions'][i]:<8}")

        # Figures → Drive
        plot_word_scores(result, idx)
        plot_cluster_scatter(result, idx)

    print("\n" + "=" * 72)
    print(f" All figures saved under  {FIGURE_DIR}")
    print("=" * 72 + "\n")


# Entry point — call this after all previous cells have executed.
# Just running this cell in Colab will trigger the full test suite.
run_all_tests()