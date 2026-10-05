"""Optimised, cached KLE utilities for the NLI-backbone swap (rebuttal analysis).

Numerically equivalent to the paper's KLE
(submission/src/generation/generate_bifurcation.py :: compute_kle_score):
  W_ij = avg(P(i|=j), P(j|=i)), W_ii = 1, rho = W / Tr(W),
  KLE  = -sum(lambda * log2(lambda)) over eigenvalues lambda > 1e-10 of rho.

Speedups over the original per-direction loop:
  * one batched bidirectional pass (all off-diagonal ordered pairs in a single call)
  * length-bucketed batching (less padding waste)
  * torch.inference_mode(), model.eval(), optional fp16 on CUDA
  * per-step W-matrix cache on disk (nothing recomputed on re-runs)
The entailment class index is auto-detected from model.config (not hardcoded to 2).

Standalone module. submission/src could import it to speed up the pipeline; we do
not do that here, to keep the paper's Appendix B timings valid.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def detect_entailment_index(model) -> int:
    """Find the ENTAILMENT logit index from the model config (robust to reordering)."""
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)
    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        if "entail" in str(label).lower():
            return int(idx)
    raise ValueError(f"No ENTAILMENT label in model config: id2label={id2label}")


class NLIScorer:
    def __init__(self, model_name, device=None, fp16=False, batch_size=64, max_len=256):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        if self.fp16:
            self.model.half()
        self.entail_idx = detect_entailment_index(self.model)

    def entailment_probs(self, a_texts, b_texts):
        """P(entailment) for each (a, b) pair. Length-bucketed batching; original order out."""
        n = len(a_texts)
        out = np.empty(n, dtype=np.float64)
        order = sorted(range(n), key=lambda k: len(a_texts[k]) + len(b_texts[k]))
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            a = [a_texts[k] for k in idx]
            b = [b_texts[k] for k in idx]
            enc = self.tokenizer(a, b, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_len).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, self.entail_idx].cpu().numpy()
            for k, p in zip(idx, probs):
                out[k] = float(p)
        return out

    def build_W(self, samples):
        """Symmetric entailment matrix W (W_ii = 1) via a single bidirectional batch."""
        n = len(samples)
        W = np.eye(n, dtype=np.float64)
        if n < 2:
            return W
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = len(pairs)
        # both directions concatenated -> one entailment_probs call
        a = [samples[i] for i, j in pairs] + [samples[j] for i, j in pairs]
        b = [samples[j] for i, j in pairs] + [samples[i] for i, j in pairs]
        probs = self.entailment_probs(a, b)
        avg = (probs[:m] + probs[m:]) / 2.0
        for k, (i, j) in enumerate(pairs):
            W[i, j] = avg[k]
            W[j, i] = avg[k]
        return W


def kle_from_W(W) -> float:
    """Von Neumann entropy (base 2) of rho = W / Tr(W)."""
    W = np.asarray(W, dtype=np.float64)
    tr = float(np.trace(W))
    if tr < 1e-10:
        return 0.0
    rho = W / tr
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-10]
    if eigs.size == 0:
        return 0.0
    return float(-np.sum(eigs * np.log2(eigs)))


def hbe(kle, dsyn, n, alpha=0.7, beta=0.3) -> float:
    """HBE = log2(N) * (alpha*KLE + beta*D_syn); matches the stored bifurcation_entropy_kle."""
    if n is None or n < 2 or dsyn is None:
        return 0.0
    return math.log2(n) * (alpha * kle + beta * dsyn)


# ---- per-step W-matrix cache -------------------------------------------------
def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(tag))


def _fname(key: str, model_tag: str) -> str:
    h = hashlib.md5(f"{key}||{model_tag}".encode()).hexdigest()[:16]
    return f"{h}__{_safe(model_tag)}.npy"


def cache_load(cache_dir, key, model_tag):
    p = Path(cache_dir) / _fname(key, model_tag)
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            return None
    return None


def cache_save(cache_dir, key, model_tag, W):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(cache_dir) / _fname(key, model_tag), np.asarray(W, dtype=np.float32))
