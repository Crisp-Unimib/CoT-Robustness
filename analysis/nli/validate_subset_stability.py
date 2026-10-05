"""
Validates KLE subset stability: shows that computing KLE on a subset S' of size m
gives statistically equivalent results to computing KLE on all 20 stored samples.

Protocol:
  1. Sample N_STEPS reasoning steps from existing bifurcation_entropy.jsonl files
     (steps that have exactly 20 stored samples, stratified across models).
  2. For each step, compute the full 20x20 pairwise entailment matrix once with DeBERTa.
  3. Derive reference KLE from the full 20x20 matrix.
  4. For each subset size m in SUBSET_SIZES, draw R random subsets and compute KLE
     from the mxm submatrix — no additional NLI calls needed.
  5. Report Spearman rho and normalised RMSE between subset KLE and reference KLE.

Outputs: validate_subset_stability_results.json
"""

import json
import glob
import random
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── config ────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parents[2]
DATA_ROOT   = str(REPO / "data" / "processed" / "blackmail_bifurcation")
NLI_MODEL   = "microsoft/deberta-large-mnli"
N_STEPS     = 300      # total steps to evaluate (stratified across models)
SUBSET_SIZES = [5, 10, 15, 20]  # 20 = reference
N_REPLICATES = 50      # random subsets per size per step (for m < 20)
BATCH_SIZE  = 32
MAX_LEN     = 256      # truncation limit for NLI (sentences are short)
MIN_SAMPLES = 20       # only use steps that have exactly 20 stored samples
SEED        = 42
OUTPUT_FILE = str(REPO / "validate_subset_stability_results.json")
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── load NLI model ────────────────────────────────────────────────────────────
print(f"Loading NLI model: {NLI_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
nli_model.eval()
# label order for deberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
ENTAILMENT_IDX = 2

# ── collect candidate steps ───────────────────────────────────────────────────
print("Collecting steps...")
per_model = defaultdict(list)
for path in glob.glob(f"{DATA_ROOT}/**/*.jsonl", recursive=True):
    model_name = Path(path).parts[-3]   # .../blackmail_bifurcation/<model>/<scenario>/file
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            samples = d.get("samples", [])
            if len(samples) != MIN_SAMPLES:
                continue
            # keep only steps with meaningful diversity
            unique_text = list(set(samples))
            if len(unique_text) < MIN_SAMPLES:
                continue
            per_model[model_name].append(samples)

print("Steps per model:", {k: len(v) for k, v in per_model.items()})

# stratified sample
models = list(per_model.keys())
per_model_quota = max(1, N_STEPS // len(models))
selected = []
for m, steps in per_model.items():
    random.shuffle(steps)
    selected.extend(steps[:per_model_quota])

# top up / trim to N_STEPS
random.shuffle(selected)
selected = selected[:N_STEPS]
print(f"Total steps selected: {len(selected)}")

# ── NLI helper ────────────────────────────────────────────────────────────────
def entailment_probs(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Return P(entailment) for each (a, b) pair, batched."""
    all_probs = []
    for start in range(0, len(texts_a), BATCH_SIZE):
        a_batch = texts_a[start:start + BATCH_SIZE]
        b_batch = texts_b[start:start + BATCH_SIZE]
        enc = tokenizer(
            a_batch, b_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)
        with torch.no_grad():
            logits = nli_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, ENTAILMENT_IDX].cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


def build_entailment_matrix(samples: list[str]) -> np.ndarray:
    """Build symmetric entailment matrix W with W_ii = 1."""
    n = len(samples)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a_texts = [samples[i] for i, j in pairs]
    b_texts = [samples[j] for i, j in pairs]

    p_ab = entailment_probs(a_texts, b_texts)
    p_ba = entailment_probs(b_texts, a_texts)

    W = np.eye(n)
    for k, (i, j) in enumerate(pairs):
        w = (p_ab[k] + p_ba[k]) / 2.0
        W[i, j] = w
        W[j, i] = w
    return W


def kle(W: np.ndarray) -> float:
    """Von Neumann entropy of density matrix rho = W / Tr(W)."""
    tr = np.trace(W)
    if tr < 1e-10:
        return 0.0
    rho = W / tr
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-10]
    return float(-np.sum(eigs * np.log2(eigs)))


# ── main loop ─────────────────────────────────────────────────────────────────
print("Computing entailment matrices and KLE estimates...")
reference_kles = []
subset_kles = defaultdict(list)   # m -> list of mean-over-replicates KLE per step

for step_idx, samples in enumerate(selected):
    if step_idx % 50 == 0:
        print(f"  Step {step_idx}/{len(selected)}")

    W_full = build_entailment_matrix(samples)
    ref_kle = kle(W_full)
    reference_kles.append(ref_kle)

    for m in SUBSET_SIZES:
        if m == 20:
            # reference == full matrix
            subset_kles[m].append(ref_kle)
            continue
        rep_kles = []
        for _ in range(N_REPLICATES):
            idx = sorted(random.sample(range(20), m))
            W_sub = W_full[np.ix_(idx, idx)]
            rep_kles.append(kle(W_sub))
        subset_kles[m].append(float(np.mean(rep_kles)))

# ── compute statistics ────────────────────────────────────────────────────────
ref = np.array(reference_kles)
ref_range = ref.max() - ref.min() if ref.max() > ref.min() else 1.0

results = {}
print("\n--- Subset Stability Results ---")
print(f"{'m':>4}  {'Spearman rho':>14}  {'Norm. RMSE':>12}")
for m in SUBSET_SIZES:
    est = np.array(subset_kles[m])
    rho, pval = spearmanr(ref, est)
    rmse = float(np.sqrt(np.mean((ref - est) ** 2)))
    norm_rmse = rmse / ref_range
    results[m] = {"spearman_rho": round(float(rho), 4),
                  "spearman_pval": float(pval),
                  "rmse": round(rmse, 4),
                  "norm_rmse": round(norm_rmse, 4)}
    print(f"{m:>4}  {rho:>14.4f}  {norm_rmse:>12.4f}")

output = {
    "n_steps": len(selected),
    "n_replicates": N_REPLICATES,
    "subset_sizes": SUBSET_SIZES,
    "results": results,
    "reference_kle_stats": {
        "mean": float(ref.mean()),
        "std": float(ref.std()),
        "min": float(ref.min()),
        "max": float(ref.max()),
    },
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")
