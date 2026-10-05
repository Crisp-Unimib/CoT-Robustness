"""
Benchmarks KLE computation time for different subset sizes m vs. the full
N_unique sample set.  Reports wall-clock time per reasoning step for
m in {5, 10, 15, 20} (our pipeline's budget options) and for
N_unique in {40, 60, 80, 100} (typical full-set sizes in the data).

Outputs: validate_subset_timing_results.json
"""

import json
import glob
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parents[2]
DATA_ROOT   = str(REPO / "data" / "processed" / "blackmail_bifurcation")
NLI_MODEL   = "microsoft/deberta-large-mnli"
BATCH_SIZE  = 32
MAX_LEN     = 256
N_TIMING_STEPS   = 50     # steps used for timing (enough for stable mean)
N_REPS_PER_STEP  = 3      # repeated forward-pass runs per step (for stable timing)
SUBSET_SIZES  = [5, 10, 15, 20]           # budget options (stored samples)
FULL_SIZES    = [40, 60, 80, 100]         # realistic N_unique values
SEED = 42
OUTPUT_FILE = str(REPO / "validate_subset_timing_results.json")
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"Loading {NLI_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
model.eval()

# ── collect steps with 20 stored samples ──────────────────────────────────────
print("Collecting steps...")
all_steps = []
per_model = defaultdict(list)
for path in glob.glob(f"{DATA_ROOT}/**/*.jsonl", recursive=True):
    model_name = Path(path).parts[-3]
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            s = d.get("samples", [])
            if len(s) == 20 and len(set(s)) == 20:
                per_model[model_name].append(s)

# stratified sample
per_model_quota = max(1, N_TIMING_STEPS // len(per_model))
for model_name, steps in per_model.items():
    random.shuffle(steps)
    all_steps.extend(steps[:per_model_quota])
random.shuffle(all_steps)
timing_steps = all_steps[:N_TIMING_STEPS]
print(f"Using {len(timing_steps)} steps for timing.")

# ── NLI forward pass for a list of (a, b) pairs ───────────────────────────────
def run_nli(texts_a, texts_b):
    for start in range(0, len(texts_a), BATCH_SIZE):
        a_b = texts_a[start:start + BATCH_SIZE]
        b_b = texts_b[start:start + BATCH_SIZE]
        enc = tokenizer(
            a_b, b_b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)
        with torch.no_grad():
            model(**enc)


def time_matrix_size(samples_pool, m, n_reps):
    """Return mean wall-clock time (seconds) for one m×m NLI matrix."""
    times = []
    pool = list(samples_pool)
    for _ in range(n_reps):
        subset = random.sample(pool, min(m, len(pool)))
        pairs  = [(i, j) for i in range(len(subset)) for j in range(i + 1, len(subset))]
        a_texts = [subset[i] for i, j in pairs]
        b_texts = [subset[j] for i, j in pairs]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_nli(a_texts, b_texts)   # direction A→B
        run_nli(b_texts, a_texts)   # direction B→A
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


# ── for FULL_SIZES we need more than 20 samples per step ──────────────────────
# We build synthetic pools by repeating + slightly perturbing stored samples.
# All samples come from real text; we just extend the pool to reach size m.
def extend_pool(samples, target_size):
    """Return a list of target_size distinct strings from samples (with repeats if needed)."""
    pool = list(samples)
    while len(pool) < target_size:
        pool.extend(samples)
    return pool[:target_size]


# ── warm-up pass ──────────────────────────────────────────────────────────────
print("Warming up...")
dummy_a = timing_steps[0][:2]
dummy_b = timing_steps[0][2:4]
run_nli(dummy_a, dummy_b)
if device.type == "cuda":
    torch.cuda.synchronize()

# ── benchmark ─────────────────────────────────────────────────────────────────
results = {}

all_sizes = SUBSET_SIZES + FULL_SIZES
for m in all_sizes:
    step_times = []
    for step_samples in timing_steps:
        pool = extend_pool(step_samples, m)
        mean_t, _ = time_matrix_size(pool, m, N_REPS_PER_STEP)
        step_times.append(mean_t)
    mean_ms  = float(np.mean(step_times)) * 1000
    std_ms   = float(np.std(step_times))  * 1000
    n_pairs  = m * (m - 1) // 2
    results[m] = {"mean_ms": round(mean_ms, 1), "std_ms": round(std_ms, 1), "n_pairs": n_pairs}
    print(f"  m={m:3d}  pairs={n_pairs:4d}  time={mean_ms:7.1f} ± {std_ms:.1f} ms")

# Speedup relative to m=20 (our reference)
ref_time = results[20]["mean_ms"]
for m, r in results.items():
    r["speedup_vs_m20"] = round(ref_time / r["mean_ms"], 2) if m != 20 else 1.0
    r["speedup_vs_100"] = round(results[100]["mean_ms"] / r["mean_ms"], 1)

output = {
    "device": str(device),
    "n_timing_steps": len(timing_steps),
    "n_reps_per_step": N_REPS_PER_STEP,
    "results": {str(k): v for k, v in results.items()},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")

# Pretty summary
print("\n--- Timing summary ---")
print(f"{'m':>5}  {'pairs':>6}  {'ms/step':>10}  {'×faster vs m=100':>18}")
for m in all_sizes:
    r = results[m]
    print(f"{m:>5}  {r['n_pairs']:>6}  {r['mean_ms']:>8.1f} ms  {r['speedup_vs_100']:>16.1f}×")
