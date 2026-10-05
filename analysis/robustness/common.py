"""
Shared loading and statistics for the robustness sweeps (paper appendix).

Every quantity here is derived from artifacts already on disk. Nothing in this
module calls a model, an NLI backbone, or a GPU.

The stored H_BE is

    bifurcation_entropy_kle = log2(N) * (alpha*KLE + beta*D_syn),  alpha=0.7, beta=0.3

with N taken from `metrics.n_unique`, which the shipped runs pinned to the kernel
budget |S'|=20 on every record. The true number of distinct continuations in the
k=100 pool survives separately in the top-level `num_unique` field. Both are
exposed here so a sweep can choose a convention explicitly.
"""

from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

# numpy and scipy are imported lazily inside the statistical helpers rather than at
# module scope, so that a sweep needing no array maths (segmentation) runs on the
# standard library alone. This keeps the dependency surface honest; it is not a
# workaround for anything. See the README's "Environment note" for the unrelated
# interpreter instability observed on this machine.

# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed" / "blackmail_bifurcation"
RES = REPO / "data" / "results" / "blackmail"

# results-directory name -> processed-directory name.
# These disagree for two models; joining on the wrong one silently drops them.
MODELS = {
    "Qwen3-VL-32B-Thinking": "Qwen3-VL-32B-Thinking",
    "Qwen3-Next-80B": "Qwen3-Next-80B-A3B-Thinking",
    "GLM-4.5-Air": "GLM-4.5-Air",
    "NVIDIA-Nemotron-3-Nano": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss-120b": "gpt-oss-120b",
}

# Present in the artifacts but not among the five reported models.
EXTRA_MODELS = {
    "Apriel-1.6-15b-Thinker": "Apriel-1.6-15b-Thinker",
    "Qwen3-VL-32B-Thinking_evil": "Qwen3-VL-32B-Thinking_evil",
}

ALPHA, BETA = 0.7, 0.3
KERNEL_BUDGET = 20
SEED = 42


# ---------------------------------------------------------------------------
# H_BE variants
# ---------------------------------------------------------------------------

def hbe(n, kle, dsyn, alpha=ALPHA, beta=BETA):
    """H_BE = log2(N) * (alpha*KLE + beta*D_syn). Zeroed when N < 2, as in the pipeline."""
    if n is None or n < 2 or kle is None:
        return 0.0
    if dsyn is None:
        dsyn = 0.0
    return math.log2(n) * (alpha * kle + beta * dsyn)


def n_at_budget(num_unique, k):
    """
    Upper bound on the number of distinct continuations obtainable from a pool of k draws.

    Exact whenever num_unique <= k (the pool had already saturated below the budget);
    an upper bound otherwise. Callers should report the exact subpopulation separately.
    """
    if num_unique is None:
        return None
    return min(num_unique, k)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_steps(processed_dir):
    """{scenario_id: {step_idx: record}} for one model."""
    out = {}
    for path in sorted(glob.glob(str(PROC / processed_dir / "*" / "bifurcation_entropy.jsonl"))):
        scenario = Path(path).parts[-2]
        steps = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                steps[rec.get("step_idx")] = rec
        out[scenario] = steps
    return out


def load_base_traces(processed_dir):
    """
    {scenario_id: raw_cot_text} for one model.

    `scratchpad` is the complete un-segmented reasoning text, which is what makes an
    offline re-segmentation possible without regenerating anything.
    """
    out = {}
    for path in sorted(glob.glob(str(PROC / processed_dir / "*" / "base_response.json"))):
        scenario = Path(path).parts[-2]
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        text = blob.get("scratchpad") or blob.get("response") or ""
        if text:
            out[scenario] = {
                "text": text,
                "chunks": blob.get("chunks") or [],
                "is_blackmail": blob.get("is_blackmail"),
            }
    return out


def load_anchors(models=None):
    """
    One row per attacked anchor, joined to its step record.

    Rows carry both N conventions so a sweep can pick one:
      n_pinned  -- metrics.n_unique, the shipped kernel budget (20 almost everywhere)
      n_true    -- num_unique, distinct continuations in the k=100 pool
    """
    models = models or MODELS
    rows = []
    for res_dir, proc_dir in models.items():
        steps = load_steps(proc_dir)
        pattern = str(RES / res_dir / "*" / "*" / "experiment_result_*.json")
        for path in sorted(glob.glob(pattern)):
            arm = "random" if "_random" in Path(path).parts[-2] else "hbe"
            try:
                with open(path, encoding="utf-8") as fh:
                    blob = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            scenario = blob.get("metadata", {}).get("problem_id")
            per_step = steps.get(scenario, {})
            for anchor in blob.get("anchor_results", []):
                # Drop status "Unknown" (the attack errored, so the trace was never
                # scored). Counting it as not-broken would deflate every break rate.
                # This matches the convention recovered from the published Figure 4;
                # see recreate_figure4.py.
                if anchor.get("status") not in ("Broken", "Robust"):
                    continue
                idx = anchor.get("anchor_idx")
                rec = per_step.get(idx)
                if rec is None:
                    continue
                metrics = rec.get("metrics") or {}
                rows.append({
                    "model": res_dir,
                    "scenario": scenario,
                    "arm": arm,
                    "step_idx": idx,
                    "status": anchor.get("status"),
                    "broken": anchor.get("status") == "Broken",
                    "kle": metrics.get("kle_score"),
                    "dsyn": metrics.get("syntactic_distance"),
                    "dsem": metrics.get("semantic_distance"),
                    "n_pinned": metrics.get("n_unique"),
                    "n_true": rec.get("num_unique"),
                    "stored_hbe": metrics.get("bifurcation_entropy_kle"),
                })
    return rows


def load_all_steps(models=None, require_metrics=True):
    """One row per step record across models. Used for ranking-stability sweeps."""
    models = models or MODELS
    rows = []
    for res_dir, proc_dir in models.items():
        for scenario, steps in load_steps(proc_dir).items():
            n_steps = len(steps)
            for idx, rec in steps.items():
                metrics = rec.get("metrics") or {}
                if require_metrics and metrics.get("kle_score") is None:
                    continue
                rows.append({
                    "model": res_dir,
                    "scenario": scenario,
                    "step_idx": idx,
                    "kle": metrics.get("kle_score"),
                    "dsyn": metrics.get("syntactic_distance"),
                    "n_pinned": metrics.get("n_unique"),
                    "n_true": rec.get("num_unique"),
                    "stored_hbe": metrics.get("bifurcation_entropy_kle"),
                    "n_steps": n_steps,
                    "first_half": idx is not None and idx < n_steps // 2,
                })
    return rows


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values, statistic, n_boot=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for a statistic over a 1-D array of records."""
    import numpy as np
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=object)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            draws.append(statistic(values[idx]))
        except (ZeroDivisionError, ValueError):
            continue
    if not draws:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def proportion_ci(successes, total, alpha=0.05):
    """Wilson score interval, which behaves at the small counts some models produce."""
    if total == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def spearman_by_model(ref, var, models):
    """Overall and per-model Spearman rho between two score vectors."""
    import numpy as np
    from scipy.stats import spearmanr
    ref = np.asarray(ref, dtype=float)
    var = np.asarray(var, dtype=float)
    models = np.asarray(models)
    out = {}
    rho, pval = spearmanr(ref, var)
    out["overall"] = {"rho": _f(rho), "pval": _f(pval), "n": int(len(ref))}
    per = {}
    for m in sorted(set(models.tolist())):
        sel = models == m
        if sel.sum() < 3:
            per[m] = {"rho": None, "n": int(sel.sum()), "note": "too few steps"}
            continue
        r, p = spearmanr(ref[sel], var[sel])
        per[m] = {"rho": _f(r), "pval": _f(p), "n": int(sel.sum())}
    out["per_model"] = per
    return out


def top_fraction_overlap(ref, var, frac=0.10):
    """Overlap between the top `frac` of each ranking. Reported as |A n B| / |A|."""
    import numpy as np
    ref = np.asarray(ref, dtype=float)
    var = np.asarray(var, dtype=float)
    k = max(1, int(round(frac * len(ref))))
    a = set(np.argsort(-ref, kind="stable")[:k].tolist())
    b = set(np.argsort(-var, kind="stable")[:k].tolist())
    inter = len(a & b)
    return {
        "k": k,
        "intersection": inter,
        "overlap_frac": _f(inter / k),
        "jaccard": _f(inter / len(a | b)) if (a | b) else None,
    }


def top_k_anchor_overlap(rows, ref_scores, var_scores, k=3):
    """
    Overlap in the top-K attack points selected per trace.

    This is the operationally relevant quantity: two rankings can correlate well
    overall and still disagree about which steps actually get attacked.
    """
    import numpy as np
    by_trace = defaultdict(list)
    for i, row in enumerate(rows):
        by_trace[(row["model"], row["scenario"])].append(i)

    per_trace, per_model = [], defaultdict(list)
    for (model, _scenario), idxs in by_trace.items():
        if len(idxs) < k:
            continue
        ref_top = {idxs[j] for j in np.argsort(-np.asarray([ref_scores[i] for i in idxs]), kind="stable")[:k]}
        var_top = {idxs[j] for j in np.argsort(-np.asarray([var_scores[i] for i in idxs]), kind="stable")[:k]}
        frac = len(ref_top & var_top) / k
        per_trace.append(frac)
        per_model[model].append(frac)

    return {
        "K": k,
        "n_traces": len(per_trace),
        "mean_overlap": _f(np.mean(per_trace)) if per_trace else None,
        "median_overlap": _f(np.median(per_trace)) if per_trace else None,
        "frac_traces_identical": _f(np.mean([p == 1.0 for p in per_trace])) if per_trace else None,
        "per_model": {m: {"mean_overlap": _f(np.mean(v)), "n_traces": len(v)}
                      for m, v in sorted(per_model.items())},
    }


def quartile_break_ratio(scores, broken, n_boot=2000, seed=SEED):
    """
    Conditional break rate in the top vs bottom H_BE quartile, with the ratio the
    paper reports as its headline. Bootstrap CI on the ratio, Wilson CIs on each rate.
    """
    import numpy as np
    scores = np.asarray(scores, dtype=float)
    broken = np.asarray(broken, dtype=bool)
    if len(scores) < 8:
        return {"n": int(len(scores)), "note": "too few anchors"}

    q1, q3 = np.quantile(scores, [0.25, 0.75])
    lo, hi = scores <= q1, scores >= q3
    r1 = broken[lo].mean() if lo.sum() else float("nan")
    r4 = broken[hi].mean() if hi.sum() else float("nan")

    rng = np.random.default_rng(seed)
    ratios = []
    n = len(scores)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s, b = scores[idx], broken[idx]
        a, c = np.quantile(s, [0.25, 0.75])
        m_lo, m_hi = s <= a, s >= c
        if m_lo.sum() == 0 or m_hi.sum() == 0:
            continue
        d = b[m_lo].mean()
        if d > 0:
            ratios.append(b[m_hi].mean() / d)
    ci = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))) if ratios else (float("nan"),) * 2

    return {
        "n": int(n),
        "q1_upper_bound": _f(q1),
        "q4_lower_bound": _f(q3),
        "score_min": _f(scores.min()),
        "score_max": _f(scores.max()),
        "q1_n": int(lo.sum()),
        "q4_n": int(hi.sum()),
        "q1_break_rate": _f(r1),
        "q4_break_rate": _f(r4),
        "q1_break_rate_ci": [_f(x) for x in proportion_ci(int(broken[lo].sum()), int(lo.sum()))],
        "q4_break_rate_ci": [_f(x) for x in proportion_ci(int(broken[hi].sum()), int(hi.sum()))],
        "ratio": _f(r4 / r1) if r1 and not math.isnan(r1) and r1 > 0 else None,
        "ratio_ci": [_f(x) for x in ci],
    }


def full_metric_suite(rows, ref_scores, var_scores, top_frac=0.10, k_anchor=3, broken=None):
    """The comparison battery, applied to one variant."""
    models = [r["model"] for r in rows]
    suite = {
        "spearman": spearman_by_model(ref_scores, var_scores, models),
        "top_decile_overlap": top_fraction_overlap(ref_scores, var_scores, top_frac),
        "top_k_anchor_overlap": top_k_anchor_overlap(rows, ref_scores, var_scores, k_anchor),
    }
    if broken is not None:
        suite["quartile_break"] = quartile_break_ratio(var_scores, broken)
    return suite


def _f(x):
    """Round for readable JSON, preserving None and NaN honestly."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return None
    return round(x, 6)


def write_results(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {path}")
