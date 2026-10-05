"""
Sweep: the number of sampled continuations per step, k in {25, 50, 100}.

The stored artifacts constrain how much of it can
be done without regenerating continuations, so this script does the part that is exact
and is explicit about the part that is not.

What varies with k, given the kernel budget |S'|=20 is held fixed:

  1. N, the branch count entering log2(N). This is the dominant, first-order effect.
  2. Which 20 continuations land in S', which perturbs KLE and D_syn second-order.

Two populations are analysed.

  A. EXACT. The 2,157 step records under Qwen3-VL-32B-Thinking_evil that retained more
     than 20 continuations, of which 289 retained the full pool of 100. For these the
     pool is on disk, so N(k) and D_syn(k) are computed exactly by subsampling. This
     model is not one of the five reported, and steps retaining all 100 are by
     construction the maximally diverse ones. Both biases are stated in the output.

  B. BOUNDED. All five reported models, using N_k = min(num_unique, k). This is exact
     whenever num_unique <= k, since the pool had already saturated below the budget,
     and an upper bound otherwise. The exactly-covered subpopulation is reported
     separately at each k so the bound is never mistaken for a measurement.

Not covered: the effect of k on KLE through S' membership. Recomputing KLE requires
forward passes through DeBERTa, which needs a GPU to be tractable. The syntactic term
is computed exactly here as a same-shaped proxy for how large that lottery is.

Outputs: sweep_continuation_budget_results.json
"""

from __future__ import annotations

import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C

OUTPUT = C.REPO / "sweep_continuation_budget_results.json"
K_VALUES = [25, 50, 100]
REPLICATES = 20
EVIL = "Qwen3-VL-32B-Thinking_evil"


def mean_syntactic_distance(samples):
    """Replica of compute_mean_syntactic_distance in the generation pipeline."""
    if len(samples) < 2:
        return 0.0
    import Levenshtein
    dists = []
    for a, b in combinations(samples, 2):
        m = max(len(a), len(b))
        dists.append(0.0 if m == 0 else Levenshtein.distance(a, b) / m)
    return float(np.mean(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# Part A: exact, on the one model that retained full pools
# ---------------------------------------------------------------------------

def load_evil_pools(min_pool):
    rows = []
    for path in sorted(glob.glob(str(C.PROC / EVIL / "*" / "bifurcation_entropy.jsonl"))):
        scenario = Path(path).parts[-2]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                samples = rec.get("samples") or []
                if len(samples) < min_pool:
                    continue
                metrics = rec.get("metrics") or {}
                rows.append({
                    "model": EVIL,
                    "scenario": scenario,
                    "step_idx": rec.get("step_idx"),
                    "pool": samples,
                    "stored_dsyn": metrics.get("syntactic_distance"),
                })
    return rows


def part_a():
    rows = load_evil_pools(min_pool=max(K_VALUES))
    print(f"\n[A] exact pools: {len(rows)} step records with all "
          f"{max(K_VALUES)} continuations retained")
    if not rows:
        return {"note": "no full pools available"}

    # Gate: reproduce the stored D_syn from the stored pool before trusting the replica.
    checks = []
    for r in rows[:200]:
        if r["stored_dsyn"] is not None:
            checks.append(abs(mean_syntactic_distance(r["pool"][:C.KERNEL_BUDGET]) - r["stored_dsyn"]))
    gate = {
        "n_checked": len(checks),
        "max_abs_diff": C._f(max(checks)) if checks else None,
        "passed": bool(checks and max(checks) < 1e-3),
    }
    print(f"    D_syn replica gate: n={gate['n_checked']} max|diff|={gate['max_abs_diff']} "
          f"passed={gate['passed']}")

    rng = np.random.default_rng(C.SEED)
    per_k = {}
    for k in K_VALUES:
        n_vals, dsyn_vals = [], []
        for r in rows:
            pool = r["pool"]
            uniq, dsyn = [], []
            for _ in range(REPLICATES):
                draw = list(rng.choice(len(pool), size=k, replace=False))
                sub = [pool[i] for i in draw]
                uniq.append(len(set(sub)))
                dsyn.append(mean_syntactic_distance(sub[:C.KERNEL_BUDGET]))
            n_vals.append(float(np.mean(uniq)))
            dsyn_vals.append(float(np.mean(dsyn)))
        per_k[k] = {"n": n_vals, "dsyn": dsyn_vals}
        print(f"    k={k:3d}  mean N={np.mean(n_vals):6.2f}  mean D_syn={np.mean(dsyn_vals):.4f}")

    ref = np.array([np.log2(n) * d for n, d in zip(per_k[100]["n"], per_k[100]["dsyn"])])
    out = {
        "population": EVIL,
        "n_steps": len(rows),
        "replicates_per_k": REPLICATES,
        "dsyn_replica_gate": gate,
        "selection_bias": (
            "Steps retaining all 100 continuations are exactly those where no two "
            "continuations were identical, so they are the most diverse steps in the "
            "corpus. This model is also not one of the five reported."
        ),
        "by_k": {},
    }
    for k in K_VALUES:
        n_arr = np.array(per_k[k]["n"])
        d_arr = np.array(per_k[k]["dsyn"])
        var = np.log2(n_arr) * d_arr
        rho, _ = spearmanr(ref, var)
        out["by_k"][str(k)] = {
            "mean_N": C._f(n_arr.mean()),
            "N_equals_k_fraction": C._f(float(np.mean(np.isclose(n_arr, k)))),
            "mean_dsyn": C._f(d_arr.mean()),
            "spearman_vs_k100": C._f(rho),
            "top_decile_overlap": C.top_fraction_overlap(ref, var, 0.10),
        }
    print(f"    Spearman vs k=100: "
          + ", ".join(f"k={k}: {out['by_k'][str(k)]['spearman_vs_k100']}" for k in K_VALUES))
    return out


# ---------------------------------------------------------------------------
# Part B: bounded, across the five reported models
# ---------------------------------------------------------------------------

def part_b():
    steps = C.load_all_steps()
    steps = [r for r in steps if r["kle"] is not None and r["n_true"] and r["n_true"] >= 2]
    anchors = C.load_anchors()
    anchors = [r for r in anchors if r["kle"] is not None and r["n_true"] and r["n_true"] >= 2]
    broken = [r["broken"] for r in anchors]
    print(f"\n[B] bounded: {len(steps)} steps, {len(anchors)} anchors "
          f"({sum(broken)} broken) across the five reported models")

    def score(rows, k):
        return [C.hbe(C.n_at_budget(r["n_true"], k), r["kle"], r["dsyn"]) for r in rows]

    ref_steps = score(steps, 100)
    ref_anchors = score(anchors, 100)

    print(f"    {'k':>4s} {'exact%':>7s} {'rho':>8s} {'top10%':>8s} {'top3':>7s} "
          f"{'Q1':>7s} {'Q4':>7s} {'ratio':>7s}")
    by_k = {}
    for k in K_VALUES:
        s_var = score(steps, k)
        a_var = score(anchors, k)
        exact = float(np.mean([r["n_true"] <= k for r in steps]))
        sp = C.spearman_by_model(ref_steps, s_var, [r["model"] for r in steps])
        dec = C.top_fraction_overlap(ref_steps, s_var, 0.10)
        top3 = C.top_k_anchor_overlap(anchors, ref_anchors, a_var, k=3)
        qb = C.quartile_break_ratio(a_var, broken)

        # The subpopulation where min(num_unique, k) is exact rather than a bound.
        sub = [i for i, r in enumerate(steps) if r["n_true"] <= k]
        sub_rho = None
        if len(sub) > 10:
            sub_rho = C._f(spearmanr([ref_steps[i] for i in sub], [s_var[i] for i in sub])[0])

        by_k[str(k)] = {
            "exact_fraction": C._f(exact),
            "n_exact_steps": len(sub),
            "spearman_vs_k100": sp,
            "spearman_on_exact_subpopulation": sub_rho,
            "top_decile_overlap": dec,
            "top_k_anchor_overlap": top3,
            "quartile_break": qb,
        }
        print(f"    {k:4d} {100*exact:6.1f}% {sp['overall']['rho']:8.4f} "
              f"{dec['overlap_frac']:8.3f} {top3['mean_overlap']:7.3f} "
              f"{qb['q1_break_rate']:7.4f} {qb['q4_break_rate']:7.4f} {str(qb['ratio']):>7s}")

    return {
        "estimator": "N_k = min(num_unique, k)",
        "estimator_note": (
            "Exact when num_unique <= k, an upper bound otherwise. The exactly covered "
            "fraction is reported at each k."
        ),
        "models": sorted(C.MODELS.keys()),
        "n_steps": len(steps),
        "n_anchors": len(anchors),
        "n_broken": int(sum(broken)),
        "by_k": by_k,
    }


def main():
    payload = {
        "sweep": "continuation_budget",
        "question": (
            "Do the step ranking, the selected attack points, and the quartile break-rate "
            "conclusion survive reducing the number of sampled continuations per step?"
        ),
        "k_values": K_VALUES,
        "requires_gpu": False,
        "requires_generation": False,
        "not_covered": (
            "The effect of k on KLE through which 20 continuations enter S'. Recomputing "
            "KLE needs DeBERTa forward passes and therefore a GPU. The syntactic term is "
            "computed exactly in Part A as a same-shaped proxy for the size of that effect."
        ),
        "part_a_exact_pools": part_a(),
        "part_b_bounded_five_models": part_b(),
    }
    C.write_results(OUTPUT, payload)


if __name__ == "__main__":
    main()
