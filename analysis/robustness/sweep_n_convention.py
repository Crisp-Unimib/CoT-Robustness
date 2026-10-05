"""
Sweep: the definition of N in H_BE = log2(N) * (alpha*KLE + beta*D_syn).

The manuscript defines N as the number of distinct branches at a step. The shipped
runs computed it from `metrics.n_unique`, which those runs pinned to the kernel
budget |S'|=20 on every record, so log2(N) is a constant 4.3219 wherever a step has
at least 20 unique continuations. The true count over the k=100 pool was retained
separately in `num_unique` and never entered the metric.

This sweep recomputes the step ranking under each convention and asks whether the
ranking, the selected attack points, and the quartile break-rate conclusion survive.
Everything is derived from stored scalars: no model, NLI backbone, or GPU is used.

Outputs: sweep_n_convention_results.json
"""

from __future__ import annotations

import numpy as np

import common as C

OUTPUT = C.REPO / "sweep_n_convention_results.json"


def variants(row):
    """Score a step under each N convention. Returns {name: score}."""
    kle, dsyn = row["kle"], row["dsyn"]
    n_pin, n_true = row["n_pinned"], row["n_true"]
    return {
        # The shipped metric: N pinned to the kernel budget, blended numerator.
        "pinned20_blend": C.hbe(n_pin, kle, dsyn),
        # N pinned, but the numerator the manuscript states (pure KLE).
        "pinned20_kle": C.hbe(n_pin, kle, dsyn, alpha=1.0, beta=0.0),
        # True branch count over the k=100 pool, blended numerator.
        "trueN_blend": C.hbe(n_true, kle, dsyn),
        # The manuscript's formula as written: log2(N_true) * KLE.
        "trueN_kle": C.hbe(n_true, kle, dsyn, alpha=1.0, beta=0.0),
    }


def main():
    print("Loading step records for the five reported models...")
    steps = C.load_all_steps()
    steps = [r for r in steps if r["kle"] is not None and r["n_true"] and r["n_true"] >= 2]
    print(f"  {len(steps)} step records with usable metrics")

    print("Loading attacked anchors...")
    anchors = C.load_anchors()
    anchors = [r for r in anchors if r["kle"] is not None and r["n_true"] and r["n_true"] >= 2]
    broken = [r["broken"] for r in anchors]
    print(f"  {len(anchors)} anchors joined, {sum(broken)} broken")

    # --- diagnostic: is log2(N) informative at all in the shipped data? -----
    n_pin = np.array([r["n_pinned"] for r in steps], dtype=float)
    n_true = np.array([r["n_true"] for r in steps], dtype=float)
    pinned_at_budget = float(np.mean(n_pin == C.KERNEL_BUDGET))
    diagnostic = {
        "n_step_records": len(steps),
        "frac_records_with_n_pinned_equal_20": round(pinned_at_budget, 6),
        "distinct_values_of_n_pinned": sorted({int(v) for v in n_pin.tolist()})[:12],
        "log2_n_pinned_std": C._f(np.std(np.log2(n_pin))),
        "log2_n_true_std": C._f(np.std(np.log2(n_true))),
        "n_true_mean": C._f(n_true.mean()),
        "n_true_min": int(n_true.min()),
        "n_true_max": int(n_true.max()),
        "note": (
            "log2(N) can only reorder steps if it varies across them. Under the shipped "
            "convention its spread is near zero, so the H_BE ranking reduces to the ranking "
            "of the numerator alone."
        ),
    }
    print(f"\n  n_pinned == 20 on {100*pinned_at_budget:.1f}% of records")
    print(f"  sd(log2 N) pinned = {diagnostic['log2_n_pinned_std']}  vs  true = {diagnostic['log2_n_true_std']}")

    # --- ranking stability across all steps ---------------------------------
    step_scores = {name: [] for name in variants(steps[0])}
    for row in steps:
        for name, val in variants(row).items():
            step_scores[name].append(val)

    ref_name = "pinned20_blend"
    ref = step_scores[ref_name]

    print(f"\nRanking stability of every convention against the shipped metric ({ref_name}):")
    ranking = {}
    for name, scores in step_scores.items():
        if name == ref_name:
            continue
        suite = {
            "spearman": C.spearman_by_model(ref, scores, [r["model"] for r in steps]),
            "top_decile_overlap": C.top_fraction_overlap(ref, scores, 0.10),
        }
        ranking[name] = suite
        rho = suite["spearman"]["overall"]["rho"]
        ov = suite["top_decile_overlap"]["overlap_frac"]
        print(f"  {name:16s} rho={rho:.4f}  top-10% overlap={ov:.3f}")

    # --- attack-point selection and the downstream conclusion ---------------
    anchor_scores = {name: [] for name in variants(anchors[0])}
    for row in anchors:
        for name, val in variants(row).items():
            anchor_scores[name].append(val)

    print("\nAttack-point selection and quartile break rate (attacked anchors only):")
    outcome = {}
    ref_anchor = anchor_scores[ref_name]
    for name, scores in anchor_scores.items():
        entry = {
            "top_k_anchor_overlap": (
                None if name == ref_name
                else C.top_k_anchor_overlap(anchors, ref_anchor, scores, k=3)
            ),
            "quartile_break": C.quartile_break_ratio(scores, broken),
        }
        outcome[name] = entry
        qb = entry["quartile_break"]
        ov = entry["top_k_anchor_overlap"]
        ovs = "  (reference)" if ov is None else f"  top-3 overlap={ov['mean_overlap']:.3f}"
        print(f"  {name:16s} Q1={qb['q1_break_rate']:.4f} Q4={qb['q4_break_rate']:.4f} "
              f"ratio={qb['ratio']}{ovs}")

    payload = {
        "sweep": "n_convention",
        "question": (
            "Does the step ranking, the top-K attack-point selection, and the quartile "
            "break-rate conclusion depend on how N is defined in H_BE?"
        ),
        "requires_gpu": False,
        "requires_generation": False,
        "coverage": {
            "models": sorted(C.MODELS.keys()),
            "n_step_records": len(steps),
            "n_anchors": len(anchors),
            "n_broken": int(sum(broken)),
            "arms": "both the H_BE-targeted and the random control arm, pooled",
        },
        "reference_variant": ref_name,
        "variant_definitions": {
            "pinned20_blend": "log2(20) * (0.7*KLE + 0.3*D_syn)  -- the shipped metric",
            "pinned20_kle": "log2(20) * KLE",
            "trueN_blend": "log2(num_unique) * (0.7*KLE + 0.3*D_syn)",
            "trueN_kle": "log2(num_unique) * KLE  -- the formula as written in the manuscript",
        },
        "diagnostic": diagnostic,
        "ranking_stability_all_steps": ranking,
        "attack_selection_and_outcome": outcome,
    }
    C.write_results(OUTPUT, payload)


if __name__ == "__main__":
    main()
