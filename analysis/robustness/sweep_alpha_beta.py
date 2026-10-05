"""
Sweep: the mixing weight between the semantic and syntactic terms of H_BE.

    H_BE = log2(N) * (alpha*KLE + (1-alpha)*D_syn)

The manuscript writes H_BE = log2(N)*KLE, i.e. alpha=1. The shipped code uses
alpha=0.7 with D_syn a mean normalised Levenshtein distance. This sweep walks alpha
from 0 (purely surface-form) to 1 (purely semantic) and reports how far the ranking,
the selected attack points, and the quartile break-rate conclusion move.

It answers whether the reported effect
depends on an undocumented weighting choice. Derived from stored scalars only, so no
model, NLI backbone, or GPU is used.

Outputs: sweep_alpha_beta_results.json
"""

from __future__ import annotations

import numpy as np

import common as C

OUTPUT = C.REPO / "sweep_alpha_beta_results.json"
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
SHIPPED_ALPHA = 0.7


def main():
    print("Loading step records...")
    steps = C.load_all_steps()
    steps = [r for r in steps if r["kle"] is not None and r["dsyn"] is not None
             and r["n_pinned"] and r["n_pinned"] >= 2]
    print(f"  {len(steps)} step records")

    anchors = C.load_anchors()
    anchors = [r for r in anchors if r["kle"] is not None and r["dsyn"] is not None
               and r["n_pinned"] and r["n_pinned"] >= 2]
    broken = [r["broken"] for r in anchors]
    print(f"  {len(anchors)} anchors, {sum(broken)} broken")

    # How correlated are the two components in the first place? If KLE and D_syn
    # already rank steps alike, alpha cannot matter much and that is worth stating.
    from scipy.stats import spearmanr
    comp_rho, _ = spearmanr([r["kle"] for r in steps], [r["dsyn"] for r in steps])
    print(f"\n  Spearman(KLE, D_syn) across steps = {comp_rho:.4f}")

    for n_conv in ("pinned", "true"):
        key = "n_pinned" if n_conv == "pinned" else "n_true"
        usable = [r for r in steps if r.get(key) and r[key] >= 2]
        print(f"  N={n_conv}: {len(usable)} usable step records")

    def score(rows, alpha, n_key):
        return [C.hbe(r[n_key], r["kle"], r["dsyn"], alpha=alpha, beta=1.0 - alpha)
                for r in rows]

    results = {}
    for n_conv, n_key in (("pinned20", "n_pinned"), ("trueN", "n_true")):
        srows = [r for r in steps if r.get(n_key) and r[n_key] >= 2]
        arows = [r for r in anchors if r.get(n_key) and r[n_key] >= 2]
        abroken = [r["broken"] for r in arows]

        ref_steps = score(srows, SHIPPED_ALPHA, n_key)
        ref_anchors = score(arows, SHIPPED_ALPHA, n_key)

        print(f"\nN convention = {n_conv} (reference alpha={SHIPPED_ALPHA}, "
              f"{len(srows)} steps / {len(arows)} anchors)")
        print(f"  {'alpha':>5s} {'rho':>8s} {'top10%':>8s} {'top3':>7s} "
              f"{'Q1':>7s} {'Q4':>7s} {'ratio':>7s}")

        per_alpha = {}
        for alpha in ALPHAS:
            s_var = score(srows, alpha, n_key)
            a_var = score(arows, alpha, n_key)
            sp = C.spearman_by_model(ref_steps, s_var, [r["model"] for r in srows])
            dec = C.top_fraction_overlap(ref_steps, s_var, 0.10)
            top3 = C.top_k_anchor_overlap(arows, ref_anchors, a_var, k=3)
            qb = C.quartile_break_ratio(a_var, abroken)
            per_alpha[f"{alpha:.1f}"] = {
                "alpha": alpha,
                "beta": round(1.0 - alpha, 1),
                "spearman": sp,
                "top_decile_overlap": dec,
                "top_k_anchor_overlap": top3,
                "quartile_break": qb,
            }
            print(f"  {alpha:5.1f} {sp['overall']['rho']:8.4f} {dec['overlap_frac']:8.3f} "
                  f"{top3['mean_overlap']:7.3f} {qb['q1_break_rate']:7.4f} "
                  f"{qb['q4_break_rate']:7.4f} {str(qb['ratio']):>7s}")

        rhos = [per_alpha[f'{a:.1f}']["spearman"]["overall"]["rho"] for a in ALPHAS]
        ratios = [per_alpha[f'{a:.1f}']["quartile_break"]["ratio"] for a in ALPHAS]
        ratios = [r for r in ratios if r is not None]
        results[n_conv] = {
            "reference_alpha": SHIPPED_ALPHA,
            "n_steps": len(srows),
            "n_anchors": len(arows),
            "n_broken": int(sum(abroken)),
            "by_alpha": per_alpha,
            "summary": {
                "min_spearman_vs_shipped": C._f(min(rhos)),
                "min_spearman_at_alpha": ALPHAS[int(np.argmin(rhos))],
                "quartile_ratio_min": C._f(min(ratios)) if ratios else None,
                "quartile_ratio_max": C._f(max(ratios)) if ratios else None,
            },
        }

    payload = {
        "sweep": "alpha_beta",
        "question": (
            "Does the H_BE conclusion depend on the 0.7/0.3 weighting between the KLE "
            "and syntactic terms, a choice the manuscript does not state?"
        ),
        "requires_gpu": False,
        "requires_generation": False,
        "formula": "H_BE = log2(N) * (alpha*KLE + (1-alpha)*D_syn)",
        "alphas": ALPHAS,
        "coverage": {
            "models": sorted(C.MODELS.keys()),
            "arms": "both the H_BE-targeted and the random control arm, pooled",
        },
        "component_rank_correlation_kle_vs_dsyn": C._f(comp_rho),
        "by_n_convention": results,
    }
    C.write_results(OUTPUT, payload)


if __name__ == "__main__":
    main()
