"""
Recreate Figure 4, "Conditional Break Rate by H_BE Quartile".

The script that produced the published figure is not in the repository. This one was
reconstructed by searching population definitions against the seven quantities readable
off the published PNG (three quartile edges, four break rates) and then checked against
the four confidence intervals, which were not used to fit. It reproduces every one of
them exactly, so the recipe below is almost certainly the original.

    RECIPE
      models        GLM-4.5-Air, Qwen3-VL-32B-Thinking, gpt-oss-120b   (three, not five)
      arm           H_BE-targeted only; the random control arm is excluded
      anchors       all 10 per scenario, status Broken or Robust
      metric        H_BE = log2(num_unique) * KLE
                    This is the formula the manuscript states. It is NOT the shipped
                    `bifurcation_entropy_kle`, which is log2(20)*(0.7*KLE + 0.3*D_syn)
                    and spans 0 to 14, not 0 to 28.6.
      filter        anchors with H_BE = 0 dropped (steps with fewer than 2 unique
                    continuations)
      bins          equal-frequency quartiles, 663 anchors each
      intervals     Wilson 95%

    REPRODUCES
      n = 2652, overall 12.22%, range 0 to 28.6, edges 18.0 / 22.9 / 25.5
      Q1 3.77% [2.57, 5.51]    published 3.8% [2.6, 5.5]
      Q2 6.94% [5.24, 9.13]    published 6.9% [5.3, 9.1]
      Q3 14.93% [12.42, 17.85] published 14.9% [12.4, 17.8]
      Q4 23.23% [20.17, 26.59] published 23.2% [20.1, 26.5]
      ratio 6.16                published 6.2

Two things follow.

First, the figure covers three models and one arm, while the abstract and introduction
attach the 6.2x to "across five reasoning models". Under the same recipe extended to all
five, the contrast falls (see the `alternative_populations` block in the output).

Second, and more seriously, the gradient is largely between-model rather than
within-model. Q1 is 69% GLM-4.5-Air and 24% gpt-oss-120b, two models that almost never
break; Q4 is 57% Qwen3-VL-32B-Thinking, which breaks 31% of the time and also has the
highest H_BE of any model. Replacing H_BE with model identity alone, discarding the
metric entirely, produces a gradient at least as steep. Within each model separately the
contrast is 1.85x for Qwen3-VL and 5.50x for GLM-4.5-Air, and undefined for gpt-oss-120b,
which never breaks at any entropy.

Outputs: recreate_figure4_results.json, and the figure as PNG and PDF.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

import common as C

OUTPUT = C.REPO / "recreate_figure4_results.json"
FIG_DIR = C.REPO / "outputs" / "plots"
FIG_STEM = "fig_hbe_break_rate_recreated"

FIGURE_MODELS = ["GLM-4.5-Air", "Qwen3-VL-32B-Thinking", "gpt-oss-120b"]

PUBLISHED = {
    "edges": [18.0, 22.9, 25.5],
    "rates": [0.038, 0.069, 0.149, 0.232],
    "cis": [(0.026, 0.055), (0.053, 0.091), (0.124, 0.178), (0.201, 0.265)],
    "overall": 0.122,
    "ratio": 6.2,
    "range": (0.0, 28.6),
}


# ---------------------------------------------------------------------------

def load_anchor_rows():
    """Attacked anchors joined to their step record, scored with log2(N_true)*KLE."""
    all_models = {**C.MODELS, **C.EXTRA_MODELS}
    rows = []
    for res_dir, proc_dir in all_models.items():
        steps = C.load_steps(proc_dir)
        for path in sorted(glob.glob(str(C.RES / res_dir / "*" / "*" / "experiment_result_*.json"))):
            arm = "random" if "_random" in Path(path).parts[-2] else "hbe"
            try:
                with open(path, encoding="utf-8") as fh:
                    blob = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            scenario = blob.get("metadata", {}).get("problem_id")
            per_step = steps.get(scenario, {})
            for anchor in blob.get("anchor_results", []):
                if anchor.get("status") not in ("Broken", "Robust"):
                    continue
                rec = per_step.get(anchor.get("anchor_idx"))
                if rec is None:
                    continue
                metrics = rec.get("metrics") or {}
                kle = metrics.get("kle_score")
                n_true = rec.get("num_unique")
                if kle is None:
                    continue
                hbe = math.log2(n_true) * kle if n_true and n_true >= 2 else 0.0
                rows.append({
                    "model": res_dir, "scenario": scenario, "arm": arm,
                    "broken": anchor.get("status") == "Broken", "hbe": hbe,
                })
    return rows


def quartile_profile(rows):
    """Equal-frequency quartiles with Wilson intervals."""
    v = np.array([r["hbe"] for r in rows])
    b = np.array([r["broken"] for r in rows])
    edges = [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75])]
    bounds = [-1e9] + edges + [1e18]
    out = []
    for i in range(4):
        m = (v > bounds[i]) & (v <= bounds[i + 1])
        k, n = int(b[m].sum()), int(m.sum())
        lo, hi = C.proportion_ci(k, n)
        out.append({
            "quartile": f"Q{i+1}", "n": n, "broken": k,
            "rate": C._f(k / n) if n else None,
            "ci": [C._f(lo), C._f(hi)],
            "hbe_lo": C._f(v[m].min()) if n else None,
            "hbe_hi": C._f(v[m].max()) if n else None,
        })
    r1, r4 = out[0]["rate"], out[3]["rate"]
    return {
        "n": len(v), "edges": [C._f(e) for e in edges],
        "hbe_min": C._f(v.min()), "hbe_max": C._f(v.max()),
        "overall_rate": C._f(b.mean()),
        "quartiles": out,
        "ratio_q4_q1": C._f(r4 / r1) if r1 else None,
    }


def check_against_published(prof):
    """Compare every published quantity, including the CIs, which were not fitted."""
    checks = []

    def add(name, got, want, tol):
        ok = got is not None and abs(got - want) <= tol
        checks.append({"quantity": name, "reproduced": C._f(got),
                       "published": want, "match": bool(ok)})

    for i, e in enumerate(PUBLISHED["edges"]):
        add(f"edge_{i+1}", prof["edges"][i], e, 0.05)
    for i, (r, (lo, hi)) in enumerate(zip(PUBLISHED["rates"], PUBLISHED["cis"])):
        q = prof["quartiles"][i]
        add(f"Q{i+1}_rate", q["rate"], r, 0.0006)
        add(f"Q{i+1}_ci_lo", q["ci"][0], lo, 0.0016)
        add(f"Q{i+1}_ci_hi", q["ci"][1], hi, 0.0016)
    add("overall_rate", prof["overall_rate"], PUBLISHED["overall"], 0.0006)
    add("hbe_max", prof["hbe_max"], PUBLISHED["range"][1], 0.05)
    add("ratio", prof["ratio_q4_q1"], PUBLISHED["ratio"], 0.05)
    return {"all_match": all(c["match"] for c in checks), "checks": checks}


def decompose(rows):
    """Is the gradient within-model or between-model?"""
    v = np.array([r["hbe"] for r in rows])
    b = np.array([r["broken"] for r in rows])
    m = np.array([r["model"] for r in rows])
    edges = np.quantile(v, [0.25, 0.5, 0.75])
    qi = np.digitize(v, edges, right=False)
    models = sorted(set(m.tolist()))

    composition = []
    for i in range(4):
        sl = qi == i
        composition.append({
            "quartile": f"Q{i+1}",
            "break_rate": C._f(b[sl].mean()),
            "model_share": {s: C._f(float(np.mean(m[sl] == s))) for s in models},
        })

    within = {}
    for s in models:
        sl = m == s
        vv, bb = v[sl], b[sl]
        lo_e, hi_e = np.quantile(vv, [0.25, 0.75])
        lo, hi = vv <= lo_e, vv >= hi_e
        r1, r4 = float(bb[lo].mean()), float(bb[hi].mean())
        within[s] = {
            "n": int(sl.sum()), "broken": int(bb.sum()),
            "overall_rate": C._f(bb.mean()), "mean_hbe": C._f(vv.mean()),
            "q1_rate": C._f(r1), "q4_rate": C._f(r4),
            "ratio": C._f(r4 / r1) if r1 > 0 else None,
            "ratio_note": None if r1 > 0 else "undefined; this model never breaks",
        }

    # Surrogate: rank anchors by model identity alone, discarding H_BE entirely.
    rate_by_model = {s: float(b[m == s].mean()) for s in models}
    order = np.argsort(np.array([rate_by_model[x] for x in m]), kind="stable")
    groups = np.array_split(order, 4)
    surrogate = [C._f(float(b[g].mean())) for g in groups]

    return {
        "quartile_composition": composition,
        "within_model": within,
        "model_identity_surrogate": {
            "description": (
                "The same four-bin plot, ranking anchors by their model's overall break "
                "rate instead of by H_BE. Uses no entropy information at all."
            ),
            "quartile_rates": surrogate,
        },
    }


def alternative_populations(rows):
    """The same recipe over the populations the manuscript's prose actually describes."""
    out = {}
    variants = {
        "figure_as_published (3 models, H_BE arm)":
            lambda r: r["model"] in FIGURE_MODELS and r["arm"] == "hbe" and r["hbe"] > 0,
        "five_models, H_BE arm":
            lambda r: r["model"] in C.MODELS and r["arm"] == "hbe" and r["hbe"] > 0,
        "five_models, both arms":
            lambda r: r["model"] in C.MODELS and r["hbe"] > 0,
        "five_models, both arms, incl. H_BE=0":
            lambda r: r["model"] in C.MODELS,
        "all six models, both arms":
            lambda r: r["hbe"] > 0,
    }
    for name, pred in variants.items():
        sel = [r for r in rows if pred(r)]
        if len(sel) < 40:
            continue
        p = quartile_profile(sel)
        out[name] = {
            "n": p["n"], "overall_rate": p["overall_rate"],
            "edges": p["edges"],
            "q1_rate": p["quartiles"][0]["rate"], "q4_rate": p["quartiles"][3]["rate"],
            "ratio": p["ratio_q4_q1"],
        }
    return out


def draw(prof, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qs = prof["quartiles"]
    rates = [q["rate"] * 100 for q in qs]
    errs = [[(q["rate"] - q["ci"][0]) * 100 for q in qs],
            [(q["ci"][1] - q["rate"]) * 100 for q in qs]]
    colours = ["#1b8a5a", "#5ecfa0", "#e05a8a", "#8f2748"]
    edges = [0.0] + prof["edges"] + [prof["hbe_max"]]
    labels = [f"Q{i+1}\n{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(4)]

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200)
    ax.bar(range(4), rates, color=colours, width=0.68, zorder=3)
    ax.errorbar(range(4), rates, yerr=errs, fmt="none", ecolor="#9e9e9e",
                elinewidth=1.6, capsize=5, capthick=1.6, zorder=4)

    overall = prof["overall_rate"] * 100
    ax.axhline(overall, ls="--", lw=1.6, color="#b0b0b0", zorder=2)
    # Sit the label over the short bars; centring it would collide with Q3.
    ax.text(1.05, overall + 0.7, f"overall {overall:.1f}%", ha="center",
            color="#8a8a8a", fontsize=11)

    top = max(r + e for r, e in zip(rates, errs[1]))
    y = top + 3.0
    ax.plot([0, 0, 3, 3], [rates[0] + 1.8, y, y, rates[3] + 3.4],
            lw=1.4, color="#4a4a4a", zorder=5)
    ax.text(1.5, y + 0.6, f"{prof['ratio_q4_q1']:.1f}× Q1", ha="center",
            fontsize=13, fontweight="bold", color="#8f2748")

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("HBE", fontsize=12)
    ax.set_ylabel("P(broken | quartile)", fontsize=12)
    ax.set_ylim(0, y + 3.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", color="#e0e0e0", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    return [str(path_stem.with_suffix(f".{ext}")) for ext in ("png", "pdf")]


def main():
    print("Loading attacked anchors...")
    rows = load_anchor_rows()
    print(f"  {len(rows)} anchors across {len({r['model'] for r in rows})} models")

    figure_rows = [r for r in rows
                   if r["model"] in FIGURE_MODELS and r["arm"] == "hbe" and r["hbe"] > 0]
    print(f"  {len(figure_rows)} in the reconstructed Figure 4 population")

    prof = quartile_profile(figure_rows)
    verdict = check_against_published(prof)

    print(f"\nReconstruction: n={prof['n']} overall={100*prof['overall_rate']:.2f}% "
          f"edges={'/'.join(f'{e:.1f}' for e in prof['edges'])} "
          f"ratio={prof['ratio_q4_q1']:.2f}")
    print(f"{'Q':>3s} {'n':>5s} {'brk':>4s} {'rate':>8s} {'95% CI':>16s}")
    for q in prof["quartiles"]:
        print(f"{q['quartile']:>3s} {q['n']:5d} {q['broken']:4d} {100*q['rate']:7.2f}% "
              f"[{100*q['ci'][0]:5.2f},{100*q['ci'][1]:5.2f}]")
    bad = [c["quantity"] for c in verdict["checks"] if not c["match"]]
    print(f"\nAgainst the published figure: "
          f"{'all 16 quantities match' if verdict['all_match'] else 'MISMATCH on ' + ', '.join(bad)}")

    dec = decompose(figure_rows)
    print("\nWhat drives the gradient:")
    for c in dec["quartile_composition"]:
        share = "  ".join(f"{k.split('-')[0][:8]} {100*v:4.1f}%"
                          for k, v in sorted(c["model_share"].items()))
        print(f"  {c['quartile']} break {100*c['break_rate']:5.2f}%   {share}")
    print("  within-model contrast:")
    for s, d in dec["within_model"].items():
        r = f"{d['ratio']:.2f}" if d["ratio"] else d["ratio_note"]
        print(f"    {s:26s} n={d['n']:4d} rate={100*d['overall_rate']:5.2f}% ratio={r}")
    print("  model identity alone, no H_BE: "
          + " ".join(f"{100*x:.2f}%" for x in dec["model_identity_surrogate"]["quartile_rates"]))

    alts = alternative_populations(rows)
    print("\nSame recipe over other populations:")
    for name, a in alts.items():
        print(f"  {name:44s} n={a['n']:5d} Q1={100*a['q1_rate']:5.2f}% "
              f"Q4={100*a['q4_rate']:5.2f}% ratio={a['ratio']}")

    figs = draw(prof, FIG_DIR / FIG_STEM)
    print(f"\nFigure written: {', '.join(figs)}")

    C.write_results(OUTPUT, {
        "purpose": "Reconstruction of Figure 4, whose generating script is absent from the repo",
        "recipe": {
            "models": FIGURE_MODELS,
            "arm": "hbe only (random control arm excluded)",
            "anchors_per_scenario": 10,
            "metric": "log2(num_unique) * KLE",
            "metric_note": (
                "The manuscript's stated formula. NOT the shipped bifurcation_entropy_kle, "
                "which is log2(20)*(0.7*KLE + 0.3*D_syn) and spans 0 to 14."
            ),
            "excluded": "anchors with H_BE = 0, and status Unknown",
            "bins": "equal-frequency quartiles",
            "intervals": "Wilson 95%",
        },
        "reconstruction": prof,
        "validation_against_published": verdict,
        "what_drives_the_gradient": dec,
        "alternative_populations": alts,
        "figure_files": figs,
    })


if __name__ == "__main__":
    main()
