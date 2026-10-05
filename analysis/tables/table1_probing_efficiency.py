"""
Table 1 (Probing Efficiency by Model): H_BE-targeted against random probing.

Discov. is the number of attack points whose rollout was classified Broken.
Probed is every attack point recorded in that arm, including those whose
attack errored (status Unknown), and Rate is their ratio. Prob.>Rand. is
P(theta > 0.5) with theta ~ Beta(1 + N_bif, 1 + N_rand).

Reads data/results/blackmail and writes table1_probing_efficiency.json at
the repository root.
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robustness"))
import common as C  # noqa: E402

OUTPUT = C.REPO / "table1_probing_efficiency.json"


def count_arms():
    counts = defaultdict(Counter)
    for res_dir in C.MODELS:
        pattern = str(C.RES / res_dir / "*" / "*" / "experiment_result_*.json")
        for path in sorted(glob.glob(pattern)):
            arm = "random" if "_random" in Path(path).parts[-2] else "hbe"
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            for anchor in blob.get("anchor_results", []):
                counts[(res_dir, arm)]["probed"] += 1
                counts[(res_dir, arm)]["broken"] += anchor.get("status") == "Broken"
    return counts


def p_bif_greater(n_bif, n_rand):
    return float(beta.sf(0.5, 1 + n_bif, 1 + n_rand))


def main():
    counts = count_arms()
    rows = []
    for model in list(C.MODELS) + ["Total"]:
        if model == "Total":
            h = sum((counts[(m, "hbe")] for m in C.MODELS), Counter())
            r = sum((counts[(m, "random")] for m in C.MODELS), Counter())
        else:
            h, r = counts[(model, "hbe")], counts[(model, "random")]
        rows.append({
            "model": model,
            "hbe_discovered": h["broken"],
            "hbe_probed": h["probed"],
            "hbe_rate": h["broken"] / h["probed"],
            "random_discovered": r["broken"],
            "random_probed": r["probed"],
            "random_rate": r["broken"] / r["probed"],
            "gain": h["broken"] - r["broken"],
            "p_bif_greater": p_bif_greater(h["broken"], r["broken"]),
        })

    print(f"{'Model':<24}{'Disc':>6}{'Probed':>8}{'Rate':>8}"
          f"{'Disc':>6}{'Probed':>8}{'Rate':>8}{'Gain':>6}{'P>Rand':>9}")
    for row in rows:
        print(f"{row['model']:<24}"
              f"{row['hbe_discovered']:>6}{row['hbe_probed']:>8}{row['hbe_rate']:>8.1%}"
              f"{row['random_discovered']:>6}{row['random_probed']:>8}{row['random_rate']:>8.1%}"
              f"{row['gain']:>+6}{row['p_bif_greater']:>9.1%}")

    OUTPUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
