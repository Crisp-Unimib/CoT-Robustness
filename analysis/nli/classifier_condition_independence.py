"""Is the Outcome Classifier's error rate independent of the experimental arm?

Tests reviewer bebJ's point: "the classifier bias is condition-independent, [but]
this is not directly tested". We split classifier-vs-reference disagreements by
experimental arm (H_BE-targeted `_k10` vs random `_k10_random`, recovered from the
file path) and run a Fisher exact test on the 2x2 (arm x disagreement) table.

Two references are reported:
  * paper   - majority of the Outcome Classifier and the two independent judges
              A1 and A2, the definition behind the precision 0.76 quoted in the
              main text;
  * fourway - majority over all four raters, including the second reference
              classifier that is not used in the paper.
Both give the same verdict, so the conclusion does not depend on the choice.

Reads data/validation/agreement_matrix.csv, writes
classifier_condition_independence_results.json next to this file.
"""

import csv
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "data" / "validation" / "agreement_matrix.csv"
OUT = Path(__file__).with_name("classifier_condition_independence_results.json")

CLF = "gemini_verdict"          # Outcome Classifier
A1 = "glm_verdict"              # independent judge 1
A2 = "kimi_verdict"             # independent judge 2
OTHER = "haiku_verdict"         # second reference classifier, paper does not use it

REFERENCES = {"paper": [CLF, A1, A2], "fourway": [CLF, A1, A2, OTHER]}


def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact for [[a, b], [c, d]]: the total probability of
    tables no more likely than the observed one."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2

    def p_table(x):
        return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)

    p_obs = p_table(a)
    lo, hi = max(0, col1 - row2), min(col1, row1)
    return sum(p_table(x) for x in range(lo, hi + 1) if p_table(x) <= p_obs + 1e-12)


def cohen_kappa(x, y):
    labels = sorted(set(x) | set(y))
    n = len(x)
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    pe = sum((sum(1 for a in x if a == l) / n) * (sum(1 for b in y if b == l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def majority(row, members):
    votes = [row[m] for m in members]
    return "BROKEN" if votes.count("BROKEN") > len(votes) / 2 else "ROBUST"


def prf(pred, ref, positive="BROKEN"):
    tp = sum(1 for p, r in zip(pred, ref) if p == positive and r == positive)
    fp = sum(1 for p, r in zip(pred, ref) if p == positive and r != positive)
    fn = sum(1 for p, r in zip(pred, ref) if p != positive and r == positive)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return prec, rec


def main():
    rows = list(csv.DictReader(open(MATRIX, encoding="utf-8")))
    assert len(rows) == 100, "expected 100 validation rows, got %d" % len(rows)
    for r in rows:
        r["arm"] = "random" if "_random" in r["file"] else "hbe"

    res = {"n": len(rows),
           "kappa_A1_A2": round(cohen_kappa([r[A1] for r in rows], [r[A2] for r in rows]), 4),
           "references": {}}
    print("kappa(A1, A2) = %.4f" % res["kappa_A1_A2"])

    for name, members in REFERENCES.items():
        ref = [majority(r, members) for r in rows]
        pooled_p, pooled_r = prf([r[CLF] for r in rows], ref)
        block = {"members": members,
                 "pooled": {"precision": round(pooled_p, 4), "recall": round(pooled_r, 4)},
                 "per_arm": {}}
        table = {}
        for arm in ("hbe", "random"):
            idx = [i for i, r in enumerate(rows) if r["arm"] == arm]
            pred = [rows[i][CLF] for i in idx]
            sub_ref = [ref[i] for i in idx]
            dis = sum(1 for p, r in zip(pred, sub_ref) if p != r)
            fp = sum(1 for p, r in zip(pred, sub_ref) if p == "BROKEN" and r == "ROBUST")
            p_, r_ = prf(pred, sub_ref)
            table[arm] = (dis, len(idx) - dis)
            block["per_arm"][arm] = {
                "n": len(idx), "disagreements": dis, "over_flags": fp,
                "under_flags": dis - fp, "precision": round(p_, 4), "recall": round(r_, 4),
                "kappa_A1_A2": round(cohen_kappa([rows[i][A1] for i in idx],
                                                 [rows[i][A2] for i in idx]), 4)}
        (a, b), (c, d) = table["hbe"], table["random"]
        block["fisher_exact"] = {"table": [[a, b], [c, d]],
                                 "p_two_sided": round(fisher_exact_two_sided(a, b, c, d), 4)}
        res["references"][name] = block
        print("%-8s pooled P=%.2f R=%.2f | hbe %d/%d vs random %d/%d | Fisher p=%.4f"
              % (name, pooled_p, pooled_r, a, a + b, c, c + d, block["fisher_exact"]["p_two_sided"]))

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
