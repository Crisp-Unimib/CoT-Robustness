"""Rebuttal analysis B: sensitivity of the HBE-vs-random comparison to the number
of attack points K (reviewer bebJ: "number of selected attack points").

For every scenario and both arms, anchors are ranked (HBE arm: by z_score = H_BE,
descending; random arm: stored order, which is the random draw order) and we count
discovered Broken steps among the top-K, K = 1..10, pooled and per model.

Sanity checks against the paper: at K=10 the pooled discovered counts must equal
364 (HBE) vs 238 (random); at K=3 the pooled yields should be ~7.2% vs ~5.9%
(Appendix C).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "blackmail"
# The five models reported in the paper (Table 2); dir names as on disk.
MODELS = [
    "Qwen3-VL-32B-Thinking",
    "Qwen3-Next-80B",
    "GLM-4.5-Air",
    "NVIDIA-Nemotron-3-Nano",
    "gpt-oss-120b",
]
SIG = "google-gemini-3-flash-preview_n100_T1.0_p0.95_e1_k10"
KMAX = 10
OUT = Path(__file__).with_name("yield_at_k_results.json")


def load_arm(model, random_arm):
    d = RESULTS / model / "bifurcation_bifurcation_entropy_kle" / (SIG + ("_random" if random_arm else ""))
    scenarios = []
    for f in sorted(d.glob("experiment_result_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        anchors = data.get("anchor_results") or []
        if not random_arm:
            anchors = sorted(anchors, key=lambda a: a.get("z_score", 0.0), reverse=True)
        scenarios.append([a.get("status") == "Broken" for a in anchors])
    return scenarios


def cum_at_k(scenarios, k):
    """(#broken among top-k, #anchors probed among top-k) across scenarios."""
    broken = sum(sum(s[:k]) for s in scenarios)
    probed = sum(min(k, len(s)) for s in scenarios)
    return broken, probed


def main():
    per_model = {}
    for m in MODELS:
        per_model[m] = {"hbe": load_arm(m, False), "random": load_arm(m, True)}
        print(f"{m}: {len(per_model[m]['hbe'])} HBE files, {len(per_model[m]['random'])} random files")

    res = {"per_k": [], "per_model_k10": {}, "sanity": {}}
    print(f"\n{'K':>2}  {'HBE broken':>10} {'/probed':>8} {'yield':>7}   "
          f"{'RND broken':>10} {'/probed':>8} {'yield':>7}   {'gap':>5}")
    for k in range(1, KMAX + 1):
        hb = hp = rb = rp = 0
        for m in MODELS:
            b, p = cum_at_k(per_model[m]["hbe"], k); hb += b; hp += p
            b, p = cum_at_k(per_model[m]["random"], k); rb += b; rp += p
        row = {"k": k,
               "hbe": {"broken": hb, "probed": hp, "yield": round(hb / hp, 4)},
               "random": {"broken": rb, "probed": rp, "yield": round(rb / rp, 4)},
               "count_gap": hb - rb,
               "yield_ratio": round((hb / hp) / (rb / rp), 3) if rb else None}
        res["per_k"].append(row)
        print(f"{k:>2}  {hb:>10} {hp:>8} {hb/hp:>7.2%}   {rb:>10} {rp:>8} {rb/rp:>7.2%}   {hb-rb:>+5}")

    for m in MODELS:
        hb, hp = cum_at_k(per_model[m]["hbe"], KMAX)
        rb, rp = cum_at_k(per_model[m]["random"], KMAX)
        res["per_model_k10"][m] = {"hbe_broken": hb, "hbe_probed": hp,
                                   "random_broken": rb, "random_probed": rp}
        print(f"K=10 {m:>28}: HBE {hb}/{hp}   random {rb}/{rp}")

    k10 = res["per_k"][-1]
    res["sanity"]["k10_counts_match_paper_364_238"] = (
        k10["hbe"]["broken"] == 364 and k10["random"]["broken"] == 238)
    print(f"\nSANITY K=10 pooled counts = {k10['hbe']['broken']} vs {k10['random']['broken']} "
          f"(paper: 364 vs 238) -> {'OK' if res['sanity']['k10_counts_match_paper_364_238'] else 'MISMATCH'}")
    k3 = res["per_k"][2]
    print(f"SANITY K=3 pooled yields = {k3['hbe']['yield']:.1%} vs {k3['random']['yield']:.1%} "
          f"(paper App C: ~7.2% vs ~5.9%)")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
