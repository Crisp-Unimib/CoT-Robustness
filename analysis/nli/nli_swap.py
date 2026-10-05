"""NLI-backbone swap: does the HBE step ranking survive changing the NLI model?

Recomputes step-level KLE with microsoft/deberta-v2-xlarge-mnli and compares the
step ranking to the paper's microsoft/deberta-large-mnli, on the same ~257-step
subset used by validate_subset_stability.py (steps with exactly 20 unique stored
samples, stratified across models, SEED=42). Reports Spearman rho (pooled + per
model) on KLE and on full HBE = log2(N)*(0.7*KLE + 0.3*D_syn). High rho => the
attack-point ranking does not depend on the NLI backbone.

Cost: the reference (large) KLE is already stored as `kle_score`. We spot-check it
by recomputing large KLE (float32) for a few steps and comparing to the stored
value. If it matches (max abs diff < 1e-3) we trust the stored value and run only
the swap model over all steps. If it does not match we recompute large for every
step, so the comparison always holds the method fixed and differs only in backbone.

Resumable: append-only checkpoint (nli_swap_progress.jsonl) + per-step W cache.
Interrupt any time; re-run skips finished steps. Aggregation reads the checkpoint
and needs no GPU.

Usage:
  python rebuttal/analysis/nli_swap.py                 # run (resumable) then aggregate
  python rebuttal/analysis/nli_swap.py --dry-run 10    # process 10 steps then stop
  python rebuttal/analysis/nli_swap.py --aggregate-only
"""
import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nli_kle import NLIScorer, kle_from_W, hbe, cache_load, cache_save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = REPO / "data" / "processed" / "blackmail_bifurcation"
PROGRESS = HERE / "nli_swap_progress.jsonl"
CACHE_DIR = HERE / "nli_swap_cache"
RESULTS = HERE / "nli_swap_results.json"

REF_MODEL = "microsoft/deberta-large-mnli"
SWAP_MODEL_DEFAULT = "microsoft/deberta-v2-xlarge-mnli"
SEED = 42
MIN_SAMPLES = 20
ALPHA, BETA = 0.7, 0.3
SPOT_TOL = 1e-3


def select_steps(n_steps):
    """Reproduce validate_subset_stability.py selection, carrying step metadata."""
    random.seed(SEED)
    per_model = defaultdict(list)
    for path in glob.glob(str(DATA_ROOT / "**" / "*.jsonl"), recursive=True):
        model_dir = Path(path).parts[-3]
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                s = d.get("samples", [])
                if len(s) != MIN_SAMPLES or len(set(s)) != MIN_SAMPLES:
                    continue
                metrics = d.get("metrics", {})
                per_model[model_dir].append({
                    "key": f'{model_dir}::{d["scenario_id"]}::{d["step_idx"]}',
                    "model_dir": model_dir,
                    "scenario_id": d["scenario_id"],
                    "step_idx": d["step_idx"],
                    "samples": s,
                    "stored_kle": metrics.get("kle_score"),
                    "dsyn": metrics.get("syntactic_distance"),
                    "num_unique": d.get("num_unique"),
                    "stored_hbe": metrics.get("bifurcation_entropy_kle"),
                })
    models = sorted(per_model)
    quota = max(1, n_steps // max(1, len(models)))
    selected = []
    for m in models:
        steps = per_model[m]
        random.shuffle(steps)
        selected.extend(steps[:quota])
    random.shuffle(selected)
    selected = selected[:n_steps]
    return selected, {m: len(v) for m, v in per_model.items()}


def load_progress():
    done = {}
    if PROGRESS.exists():
        with open(PROGRESS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["key"]] = r
    return done


def append_progress(rec):
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def kle_cached(step, scorer, model_tag):
    W = cache_load(CACHE_DIR, step["key"], model_tag)
    if W is None:
        W = scorer.build_W(step["samples"])
        cache_save(CACHE_DIR, step["key"], model_tag, W)
    return kle_from_W(W)


def run(args):
    swap_model = args.swap_model
    swap_tag = "v2xlarge" if "v2-xlarge" in swap_model else swap_model.split("/")[-1]

    selected, avail = select_steps(args.n_steps)
    print(f"Selected {len(selected)} steps (exactly {MIN_SAMPLES} unique samples).")
    print("Available per model:", avail)

    done = load_progress() if args.resume else {}
    todo = [s for s in selected if s["key"] not in done]
    print(f"{len(done)} already done, {len(todo)} to do.")

    spot = None
    use_stored_large = True
    if todo:
        # spot-check: our large KLE (float32) vs stored kle_score
        if args.validate_large > 0:
            print(f"Spot-check: large KLE (float32) vs stored for {args.validate_large} steps ...")
            sc = NLIScorer(REF_MODEL, fp16=False, batch_size=args.batch_size)
            print(f"  large entailment index = {sc.entail_idx}")
            diffs = []
            for s in selected[:args.validate_large]:
                if s["stored_kle"] is None:
                    continue
                diffs.append(abs(kle_from_W(sc.build_W(s["samples"])) - s["stored_kle"]))
            del sc
            if diffs:
                spot = {"n": len(diffs), "max_abs_diff": float(max(diffs)),
                        "mean_abs_diff": float(np.mean(diffs)), "tol": SPOT_TOL}
                print(f"  max|our_large - stored| = {spot['max_abs_diff']:.6f} (tol {SPOT_TOL})")
                use_stored_large = spot["max_abs_diff"] <= SPOT_TOL
        if not use_stored_large:
            print("  spot-check FAILED -> recomputing large for every step (method held fixed).")
        else:
            print("  spot-check OK -> using stored kle_score as the large reference.")

        # load models needed for the main loop
        print(f"Loading swap model {swap_model} (fp16={args.fp16}) ...")
        swap_scorer = NLIScorer(swap_model, fp16=args.fp16, batch_size=args.batch_size)
        print(f"  swap entailment index = {swap_scorer.entail_idx}")
        ref_scorer = None
        if not use_stored_large:
            ref_scorer = NLIScorer(REF_MODEL, fp16=args.fp16, batch_size=args.batch_size)

        for i, s in enumerate(todo):
            if args.dry_run and i >= args.dry_run:
                print(f"Dry-run limit {args.dry_run} reached; stopping.")
                break
            if use_stored_large and s["stored_kle"] is not None:
                kle_large = float(s["stored_kle"])
                large_src = "stored"
            else:
                if ref_scorer is None:
                    ref_scorer = NLIScorer(REF_MODEL, fp16=args.fp16, batch_size=args.batch_size)
                kle_large = kle_cached(s, ref_scorer, "large")
                large_src = "recompute"
            kle_swap = kle_cached(s, swap_scorer, swap_tag)
            n = len(s["samples"])
            append_progress({
                "key": s["key"], "model_dir": s["model_dir"], "scenario_id": s["scenario_id"],
                "step_idx": s["step_idx"], "n_used": n, "dsyn": s["dsyn"],
                "kle_large": kle_large, "kle_swap": kle_swap, "large_src": large_src,
                "hbe_large": hbe(kle_large, s["dsyn"], n, ALPHA, BETA),
                "hbe_swap": hbe(kle_swap, s["dsyn"], n, ALPHA, BETA),
                "stored_kle": s["stored_kle"], "stored_hbe": s["stored_hbe"],
                "swap_model": swap_model, "fp16": bool(args.fp16),
            })
            if i == 0 or (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(todo)}  (KLE large={kle_large:.3f} swap={kle_swap:.3f})")

    aggregate(spot=spot, swap_model=swap_model)


def _spearman(a, b):
    if len(a) < 3:
        return None
    rho, p = spearmanr(a, b)
    return {"rho": round(float(rho), 4), "p": float(p), "n": len(a)}


def aggregate(spot=None, swap_model=None):
    recs = []
    if PROGRESS.exists():
        with open(PROGRESS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    if not recs:
        print("No records to aggregate.")
        return

    kl = [r["kle_large"] for r in recs]
    ks = [r["kle_swap"] for r in recs]
    hl = [r["hbe_large"] for r in recs]
    hs = [r["hbe_swap"] for r in recs]

    out = {
        "n_steps": len(recs),
        "ref_model": REF_MODEL,
        "swap_model": swap_model or recs[0].get("swap_model"),
        "fp16": recs[0].get("fp16"),
        "large_source": recs[0].get("large_src"),
        "spot_check_large_vs_stored": spot,
        "kle_spearman_pooled": _spearman(kl, ks),
        "hbe_spearman_pooled": _spearman(hl, hs),
        "kle_pearson_pooled": {"r": round(float(pearsonr(kl, ks)[0]), 4)} if len(kl) >= 3 else None,
        "per_model": {},
    }

    # method sanity: our large vs stored across the whole set
    pairs = [(r["kle_large"], r["stored_kle"]) for r in recs if r.get("stored_kle") is not None]
    if len(pairs) >= 3:
        a, b = zip(*pairs)
        out["our_large_vs_stored_spearman"] = _spearman(list(a), list(b))
        out["our_large_vs_stored_mean_abs_diff"] = round(float(np.mean([abs(x - y) for x, y in pairs])), 5)

    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model_dir"]].append(r)
    for m, rs in sorted(by_model.items()):
        out["per_model"][m] = {
            "n": len(rs),
            "kle_spearman": _spearman([r["kle_large"] for r in rs], [r["kle_swap"] for r in rs]),
            "hbe_spearman": _spearman([r["hbe_large"] for r in rs], [r["hbe_swap"] for r in rs]),
        }

    # pooled top-10% HBE overlap: proxy for "same steps flagged for attack"
    k = max(1, len(recs) // 10)
    top_l = set(sorted(range(len(recs)), key=lambda i: hl[i], reverse=True)[:k])
    top_s = set(sorted(range(len(recs)), key=lambda i: hs[i], reverse=True)[:k])
    union = len(top_l | top_s)
    out["top10pct_hbe_overlap"] = {
        "k": k, "intersection": len(top_l & top_s),
        "overlap_frac": round(len(top_l & top_s) / k, 4),
        "jaccard": round(len(top_l & top_s) / union, 4) if union else None,
    }

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== NLI SWAP RESULTS ===")
    print(f"n_steps = {out['n_steps']}   swap = {out['swap_model']}   fp16={out['fp16']}   large_source={out['large_source']}")
    if spot:
        print(f"spot-check our_large vs stored: max|d|={spot['max_abs_diff']:.6f} (tol {spot['tol']})")
    if out.get("our_large_vs_stored_spearman"):
        print(f"our_large vs stored (method sanity): rho={out['our_large_vs_stored_spearman']['rho']}, "
              f"mean|d|={out.get('our_large_vs_stored_mean_abs_diff')}")
    print(f"KLE Spearman (pooled): {out['kle_spearman_pooled']}")
    print(f"HBE Spearman (pooled): {out['hbe_spearman_pooled']}")
    print(f"top-10% HBE overlap:   {out['top10pct_hbe_overlap']}")
    for m, d in out["per_model"].items():
        kr = d["kle_spearman"]["rho"] if d["kle_spearman"] else "NA"
        hr = d["hbe_spearman"]["rho"] if d["hbe_spearman"] else "NA"
        print(f"  {m:>34}: n={d['n']:3d}  KLE rho={kr}  HBE rho={hr}")
    print(f"\nsaved -> {RESULTS}")


def main():
    ap = argparse.ArgumentParser(description="NLI-backbone swap for HBE step ranking.")
    ap.add_argument("--swap-model", default=SWAP_MODEL_DEFAULT)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", help="fp16 main passes (default float32 for a clean A/B)")
    ap.add_argument("--validate-large", type=int, default=20, help="steps for the large-vs-stored spot-check")
    ap.add_argument("--dry-run", type=int, default=0, help="process at most N steps then stop")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.set_defaults(resume=True)
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate()
    else:
        run(args)


if __name__ == "__main__":
    main()
