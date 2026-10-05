"""
python print_metrics.py --run_dir user/uncertainty/wandb/run-20260219_215552-8amyqt0h --best

"""


import os
import argparse
import json
import ast
import numpy as np
from math import nan

def main():
    parser = argparse.ArgumentParser(description='Print metrics from wandb-summary.json or a given dict string')
    parser.add_argument('--run_dir', type=str, required=False, help='Path to wandb run directory')
    parser.add_argument('--dict_str', type=str, required=False, help='Raw string representation of the results dict')
    parser.add_argument('--filter', type=str, default=None, help='Substring to include measure names (case-insensitive)')
    parser.add_argument('--exclude', type=str, default=None, help='Substring to exclude measure names (case-insensitive, comma-separated)')
    parser.add_argument('--best', action='store_true', help='Only print the best KLE-HEAT result, ignoring inverted/normalized and UNANSWERABLE variants.')
    args = parser.parse_args()

    if args.dict_str:
        print("Parsing dict string directly...")
        # Evaluate safely considering np.float64 and nan
        data = eval(args.dict_str, {"np": np, "nan": nan})
    elif args.run_dir:
        json_path = os.path.join(args.run_dir, 'files', 'wandb-summary.json')
        if not os.path.exists(json_path):
            # Fallback: check root of run_dir if not in files/
            json_path = os.path.join(args.run_dir, 'wandb-summary.json')
        
        if not os.path.exists(json_path):
            print(f"Error: Could not find wandb-summary.json in {args.run_dir}")
            return

        print(f"Loading metrics from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        print("Error: Must provide either --run_dir or --dict_str")
        return

    if 'uncertainty' not in data:
        # Fallback: handle flat reasoning_auroc_* keys from generate_answers_reasoning_kle.py
        auroc_keys = {k: v for k, v in data.items() if k.startswith('reasoning_auroc_') and isinstance(v, (int, float))}
        if not auroc_keys:
            print("Error: 'uncertainty' key not found and no reasoning_auroc_* keys found.")
            return

        print(f"\nFound {len(auroc_keys)} inline reasoning AUROC metrics (flat format).\n")
        print("=" * 80)
        print(f"{'Measure':<60} | {'AUROC':>10}")
        print("-" * 80)

        best_name, best_val = None, -1
        sorted_keys = sorted(auroc_keys.items(), key=lambda x: x[1], reverse=True)

        for key, val in sorted_keys:
            measure = key.replace('reasoning_auroc_', '')
            # Apply --best filter: skip heatn and unanswerable
            if args.best:
                if 'heatn' in measure.lower() or 'UNANSWERABLE' in measure:
                    continue
                if val > best_val:
                    best_val = val
                    best_name = measure
            else:
                if args.filter and args.filter.lower() not in measure.lower():
                    continue
                if args.exclude:
                    excludes = [e.strip().lower() for e in args.exclude.split(',')]
                    if any(e in measure.lower() for e in excludes):
                        continue
                print(f"{measure:<60} | {val:>10.4f}")

        if args.best and best_name:
            print("🏆 Your Best Result 🏆")
            print("-" * 80)
            print(f"{best_name:<60} | {best_val:>10.4f}")

        print("=" * 80)

        # Print accuracy if available
        if 'validation_accuracy' in data:
            print(f"\nValidation Accuracy: {data['validation_accuracy']:.4f}")
        return

    print("\n" + "="*110)
    print(f"{'Measure':<45} | {'AUROC mean [95% CI]':<30} | {'AUARC mean [95% CI]':<30}")
    print("-" * 110)

    best_measure = None
    best_auroc = -1
    best_str_auroc = ""
    best_str_auarc = ""

    keys = sorted(data['uncertainty'].keys())
    for measure_name in keys:
        if args.filter and args.filter.lower() not in measure_name.lower():
            continue
        
        if args.exclude:
            excludes = [e.strip().lower() for e in args.exclude.split(',')]
            if any(e in measure_name.lower() for e in excludes):
                continue

        # Ignore heatn and UNANSWERABLE if searching for the best standard metric
        if args.best:
            if 'UNANSWERABLE' in measure_name or 'heatn' in measure_name:
                continue
            
        metrics = data['uncertainty'][measure_name]
        if 'AUROC' not in metrics:
            continue
        
        # AUROC
        def get_fmt(m):
            if isinstance(m['mean'], float) and str(m['mean']).lower() == 'nan':
                return "NaN"
            # It could be missing or None
            if m.get('mean') is None:
                return "N/A"
            mean = m['mean']
            low = m['bootstrap'].get('low', float('nan'))
            high = m['bootstrap'].get('high', float('nan'))
            return f"{mean:.4f} [{low:.4f}, {high:.4f}]"

        try:
            auroc_str = get_fmt(metrics['AUROC'])
            auarc_str = get_fmt(metrics['area_under_thresholded_accuracy'])
            auroc_val = metrics['AUROC']['mean'] if not (isinstance(metrics['AUROC']['mean'], float) and str(metrics['AUROC']['mean']).lower() == 'nan') else -1
        except Exception as e:
            auroc_str = "Error formatting"
            auarc_str = "Error formatting"
            auroc_val = -1

        if args.best:
            if auroc_val > best_auroc:
                best_auroc = auroc_val
                best_measure = measure_name
                best_str_auroc = auroc_str
                best_str_auarc = auarc_str
        else:
            print(f"{measure_name:<45} | {auroc_str:<30} | {auarc_str:<30}")
    
    if args.best and best_measure:
        print("🏆 Your Best Result 🏆")
        print("-" * 110)
        print(f"{best_measure:<45} | {best_str_auroc:<30} | {best_str_auarc:<30}")

    print("="*110 + "\n")

    # Check for potential issues
    uncertainties = data['uncertainty']
    first_key = list(uncertainties.keys())[0]
    if uncertainties[first_key]['AUROC']['mean'] == 0.5 and uncertainties[first_key]['AUROC']['bootstrap']['std_err'] == 0.0:
        print("⚠️  WARNING: AUROC is exactly 0.5 with 0 stderr. This indicates that the uncertainty scores might be constant or failing to discriminate correct/incorrect answers entirely.")

if __name__ == "__main__":
    main()
