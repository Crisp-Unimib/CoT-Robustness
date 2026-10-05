import os
import argparse
import ast
import numpy as np
from math import nan

if __name__ == "__main__":
    import sys
    with open("user_metrics.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract just the dictionary part if there's prefix text
    if "Full results dict: " in content:
        dict_str = content.split("Full results dict: ")[1]
    else:
        dict_str = content
        
    data = eval(dict_str, {"np": np, "nan": nan})
    print("\n" + "="*120)
    print(f"{'Measure':<45} | {'AUROC mean [95% CI]':<35} | {'AUARC mean [95% CI]':<35}")
    print("-" * 120)

    keys = sorted(data['uncertainty'].keys())
    for measure_name in keys:
        if "UNANSWERABLE" in measure_name:
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
        except Exception as e:
            auroc_str = "Error formatting"
            auarc_str = "Error formatting"

        print(f"{measure_name:<45} | {auroc_str:<35} | {auarc_str:<35}")
    
    print("="*120 + "\n")
