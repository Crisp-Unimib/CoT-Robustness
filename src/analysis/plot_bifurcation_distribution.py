#!/usr/bin/env python3
"""
Generate a distribution (area/density) plot of Bifurcation Entropy KLE values
aggregated across all models and scenarios.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(data_dir: str) -> list[float]:
    """Load bifurcation_entropy_kle values from all jsonl files."""
    values = []
    
    pattern = os.path.join(data_dir, "**", "bifurcation_entropy.jsonl")
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(json_files)} result files.")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        metrics = record.get("metrics", {})
                        if metrics:
                            val = metrics.get("bifurcation_entropy_kle")
                            if val is not None:
                                values.append(val)
                        # Fallback for older format if exists? 
                        # Assuming consistent format based on view_file

                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
            
    return values

def create_density_plot(values: list[float], output_path: str):
    """Create a styled density area plot."""
    
    # Filter out None or non-finite values just in case
    clean_values = [v for v in values if v is not None and np.isfinite(v)]
    
    print(f"Plotting {len(clean_values)} values.")
    
    if not clean_values:
        print("No valid data to plot.")
        return

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Color palette (Sea Green / Teal similar to 'Robust' in reference)
    fill_color = '#20B2AA'  # Light sea green
    edge_color = '#008B8B'  # Dark cyan
    
    # Plot KDE (Kernel Density Estimate)
    sns.kdeplot(
        clean_values,
        fill=True,
        color=fill_color,
        edgecolor=edge_color,
        alpha=0.6,
        linewidth=2,
        ax=ax,
        cut=0 # Don't extend past data limits artificially if not needed, or distinct cut
    )
    
    # Create a filled area look manually if sns.kdeplot fill isn't enough, 
    # but sns.kdeplot(fill=True) creates the area graph loop.
    
    # Customize axes
    ax.set_xlabel('Bifurcation Entropy KLE', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)

    
    # Add horizontal grid lines only
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Set limits and margins to remove gaps
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(0)


    # Stats text removed per user request


    # Mark mean/median on the plot (optional, but good for context)
    # ax.axvline(mean_val, color='#8B0000', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # Also save as PDF
    plt.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved density plot to: {output_path}")

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "processed" / "blackmail_bifurcation"
    output_dir = project_root / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_dir}")
    values = load_data(str(data_dir))
    
    output_path = output_dir / "bifurcation_distribution.png"
    create_density_plot(values, str(output_path))

if __name__ == "__main__":
    main()
