#!/usr/bin/env python3
"""
Generate a 2-panel "Phase Fragility & Breakage Profile" plot.
Top: Split Violin Plot of Fragility (Broken vs Robust) by Phase.
Bottom: Breakage Frequency by Phase.

Matches the EXACT style of urgency_violin_distribution.png:
- White Violin Bodies with Black Edges
- Jittered Scatter Points (Teal/Pink)
- Mean Markers (Red Circles)
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set consistent style parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def load_data(results_base_dir: str, classifications_base_dir: str) -> pd.DataFrame:
    """
    Load data from experiment results and link with classifications.
    Iterates over all models and scenarios.
    """
    data = []
    print(f"Scanning for result files in {results_base_dir}...")
    
    pattern = os.path.join(results_base_dir, "*", "bifurcation_bifurcation_entropy_kle", "*", "experiment_result_*.json")
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(json_files)} experiment result files.")
    
    for json_file in json_files:
        try:
            # Extract Model Name
            path_parts = Path(json_file).parts
            try:
                blackmail_idx = path_parts.index('blackmail')
                model_name = path_parts[blackmail_idx + 1]
            except ValueError:
                continue
                
            filename = os.path.basename(json_file)
            if not filename.startswith("experiment_result_"):
                continue
                
            scenario_name = filename.replace("experiment_result_", "").replace(".json", "")
            
            # Classification file path
            classification_file = os.path.join(classifications_base_dir, model_name, f"{scenario_name}.json")
            if not os.path.exists(classification_file):
                continue
                
            # Load Data
            with open(json_file, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
            with open(classification_file, 'r', encoding='utf-8') as f:
                class_data = json.load(f)
            
            classifications = class_data.get("classification", {})
            anchor_results = exp_data.get("anchor_results", [])
            
            for anchor in anchor_results:
                anchor_idx = str(anchor.get("anchor_idx"))
                z_score = anchor.get("z_score")
                status = anchor.get("status") 
                
                if z_score is None or not status:
                    continue
                    
                # Get Functional Tag
                if anchor_idx in classifications:
                    tags = classifications[anchor_idx].get("function_tags", [])
                    if isinstance(tags, str):
                        primary_tag = tags
                    elif isinstance(tags, list) and len(tags) > 0:
                        primary_tag = tags[0]
                    else:
                        primary_tag = "Unclassified"
                else:
                    primary_tag = "Unclassified"
                
                data.append({
                    "Model": model_name,
                    "Scenario": scenario_name,
                    "Anchor_Idx": anchor_idx,
                    "Phase": primary_tag,
                    "Status": status,
                    "Z_Score": z_score
                })
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
            
    return pd.DataFrame(data)

def style_axis(ax):
    """Apply common axis styling."""
    ax.set_facecolor('white')
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_fragility_and_breakage(df: pd.DataFrame, output_path: str):
    """
    Create 2-panel plot with advanced styling matching urgency_violin_distribution.png.
    """
    if df.empty:
        print("No data for plot.")
        return

    # Filter and Normalize
    plot_data = df.dropna(subset=["Z_Score", "Phase", "Status"])
    plot_data = plot_data[plot_data["Status"].isin(["Broken", "Robust"])]
    
    # Filter out Zero Z-Scores (as requested)
    plot_data = plot_data[plot_data["Z_Score"] > 0]
    
    # Clean Phase Names
    plot_data["Phase"] = plot_data["Phase"].str.replace("_", " ").str.title()
    
    # Sorting Order: Descending by ABSOLUTE Breakage Count
    summary_counts = plot_data.groupby("Phase")["Status"].value_counts().unstack(fill_value=0)
    if "Broken" not in summary_counts.columns:
        summary_counts["Broken"] = 0
        
    summary_counts = summary_counts.sort_values("Broken", ascending=False)
    order = summary_counts.index.tolist()
    
    print(f"Phases (sorted by Broken Count): {order}")

    # Set up Figure
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.1)
    
    ax_violin = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax_violin)
    
    fig.patch.set_facecolor('white')
    
    # --- TOP PANEL: Styled Split Violin ---
    palette = {"Robust": "#20B2AA", "Broken": "#FF69B4"} # Teal, Pink
    
    # 1. Draw standard split violin to get shapes (with inner=None)
    sns.violinplot(
        data=plot_data,
        x="Phase",
        y="Z_Score",
        hue="Status",
        hue_order=["Robust", "Broken"],
        order=order,
        split=True,
        inner=None, # We draw our own stats
        palette=palette, # Will be overridden to white
        cut=0,
        ax=ax_violin,
        linewidth=1.2
    )
    
    # 2. Force White Body + Black Edge
    for collection in ax_violin.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_facecolor('white')
            collection.set_edgecolor('black')
            collection.set_linewidth(1.2)
            collection.set_alpha(1.0)
            
    # 3. Add Jittered Scatter Points Manually
    np.random.seed(42)
    MAX_POINTS = 200 # Per group per phase to avoid clutter
    
    for i, phase in enumerate(order):
        subset = plot_data[plot_data["Phase"] == phase]
        
        for status, offset_dir in [("Robust", -1), ("Broken", 1)]:
            group_data = subset[subset["Status"] == status]["Z_Score"].values
            if len(group_data) == 0: continue
            
            # Subsample if needed
            if len(group_data) > MAX_POINTS:
                display_data = np.random.choice(group_data, size=MAX_POINTS, replace=False)
            else:
                display_data = group_data
            
            # Jitter
            jitter = np.random.uniform(0.05, 0.25, size=len(display_data)) * offset_dir
            x_pos = i + jitter
            
            # Color
            color = palette[status]
            edge = 'black' if status == 'Robust' else '#8B0000' # Darker edge for pink
            if status == 'Robust': edge = '#006666' # Darker teal
            
            ax_violin.scatter(
                x_pos, display_data,
                c=color,
                s=20,
                alpha=0.6,
                edgecolors=edge,
                linewidths=0.5,
                zorder=5
            )
            
            # 4. Add Mean Markers
            mean_val = np.mean(group_data)
            mean_x = i + (0.25 * offset_dir) # Center of the half
            
            # Marker
            ax_violin.scatter([mean_x], [mean_val], s=100, c='#8B0000', edgecolors='white', linewidths=1.5, zorder=10)

    ax_violin.set_ylabel("Bifurcation Entropy KLE", fontsize=13)
    ax_violin.set_xlabel("")
    ax_violin.tick_params(labelbottom=False)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["Robust"], markersize=10, label='Robust'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette["Broken"], markersize=10, label='Broken')
    ]
    ax_violin.legend(handles=legend_elements, title="Anchor Status", loc='upper right', bbox_to_anchor=(1.0, 1.05), frameon=False)
    # ax_violin.set_title("Structural Fragility & Breakage Profile by Phase", fontsize=16, pad=20)
    
    style_axis(ax_violin)
    
    # --- BOTTOM PANEL: Absolute Breakage Count ---
    
    broken_counts = summary_counts["Broken"]
    broken_counts = broken_counts.reindex(order)
    
    bars = ax_bar.bar(
        order,
        broken_counts,
        color=palette["Broken"],
        edgecolor='black',
        linewidth=1.0,
        width=0.6,
        alpha=0.8
    )
    
    ax_bar.set_ylabel("Number of Broken Anchors", fontsize=13)
    ax_bar.set_xlabel("Functional Phase", fontsize=13)
    # ax_bar.set_ylim(0, 100) # No longer percentage
    ax_bar.set_xticklabels(order, rotation=45, ha='right', fontsize=12)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + (max(broken_counts)*0.02),
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)
    
    style_axis(ax_bar)
    
    # Annotate Total Counts on X-axis labels
    counts = plot_data["Phase"].value_counts().reindex(order)
    new_labels = [f"{label}\n(n={counts[label]})" for label in order]
    ax_bar.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=11)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved plot to {output_path}")

    # Stats Export
    stats_file = Path(output_path).parent / "fragility_breakage_stats.txt"
    with open(stats_file, "w") as f:
        f.write("PHASE STATISTICS\n")
        f.write("Phase | Total Count | Broken Count | Broken % | Mean Z (Broken) | Mean Z (Robust)\n")
        f.write("-" * 90 + "\n")
        for phase in order:
            subset = plot_data[plot_data["Phase"] == phase]
            total_count = len(subset)
            broken_cnt = broken_counts[phase]
            broken_pct = (broken_cnt / total_count * 100) if total_count > 0 else 0
            
            mean_broken = subset[subset["Status"] == "Broken"]["Z_Score"].mean()
            mean_robust = subset[subset["Status"] == "Robust"]["Z_Score"].mean()
            f.write(f"{phase:<25} | {total_count:<5} | {broken_cnt:<5} | {broken_pct:>6.2f}% | {mean_broken:>14.4f} | {mean_robust:>14.4f}\n")
    print(f"Saved stats to {stats_file}")

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_base_dir = project_root / "data" / "results" / "blackmail"
    classifications_base_dir = project_root / "outputs" / "classifications"
    output_dir = project_root / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data(str(results_base_dir), str(classifications_base_dir))
    
    if not df.empty:
        import matplotlib
        import matplotlib.collections # Needed for type checking
        plot_fragility_and_breakage(df, str(output_dir / "fragility_breakage_profile.png"))
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
