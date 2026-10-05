#!/usr/bin/env python3
"""
Generate Merged Plots comparing Bifurcation Entropy KLE across Urgency levels,
split by Broken vs Robust status.
Includes:
1. Split Violin Plot (Existing)
2. Faceted Density Plot (New)
3. Interaction Point Plot (New)
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set consistent style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def load_data(base_dir: str) -> pd.DataFrame:
    """
    Load data from experiment result files and categorize by urgency and status.
    """
    data = []
    print(f"Scanning for result files in {base_dir}...")
    
    pattern = os.path.join(base_dir, "**", "experiment_result_*.json")
    json_files = glob.glob(pattern, recursive=True)
    
    # Filter out random if we want strictly comparable sets
    # json_files = [f for f in json_files if '_random' not in f]
    
    print(f"Found {len(json_files)} files.")
    
    for json_file in json_files:
        try:
            filename = os.path.basename(json_file)
            if filename.endswith("_replacement.json"):
                urgency = "Replacement"
            elif filename.endswith("_restriction.json"):
                urgency = "Restriction"
            elif filename.endswith("_none.json"):
                urgency = "None"
            else:
                continue
                
            model_name = Path(json_file).parent.parent.parent.name
            
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            anchor_results = content.get("anchor_results", [])
            
            for anchor in anchor_results:
                status = anchor.get("status")
                z_score = anchor.get("z_score")
                
                if not status or z_score is None:
                    continue
                    
                data.append({
                    "Model": model_name,
                    "Urgency": urgency,
                    "Status": status,
                    "Z_Score": z_score
                })
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
            
    return pd.DataFrame(data)

def get_palette():
    return {"Robust": "#20B2AA", "Broken": "#FF69B4"}

def create_split_violin_plot(df: pd.DataFrame, output_path: str):
    """
    Create a split violin plot: X=Urgency, Y=Score, Hue=Status (Split).
    """
    if df.empty: return

    plot_data = df.dropna(subset=["Z_Score"])
    plot_data = plot_data[plot_data["Z_Score"] > 0]
    plot_data = plot_data[plot_data["Status"].isin(["Broken", "Robust"])]
    
    if plot_data.empty: return

    order = ["Replacement", "Restriction", "None"]
    palette = get_palette()

    plt.figure(figsize=(10, 7), dpi=150)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')

    sns.violinplot(
        data=plot_data,
        x="Urgency",
        y="Z_Score",
        hue="Status",
        order=order,
        hue_order=["Robust", "Broken"], 
        split=True,
        inner="quartiles",
        palette=palette,
        cut=0, 
        linewidth=1.2,
        ax=ax
    )

    ax.set_title("Bifurcation Entropy by Urgency (Split by Status)", fontsize=14, pad=15)
    ax.set_xlabel("Urgency Level", fontsize=13)
    ax.set_ylabel("Bifurcation Entropy KLE", fontsize=13)
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    sns.despine(trim=False)
    
    plt.legend(title="Anchor Status", loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Split violin plot saved to {output_path}")

def create_faceted_density_plot(df: pd.DataFrame, output_path: str):
    """
    Create small multiples (facets) by Urgency, overlapping distributions of Broken vs Robust.
    """
    plot_data = df.dropna(subset=["Z_Score"])
    plot_data = plot_data[plot_data["Z_Score"] > 0]
    plot_data = plot_data[plot_data["Status"].isin(["Broken", "Robust"])]
    
    if plot_data.empty: return
    
    order = ["Replacement", "Restriction", "None"]
    palette = get_palette()
    
    # FacetGrid
    g = sns.FacetGrid(
        plot_data, 
        col="Urgency", 
        col_order=order, 
        hue="Status", 
        hue_order=["Robust", "Broken"],
        palette=palette,
        height=5, 
        aspect=0.8,
        sharex=True,
        sharey=True
    )
    
    g.map(sns.kdeplot, "Z_Score", fill=True, alpha=0.5, linewidth=1.5)
    g.add_legend(title="Anchor Status", frameon=False)
    
    g.set_axis_labels("Bifurcation Entropy KLE", "Density")
    g.set_titles("{col_name}")
    
    for ax in g.axes.flatten():
        ax.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')
        ax.set_facecolor('white')
        
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Bifurcation Entropy Distributions by Urgency", fontsize=16)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Faceted density plot saved to {output_path}")

def create_interaction_point_plot(df: pd.DataFrame, output_path: str):
    """
    Create a point plot showing Means and Confidence Intervals.
    Good for showing if urgency affects Broken models differently than Robust ones.
    """
    plot_data = df.dropna(subset=["Z_Score"])
    plot_data = plot_data[plot_data["Z_Score"] > 0]
    plot_data = plot_data[plot_data["Status"].isin(["Broken", "Robust"])]
    
    if plot_data.empty: return
    
    order = ["Replacement", "Restriction", "None"]
    palette = get_palette()
    
    plt.figure(figsize=(8, 6), dpi=150)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    sns.pointplot(
        data=plot_data,
        x="Urgency",
        y="Z_Score",
        hue="Status",
        order=order,
        hue_order=["Robust", "Broken"],
        palette=palette,
        markers=["o", "D"],
        linestyles=["-", "--"],
        errorbar=("ci", 95), # 95% Confidence Interval
        capsize=0.1,
        dodge=True,
        ax=ax
    )
    
    ax.set_title("Interaction Effect: Urgency vs Status", fontsize=14, pad=15)
    ax.set_xlabel("Urgency Level", fontsize=13)
    ax.set_ylabel("Mean Bifurcation Entropy KLE (with 95% CI)", fontsize=13)
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    sns.despine(trim=False)
    
    plt.legend(title="Anchor Status", loc='best', frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Interaction point plot saved to {output_path}")

def create_interaction_boxplot(df: pd.DataFrame, output_path: str):
    """
    Create a Means with 95% CI plot (Error Bar style) matching the reference image.
    - Square markers
    - No connecting lines
    - Colors: Teal (Robust), Pink (Broken)
    """
    from scipy import stats as scipy_stats
    
    plot_data = df.dropna(subset=["Z_Score"])
    plot_data = plot_data[plot_data["Z_Score"] > 0]
    plot_data = plot_data[plot_data["Status"].isin(["Broken", "Robust"])]
    
    if plot_data.empty: return
    
    order = ["Replacement", "Restriction", "None"]
    palette = get_palette() # {"Robust": "#20B2AA", "Broken": "#FF69B4"}
    
    # Figure setup
    plt.figure(figsize=(10, 7), dpi=150)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Compute stats
    dodge = 0.15
    
    for i, urgency in enumerate(order):
        for status in ["Robust", "Broken"]:
            subset = plot_data[(plot_data["Urgency"] == urgency) & (plot_data["Status"] == status)]["Z_Score"]
            if len(subset) == 0:
                continue
            
            mean = subset.mean()
            sem = subset.sem()
            n = len(subset)
            
            # 95% CI
            if n > 1:
                ci = scipy_stats.t.ppf(0.975, n-1) * sem
            else:
                ci = 0
                
            # X Position
            x_pos = i + (dodge if status == "Broken" else -dodge)
            color = palette[status]
            
            # Plot Error Bar
            ax.errorbar(
                x_pos, mean,
                yerr=ci,
                fmt='s', # Square marker
                markersize=10,
                color=color,
                ecolor=color,
                elinewidth=2.5,
                capsize=10,
                capthick=2.5,
                zorder=5
            )

    # Axes styling
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=12)
    ax.set_xlim(-0.5, len(order) - 0.5)
    
    # ax.set_title("Bifurcation Entropy by Urgency and Status", fontsize=16, pad=20) # REMOVED per user request
    ax.set_xlabel("Urgency Level", fontsize=14)
    ax.set_ylabel("Mean Bifurcation Entropy KLE (with 95% CI)", fontsize=14)
    
    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    sns.despine(trim=False)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=palette["Robust"], markersize=10, label='Robust'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=palette["Broken"], markersize=10, label='Broken')
    ]
    ax.legend(handles=legend_elements, title="Anchor Status", loc='upper left', frameon=False, title_fontsize=12, fontsize=11)
    
    # Axis Break Indicator
    ymin, ymax = ax.get_ylim()
    # Add a bit of space at the bottom if needed, or rely on auto-scaling
    # Drawing break lines at bottom left
    d = 0.015 # Size of diagonal lines
    kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, linewidth=1.5)
    ax.plot((-d, +d), (0.02 - d, 0.02 + d), **kwargs) # Bottom-left diagonal
    ax.plot((-d, +d), (0.04 - d, 0.04 + d), **kwargs) # Top-left diagonal (slightly higher)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Interaction boxplot saved to {output_path}")



def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    base_dir = project_root / "data" / "results" / "blackmail"
    output_dir = project_root / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data(str(base_dir))
    
    if not df.empty:
        # Generate all plots
        create_split_violin_plot(df, str(output_dir / "merged_urgency_violin.png"))
        create_faceted_density_plot(df, str(output_dir / "urgency_status_density_facet.png"))
        create_interaction_point_plot(df, str(output_dir / "urgency_status_point_interaction.png"))
        create_interaction_boxplot(df, str(output_dir / "urgency_status_boxplot.png"))
    else:
        print("No data found.")

if __name__ == "__main__":
    main()

