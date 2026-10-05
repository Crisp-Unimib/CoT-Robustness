#!/usr/bin/env python3
"""
Generate a LaTeX table showing the COUNT of broken anchors
across different models and scenarios (Selected vs Random), matching the user's request.
"""

import os
import json
import glob
import pandas as pd
from pathlib import Path

import scipy.stats as stats

def load_data(base_dir: str):
    data = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return pd.DataFrame()

    # Iterate over models
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        # Check for bifurcation_bifurcation_entropy_kle
        bifurcation_path = os.path.join(model_path, "bifurcation_bifurcation_entropy_kle")
        
        if not os.path.exists(bifurcation_path):
            continue
            
        # Iterate over experiment directories
        for exp_dir in os.listdir(bifurcation_path):
            exp_path = os.path.join(bifurcation_path, exp_dir)
            if not os.path.isdir(exp_path):
                continue
                
            # Determine type
            is_random = exp_dir.endswith("_random")
            scenario_type = "Random" if is_random else "Bifurcation"
            
            # Find result json
            json_files = glob.glob(os.path.join(exp_path, "experiment_result_*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        
                    anchor_results = content.get("anchor_results", [])
                    total_count = len(anchor_results)
                    broken_count = sum(1 for anchor in anchor_results if anchor.get("status") == "Broken")
                    
                    data.append({
                        "Model": model_name,
                        "Type": scenario_type,
                        "Broken Count": broken_count,
                        "Total Count": total_count
                    })
                except Exception as e:
                    # simplistic error logging
                    pass

    return pd.DataFrame(data)

def format_discovery_gap(bif_broken, bif_total, rand_broken, rand_total):
    # Calculate gap
    if rand_broken == 0:
        gap_str = f"+{bif_broken - rand_broken}"
    else:
        gap_percent = ((bif_broken - rand_broken) / rand_broken) * 100
        gap_str = f"{gap_percent:+.1f}\\%"

    # Calculate statistical significance (Binomial Test on Counts)
    # We test if the counts deviate significantly from a 50/50 split.
    # We use alternative='greater' to test if Bifurcation is significantly BETTER (higher count) than Random.
    total_broken = bif_broken + rand_broken
    if total_broken > 0:
        # Scipy 1.7.0+ has binomtest. Fallback to binom_test if needed (but we verified 1.15.3).
        if hasattr(stats, 'binomtest'):
            res = stats.binomtest(bif_broken, n=total_broken, p=0.5, alternative='greater')
            p_value = res.pvalue
        else:
            p_value = stats.binom_test(bif_broken, n=total_broken, p=0.5, alternative='greater')
    else:
        p_value = 1.0
    
    significance = ""
    if p_value < 0.001:
        significance = " \\textbf{(***)}"
    elif p_value < 0.01:
        significance = " \\textbf{(**)}"
    elif p_value < 0.05:
        significance = " \\textbf{(*)}"
        
    return f"{gap_str}{significance}"

def calculate_prob_superiority(bif_broken, bif_total, rand_broken, rand_total, num_samples=100000):
    """
    Calculate P(Theta_bif > Theta_rand) using Monte Carlo simulation from Beta distributions.
    Prior: Beta(1, 1) (Uniform)
    Posterior: Beta(1 + hits, 1 + misses)
    """
    # Bifurcation posterior
    alpha_b = 1 + bif_broken
    beta_b = 1 + (bif_total - bif_broken)
    
    # Random posterior
    alpha_r = 1 + rand_broken
    beta_r = 1 + (rand_total - rand_broken)
    
    # Sample
    samples_b = stats.beta.rvs(alpha_b, beta_b, size=num_samples)
    samples_r = stats.beta.rvs(alpha_r, beta_r, size=num_samples)
    
    # Fraction where Bifurcation > Random
    prob = (samples_b > samples_r).mean()
    
    # Formatting
    prob_percent = prob * 100
    
    # Heuristic for highlighting reliable improvements
    if prob > 0.95:
        return f"\\textbf{{{prob_percent:.1f}\\%}}"
    elif prob > 0.90:
        return f"{prob_percent:.1f}\\%"
    else:
        return f"{prob_percent:.1f}\\%"

def generate_latex_table(df: pd.DataFrame):
    if df.empty:
        print("No data found.")
        return

    # Normalize model names
    df["Model"] = df["Model"].replace("Qwen3-VL-32B-Thinking", "Qwen3-VL-32B")

    # Aggregate by Model and Type
    # Summing up Broken Count and Total Count
    df_agg = df.groupby(["Model", "Type"], as_index=False)[["Broken Count", "Total Count"]].sum()

    # Pivot to get columns for Bifurcation and Random
    pivot_df = df_agg.pivot(index="Model", columns="Type", values=["Broken Count", "Total Count"])
    
    # Flatten columns
    pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
    
    # Fill NaN with 0 if any missing data (unlikely if data is complete)
    pivot_df = pivot_df.fillna(0)
    
    # Calculate Totals for the bottom row
    total_bif_broken = int(pivot_df["Bifurcation Broken Count"].sum())
    total_bif_total = int(pivot_df["Bifurcation Total Count"].sum())
    total_rand_broken = int(pivot_df["Random Broken Count"].sum())
    total_rand_total = int(pivot_df["Random Total Count"].sum())
    
    # Format the table rows
    latex_rows = []
    
    # Reset index to iterate
    pivot_df = pivot_df.reset_index()
    
    # Sort by Bifurcation Broken Count descending
    pivot_df["SortKey"] = pivot_df["Bifurcation Broken Count"]
    pivot_df = pivot_df.sort_values("SortKey", ascending=False)

    for _, row in pivot_df.iterrows():
        model = row["Model"]
        bif_broken = int(row["Bifurcation Broken Count"])
        bif_total = int(row["Bifurcation Total Count"])
        rand_broken = int(row["Random Broken Count"])
        rand_total = int(row["Random Total Count"])
    
    return f"{gap_str}{significance}"

def calculate_prob_superiority_counts(bif_broken, rand_broken):
    """
    Calculate Probability that Bifurcation Count > Random Count
    assuming they are Competing Processes with equal base probability (Null H0: p=0.5).
    This answers: "How sure are we that Bifurcation finds MORE bugs?"
    """
    total = bif_broken + rand_broken
    if total == 0:
        return "50.0\\%"
    
    # We use the survival function (1 - cdf) of the binomial distribution
    # binom.sf(k, n, p) = P(X > k)
    # We want P(X_bif > X_rand). In a 50/50 split, expected X_bif is total/2.
    # We actually want the posterior probability that p_bif > 0.5 given the counts.
    # Using a Beta(1,1) prior on the "Share of Bugs" p_bif:
    # Posterior for p_bif is Beta(1 + bif_broken, 1 + rand_broken).
    # We calculate P(p_bif > 0.5).
    
    alpha = 1 + bif_broken
    beta_param = 1 + rand_broken
    
    # Cumulative probability up to 0.5 is P(p_bif <= 0.5)
    # We want P(p_bif > 0.5) = 1 - CDF(0.5)
    prob = 1 - stats.beta.cdf(0.5, alpha, beta_param)
    
    prob_percent = prob * 100
    
    if prob > 0.95:
        return f"\\textbf{{{prob_percent:.1f}\\%}}"
    elif prob > 0.80: # Slightly relaxed threshold? No, keep it rigorous.
        return f"{prob_percent:.1f}\\%"
    else:
        return f"{prob_percent:.1f}\\%"

def generate_latex_table(df: pd.DataFrame):
    if df.empty:
        print("No data found.")
        return

    # Normalize model names
    df["Model"] = df["Model"].replace("Qwen3-VL-32B-Thinking", "Qwen3-VL-32B")

    # Aggregate by Model and Type
    df_agg = df.groupby(["Model", "Type"], as_index=False)[["Broken Count", "Total Count"]].sum()

    # Pivot
    pivot_df = df_agg.pivot(index="Model", columns="Type", values=["Broken Count", "Total Count"])
    pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.fillna(0)
    
    # Calculate Totals
    total_bif_broken = int(pivot_df["Bifurcation Broken Count"].sum())
    total_bif_total = int(pivot_df["Bifurcation Total Count"].sum())
    total_rand_broken = int(pivot_df["Random Broken Count"].sum())
    total_rand_total = int(pivot_df["Random Total Count"].sum())
    
    # Format rows
    latex_rows = []
    pivot_df = pivot_df.reset_index()
    pivot_df["SortKey"] = pivot_df["Bifurcation Broken Count"]
    pivot_df = pivot_df.sort_values("SortKey", ascending=False)

    for _, row in pivot_df.iterrows():
        model = row["Model"]
        bif_broken = int(row["Bifurcation Broken Count"])
        bif_total = int(row["Bifurcation Total Count"])
        rand_broken = int(row["Random Broken Count"])
        rand_total = int(row["Random Total Count"])
        
        bif_yield = f"{bif_broken} / {bif_total}"
        rand_yield = f"{rand_broken} / {rand_total}"
        # gap = format_discovery_gap(bif_broken, bif_total, rand_broken, rand_total) # REMOVED
        prob_sup = calculate_prob_superiority_counts(bif_broken, rand_broken)
        
        latex_rows.append(f"        {model} & {bif_yield} & {rand_yield} & {prob_sup} \\\\")

    # Total Row
    total_gap = format_discovery_gap(total_bif_broken, total_bif_total, total_rand_broken, total_rand_total) # Not used in output
    total_prob = calculate_prob_superiority_counts(total_bif_broken, total_rand_broken)
    total_row = f"        \\textbf{{Total}} & \\textbf{{{total_bif_broken} / {total_bif_total}}} & \\textbf{{{total_rand_broken} / {total_rand_total}}} & \\textbf{{{total_prob}}} \\\\"

    # LaTeX Table
    latex_table = f"""\\begin{{table*}}[t]
    \\centering
    \\caption{{Comparison of Vulnerability Discovery Yields. 'Yield' columns show the number of broken anchors found out of total attempts ($N_{{broken}} / N_{{total}}$). The 'Prob. Counts $>$ Random' column reports the probability that the Bifurcation method discovers a greater absolute number of vulnerabilities than the Random baseline ($P(N_{{bif}} > N_{{rand}})$).}}
    \\label{{tab:bif_vs_rand}}
    \\begin{{tabular}}{{lccc}}
        \\toprule
        \\textbf{{Model}} & \\textbf{{Bifurcation Yield}} & \\textbf{{Random Yield}} & \\textbf{{Prob. Counts > Random}} \\\\
        \\midrule
{chr(10).join(latex_rows)}
        
        \\midrule
{total_row}
        \\bottomrule
    \\end{{tabular}}
\\end{{table*}}"""

    print(latex_table)

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    base_dir = project_root / "data" / "results" / "blackmail"
    
    df = load_data(str(base_dir))
    generate_latex_table(df)

if __name__ == "__main__":
    main()
