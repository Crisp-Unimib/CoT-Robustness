#!/usr/bin/env python3
"""
Script to update bifurcation_entropy_kle (z_score) values in experiment result JSON files
using the updated values from the processed bifurcation entropy JSONL files.

This script:
1. Reads the bifurcation_entropy_kle values from JSONL source files
2. Updates the z_score values in the experiment result JSON files
3. Matches entries by step_idx (in source) to anchor_idx (in target)
"""

import json
import os
from pathlib import Path
from typing import Dict

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = BASE_DIR / "data" / "processed" / "blackmail_bifurcation" / "Qwen3-Next-80B-A3B-Thinking"
TARGET_DIRS = [
    BASE_DIR / "data" / "results" / "blackmail" / "Qwen3-Next-80B" / "bifurcation_bifurcation_entropy_kle" / "google-gemini-3-flash-preview_n100_T1.0_p0.95_e1_k10",
    BASE_DIR / "data" / "results" / "blackmail" / "Qwen3-Next-80B" / "bifurcation_bifurcation_entropy_kle" / "google-gemini-3-flash-preview_n100_T1.0_p0.95_e1_k10_random",
]


def load_bifurcation_kle_values(scenario_dir: Path) -> Dict[int, float]:
    """
    Load bifurcation_entropy_kle values from a scenario's bifurcation_entropy.jsonl file.
    
    Args:
        scenario_dir: Directory containing the bifurcation_entropy.jsonl file
        
    Returns:
        Dictionary mapping step_idx to bifurcation_entropy_kle value
    """
    jsonl_path = scenario_dir / "bifurcation_entropy.jsonl"
    if not jsonl_path.exists():
        print(f"  Warning: {jsonl_path} does not exist")
        return {}
    
    values = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                step_idx = entry.get("step_idx")
                # The bifurcation_entropy_kle value is nested inside the metrics dictionary
                metrics = entry.get("metrics", {})
                kle_value = metrics.get("bifurcation_entropy_kle")
                if step_idx is not None and kle_value is not None:
                    values[step_idx] = kle_value
    return values


def update_result_file(result_path: Path, kle_values: Dict[int, float]) -> int:
    """
    Update the z_score values in an experiment result JSON file.
    
    Args:
        result_path: Path to the experiment result JSON file
        kle_values: Dictionary mapping step_idx to bifurcation_entropy_kle value
        
    Returns:
        Number of values updated
    """
    if not result_path.exists():
        print(f"  Warning: {result_path} does not exist")
        return 0
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updates_count = 0
    anchor_results = data.get("anchor_results", [])
    
    for anchor in anchor_results:
        anchor_idx = anchor.get("anchor_idx")
        if anchor_idx is not None and anchor_idx in kle_values:
            old_value = anchor.get("z_score")
            new_value = kle_values[anchor_idx]
            if old_value != new_value:
                anchor["z_score"] = new_value
                updates_count += 1
                print(f"    anchor_idx {anchor_idx}: {old_value} -> {new_value}")
    
    if updates_count > 0:
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    return updates_count


def get_scenario_name_from_filename(filename: str) -> str:
    """
    Extract the scenario name from the experiment result filename.
    
    Example: "experiment_result_blackmail_ambiguous-none_none.json" -> "blackmail_ambiguous-none_none"
    """
    # Remove prefix and extension
    name = filename.replace("experiment_result_", "").replace(".json", "")
    return name


def main():
    print("=" * 80)
    print("Updating bifurcation_entropy_kle (z_score) values")
    print("=" * 80)
    
    # Get all scenario directories from source
    scenario_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    print(f"\nFound {len(scenario_dirs)} scenario directories in source")
    
    # Build mapping of scenario_id to kle values
    scenario_kle_values: Dict[str, Dict[int, float]] = {}
    for scenario_dir in scenario_dirs:
        scenario_id = scenario_dir.name
        kle_values = load_bifurcation_kle_values(scenario_dir)
        if kle_values:
            scenario_kle_values[scenario_id] = kle_values
            print(f"  Loaded {len(kle_values)} KLE values for {scenario_id}")
    
    print(f"\nTotal scenarios with KLE values: {len(scenario_kle_values)}")
    
    # Update result files in each target directory
    total_updates = 0
    for target_dir in TARGET_DIRS:
        print(f"\n{'=' * 60}")
        print(f"Processing target directory: {target_dir.name}")
        print("=" * 60)
        
        if not target_dir.exists():
            print(f"  Warning: Target directory does not exist: {target_dir}")
            continue
        
        # Find all experiment result JSON files
        result_files = list(target_dir.glob("experiment_result_*.json"))
        print(f"  Found {len(result_files)} result files")
        
        dir_updates = 0
        for result_file in result_files:
            scenario_id = get_scenario_name_from_filename(result_file.name)
            
            if scenario_id not in scenario_kle_values:
                print(f"  Skipping {result_file.name}: No source KLE values found for {scenario_id}")
                continue
            
            print(f"\n  Updating {result_file.name}:")
            updates = update_result_file(result_file, scenario_kle_values[scenario_id])
            dir_updates += updates
            if updates == 0:
                print(f"    No updates needed")
        
        print(f"\n  Total updates in this directory: {dir_updates}")
        total_updates += dir_updates
    
    print(f"\n{'=' * 80}")
    print(f"DONE! Total values updated: {total_updates}")
    print("=" * 80)


if __name__ == "__main__":
    main()
