
import os
import json
import glob
import argparse
import sys
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_utils.config import ExperimentConfig
from pipeline_utils.llm_client import LLMClient
from analysis.blackmail_prompts import DAG_PROMPT
from rich.console import Console
from rich.progress import Progress

import random

@dataclass
class ClassificationTask:
    file_path: str
    problem_id: str
    model_name: str
    question: str
    original_trace: List[str]
    output_file: str
    force: bool = False  # Add force flag to task


def classify_original_trace(
    task: ClassificationTask,
    llm_client: LLMClient, 
    classifier_model: str,
) -> Dict[str, Any]:
    """
    Classifies the original trace chunks using the DAG_PROMPT.
    Returns a dict with status and result/error.
    """
    # Format chunks with indices for the prompt
    full_chunked_text = ""
    for i, chunk in enumerate(task.original_trace):
        full_chunked_text += f"[{i}] {chunk}\n"
    
    prompt = DAG_PROMPT.format(
        problem_text=task.question,
        full_chunked_text=full_chunked_text
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = llm_client.query_openrouter(
            messages, 
            response_format={"type": "json_object"},
            model=classifier_model
        )
        if isinstance(response, list) and len(response) == 1 and isinstance(response[0], dict):
            response = response[0]

        # Check for error passed from LLMClient (custom error dict)
        if isinstance(response, dict) and response.get("error") == "json_parse_error":
             raise ValueError(f"JSON parsing failed. Raw content: {response.get('raw_content')}")

        if not isinstance(response, dict):
             raise ValueError(f"Invalid response format: {response}")
        if not response:
             raise ValueError("Empty response received (likely JSON parsing failure)")
        return {"status": "success", "result": response, "task": task}
    except Exception as e:
        return {"status": "error", "error": str(e), "task": task}


def process_task(task: ClassificationTask, llm_client: LLMClient, classifier_model: str) -> Dict[str, Any]:
    """Wrapper to process a single task and save the result."""
    result = classify_original_trace(task, llm_client, classifier_model)
    
    if result["status"] == "success":
        output_data = {
            "file_id": task.problem_id,
            "model_name": task.model_name,
            "source_file": task.file_path,
            "classification": result["result"]
        }
        os.makedirs(os.path.dirname(task.output_file), exist_ok=True)
        with open(task.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
    return result


def main():
    parser = argparse.ArgumentParser(description="Classify original traces from experiment results.")
    parser.add_argument("--results_dir", type=str, default="data/results/blackmail", help="Base directory for results")
    parser.add_argument("--output_dir", type=str, default="outputs/classifications", help="Directory to save classifications")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview", help="Model to use for classification")
    parser.add_argument("--pattern", type=str, default="**/bifurcation_bifurcation_entropy_kle/**/experiment_result_*.json", help="Glob pattern to find result files")
    parser.add_argument("--dry_run", action="store_true", help="Run without calling LLM")
    parser.add_argument("--max_workers", type=int, default=20, help="Max parallel workers")
    parser.add_argument("--test_one", action="store_true", help="Process a random file for testing (overwrites existing)")
    
    args = parser.parse_args()
    
    console = Console()
    config = ExperimentConfig()
    llm_client = LLMClient(config)
    
    # 1. Find all experiment result files
    search_pattern = os.path.join(args.results_dir, args.pattern)
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        console.print(f"[bold red]No files found matching pattern:[/bold red] {search_pattern}")
        return

    # Random selection for test mode
    if args.test_one:
        import random
        selected_file = random.choice(files)
        files = [selected_file]
        console.print(f"[bold yellow]Test Mode: Selected random file {selected_file}[/bold yellow]")

    console.print(f"[bold green]Found {len(files)} files to scan.[/bold green]")
    
    # 2. Prepare tasks
    tasks: List[ClassificationTask] = []
    skipped_count = 0
    scheduled_output_files = set()
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problem_data = data.get("problem", {})
            metadata = data.get("metadata", {})
            
            problem_id = metadata.get("problem_id") or os.path.splitext(os.path.basename(file_path))[0].replace("experiment_result_", "")
            model_name = metadata.get("model_name", "unknown_model")
            
            question = problem_data.get("question")
            original_trace = problem_data.get("original_trace")
            
            if not question or not original_trace:
                skipped_count += 1
                continue
                
            model_output_dir = os.path.join(args.output_dir, model_name)
            output_file = os.path.join(model_output_dir, f"{problem_id}.json")
            
            if os.path.exists(output_file) and not args.test_one:
                try:
                    with open(output_file, 'r', encoding='utf-8') as f_out:
                        existing_data = json.load(f_out)
                    
                    classification = existing_data.get("classification")
                    
                    # detailed check for validity
                    is_valid = False
                    if classification:
                        # Check if it's explicitly an error string
                        if isinstance(classification, str) and "error" in classification.lower():
                            is_valid = False
                        # Check if it's a dict with an error key
                        elif isinstance(classification, dict) and classification.get("error"):
                            is_valid = False
                        else:
                            is_valid = True
                            
                    if is_valid:
                        skipped_count += 1
                        continue
                except Exception:
                    # If we can't read/parse the file, assume it's invalid and needs reprocessing
                    pass

            if output_file in scheduled_output_files:
                 skipped_count += 1
                 continue
            
            scheduled_output_files.add(output_file)

            tasks.append(ClassificationTask(
                file_path=file_path,
                problem_id=problem_id,
                model_name=model_name,
                question=question,
                original_trace=original_trace,
                output_file=output_file,
                force=args.test_one
            ))
        except Exception as e:
            console.print(f"[bold red]Error reading file {file_path}:[/bold red] {e}")
            
    console.print(f"[bold green]Tasks to process: {len(tasks)}[/bold green]")
    console.print(f"[dim]Skipped (existing or invalid): {skipped_count}[/dim]")

    if args.dry_run:
        for task in tasks:
            console.print(f"[dim]Dry Run: Would process {task.problem_id} for model {task.model_name}[/dim]")
        return
    
    if not tasks:
        console.print("[bold yellow]No tasks to process.[/bold yellow]")
        return

    # 3. Process tasks in parallel
    success_count = 0
    error_count = 0
    
    with Progress(console=console) as progress:
        task_progress = progress.add_task("[cyan]Classifying...", total=len(tasks))
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_task, task, llm_client, args.model): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                    if args.test_one:
                         console.print(f"[bold green]Test Result Saved:[/bold green] {result['task'].output_file}")
                else:
                    error_count += 1
                    console.print(f"[red]Error for {result['task'].problem_id}: {result.get('error', 'Unknown error')}[/red]")
                progress.update(task_progress, advance=1)
                
    console.print(f"[bold green]Completed: {success_count} success, {error_count} errors.[/bold green]")


if __name__ == "__main__":
    main()
