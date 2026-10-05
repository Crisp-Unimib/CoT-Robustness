import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pipeline_utils.config import ExperimentConfig
from generation.utils import split_solution_into_chunks

class DatasetLoader:
    """
    Handles loading and processing of data from local directories.
    """
    def __init__(self, config: ExperimentConfig, console: Console):
        self.config = config
        self.console = console

    def _get_problem_dir(self) -> str:
        """
        Constructs the path to the problem directory.
        Math: math_rollouts/Qwen3.../temperature_0.6_top_p_0.95/correct_base_solution/problem_6596/
        Blackmail: data/processed/blackmail_bifurcation/Qwen3.../scenario_id/
        """
        if self.config.is_blackmail:
            return os.path.join(
                "data/processed/blackmail_bifurcation",
                self.config.model_name.split('/')[-1], # Use basename usually
                self.config.scenario_id
            )
            
        temp = self.config.vllm_temperature
        top_p = self.config.vllm_top_p
        temp_dir = f"temperature_{temp}_top_p_{top_p}"
        
        return os.path.join(
            self.config.math_rollouts_path,
            self.config.model_name,
            temp_dir,
            self.config.solution_type,
            f"problem_{self.config.problem_id}"
        )

    def _get_analysis_results_path(self) -> str:
        """
        Constructs the path to analysis_results.json in analysis directory.
        E.g., analysis/basic/Qwen3.../alpha_1.0/correct_base_solution/analysis_results.json
        """
        return os.path.join(
            self.config.analysis_path,
            self.config.model_name,
            "alpha_1.0",
            self.config.solution_type,
            "analysis_results.json"
        )

    def _get_bifurcation_results_path(self) -> str:
        """
        Constructs the path to bifurcation_entropy.jsonl.
        Math: bifurcation_results_compare/Qwen3.../problem_2238/bifurcation_entropy.jsonl
        Blackmail: inside the problem dir
        """
        if self.config.is_blackmail:
             return os.path.join(
                self._get_problem_dir(),
                "bifurcation_entropy.jsonl"
            )

        return os.path.join(
            "data/processed/bifurcation_results_compare",
            self.config.model_name,
            f"problem_{self.config.problem_id}",
            "bifurcation_entropy.jsonl"
        )

    def get_problem_data(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
        """
        Retrieves chunks, the question, and the correct answer.
        Reads from local files instead of HuggingFace dataset.
        Returns: (chunks, question, correct_answer)
        """
        problem_dir = self._get_problem_dir()
        
        # Handle Blackmail Mode
        if self.config.is_blackmail:
            base_response_path = os.path.join(problem_dir, "base_response.json")
            if os.path.exists(base_response_path):
                try:
                    with open(base_response_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    chunks = data.get('chunks', [])
                    # Fallback to reconstructing chunks if missing but scratchpad exists
                    if not chunks and 'scratchpad' in data:
                        chunks = split_solution_into_chunks(data['scratchpad'])
                        
                    # In generated JSON, chunks might be list of strings or list of dicts? 
                    # Usually split_solution lists strings. pipeline expects dicts with 'chunk' key sometimes?
                    # Let's check: in math pipeline, chunks_labeled.json usually has objects if labeled, 
                    # but simple split returns strings.
                    # The pipeline usage: c['chunk'] - indicates it expects dicts.
                    # Let's normalize chunks to dicts if they are strings.
                    
                    normalized_chunks = []
                    for i, c in enumerate(chunks):
                        if isinstance(c, str):
                            normalized_chunks.append({'chunk': c, 'step': i})
                        elif isinstance(c, dict):
                            normalized_chunks.append(c)
                    
                    chunks = normalized_chunks
                    
                    # For GLM models, normalize <thinking> tags to <think> in chunks
                    # Cached data may have been generated with wrong tag format
                    model_name = self.config.model_name.lower()
                    if 'glm' in model_name:
                        for chunk in chunks:
                            if 'chunk' in chunk:
                                chunk['chunk'] = chunk['chunk'].replace('<thinking>', '<think>').replace('</thinking>', '</think>')
                        self.console.print(f"[dim]DEBUG: Normalized <thinking> tags to <think> for GLM model[/]")
                    
                    question = data.get('prompt', '')
                    
                    # Strip old reasoning tags from cached prompts
                    # Cached base_response.json files may have been generated with old templates
                    # that injected reasoning tags. Clean them up for consistency.
                    # Include all known reasoning tag formats from different templates.
                    old_tags = [
                        '<think>\n', '<thinking>\n', '<think>', '<thinking>',  # Qwen/GLM style
                        'Here are my reasoning steps:\n', 'Here are my reasoning steps:',  # Apriel style
                    ]
                    for old_tag in old_tags:
                        if question.endswith(old_tag):
                            question = question[:-len(old_tag)]
                            self.console.print(f"[dim]DEBUG: Stripped old tag {old_tag!r} from cached prompt[/]")
                    question = question.rstrip()
                    
                    correct_answer = "SAFE (Non-blackmail)" # Placeholder
                    
                    return chunks, question, correct_answer
                    
                except json.JSONDecodeError as e:
                    self.console.print(f"[bold red]Error decoding JSON for {base_response_path}: {e}[/]")
            else:
                 self.console.print(f"[bold red]base_response.json not found at {base_response_path}[/]")
            
            return None, None, None

        # Standard Math Mode
        chunks_path = os.path.join(problem_dir, "chunks_labeled.json")
        base_solution_path = os.path.join(problem_dir, "base_solution.json")
        
        chunks = None
        question = None
        correct_answer = None
        
        # Load chunks_labeled.json
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            except json.JSONDecodeError as e:
                self.console.print(f"[bold red]Error decoding JSON for {chunks_path}: {e}[/]")
        else:
            self.console.print(f"[bold red]chunks_labeled.json not found at {chunks_path}[/]")
        
        # Load base_solution.json for question and correct answer
        if os.path.exists(base_solution_path):
            try:
                with open(base_solution_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    prompt = content.get('prompt', '')
                    if "Problem: " in prompt:
                        question = prompt.split("Problem: ", 1)[1].strip()
                        # Remove Solution: suffix if present
                        if " Solution:" in question:
                            question = question.split(" Solution:")[0].strip()
                    else:
                        question = prompt
                    # Get the correct answer
                    correct_answer = content.get('answer', None)
            except json.JSONDecodeError as e:
                self.console.print(f"[bold red]Error decoding JSON for {base_solution_path}: {e}[/]")
        else:
            self.console.print(f"[bold red]base_solution.json not found at {base_solution_path}[/]")
                    
        return chunks, question, correct_answer

    def load_analysis_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads analysis_results.json and extracts chunk importance data per problem.
        Returns dict mapping problem_idx (str) -> list of {chunk_idx, resampling_importance_accuracy, ...}
        """
        analysis_path = self._get_analysis_results_path()
        result = {}
        
        if not os.path.exists(analysis_path):
            self.console.print(f"[yellow]Analysis results not found: {analysis_path}[/]")
            return result
        
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for problem in data:
                problem_idx = str(problem.get('problem_idx'))  # Keep as string for consistency
                labeled_chunks = problem.get('labeled_chunks', [])
                
                chunks_info = []
                for chunk in labeled_chunks:
                    chunks_info.append({
                        'chunk_idx': chunk.get('chunk_idx'),
                        'resampling_importance_accuracy': chunk.get('resampling_importance_accuracy', 0),
                        'counterfactual_importance_accuracy': chunk.get('counterfactual_importance_accuracy', 0),
                        'different_trajectories_fraction': chunk.get('different_trajectories_fraction', 0)
                    })
                
                if chunks_info:
                    result[problem_idx] = chunks_info
                    
        except Exception as e:
            self.console.print(f"[bold red]Error loading analysis results: {e}[/]")
        
        return result

    def load_bifurcation_results(self) -> Dict[int, float]:
        """
        Loads bifurcation_entropy.jsonl and extracts entropy values per step.
        Returns dict mapping step_idx (int) -> bifurcation_entropy (float)
        """
        bifurcation_path = self._get_bifurcation_results_path()
        result = {}

        if not os.path.exists(bifurcation_path):
            self.console.print(f"[yellow]Bifurcation results not found: {bifurcation_path}[/]")
            return result

        try:
            with open(bifurcation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        step_idx = data.get('step_idx')
                        metrics = data.get('metrics', {})
                        entropy = metrics.get(self.config.bifurcation_metric, 0.0)
                        
                        if step_idx is not None:
                            result[step_idx] = entropy
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.console.print(f"[bold red]Error loading bifurcation results: {e}[/]")

        return result



    def find_thought_anchor_bifurcation(self, chunks: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Identifies the 'Thought Anchor' using bifurcation_entropy.jsonl data.
        Uses highest bifurcation_entropy to select anchor.
        """
        if not chunks:
            return None, -1
            
        bifurcation_data = self.load_bifurcation_results()
        
        if bifurcation_data:
            # Sort steps by entropy (descending)
            sorted_steps = sorted(bifurcation_data.items(), key=lambda x: x[1], reverse=True)
            
            for step_idx, entropy in sorted_steps:
                if 0 <= step_idx < len(chunks):
                    anchor_chunk = chunks[step_idx]
                    self.console.print(f"[green]Using bifurcation results: Chunk {step_idx} (entropy: {entropy:.4f})[/]")
                    return anchor_chunk, step_idx
        
        self.console.print("[yellow]Bifurcation data missing or invalid, using first chunk fallback[/]")
        return (chunks[0], 0) if chunks else (None, -1)

    def find_thought_anchor(self, chunks: List[Dict[str, Any]], min_diversity: float = 0.1) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Identifies the 'Thought Anchor' based on configured method.
        """
        if self.config.anchor_selection_method == "bifurcation":
            return self.find_thought_anchor_bifurcation(chunks)

        # Default to importance method
        """
        Identifies the 'Thought Anchor' using analysis_results.json data.
        Uses highest resampling_importance_accuracy to select anchor.
        """
        if not chunks:
            return None, -1
        
        # Load analysis results
        problem_id = str(self.config.problem_id)
        analysis_data = self.load_analysis_results()
        
        if problem_id in analysis_data:
            # Use the chunk with highest resampling_importance_accuracy
            chunks_info = analysis_data[problem_id]
            # Sort by resampling_importance_accuracy (descending)
            chunks_info.sort(key=lambda x: x['resampling_importance_accuracy'], reverse=True)
            
            for info in chunks_info:
                chunk_idx = info['chunk_idx']
                if 0 <= chunk_idx < len(chunks):
                    anchor_chunk = chunks[chunk_idx]
                    importance = info['resampling_importance_accuracy']
                    self.console.print(f"[green]Using analysis results: Chunk {chunk_idx} (resampling_importance: {importance:.4f})[/]")
                    return anchor_chunk, chunk_idx
        
        # Fallback: use resampling_importance_accuracy from chunks directly
        self.console.print("[yellow]Falling back to resampling_importance_accuracy from chunks[/]")
        max_score = -float('inf')
        anchor_chunk = None
        anchor_idx = -1
        
        for i, chunk in enumerate(chunks):
            diversity = float(chunk.get('different_trajectories_fraction', 0.0))
            if diversity < min_diversity:
                continue
                
            acc_imp = float(chunk.get('resampling_importance_accuracy', -float('inf')))
            
            if acc_imp > max_score:
                max_score = acc_imp
                anchor_chunk = chunk
                anchor_idx = i
                        
        return anchor_chunk, anchor_idx



    def find_top_k_anchors_bifurcation(self, chunks: List[Dict[str, Any]], k: int = 3) -> List[Tuple[Dict[str, Any], int, float]]:
        """
        Identifies top-K anchors using bifurcation entropy.
        """
        if not chunks:
            return []

        bifurcation_data = self.load_bifurcation_results()
        #select only frist 50% steps
        bifurcation_data = {k: v for k, v in bifurcation_data.items() if k < len(chunks) // 2}
        results = []

        if bifurcation_data:
            # Sort by entropy (descending)
            sorted_steps = sorted(bifurcation_data.items(), key=lambda x: x[1], reverse=True)
            
            for step_idx, entropy in sorted_steps:
                if len(results) >= k:
                    break
                if 0 <= step_idx < len(chunks):
                    anchor_chunk = chunks[step_idx]
                    results.append((anchor_chunk, step_idx, entropy))
                    self.console.print(f"[green]Top-K anchor {len(results)}: Chunk {step_idx} (entropy: {entropy:.4f})[/]")
        
        if not results:
             self.console.print("[yellow]No bifurcation data found, falling back to empty list[/]")
             
        # If we have results but less than k? that's fine.
        return results

    def find_top_k_anchors_random(self, chunks: List[Dict[str, Any]], k: int = 3) -> List[Tuple[Dict[str, Any], int, float]]:
        """
        Randomly selects K anchors from the first 50% of chunks, EXCLUDING those
        that would be selected by the standard method (bifurcation or importance).
        This serves as a control group for comparison.
        """
        if not chunks:
            return []
        
        # Get first 50% of chunk indices
        half_len = len(chunks) // 2
        available_indices = set(range(half_len))
        
        # Get indices selected by standard method and exclude them
        if self.config.anchor_selection_method == "bifurcation":
            standard_anchors = self.find_top_k_anchors_bifurcation(chunks, k)
        else:
            # Temporarily disable random to get standard anchors
            self.config.random_anchor = False
            standard_anchors = self.find_top_k_anchors(chunks, k)
            self.config.random_anchor = True
        
        excluded_indices = {anchor[1] for anchor in standard_anchors}
        available_indices -= excluded_indices
        
        self.console.print(f"[cyan]Random anchor selection: {len(available_indices)} candidates after excluding {len(excluded_indices)} standard anchors[/]")
        
        if len(available_indices) == 0:
            self.console.print("[yellow]No remaining anchors available for random selection[/]")
            return []
        
        # Randomly sample k indices (or all available if fewer)
        sample_size = min(k, len(available_indices))
        selected_indices = random.sample(list(available_indices), sample_size)
        
        results = []
        for idx in selected_indices:
            anchor_chunk = chunks[idx]
            results.append((anchor_chunk, idx, 0.0))  # 0.0 as placeholder score
            self.console.print(f"[magenta]Random anchor {len(results)}: Chunk {idx} (random selection)[/]")
        
        return results

    def find_top_k_anchors(self, chunks: List[Dict[str, Any]], k: int = 3, min_diversity: float = 0.1) -> List[Tuple[Dict[str, Any], int, float]]:
        """
        Identifies the top-K 'Thought Anchors' based on configured method.
        Returns list of (anchor_chunk, idx, score) tuples.
        """
        # Check for random anchor mode first
        if self.config.random_anchor:
            return self.find_top_k_anchors_random(chunks, k)
        
        if self.config.anchor_selection_method == "bifurcation":
            return self.find_top_k_anchors_bifurcation(chunks, k)

        # Default to importance method
        if not chunks:
            return []
        
        problem_id = str(self.config.problem_id)
        analysis_data = self.load_analysis_results()
        
        results = []
        
        if problem_id in analysis_data:
            chunks_info = analysis_data[problem_id]
            # select frist 50% chunks
            chunks_info = chunks_info[:len(chunks_info) // 2]
            # Sort by resampling_importance_accuracy (descending)
            chunks_info.sort(key=lambda x: x['resampling_importance_accuracy'], reverse=True)
            
            for info in chunks_info:
                if len(results) >= k:
                    break
                chunk_idx = info['chunk_idx']
                if 0 <= chunk_idx < len(chunks):
                    anchor_chunk = chunks[chunk_idx]
                    importance = info['resampling_importance_accuracy']
                    results.append((anchor_chunk, chunk_idx, importance))
                    self.console.print(f"[green]Top-K anchor {len(results)}: Chunk {chunk_idx} (resampling_importance: {importance:.4f})[/]")
        
        if not results:
            # Fallback: use resampling_importance_accuracy from chunks directly
            self.console.print("[yellow]No analysis data found, falling back to chunks data[/]")
            scored = []
            for i, chunk in enumerate(chunks):
                diversity = float(chunk.get('different_trajectories_fraction', 0.0))
                if diversity < min_diversity:
                    continue
                acc_imp = float(chunk.get('resampling_importance_accuracy', -float('inf')))
                scored.append((chunk, i, acc_imp))
            
            scored.sort(key=lambda x: x[2], reverse=True)  # Sort by importance descending
            results = scored[:k]
        
        return results

    def select_and_load(self) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
        """
        Executes Step 1: Select & Load from local directories.
        Returns: (anchor, chunks, question, correct_answer)
        """
        problem_dir = self._get_problem_dir()
        self.console.print(f"[bold blue]Loading data from {problem_dir}...[/]")
        
        if not os.path.exists(problem_dir):
            self.console.print(f"[bold red]Problem directory not found: {problem_dir}[/]")
            return None, None, None, None
        
        self.console.print(f"[bold blue]Loading Problem {self.config.problem_id} (Model: {self.config.model_name})...[/]")
        chunks, question, correct_answer = self.get_problem_data()
        
        if not chunks:
            self.console.print(f"[bold red]No data found for Problem {self.config.problem_id}.[/]")
            return None, None, None, None
            
        if question:
            self.console.print(Panel(Markdown(question), title="Problem Statement", border_style="green"))
        else:
            self.console.print(Panel("(Question not found)", title="Problem Statement", border_style="red"))
        
        if correct_answer:
            self.console.print(f"[bold green]Correct Answer: {correct_answer}[/]")
            
        self.console.print(f"Found {len(chunks)} chunks. Identifying Thought Anchor...")

        anchor, idx = self.find_thought_anchor(chunks)
        
        if anchor:
            self.console.print(Panel(Markdown(anchor.get('chunk')), title=f"Thought Anchor Identified (Step {idx})", subtitle=f"KL: {anchor.get('resampling_importance_kl', 0):.4f}", border_style="yellow"))
            return anchor, chunks, question, correct_answer
        else:
            self.console.print("[bold red]Could not identify Thought Anchor.[/]")
            return None, chunks, question, correct_answer
