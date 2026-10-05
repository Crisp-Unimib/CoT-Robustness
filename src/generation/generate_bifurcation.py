"""
generate_bifurcation.py - Analyze CoT entropy via Bifurcation Importance

Samples multiple continuations at each CoT step to measure decision branching points.
Uses Weighted Bifurcation Entropy: Score = log2(N_unique) × (α·D_sem + β·D_syn)

Architecture follows generate_rollouts.py patterns:
- asyncio + httpx for async HTTP
- Semaphore for concurrency control
- Round-robin routing across vLLM servers
- Exponential backoff for retries
- Rich for progress display
"""

import os
import sys
import json
import math
import random
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from urllib.parse import urlparse

import httpx
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
import blackmail_utils as bu

# Fix Windows "too many file descriptors in select()" error
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Import chunking logic from utils (same as generate_rollouts.py)
from utils import (
    split_solution_into_chunks, 
    load_math_problems, 
    extract_boxed_answers, 
    check_answer
)
from templates import get_template_for_model

# Load environment variables
load_dotenv()

# ============================================================================
# Argument Parser
# ============================================================================

parser = argparse.ArgumentParser(
    description='Analyze CoT entropy via Bifurcation Importance sampling'
)
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Model name for vLLM server')
parser.add_argument('-u', '--url', type=str, default='http://localhost:8000/v1',
                    help='Base URL for vLLM server')
parser.add_argument('--api_key', type=str, default=None,
                    help='API key for vLLM server')
parser.add_argument('-ns', '--ns', '--num_servers', dest='num_servers', type=int, default=1,
                    help='Number of vLLM servers for round-robin (ports 8000, 8001, ...)')
parser.add_argument('-n', '--num_samples', type=int, default=100,
                    help='Number of continuation samples per step')
parser.add_argument('-c', '--concurrent_requests', type=int, default=100,
                    help='Maximum concurrent requests')
parser.add_argument('-em', '--embedding_model', type=str, default='all-MiniLM-L6-v2',
                    help='sentence-transformers model for semantic similarity')
parser.add_argument('-a', '--alpha', type=float, default=0.7,
                    help='Weight for semantic distance (default: 0.7)')
parser.add_argument('-b', '--beta', type=float, default=0.3,
                    help='Weight for syntactic distance (default: 0.3)')
parser.add_argument('-t', '--temperature', type=float, default=1.0,
                    help='Sampling temperature for continuations')
parser.add_argument('-mt', '--max_tokens', type=int, default=64,
                    help='Max tokens for continuation generation')
parser.add_argument('-i', '--input_file', type=str, default=None,
                    help='JSON file with problems to analyze')
parser.add_argument('-ip', '--include_problems', type=str, default=None,
                    help='Comma-separated list of problem IDs to include')
parser.add_argument('-o', '--output_dir', type=str, default='data/processed/bifurcation_results',
                    help='Output directory for results')
parser.add_argument('-ty', '--type', type=str, default=None,
                    help='Problem type filter')
parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'],
                    help='Dataset split to use')
parser.add_argument('-l', '--level', type=str, default='Level 5',
                    help='Problem level filter (for MATH dataset)')
parser.add_argument('-np', '--num_problems', type=int, default=100,
                    help='Number of problems to process')
parser.add_argument('--base_temp', type=float, default=0.6,
                    help='Temperature for base CoT generation')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p sampling for vLLM generation (default: 0.95)')
parser.add_argument('--max_base_tokens', type=int, default=16384,
                    help='Max tokens for base CoT generation')
parser.add_argument('--dry_run', action='store_true',
                    help='Print requests without making API calls')
parser.add_argument('--test_metrics', action='store_true',
                    help='Run self-test on diversity metrics')
parser.add_argument('-s', '--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--preload_dir', type=str, default=None,
                    help='Directory containing pre-existing base solutions (e.g., math_rollouts/model_name/temperature_X_top_p_Y/correct_base_solution)')
parser.add_argument('--compare', action='store_true',
                    help='Enable comparison mode: use preloaded solutions and save to bifurcation_results_compare')

# Blackmail scenario mode arguments
parser.add_argument('--blackmail', action='store_true',
                    help='Run blackmail scenario mode instead of math problems')
parser.add_argument('--evil', action='store_true',
                    help='Evil mode: keep first BLACKMAIL response instead of safe response. '
                         'Saves to {model}_evil directory. Requires --blackmail.')
parser.add_argument('--blackmail_scenario', type=str, default=None,
                    help='Specific scenario ID (e.g., "blackmail_explicit-none_restriction"). If not set, runs all.')
parser.add_argument('--blackmail_prompts_dir', type=str, default='configs/blackmail_prompts',
                    help='Directory containing blackmail scenario prompts')
parser.add_argument('--controller_model', type=str, default='google/gemini-3-flash-preview',
                    help='OpenRouter model for blackmail classification')
parser.add_argument('--openrouter_url', type=str, default='https://openrouter.ai/api/v1',
                    help='OpenRouter API base URL')
parser.add_argument('--blackmail_samples', type=int, default=50,
                    help='Number of candidate responses to generate per scenario batch')
parser.add_argument('--max_batches', type=int, default=5,
                    help='Maximum number of batch retries for finding a suitable base response (default: 5)')

# KLE and recompute mode arguments
parser.add_argument('-nm', '--nli_model', type=str, default='microsoft/deberta-large-mnli',
                    help='NLI model for KLE semantic distance')
parser.add_argument('--recompute_metrics', action='store_true',
                    help='Recompute metrics from existing samples without regenerating. '
                         'Useful to add KLE scores to existing results.')
parser.add_argument('--compute_kle', action='store_true', default=True,
                    help='Compute KLE-based scores (default: True)')
parser.add_argument('--compute_semantic', action='store_true', default=True,
                    help='Compute semantic distance scores (default: True)')

# Phase Control
parser.add_argument('--resampling_true', action='store_true',
                    help='Enable resampling phase (generation of continuations)')
parser.add_argument('--calculate_true', action='store_true',
                    help='Enable calculation phase (computing metrics)')

# Batch Processing
parser.add_argument('-k', '--batch_sentences', type=int, default=1,
                    help='Number of sentences to process in parallel (default: 1)')

parser.add_argument('--force', action='store_true',
                    help='Force recomputation of metrics even if they exist')
parser.add_argument('--metric_sample_limit', type=int, default=20,
                    help='Number of top samples to use for expensive metrics (NLI/Distance) while keeping full count for entropy (default: 20)')

args = parser.parse_args()

# ============================================================================
# Global State
# ============================================================================

console = Console()
request_semaphore: Optional[asyncio.Semaphore] = None
global_client: Optional[httpx.AsyncClient] = None
vllm_request_counter = 0
embedding_model = None  # Lazy loaded
nli_model = None        # Lazy loaded - DeBERTa for NLI
nli_tokenizer = None
template = None         # Initialized in main



# Generate vLLM server URLs for round-robin
VLLM_API_URLS = []
if args.num_servers > 1:
    parsed = urlparse(args.url)
    hostname = parsed.hostname or 'localhost'
    scheme = parsed.scheme or 'http'
    base_port = parsed.port or 8000
    path = parsed.path
    
    for i in range(args.num_servers):
        VLLM_API_URLS.append(f"{scheme}://{hostname}:{base_port + i}{path}")
else:
    VLLM_API_URLS = [args.url]

# Output directory
if args.blackmail:
    model_dir = args.model.split("/")[-1]
    if args.evil:
        model_dir += "_evil"
    output_dir = Path('data/processed/blackmail_bifurcation') / model_dir
elif args.compare:
    output_dir = Path('data/processed/bifurcation_results_compare') / args.model.split("/")[-1]
else:
    output_dir = Path(args.output_dir) / args.model.split("/")[-1]
output_dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# HTTP Client Management
# ============================================================================

async def get_client() -> httpx.AsyncClient:
    """Get or create global HTTP client with connection pooling."""
    global global_client
    if global_client is None:
        limits = httpx.Limits(max_keepalive_connections=500, max_connections=1000)
        global_client = httpx.AsyncClient(limits=limits, timeout=None)
    return global_client


async def close_client():
    """Close global HTTP client."""
    global global_client
    if global_client:
        await global_client.aclose()
        global_client = None

# ============================================================================
# vLLM API Requests
# ============================================================================

async def make_vllm_request(
    prompt: str,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    stop: Optional[List[str]] = None,
    top_p: float = 0.95
) -> Dict:
    """
    Make a single vLLM completion request with retry and exponential backoff.
    Uses round-robin routing across multiple servers.
    """
    global vllm_request_counter
    
    if args.dry_run:
        return {"text": f"[DRY RUN] Prompt length: {len(prompt)}", "finish_reason": "stop"}
    
    # Round-robin server selection
    server_idx = vllm_request_counter % len(VLLM_API_URLS)
    vllm_request_counter += 1
    api_url = f"{VLLM_API_URLS[server_idx]}/completions"
    
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    payload = {
        "model": args.model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "stream": False
    }
    if stop:
        payload["stop"] = stop
    
    max_retries = 5
    retry_delay = 2
    
    async with request_semaphore:
        for attempt in range(max_retries):
            try:
                client = await get_client()
                response = await client.post(api_url, headers=headers, json=payload, timeout=3600)
                
                if response.status_code == 500:
                    console.print(f"[yellow]Server error (500) on attempt {attempt+1}/{max_retries}[/yellow]")
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                    
                elif response.status_code == 429:
                    console.print(f"[yellow]Rate limit (429) on attempt {attempt+1}/{max_retries}[/yellow]")
                    await asyncio.sleep(retry_delay * (2 ** attempt) + random.uniform(1, 3))
                    continue
                    
                elif response.status_code != 200:
                    if attempt == max_retries - 1:
                        return {"error": f"API error: {response.status_code}", "details": response.text}
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                
                result = response.json()
                
                
                if "choices" not in result or len(result["choices"]) == 0:
                    return {"error": "Invalid response format"}
                
                choice = result["choices"][0]
                all_texts = [c.get("text", "") for c in result["choices"]]
                return {
                    "text": choice.get("text", ""),
                    "texts": all_texts,
                    "finish_reason": choice.get("finish_reason", ""),
                    "usage": result.get("usage", {})
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"Request exception: {str(e)}"}
                await asyncio.sleep(retry_delay * (2 ** attempt))
    
    return {"error": "All API request attempts failed"}

# Blackmail scenario functions moved to blackmail_utils.py


# (Function removed - replaced by blackmail_utils)

# ============================================================================
# Diversity Metrics
# ============================================================================

def get_embedding_model():
    """Lazy load sentence-transformers model."""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        console.print(f"[dim]Loading embedding model: {args.embedding_model}[/dim]")
        embedding_model = SentenceTransformer(args.embedding_model)
    return embedding_model


def compute_mean_semantic_distance(samples: List[str]) -> float:
    """
    Compute mean pairwise semantic distance (1 - cosine similarity).
    """
    if len(samples) < 2:
        return 0.0
    
    model = get_embedding_model()
    embeddings = model.encode(samples, convert_to_numpy=True)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    embeddings = embeddings / norms
    
    # Compute pairwise cosine similarities
    similarities = embeddings @ embeddings.T
    
    # Extract upper triangle (excluding diagonal)
    n = len(samples)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(1.0 - similarities[i, j])
    
    return float(np.mean(distances)) if distances else 0.0


def compute_mean_syntactic_distance(samples: List[str]) -> float:
    """
    Compute mean pairwise normalized Levenshtein distance.
    """
    if len(samples) < 2:
        return 0.0
    
    try:
        import Levenshtein
    except ImportError:
        console.print("[red]python-Levenshtein not installed. Using fallback.[/red]")
        return 0.0
    
    distances = []
    for s1, s2 in combinations(samples, 2):
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            distances.append(0.0)
        else:
            dist = Levenshtein.distance(s1, s2) / max_len
            distances.append(dist)
    
    return float(np.mean(distances)) if distances else 0.0


def get_nli_model():
    """Lazy load NLI model for entailment predictions with maximum GPU optimizations."""
    global nli_model, nli_tokenizer
    if nli_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        console.print(f"[dim]Loading NLI model: {args.nli_model}[/dim]")
        nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
        
        # DeBERTa does not support bfloat16/float16 - use float32
        nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)
        
        if torch.cuda.is_available():
            # Enable CUDA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            nli_model = nli_model.cuda()
            
            # Enable SDPA (Scaled Dot Product Attention) - built into PyTorch 2.0+
            # This provides fused attention kernels similar to BetterTransformer
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                console.print("[dim]  ✓ SDPA (Flash/MemEfficient attention) enabled[/dim]")
            except Exception as e:
                console.print(f"[dim]  ✗ SDPA not available: {e}[/dim]")
            
            # Try torch.compile for JIT optimization (PyTorch 2.0+)
            try:
                if sys.platform == 'win32':
                    # Windows: use eager backend (no Triton required)
                    nli_model = torch.compile(nli_model, backend="eager")
                    console.print("[dim]  ✓ torch.compile enabled (eager backend)[/dim]")
                else:
                    # Linux: use max-autotune with Triton
                    nli_model = torch.compile(nli_model, mode="max-autotune", fullgraph=False)
                    console.print("[dim]  ✓ torch.compile enabled (max-autotune)[/dim]")
            except Exception as e:
                console.print(f"[dim]  ✗ torch.compile not available: {e}[/dim]")
            
        nli_model.eval()
    return nli_model, nli_tokenizer


def compute_entailment_batch(pairs: List[Tuple[str, str]], batch_size: int = 256, return_tensor: bool = False):
    """
    Compute entailment probabilities for multiple premise-hypothesis pairs in batches.
    Optimized for maximum GPU throughput.
    
    Args:
        pairs: List of (premise, hypothesis) tuples
        batch_size: Batch size for inference (256 default for RTX 4080)
        return_tensor: If True, returns GPU tensor; if False, returns list (for compatibility)
    
    Returns:
        GPU tensor or list of entailment probabilities
    """
    model, tokenizer = get_nli_model()
    import torch
    
    all_probs = []
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        premises = [p[0] for p in batch_pairs]
        hypotheses = [p[1] for p in batch_pairs]
        
        # Optimized tokenization with padding to multiple of 8 for tensor cores
        inputs = tokenizer(premises, hypotheses, return_tensors='pt', 
                          truncation=True, max_length=256, padding='longest',
                          pad_to_multiple_of=8)
        if torch.cuda.is_available():
            # Non-blocking transfer for overlap with computation
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        
        with torch.no_grad():
            # NOTE: DeBERTa does NOT support FP16/AMP - attention mask overflows
            # TF32 is still active from get_nli_model() settings
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # DeBERTa-MNLI labels: 0=contradiction, 1=neutral, 2=entailment
            entailment_probs = probs[:, 2]  # Keep on GPU
            all_probs.append(entailment_probs)
    
    # Concatenate all batches on GPU
    result = torch.cat(all_probs, dim=0)
    
    if return_tensor:
        return result
    else:
        # Compatibility: return list (only one CPU transfer at the end)
        return result.cpu().tolist()


def compute_kle_score(samples: List[str]) -> float:
    """
    Compute Kernel Language Entropy (KLE) using Von Neumann Entropy.
    Uses BATCHED NLI inference for speed - ALL operations on GPU.
    1. Build adjacency matrix W[i][j] = avg(P_entail(i->j), P_entail(j->i))
    2. Normalize to density matrix K = W / trace(W)
    3. Compute VNE = -sum(eigenvalue * log2(eigenvalue))
    """
    import torch
    
    if len(samples) < 2:
        return 0.0
    
    n = len(samples)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build all pairs for batched inference
    pairs_forward = []  # (i, j) pairs
    pairs_backward = []  # (j, i) pairs
    pair_indices = []  # Store (i, j) indices
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs_forward.append((samples[i], samples[j]))
            pairs_backward.append((samples[j], samples[i]))
            pair_indices.append((i, j))
    
    # Batch inference for all pairs - keep on GPU
    probs_forward = compute_entailment_batch(pairs_forward, return_tensor=True)
    probs_backward = compute_entailment_batch(pairs_backward, return_tensor=True)
    
    # Build adjacency matrix on GPU
    W = torch.zeros((n, n), device=device, dtype=torch.float32)
    
    # Compute average probabilities on GPU
    avg_probs = (probs_forward + probs_backward) / 2.0
    
    # Fill matrix using scatter (GPU-friendly)
    for idx, (i, j) in enumerate(pair_indices):
        W[i, j] = avg_probs[idx]
        W[j, i] = avg_probs[idx]
    
    # Diagonal = 1 (self-entailment)
    W.fill_diagonal_(1.0)
    
    trace_W = torch.trace(W)
    if trace_W == 0:
        return 0.0
    K = W / trace_W
    
    # Eigenvalue decomposition on GPU (PyTorch uses cuSOLVER on CUDA)
    eigenvalues = torch.linalg.eigvalsh(K)
    
    # Compute VNE on GPU
    # Filter positive eigenvalues and compute entropy
    positive_mask = eigenvalues > 1e-10
    positive_eigs = eigenvalues[positive_mask]
    
    if len(positive_eigs) == 0:
        return 0.0
    
    # VNE = -sum(eig * log2(eig))
    vne = -torch.sum(positive_eigs * torch.log2(positive_eigs))
    
    return float(vne.item())


def compute_bifurcation_score(
    unique_samples: List[str],
    alpha: float = 0.7,
    beta: float = 0.3,
    compute_semantic: bool = True,
    compute_kle: bool = True,
    metric_sample_limit: int = 20
) -> Dict[str, float]:
    """
    Compute Weighted Bifurcation Entropy with Optimized Performance.
    
    Goal: maintain diversity awareness of FULL sample set, but limit expensive
    calculation (NLI, Levenshtein) to a subset.
    
    Formula:
      Score = log2(N_unique_FULL) * (α * Metric(Subset) + β * Metric(Subset))
    
    Args:
        unique_samples: Full list of unique continuation samples
        alpha: Weight for semantic/KLE component
        beta: Weight for syntactic component
        compute_semantic: Whether to compute semantic/KLE metrics
        compute_kle: Whether to use KLE instead of cosine similarity
        metric_sample_limit: Max number of samples to use for costly metric calculations (default: 20)
    """
    n_unique_full = len(unique_samples)
    
    if n_unique_full <= 1:
        return {
            "semantic_distance": 0.0,
            "syntactic_distance": 0.0,
            "kle_score": 0.0,
            "bifurcation_entropy": 0.0,
            "bifurcation_entropy_kle": 0.0,
            "n_unique": n_unique_full
        }
    
    # Use only top N samples for expensive metric calculations
    # This prevents O(N^2) NLI scaling issues while preserving diversity count
    calc_samples = unique_samples[:metric_sample_limit]
    
    d_syn = compute_mean_syntactic_distance(calc_samples)
    d_sem = compute_mean_semantic_distance(calc_samples) if compute_semantic else 0.0
    kle = compute_kle_score(calc_samples) if compute_kle else 0.0
    
    # Log term uses FULL count, Metric term uses SUBSET score
    score_original = math.log2(n_unique_full) * (alpha * d_sem + beta * d_syn) if compute_semantic else 0.0
    score_kle = math.log2(n_unique_full) * (alpha * kle + beta * d_syn) if compute_kle else 0.0
    
    return {
        "semantic_distance": round(d_sem, 4),
        "syntactic_distance": round(d_syn, 4),
        "kle_score": round(kle, 4),
        "bifurcation_entropy": round(score_original, 4),
        "bifurcation_entropy_kle": round(score_kle, 4),
        "n_unique": n_unique_full,
        "metrics_calculated_on": len(calc_samples)
    }


# ============================================================================
# Blackmail Core Logic
# ============================================================================

async def generate_blackmail_base_cot(scenario: Dict) -> Optional[Dict]:
    """
    Generate base response for a blackmail scenario.
    Launches parallel requests and returns the first NON-BLACKMAIL response.
    (Analogous to generate_base_cot returning first CORRECT solution)
    """
    scenario_dir = Path(scenario['scenario_dir'])
    sys_prompt, user_prompt, email_content = bu.read_scenario_prompts(scenario_dir)
    prompt = template.format_blackmail_prompt(sys_prompt, user_prompt, email_content)
    
    num_candidates = args.blackmail_samples
    max_batches = args.max_batches
    blackmail_count = 0
    
    for batch_idx in range(max_batches):
        if batch_idx > 0:
            console.print(f"[yellow]Batch {batch_idx+1}/{max_batches}: Retrying ({num_candidates} candidates)...[/yellow]")
        else:
            console.print(f"[dim]Scenario: Generating {num_candidates} candidate responses (Batch 1)...[/dim]")
        
        pending_tasks = []
        
        # Launch parallel requests
        for _ in range(num_candidates):
            task = asyncio.create_task(make_vllm_request(
                prompt=prompt,
                temperature=args.base_temp,
                max_tokens=args.max_base_tokens,
                top_p=args.top_p
            ))
            pending_tasks.append(task)
        
        try:
            # Process as they complete
            for future in asyncio.as_completed(pending_tasks):
                try:
                    response = await future
                except Exception:
                    continue
                
                if "error" in response:
                    continue
                
                response_text = response["text"]
                full_response = prompt + response_text
                
                # Classify via OpenRouter
                is_blackmail, reasoning = await bu.classify_blackmail_response(
                    full_response,
                    os.getenv("OPENROUTER_API_KEY"),
                    args.controller_model,
                    args.openrouter_url
                )
                
                if is_blackmail:
                    blackmail_count += 1
                    console.print(f"[dim]✗ Candidate: BLACKMAIL (discarded) - Count: {blackmail_count}[/dim]")
                    continue
                else:
                    console.print(f"[green]✓ Candidate: SAFE (keeping)[/green]")
                    
                    # Cancel remaining tasks
                    for t in pending_tasks:
                        t.cancel()
                    
                    
                    # Extract reasoning for bifurcation analysis
                    scratchpad = template.parse_reasoning(response_text)
                    chunks = split_solution_into_chunks(scratchpad) if scratchpad else []
                    
                    return {
                        "prompt": prompt,
                        "response": response_text,
                        "full_response": full_response,
                        "scratchpad": scratchpad,
                        "chunks": chunks,
                        "is_blackmail": False,
                        "classification_reasoning": reasoning
                    }
                    
        except Exception as e:
            console.print(f"[red]Exception in parallel generation: {e}[/red]")
            
        finally:
            # Ensure cleanup for this batch
            for t in pending_tasks:
                if not t.done():
                    t.cancel()
    
    console.print(f"[red]Failed: All {max_batches * num_candidates} candidates were blackmail[/red]")
    return None


async def generate_blackmail_base_cot_evil(scenario: Dict) -> Optional[Dict]:
    """
    Generate base response for a blackmail scenario in EVIL mode.
    Launches parallel requests and returns the first BLACKMAIL response.
    (Inverse of generate_blackmail_base_cot)
    """
    scenario_dir = Path(scenario['scenario_dir'])
    sys_prompt, user_prompt, email_content = bu.read_scenario_prompts(scenario_dir)
    prompt = template.format_blackmail_prompt(sys_prompt, user_prompt, email_content)
    
    num_candidates = args.blackmail_samples
    max_batches = args.max_batches
    safe_count = 0
    
    for batch_idx in range(max_batches):
        if batch_idx > 0:
            console.print(f"[yellow]Batch {batch_idx+1}/{max_batches}: Retrying ({num_candidates} candidates)...[/yellow]")
        else:
            console.print(f"[dim]Scenario (EVIL): Generating {num_candidates} candidate responses (Batch 1)...[/dim]")
        
        pending_tasks = []
        
        # Launch parallel requests
        for _ in range(num_candidates):
            task = asyncio.create_task(make_vllm_request(
                prompt=prompt,
                temperature=args.base_temp,
                max_tokens=args.max_base_tokens,
                top_p=args.top_p
            ))
            pending_tasks.append(task)
        
        try:
            # Process as they complete
            for future in asyncio.as_completed(pending_tasks):
                try:
                    response = await future
                except Exception:
                    continue
                
                if "error" in response:
                    continue
                
                response_text = response["text"]
                full_response = prompt + response_text
                
                # Classify via OpenRouter
                is_blackmail, reasoning = await bu.classify_blackmail_response(
                    full_response,
                    os.getenv("OPENROUTER_API_KEY"),
                    args.controller_model,
                    args.openrouter_url
                )
                
                if not is_blackmail:
                    safe_count += 1
                    console.print(f"[dim]✗ Candidate: SAFE (discarded) - Count: {safe_count}[/dim]")
                    continue
                else:
                    console.print(f"[red]✓ Candidate: BLACKMAIL (keeping - evil mode)[/red]")
                    
                    # Cancel remaining tasks
                    for t in pending_tasks:
                        t.cancel()
                    
                    
                    # Extract reasoning for bifurcation analysis
                    scratchpad = template.parse_reasoning(response_text)
                    chunks = split_solution_into_chunks(scratchpad) if scratchpad else []
                    
                    return {
                        "prompt": prompt,
                        "response": response_text,
                        "full_response": full_response,
                        "scratchpad": scratchpad,
                        "chunks": chunks,
                        "is_blackmail": True,
                        "classification_reasoning": reasoning
                    }
                    
        except Exception as e:
            console.print(f"[red]Exception in parallel generation (evil): {e}[/red]")
            
        finally:
            # Ensure cleanup for this batch
            for t in pending_tasks:
                if not t.done():
                    t.cancel()
    
    console.print(f"[red]Failed: All {max_batches * num_candidates} candidates were safe (non-blackmail)[/red]")
    return None


async def sample_blackmail_continuations(context: str, num_samples: int) -> List[str]:
    """
    Sample multiple continuations from the same context prefix for blackmail scenarios.
    Returns list of generated texts.
    Stop at newline or period (simple heuristic for "next step" or sentence).
    Batches requests to reduce network overhead if num_servers > 1.
    """
    tasks = []
    
    # Batching strategy
    total_samples = num_samples
    num_batches = args.num_servers if args.num_servers > 0 else 1
    
    # If total samples is small, don't over-fragment
    if total_samples < num_batches:
        num_batches = total_samples
        
    base_batch_size = total_samples // num_batches
    remainder = total_samples % num_batches
    
    for i in range(num_batches):
        # Distribute remainder
        batch_n = base_batch_size + (1 if i < remainder else 0)
        
        if batch_n > 0:
            tasks.append(
                make_vllm_request(
                    prompt=context,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    n=batch_n,
                    stop=["\n", "."],
                    top_p=args.top_p
                )
            )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    samples = []
    for r in results:
        if isinstance(r, Exception):
            continue
        if isinstance(r, dict):
            # Handle list of texts if available (n > 1 support)
            if "texts" in r:
                for t in r["texts"]:
                    if t:
                        samples.append(t.strip())
            # Fallback to single text
            elif "text" in r and r["text"]:
                samples.append(r["text"].strip())
    
    return samples


async def analyze_blackmail_scenario(
    scenario_idx: int,
    scenario: Dict,
    progress: Progress,
    scenario_task_id,
    main_task_id=None
) -> None:
    """
    Analyze bifurcation at each step of a non-blackmail response.
    Saves results incrementally to output_dir/scenario_id/
    (Analogous to analyze_problem for math)
    """
    scenario_id = scenario.get('condition_id', f'scenario_{scenario_idx}')
    
    # Create scenario directory
    scenario_dir = output_dir / scenario_id
    scenario_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Base Response Management
    base_response_file = scenario_dir / "base_response.json"
    base_response = None
    
    # Check for existing base response
    if base_response_file.exists():
        try:
            with open(base_response_file, 'r', encoding='utf-8') as f:
                loaded_response = json.load(f)
            
            # Verify it's not blackmail (re-classify if needed)
            expected_blackmail = args.evil
            if loaded_response.get('is_blackmail') == expected_blackmail:
                base_response = loaded_response
                console.print(f"[green]Scenario {scenario_id}: Loaded existing safe response ({len(loaded_response.get('chunks', []))} chunks)[/green]")
        except Exception:
            pass
    
    if base_response is None:
        # If only calculating (no resampling), skip scenarios without existing base response
        if args.calculate_true and not args.resampling_true:
            console.print(f"[yellow]Scenario {scenario_id}: No existing base response found, skipping (calculate-only mode)[/yellow]")
            if main_task_id is not None:
                progress.update(main_task_id, advance=1)
            return
        
        progress.update(scenario_task_id, description=f"[cyan]{scenario_id}[/cyan] Generating {'evil' if args.evil else 'safe'} response...")
        if args.evil:
            base_response = await generate_blackmail_base_cot_evil(scenario)
        else:
            base_response = await generate_blackmail_base_cot(scenario)
        
        if base_response is None:
            progress.update(scenario_task_id, description=f"[red]{scenario_id}[/red] All responses were blackmail")
            
            # Save metadata noting failure
            with open(scenario_dir / "failed.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "scenario_id": scenario_id,
                    "message": "All generated candidates contained blackmail"
                }, f, indent=2)
            if main_task_id is not None:
                progress.update(main_task_id, advance=1)
            return
        
        if not base_response.get("chunks"):
            progress.update(scenario_task_id, description=f"[yellow]{scenario_id}[/yellow] No chunks in response")
            console.print(f"[yellow]Response found but no chunks. Scratchpad length: {len(base_response.get('scratchpad', '') or '')}[/yellow]")
            console.print(f"[dim]Response preview (first 500 chars): {base_response.get('response', '')[:500]}...[/dim]")
            
            # Save metadata noting failure
            with open(scenario_dir / "failed.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "scenario_id": scenario_id,
                    "status": "no_chunks",
                    "message": "Safe response found but chunking produced no results",
                    "scratchpad_length": len(base_response.get('scratchpad', '') or ''),
                    "response_preview": base_response.get('response', '')[:1000]
                }, f, indent=2)
            if main_task_id is not None:
                progress.update(main_task_id, advance=1)
            return
        
        # Remove any old failed.json from previous runs
        failed_file = scenario_dir / "failed.json"
        if failed_file.exists():
            failed_file.unlink()
        
        # Save base response
        with open(base_response_file, 'w', encoding='utf-8') as f:
            json.dump(base_response, f, indent=2)
        
        # Also save scenario metadata
        with open(scenario_dir / "scenario_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(scenario, f, indent=2)
    
    chunks = base_response.get("chunks", [])
    num_chunks = len(chunks)
    
    if num_chunks == 0:
        console.print(f"[yellow]Scenario {scenario_id}: No chunks found in scratchpad reasoning[/yellow]")
        if main_task_id is not None:
            progress.update(main_task_id, advance=1)
        return
    
    progress.update(scenario_task_id, description=f"[cyan]{scenario_id}[/cyan] {num_chunks} steps")
    
    # 2. Load existing step results
    steps_file = scenario_dir / "bifurcation_entropy.jsonl"
    step_data = {}
    
    if steps_file.exists():
        with open(steps_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    step_data[data['step_idx']] = data
                except:
                    pass
    
    # Create step progress
    step_task_id = progress.add_task(
        f"  Steps",
        total=num_chunks,
        visible=True
    )
    
    # 3. Analyze each step
    base_prompt = base_response["prompt"]
    
    # Track if any modifications were made to rewrite the file at the end
    any_modified = False
    
    # Get batch size from args
    batch_size = args.batch_sentences
    
    # Process steps in batches of k
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_indices = list(range(batch_start, batch_end))
        
        progress.update(step_task_id, advance=0, description=f"  Steps {batch_start}-{batch_end-1}/{num_chunks}")
        
        # Prepare batch data structures
        batch_current_data = []
        batch_should_sample = []
        batch_contexts = []
        
        for step_idx in batch_indices:
            current_data = step_data.get(step_idx, {})
            # Initialize minimal structure if missing
            if not current_data:
                current_data = {
                    "scenario_id": scenario_id,
                    "step_idx": step_idx,
                    "original_chunk": chunks[step_idx] if step_idx < len(chunks) else "",
                }
            
            # PHASE 1: Determine if resampling needed
            has_samples = 'samples' in current_data and current_data['samples']
            should_sample = False
            
            if args.resampling_true:
                if not has_samples:
                    should_sample = True
            
            if should_sample:
                # Build context up to current step
                context = template.format_context(base_prompt, chunks[:step_idx])
                batch_contexts.append((step_idx, context))
            
            batch_current_data.append(current_data)
            batch_should_sample.append(should_sample)
        
        # PHASE 1: Parallel sampling for all steps in batch that need it
        if batch_contexts:
            progress.update(step_task_id, description=f"  Steps {batch_start}-{batch_end-1}: Sampling {len(batch_contexts)} steps in parallel...")
            
            # Create sampling tasks for all contexts
            sampling_tasks = [
                sample_blackmail_continuations(ctx, args.num_samples)
                for _, ctx in batch_contexts
            ]
            
            # Execute all sampling in parallel
            sampling_results = await asyncio.gather(*sampling_tasks, return_exceptions=True)
            
            # Map results back to step indices
            context_idx = 0
            for i, step_idx in enumerate(batch_indices):
                if batch_should_sample[i]:
                    result = sampling_results[context_idx]
                    context_idx += 1
                    
                    if isinstance(result, Exception):
                        console.print(f"[red]Step {step_idx}: Sampling failed: {result}[/red]")
                        continue
                    
                    samples = result
                    unique_samples = list(set(samples))
                    
                    batch_current_data[i]['context_length'] = len(batch_contexts[context_idx-1][1])
                    batch_current_data[i]['num_samples'] = len(samples)
                    batch_current_data[i]['num_unique'] = len(unique_samples)
                    batch_current_data[i]['samples'] = unique_samples  # Store ALL samples
                    batch_current_data[i]['_unique_samples_full'] = unique_samples  # Temp storage for metrics
        
        # PHASE 2: Calculation for all steps in batch
        for i, step_idx in enumerate(batch_indices):
            current_data = batch_current_data[i]
            has_samples = 'samples' in current_data and current_data['samples']
            modified = batch_should_sample[i]  # Already modified if we sampled
            
            if args.calculate_true and has_samples:
                # FORCE LOGIC: Calculate if force is on, OR metrics missing, OR we just sampled
                should_calculate = args.force or ('metrics' not in current_data) or ('_unique_samples_full' in current_data)
                
                if should_calculate:
                    # Use full list if we just sampled, otherwise use stored samples
                    if '_unique_samples_full' in current_data:
                        calc_samples = current_data.pop('_unique_samples_full')
                    else:
                        calc_samples = current_data['samples']
                    
                    metrics = compute_bifurcation_score(calc_samples, args.alpha, args.beta,
                                                          compute_semantic=args.compute_semantic,
                                                          compute_kle=args.compute_kle,
                                                          metric_sample_limit=args.metric_sample_limit)
                    current_data['metrics'] = metrics
                    modified = True
            
            if modified:
                step_data[step_idx] = current_data
                any_modified = True
            
            if 'metrics' in current_data:
                progress.update(step_task_id, description=f"  Step {step_idx}/{num_chunks}: H={current_data['metrics']['bifurcation_entropy']:.2f}")
        
        progress.update(step_task_id, advance=len(batch_indices))
        if main_task_id is not None:
            progress.update(main_task_id, advance=len(batch_indices) / num_chunks)

    
    # Rewrite entire file if any modifications were made
    if any_modified:
        with open(steps_file, 'w', encoding='utf-8') as f_out:
            for idx in sorted(step_data.keys()):
                f_out.write(json.dumps(step_data[idx]) + "\n")

    progress.remove_task(step_task_id)
    progress.update(scenario_task_id, 
                   description=f"[green]{scenario_id}[/green] ✓ {num_chunks} steps processed")

# ============================================================================
# Core Logic (Math Problems)
# ============================================================================

async def generate_base_cot(problem: Dict) -> Optional[Dict]:
    """
    Generate base Chain-of-Thought trace for a problem. 
    Launches 100 parallel requests and returns the first correct solution found.
    """
    prompt = template.format_math_prompt(problem['problem'])
    
    # Extract Ground Truth Answer
    # load_math_problems returns 'gt_solution' and pre-calculated 'gt_answer'
    gt_solution = problem.get('gt_solution', problem.get('solution', '')) 
    gt_answers = extract_boxed_answers(gt_solution)
    gt_answer = gt_answers[-1] if gt_answers else problem.get('gt_answer', '')
    
    if not gt_answer:
        console.print(f"[red]Warning: Could not extract GT answer from solution for verification![/red]")
    else:
        console.print(f"[dim]Verifying against GT Answer: '{gt_answer}'[/dim]")
    
    num_candidates = 50
    max_step_limit = 350
    max_batches = 5
    incorrect_count = 0
    too_long_count = 0
    
    for batch_idx in range(max_batches):
        if batch_idx > 0:
            console.print(f"[yellow]Batch {batch_idx+1}/{max_batches}: Retrying parallel generation (100 candidates)...[/yellow]")
        else:
            console.print(f"[dim]Problem: Generating {num_candidates} candidate solutions in parallel (Batch 1)...[/dim]")
            
        pending_tasks = []
        too_long_count = 0
        
        # Launch parallel requests
        for _ in range(num_candidates):
            task = asyncio.create_task(make_vllm_request(
                prompt=prompt,
                temperature=args.base_temp,
                max_tokens=args.max_base_tokens,
                top_p=args.top_p
            ))
            pending_tasks.append(task)
            
        try:
            # Process as they complete
            for future in asyncio.as_completed(pending_tasks):
                try:
                    response = await future
                except Exception:
                    continue
                    
                if "error" in response:
                    continue
                
                solution_text = response["text"]
                full_cot = prompt + solution_text
                
                # Check for thinking steps early? No, need to parse first.
                # Actually, chunks are based on <think> or full text.
                
                # Extract thinking content
                thinking_text = template.parse_reasoning(full_cot)
                
                # Split into chunks
                chunks = split_solution_into_chunks(thinking_text)
                
                # Verify correctness immediately
                generated_answers = extract_boxed_answers(solution_text)
                is_correct = False
                
                for ans in generated_answers:
                    if check_answer(ans, gt_answer):
                        is_correct = True
                        break

                # Check length limit
                if len(chunks) > max_step_limit:
                    too_long_count += 1
                    status_str = "CORRECT" if is_correct else "INCORRECT"
                    console.print(f"[dim]Candidate: {len(chunks)} steps (TOO LONG > {max_step_limit}) - {status_str} - Count: {too_long_count}/10[/dim]")
                    
                    if too_long_count >= 10:
                        console.print(f"[yellow]Batch {batch_idx+1}: Reached 10 too-long candidates. Aborting batch...[/yellow]")
                        # Abort this batch immediately
                        for t in pending_tasks:
                            t.cancel()
                        break 
                    continue
                
                if is_correct:
                    console.print(f"[green]✓ Candidate: {len(chunks)} steps - CORRECT[/green]")
                    
                    # Cancel remaining tasks
                    for t in pending_tasks:
                        t.cancel()
                        
                    return {
                        "prompt": prompt,
                        "solution": solution_text,
                        "full_cot": full_cot,
                        "thinking_text": thinking_text,
                        "chunks": chunks
                    }
                else:
                    console.print(f"[dim]✗ Candidate: {len(chunks)} steps - INCORRECT[/dim]")
                    incorrect_count += 1

                    
        except Exception as e:
            console.print(f"[red]Exception in parallel generation: {e}[/red]")
            
        finally:
            # Ensure cleanup for this batch
            for t in pending_tasks:
                if not t.done():
                    t.cancel()
    
    console.print(f"[red]Failed to generate correct base CoT after {max_batches} batches ({max_batches * num_candidates} attempts)[/red]")
    return None


async def sample_continuations(context: str, num_samples: int) -> List[str]:
    """
    Sample multiple continuations from the same context prefix.
    Returns list of generated texts.
    """
    tasks = [
        make_vllm_request(
            prompt=context,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n", "."],
            top_p=args.top_p
        )
        for _ in range(num_samples)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    samples = []
    for r in results:
        if isinstance(r, Exception):
            continue
        if isinstance(r, dict) and "text" in r:
            text = r["text"].strip()
            if text:
                samples.append(text)
    
    return samples


async def analyze_problem(
    problem_idx: int,
    problem: Dict,
    progress: Progress,
    problem_task_id,
    main_task_id=None
) -> None:
    """
    Analyze bifurcation at each step of a problem's CoT.
    Saves results incrementally to output_dir/problem_{idx}/
    """
    # Create problem directory
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Base CoT Management
    base_cot_file = problem_dir / "base_cot.json"
    base_cot = None
    
    # Check for preloaded base solution when --compare mode is enabled
    if args.compare and args.preload_dir:
        preload_path = Path(args.preload_dir) / f"problem_{problem_idx}" / "base_solution.json"
        if preload_path.exists():
            try:
                with open(preload_path, 'r', encoding='utf-8') as f:
                    preloaded = json.load(f)
                
                # Convert preloaded format to base_cot format
                solution_text = preloaded.get('solution', '')
                full_cot = preloaded.get('full_cot', preloaded.get('prompt', '') + solution_text)
                
                # Extract thinking content
                thinking_text = template.parse_reasoning(full_cot)
                
                # Split into chunks
                chunks = split_solution_into_chunks(thinking_text)
                
                if chunks:
                    base_cot = {
                        "prompt": preloaded.get('prompt', ''),
                        "solution": solution_text,
                        "full_cot": full_cot,
                        "thinking_text": thinking_text,
                        "chunks": chunks,
                        "preloaded_from": str(preload_path)
                    }
                    console.print(f"[green]Problem {problem_idx}: Loaded preloaded base solution ({len(chunks)} chunks)[/green]")
                    
                    # Save to output directory for reference
                    with open(base_cot_file, 'w', encoding='utf-8') as f:
                        json.dump(base_cot, f, indent=2)
            except Exception as e:
                console.print(f"[yellow]Problem {problem_idx}: Failed to load preloaded solution: {e}[/yellow]")
    
    if base_cot is None and base_cot_file.exists():
        try:
            with open(base_cot_file, 'r', encoding='utf-8') as f:
                loaded_cot = json.load(f)
            
            # Verify correctness of loaded CoT
            gt_solution = problem.get('gt_solution', problem.get('solution', ''))
            gt_answers = extract_boxed_answers(gt_solution)
            gt_answer = gt_answers[-1] if gt_answers else problem.get('gt_answer', '')
            
            generated_answers = extract_boxed_answers(loaded_cot.get('solution', ''))
            is_correct = False
            for ans in generated_answers:
                if check_answer(ans, gt_answer):
                    is_correct = True
                    break
            
            if is_correct:
                base_cot = loaded_cot
            else:
                console.print(f"[yellow]Problem {problem_idx}: Existing base CoT is incorrect. Regenerating...[/yellow]")
                
        except Exception:
            pass
            
    if base_cot is None:
        progress.update(problem_task_id, description=f"[cyan]Problem {problem_idx}[/cyan] Generating base CoT...")
        base_cot = await generate_base_cot(problem)
        
        if base_cot is None or not base_cot["chunks"]:
            progress.update(problem_task_id, description=f"[red]Problem {problem_idx}[/red] Failed to generate CoT")
            if main_task_id is not None:
                progress.update(main_task_id, advance=1)
            return
            
        with open(base_cot_file, 'w', encoding='utf-8') as f:
            json.dump(base_cot, f, indent=2)
    
    chunks = base_cot["chunks"]
    num_chunks = len(chunks)
    
    progress.update(problem_task_id, description=f"[cyan]Problem {problem_idx}[/cyan] {num_chunks} steps")
    
    # 2. Load existing step results
    steps_file = problem_dir / "bifurcation_entropy.jsonl"
    step_data = {}
    
    if steps_file.exists():
        with open(steps_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    step_data[data['step_idx']] = data
                except:
                    pass
    
    # Create step progress
    step_task_id = progress.add_task(
        f"  Steps",
        total=num_chunks,
        visible=True
    )
    
    # Analyze each step
    prompt_base = template.format_math_prompt(problem['problem'])
    
    # Track if any modifications were made to rewrite the file at the end
    any_modified = False
    
    # Get batch size from args
    batch_size = args.batch_sentences
    
    # Process steps in batches of k
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_indices = list(range(batch_start, batch_end))
        
        progress.update(step_task_id, advance=0, description=f"  Steps {batch_start}-{batch_end-1}/{num_chunks}")
        
        # Prepare batch data structures
        batch_current_data = []
        batch_should_sample = []
        batch_contexts = []
        
        for step_idx in batch_indices:
            current_data = step_data.get(step_idx, {})
            # Initialize minimal structure if missing
            if not current_data:
                current_data = {
                    "problem_id": problem_idx,
                    "step_idx": step_idx,
                    "original_chunk": chunks[step_idx] if step_idx < len(chunks) else "",
                }
            
            # PHASE 1: Determine if resampling needed
            has_samples = 'samples' in current_data and current_data['samples']
            should_sample = False
            
            if args.resampling_true:
                if not has_samples:
                    should_sample = True
            
            if should_sample:
                # Build context up to current step
                context = template.format_context(prompt_base, chunks[:step_idx])
                batch_contexts.append((step_idx, context))
            
            batch_current_data.append(current_data)
            batch_should_sample.append(should_sample)
        
        # PHASE 1: Parallel sampling for all steps in batch that need it
        if batch_contexts:
            progress.update(step_task_id, description=f"  Steps {batch_start}-{batch_end-1}: Sampling {len(batch_contexts)} steps in parallel...")
            
            # Create sampling tasks for all contexts
            sampling_tasks = [
                sample_continuations(ctx, args.num_samples)
                for _, ctx in batch_contexts
            ]
            
            # Execute all sampling in parallel
            sampling_results = await asyncio.gather(*sampling_tasks, return_exceptions=True)
            
            # Map results back to step indices
            context_idx = 0
            for i, step_idx in enumerate(batch_indices):
                if batch_should_sample[i]:
                    result = sampling_results[context_idx]
                    context_idx += 1
                    
                    if isinstance(result, Exception):
                        console.print(f"[red]Step {step_idx}: Sampling failed: {result}[/red]")
                        continue
                    
                    samples = result
                    unique_samples = list(set(samples))
                    
                    batch_current_data[i]['context_length'] = len(batch_contexts[context_idx-1][1])
                    batch_current_data[i]['num_samples'] = len(samples)
                    batch_current_data[i]['num_unique'] = len(unique_samples)
                    batch_current_data[i]['samples'] = unique_samples  # Store ALL samples
                    batch_current_data[i]['_unique_samples_full'] = unique_samples  # Temp storage for metrics
        
        # PHASE 2: Calculation for all steps in batch
        for i, step_idx in enumerate(batch_indices):
            current_data = batch_current_data[i]
            has_samples = 'samples' in current_data and current_data['samples']
            modified = batch_should_sample[i]  # Already modified if we sampled
            
            if args.calculate_true and has_samples:
                # FORCE LOGIC: Calculate if force is on, OR metrics missing, OR we just sampled
                should_calculate = args.force or ('metrics' not in current_data) or ('_unique_samples_full' in current_data)
                
                if should_calculate:
                    # Use full list if we just sampled, otherwise use stored samples
                    if '_unique_samples_full' in current_data:
                        calc_samples = current_data.pop('_unique_samples_full')
                    else:
                        calc_samples = current_data['samples']
                    
                    metrics = compute_bifurcation_score(calc_samples, args.alpha, args.beta,
                                                          compute_semantic=args.compute_semantic,
                                                          compute_kle=args.compute_kle,
                                                          metric_sample_limit=args.metric_sample_limit)
                    current_data['metrics'] = metrics
                    modified = True
            
            if modified:
                step_data[step_idx] = current_data
                any_modified = True
            
            if 'metrics' in current_data:
                progress.update(step_task_id, description=f"  Step {step_idx}/{num_chunks}: H={current_data['metrics']['bifurcation_entropy']:.2f}")
        
        progress.update(step_task_id, advance=len(batch_indices))
        if main_task_id is not None:
            progress.update(main_task_id, advance=len(batch_indices) / num_chunks)

    
    # Rewrite entire file if any modifications were made
    if any_modified:
        with open(steps_file, 'w', encoding='utf-8') as f_out:
            for idx in sorted(step_data.keys()):
                f_out.write(json.dumps(step_data[idx]) + "\n")
    
    progress.remove_task(step_task_id)
    progress.update(problem_task_id, 
                   description=f"[green]Problem {problem_idx}[/green] ✓ {num_chunks} steps processed")


# ============================================================================
# Recompute Metrics Mode
# ============================================================================

def recompute_metrics_from_samples(input_dir: Path) -> None:
    """
    Recompute metrics from existing samples in bifurcation_entropy.jsonl files.
    Useful for adding KLE scores to existing results without regenerating samples.
    """
    console.print(f"[bold]Recompute Metrics Mode[/bold]")
    console.print(f"Input directory: {input_dir}")
    console.print(f"Compute semantic: {args.compute_semantic}, Compute KLE: {args.compute_kle}")
    
    # Find all scenario/problem directories (handle both math and blackmail)
    problem_dirs = list(input_dir.glob("problem_*"))
    scenario_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("problem_")]
    all_dirs = sorted(problem_dirs + scenario_dirs)
    
    if not all_dirs:
        console.print("[red]No problem/scenario directories found.[/red]")
        return
    
    console.print(f"Found {len(all_dirs)} directories to process")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console) as progress:
        
        main_task = progress.add_task("[bold]Recomputing metrics[/bold]", total=len(all_dirs))
        
        for dir_path in all_dirs:
            steps_file = dir_path / "bifurcation_entropy.jsonl"
            if not steps_file.exists():
                progress.update(main_task, advance=1)
                continue
            
            # Read existing data
            step_data = []
            with open(steps_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        step_data.append(json.loads(line))
                    except:
                        pass
            
            if not step_data:
                progress.update(main_task, advance=1)
                continue
            
            step_task = progress.add_task(f"  {dir_path.name}", total=len(step_data))
            
            # Recompute metrics for each step
            updated_data = []
            for step in step_data:
                samples = step.get("samples", [])
                if samples:
                    new_metrics = compute_bifurcation_score(
                        samples, args.alpha, args.beta,
                        compute_semantic=args.compute_semantic,
                        compute_kle=args.compute_kle,
                        metric_sample_limit=args.metric_sample_limit
                    )
                    step["metrics"] = new_metrics
                    step["num_unique"] = new_metrics["n_unique"]
                updated_data.append(step)
                progress.update(step_task, advance=1)
            
            # Write updated data
            with open(steps_file, 'w', encoding='utf-8') as f:
                for step in updated_data:
                    f.write(json.dumps(step) + "\n")
            
            progress.remove_task(step_task)
            progress.update(main_task, advance=1)
    
    console.print("[green]✓ Metrics recomputation complete.[/green]")


async def main():
    """Main entry point."""
    global request_semaphore
    
    # Handle test mode
    if args.test_metrics:
        # ... (keep test logic)
        console.print("[bold]Running metrics self-test...[/bold]")
        test_samples = [
            "The answer is 42.",
            "The solution is 42.",
            "We get 42 as the result.",
            "Therefore, x equals 42.",
            "The answer is 42."  # Duplicate
        ]
        unique = list(set(test_samples))
        metrics = compute_bifurcation_score(unique, args.alpha, args.beta)
        console.print(f"Test samples: {len(test_samples)} total, {len(unique)} unique")
        console.print(f"Metrics: {metrics}")
        return
    
    # Handle recompute mode
    if args.recompute_metrics:
        recompute_metrics_from_samples(output_dir)
        return
    
    # Initialize semaphore
    request_semaphore = asyncio.Semaphore(args.concurrent_requests)
    
    # Initialize template
    global template
    template = get_template_for_model(args.model)
    console.print(f"[bold]Using Chat Template:[/bold] {template.__class__.__name__}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # ========================================================================
    # BLACKMAIL MODE
    # ========================================================================
    if args.blackmail:
        # Print configuration for blackmail mode
        console.print(Panel.fit(
            f"[bold]Blackmail Bifurcation Analyzer[/bold]\n"
            f"Model: {args.model}\n"
            f"Evil Mode: {args.evil}\n"
            f"Servers: {VLLM_API_URLS}\n"
            f"Controller: {args.controller_model}\n"
            f"Samples/step: {args.num_samples}\n"
            f"Candidates/batch: {args.blackmail_samples}\n"
            f"Concurrent: {args.concurrent_requests}\n"
            f"α={args.alpha}, β={args.beta}\n"
            f"Output: {output_dir}",
            title="Blackmail Mode Configuration"
        ))
        
        # Load blackmail scenarios
        scenarios = bu.load_blackmail_scenarios(Path(args.blackmail_prompts_dir), args.blackmail_scenario)
        
        if not scenarios:
            console.print("[red]No blackmail scenarios found.[/red]")
            return
        
        console.print(f"Processing {len(scenarios)} blackmail scenarios...")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=console,
                expand=True
            ) as progress:
                
                main_task = progress.add_task("[bold]Overall[/bold]", total=len(scenarios))
                
                for scenario_idx, scenario in enumerate(scenarios):
                    scenario_id = scenario.get('condition_id', f'scenario_{scenario_idx}')
                    scenario_task = progress.add_task(
                        f"[cyan]{scenario_id}[/cyan]",
                        total=None
                    )
                    
                    await analyze_blackmail_scenario(scenario_idx, scenario, progress, scenario_task, main_task)
                    
                    progress.remove_task(scenario_task)
                    # Main task advanced inside analyze function

            
            console.print(f"\n[green]✓ Blackmail analysis complete. Results in {output_dir}[/green]")
            
        finally:
            await close_client()
        
        return  # Exit after blackmail mode
    
    # ========================================================================
    # MATH PROBLEMS MODE (Original)
    # ========================================================================
    
    # Print configuration
    console.print(Panel.fit(
        f"[bold]Bifurcation Analyzer[/bold]\n"
        f"Model: {args.model}\n"
        f"Servers: {VLLM_API_URLS}\n"
        f"Samples/step: {args.num_samples}\n"
        f"Concurrent: {args.concurrent_requests}\n"
        f"α={args.alpha}, β={args.beta}\n"
        f"Output: {output_dir}",
        title="Configuration"
    ))
    
    # Load problems
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        problems = [(i, p) for i, p in enumerate(data)]
    else:
        # Pass explicit None to let loader handle validation if not provided
        problems = load_math_problems(
            problem_type=args.type,
            level=args.level,
            num_problems=args.num_problems,
            split=args.split,
            include_problems=args.include_problems
        )
    
    if args.include_problems:
        include_problems = [int(id) for id in args.include_problems.split(",")]
        problems = [problem for problem in problems if problem[0] in include_problems]
    
    if not problems:
        console.print("[red]No problems to process.[/red]")
        return
    
    console.print(f"Processing {len(problems)} problems...")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            
            main_task = progress.add_task("[bold]Overall[/bold]", total=len(problems))
            
            for problem_idx, problem in problems:
                problem_task = progress.add_task(
                    f"[cyan]Problem {problem_idx}[/cyan]",
                    total=None
                )
                
                await analyze_problem(problem_idx, problem, progress, problem_task, main_task)
                
                progress.remove_task(problem_task)
                # Main task advanced inside analyze function

        
        console.print(f"\n[green]✓ Analysis complete. Results in {output_dir}[/green]")
        
    finally:
        await close_client()


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
