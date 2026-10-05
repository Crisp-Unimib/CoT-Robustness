import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from openrouter import OpenRouter
from dotenv import load_dotenv
from prompts import DAG_PROMPT
import re
from collections import Counter, defaultdict
from utils import normalize_answer, split_solution_into_chunks
import math
import multiprocessing as mp
from functools import partial
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
from concurrent.futures import ThreadPoolExecutor, as_completed


# Class to hold arguments for importance calculation functions
class ImportanceArgs:
    """Class to hold arguments for importance calculation functions."""

    def __init__(
        self,
        use_absolute=False,
        forced_answer_dir=None,
        similarity_threshold=0.8,
        use_similar_chunks=True,
        use_abs_importance=False,
        top_chunks=100,
        use_prob_true=True,
        global_vocab=None,
        laplace_alpha=1e-9,
        cf_fraction_threshold=0.05,
    ):
        self.use_absolute = use_absolute
        self.forced_answer_dir = forced_answer_dir
        self.similarity_threshold = similarity_threshold
        self.use_similar_chunks = use_similar_chunks
        self.use_abs_importance = use_abs_importance
        self.top_chunks = top_chunks
        self.use_prob_true = use_prob_true
        self.global_vocab = global_vocab
        self.laplace_alpha = laplace_alpha
        # If > 0, when different_trajectories_fraction < this threshold, fallback CF metrics to resampling metrics
        self.cf_fraction_threshold = cf_fraction_threshold


# Set tokenizers parallelism to false to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache for the embedding model
embedding_model_cache = {}

IMPORTANCE_METRICS = [
    "resampling_importance_accuracy",
    "resampling_importance_kl",
    "counterfactual_importance_accuracy",
    "counterfactual_importance_kl",
    "forced_importance_accuracy",
    "forced_importance_kl",
]

# Math error categories for multiclass tracking (taxonomy of reasoning errors)
MATH_ERROR_CATEGORIES = [
    "Calculation Error (CA)",   # Calculation group - right formula, wrong result
    "Counting Error (CO)",      # Calculation group
    "Operator Error (OP)",      # Calculation group
    "Context Value Error (CV)", # Semantic group - hallucinating numbers
    "Unit Conversion Error (UC)",  # Semantic group
    "Formula Confusion (FC)",   # Semantic group
    "Missing Step (MS)",        # Logical group
    "Contradictory Step (CS)",  # Logical group
    "Hallucination (HA)"        # Logical group - adding fictitious constraints
]

# Color mapping for error category groups
MATH_ERROR_CATEGORY_COLORS = {
    # Calculation & Execution Errors (Reds/Oranges)
    "Calculation Error (CA)": "#e53935",   # Red
    "Counting Error (CO)": "#ff7043",      # Deep Orange
    "Operator Error (OP)": "#ff9800",      # Orange
    # Semantic & Contextual Errors (Blues/Purples)
    "Context Value Error (CV)": "#5c6bc0", # Indigo
    "Unit Conversion Error (UC)": "#42a5f5",  # Blue
    "Formula Confusion (FC)": "#ab47bc",   # Purple
    # Logical & Structural Errors (Greens/Teals)
    "Missing Step (MS)": "#26a69a",        # Teal
    "Contradictory Step (CS)": "#66bb6a",  # Green
    "Hallucination (HA)": "#009688"        # Dark Teal
}

model = "gpt-oss-120b"
# model = "deepseek-r1-distill-llama-8b"
smoothing = 1e-9
# smoothing = 0.5
smoothing = 1.0

parser = argparse.ArgumentParser(
    description="Analyze rollout data and label chunks"
)
parser.add_argument(
    "-ic",
    "--correct_rollouts_dir",
    type=str,
    default=f"math_rollouts/{model}/temperature_0.6_top_p_0.95/correct_base_solution",
    help="Directory containing correct rollout data",
)
parser.add_argument(
    "-ii",
    "--incorrect_rollouts_dir",
    type=str,
    default=f"math_rollouts/{model}/temperature_0.6_top_p_0.95/incorrect_base_solution",
    help="Directory containing incorrect rollout data",
)
parser.add_argument(
    "-icf",
    "--correct_forced_answer_rollouts_dir",
    type=str,
    default=f"math_rollouts/{model}/temperature_0.6_top_p_0.95/correct_base_solution_forced_answer",
    help="Directory containing correct rollout data with forced answers",
)
parser.add_argument(
    "-iif",
    "--incorrect_forced_answer_rollouts_dir",
    type=str,
    default=f"math_rollouts/{model}/temperature_0.6_top_p_0.95/incorrect_base_solution_forced_answer",
    help="Directory containing incorrect rollout data with forced answers",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=f"analysis/basic/{model}/alpha_{smoothing}",
    help="Directory to save analysis results (defaults to rollouts_dir)",
)
parser.add_argument(
    "-p",
    "--problems",
    type=str,
    default=None,
    help="Comma-separated list of problem indices to analyze (default: all)",
)
parser.add_argument(
    "-m",
    "--max_problems",
    type=int,
    default=None,
    help="Maximum number of problems to analyze",
)
parser.add_argument(
    "-a",
    "--absolute",
    default=False,
    action="store_true",
    help="Use absolute value for importance calculation",
)
parser.add_argument(
    "-f",
    "--force_relabel",
    default=False,
    action="store_true",
    help="Force relabeling of chunks",
)
parser.add_argument(
    "-fm",
    "--force_metadata",
    default=False,
    action="store_true",
    help="Force regeneration of chunk summaries and problem nicknames",
)
parser.add_argument(
    "-d",
    "--dag_dir",
    type=str,
    default=f"archive/analysis/math/{model}",
    help="Directory containing DAG-improved chunks for token frequency analysis",
)
parser.add_argument(
    "-t",
    "--token_analysis_source",
    type=str,
    default="dag",
    choices=["dag", "rollouts"],
    help='Source for token frequency analysis: "dag" for DAG-improved chunks or "rollouts" for rollout data',
)
parser.add_argument(
    "-tf",
    "--get_token_frequencies",
    default=False,
    action="store_true",
    help="Get token frequencies",
)
parser.add_argument(
    "-mc",
    "--max_chunks_to_show",
    type=int,
    default=100,
    help="Maximum number of chunks to show in plots",
)
parser.add_argument(
    "-tc",
    "--top_chunks",
    type=int,
    default=500,
    help="Number of top chunks to use for similar and dissimilar during counterfactual importance calculation",
)
parser.add_argument(
    "-u",
    "--use_existing_metrics",
    default=False,
    action="store_true",
    help="Use existing metrics from chunks_labeled.json without recalculating",
)
parser.add_argument(
    "-im",
    "--importance_metric",
    type=str,
    default="counterfactual_importance_accuracy",
    choices=IMPORTANCE_METRICS,
    help="Which importance metric to use for plotting and analysis",
)
parser.add_argument(
    "-sm",
    "--sentence_model",
    type=str,
    default="sentence-transformers/all-minilm-l6-v2",
    help="OpenRouter embedding model to use (e.g., 'Taurus')",
)
parser.add_argument(
    "-st",
    "--similarity_threshold",
    type=float,
    default=0.8,
    help="Similarity threshold for determining different chunks",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for embedding model",
)
parser.add_argument(
    "-us",
    "--use_similar_chunks",
    default=True,
    action="store_true",
    help="Use similar chunks for importance calculation",
)
parser.add_argument(
    "-np",
    "--num_processes",
    type=int,
    default=1,
    help="Number of parallel processes for chunk processing",
)
parser.add_argument(
    "-pt",
    "--use_prob_true",
    default=False,
    action="store_false",
    help="Use probability of correct answer (P(true)) instead of answer distribution for KL divergence calculations",
)
parser.add_argument(
    "--laplace_alpha",
    type=float,
    default=smoothing,
    help="Smoothing parameter for KL divergence calculation. Use 1e-9 (default) for minimal smoothing to prevent division by zero, or 1.0 for true Laplace smoothing with add-one smoothing",
)
parser.add_argument(
    "--use_global_vocab",
    default=False,
    action="store_true",
    help="If True, use global vocabulary from entire dataset for consistent Laplace smoothing across all KL divergence calculations. If False (default), use local vocabulary (union of answers in each pair of chunks)",
)
parser.add_argument(
    "-fce",
    "--force_classify_error",
    default=False,
    action="store_true",
    help="Force reclassification of all error types, even if already classified",
)
parser.add_argument(
    "--llm_provider",
    type=str,
    default="openrouter",
    choices=["openrouter", "vllm"],
    help="LLM provider to use for API calls (default: openrouter)",
)
parser.add_argument(
    "--vllm_base_url",
    type=str,
    default="http://localhost:8000/v1",
    help="Base URL for vLLM server (e.g., http://localhost:8000/v1)",
)
parser.add_argument(
    "--vllm_api_key",
    type=str,
    default=None,
    help="API key/bearer token for vLLM server (optional, uses 'EMPTY' if not specified)",
)
parser.add_argument(
    "--vllm_model",
    type=str,
    default=None,
    help="Model name to use with vLLM (required when using vLLM provider)",
)
parser.add_argument(
    "-fa",
    "--force_all",
    default=False,
    action="store_true",
    help="Force reclassification of everything: chunk labels, summaries, nicknames, and error types",
)

args = parser.parse_args()

# If force_all is set, enable all force flags
if args.force_all:
    args.force_relabel = True
    args.force_metadata = True
    args.force_classify_error = True
# print(f"{args=}")
# quit()
# min(100, mp.cpu_count()),
# Set consistent font size for all plots
FONT_SIZE = 20
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 4,
        "axes.labelsize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.titlesize": FONT_SIZE + 6,
    }
)
FIGSIZE = (20, 7)

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams.update(
    {
        "axes.labelpad": 20,  # Padding for axis labels
        "axes.titlepad": 20,  # Padding for plot titles
        "axes.spines.top": False,  # Hide top spine
        "axes.spines.right": False,  # Hide right spine
    }
)

# Load environment variables
load_dotenv()

# Set up OpenRouter client (for chat completions)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# OpenRouter headers for HTTP requests
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/trojan-thought",
    "X-Title": "Trojan Thought Analysis",
}

# Set up vLLM client if that provider is selected
vllm_client = None
vllm_semaphore = None  # Semaphore to limit concurrent vLLM requests
if args.llm_provider == "vllm":
    if not args.vllm_model:
        raise ValueError("--vllm_model is required when using vLLM provider")
    import threading
    vllm_semaphore = threading.Semaphore(100)  # Limit to 5 concurrent vLLM requests
    vllm_client = OpenAI(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key or "EMPTY",  # vLLM often doesn't require auth
        timeout=3600.0,  # 1 hour timeout
    )
    print(f"Using vLLM provider at {args.vllm_base_url} with model {args.vllm_model} (max 5 concurrent requests)")
else:
    if not openrouter_client.api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    print("Using OpenRouter provider")


def llm_chat_completion(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    response_format: dict = None,
) -> str:
    """
    Make a chat completion request to the active LLM provider.
    Handles differences between OpenRouter and vLLM.
    
    Args:
        prompt: The user prompt to send
        model: Model name (used for OpenRouter, overridden by --vllm_model for vLLM)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        response_format: Optional response format (e.g., {"type": "json_object"})
    
    Returns:
        The response content as a string
    """
    kwargs = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    if args.llm_provider == "vllm":
        # Use vLLM client
        kwargs["model"] = args.vllm_model
        kwargs["temperature"] = 0.6  # Override temperature for vLLM
        # Enable thinking and set low reasoning effort
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_effort": "low"
        }
        # Do not impose max_tokens for vLLM
        client_to_use = vllm_client
        
        # Use semaphore to limit concurrent vLLM requests
        with vllm_semaphore:
            if response_format:
                kwargs["response_format"] = response_format
            response = client_to_use.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
    else:
        # Use OpenRouter client
        kwargs["model"] = model
        kwargs["max_tokens"] = max_tokens  # Only set max_tokens for OpenRouter
        kwargs["extra_headers"] = OPENROUTER_HEADERS
        # Disable reasoning for OpenRouter
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        client_to_use = openrouter_client
    
    if response_format:
        kwargs["response_format"] = response_format
    
    response = client_to_use.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()

# Initialize the r1-distill-qwen-14b tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-r1-distill-qwen-14b"
)

# Define stopwords to filter out
stopwords = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "at",
    "from",
    "for",
    "with",
    "about",
    "into",
    "through",
    "above",
    "ve",
    "below",
    "under",
    "again",
    "further",
    "here",
    "there",
    "all",
    "most",
    "other",
    "some",
    "such",
    "to",
    "on",
    "only",
    "own",
    "too",
    "very",
    "will",
    "wasn",
    "weren",
    "wouldn",
    "this",
    "that",
    "these",
    "those",
    "of",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "got",
    "does",
    "did",
    "doing",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "get",
    "in",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "so",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "as",
    "us",
}


def count_tokens(text: str, approximate: bool = True) -> int:
    """Count the number of tokens in a text string using the r1-distill-qwen-14b tokenizer."""
    if approximate:
        return len(text) // 4
    else:
        return len(tokenizer.encode(text))


def get_openrouter_embeddings(
    texts: List[str], 
    model_name: str = "Taurus", 
    batch_size: int = 32,
    max_workers: int = 100
) -> np.ndarray:
    """Get embeddings from OpenRouter API using parallel requests.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the OpenRouter embedding model to use
        batch_size: Number of texts to process per API call
        max_workers: Maximum number of parallel threads
        
    Returns:
        numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])
    
    all_embeddings = {}
    
    def process_batch(batch_indices, batch_texts):
        batch_results = []
        with OpenRouter(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ) as open_router:
            for text in batch_texts:
                retries = 3
                backoff = 1.0
                embedding = None
                
                for attempt in range(retries):
                    try:
                        res = open_router.embeddings.generate(
                            input=text,
                            model=model_name
                        )
                        # Extract embedding from response
                        if hasattr(res, 'data') and len(res.data) > 0:
                            embedding = res.data[0].embedding
                        elif isinstance(res, dict) and 'data' in res and len(res['data']) > 0:
                            embedding = res['data'][0]['embedding']
                        else:
                            # Fallback: try to access directly
                            embedding = res.embedding if hasattr(res, 'embedding') else res
                        
                        # If successful, break retry loop
                        break
                    except Exception as e:
                        if attempt < retries - 1:
                            print(f"Error getting embedding (attempt {attempt+1}/{retries}): {e}. Retrying in {backoff}s...")
                            time.sleep(backoff)
                            backoff *= 2
                        else:
                            print(f"Failed to get embedding after {retries} attempts: {e}")

                if embedding is not None:
                    batch_results.append(np.array(embedding, dtype=np.float32))
                else:
                    # Return zero vector as fallback
                    batch_results.append(np.zeros(768, dtype=np.float32))
        return batch_indices, batch_results

    # Create batches
    batches = []
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_indices = list(range(i, end_idx))
        batch_texts = texts[i:end_idx]
        batches.append((batch_indices, batch_texts))

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, b[0], b[1]) for b in batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings (parallel)"):
            try:
                indices, results = future.result()
                for idx, embedding in zip(indices, results):
                    all_embeddings[idx] = embedding
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    # Determine dimension from first valid embedding
    embedding_dim = 384 # Default fallback for miniLM
    for idx in sorted(all_embeddings.keys()):
        emb = all_embeddings[idx]
        if isinstance(emb, np.ndarray) and emb.shape != (0,) and np.any(emb):
             if len(emb.shape) == 1:
                 embedding_dim = emb.shape[0]
                 break
    
    print(f"Detected embedding dimension: {embedding_dim}")

    # Reassemble and validate
    final_embeddings = []
    for i in range(len(texts)):
        if i in all_embeddings:
            emb = all_embeddings[i]
            # Check shape
            if isinstance(emb, np.ndarray) and emb.shape == (embedding_dim,):
                final_embeddings.append(emb)
            else:
                 # Wrong shape or invalid
                 final_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
        else:
            # Missing
            final_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))

    return np.array(final_embeddings, dtype=np.float32)


def cluster_texts_by_semantics(
    texts: List[str],
    chunk_embedding_cache: Dict[str, np.ndarray],
    similarity_threshold: float,
    model_name: str = "Taurus",
) -> List[List[int]]:
    """Greedy clustering based on cosine similarity of sentence embeddings."""
    if not texts:
        return []
    
    # Get embeddings for texts from cache or compute new ones
    embeddings_list = []
    texts_to_embed = []
    text_indices = []
    
    for i, text in enumerate(texts):
        if text in chunk_embedding_cache:
            embeddings_list.append((i, chunk_embedding_cache[text]))
        else:
            texts_to_embed.append(text)
            text_indices.append(i)
    
    # Get embeddings for texts not in cache
    if texts_to_embed:
        new_embeddings = get_openrouter_embeddings(texts_to_embed, model_name=model_name)
        for i, (text, emb) in enumerate(zip(texts_to_embed, new_embeddings)):
            chunk_embedding_cache[text] = emb
            embeddings_list.append((text_indices[i], emb))
    
    # Sort by original index and extract embeddings
    embeddings_list.sort(key=lambda x: x[0])
    embeddings = np.array([emb for _, emb in embeddings_list], dtype=np.float32)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    clusters: List[List[int]] = []
    representatives: List[int] = []
    for idx, emb in enumerate(embeddings):
        best_cluster = -1
        best_sim = -1.0
        for c_idx, rep_idx in enumerate(representatives):
            sim = float(np.dot(emb, embeddings[rep_idx]))
            if sim > best_sim:
                best_sim = sim
                best_cluster = c_idx
        if best_cluster >= 0 and best_sim >= similarity_threshold:
            clusters[best_cluster].append(idx)
        else:
            clusters.append([idx])
            representatives.append(idx)
    return clusters


def compute_semantic_entropy_for_texts(
    texts: List[str],
    chunk_embedding_cache: Dict[str, np.ndarray],
    similarity_threshold: float = 0.85,
    model_name: str = "Taurus",
) -> float:
    """Compute semantic entropy over clusters approximating meanings."""
    filtered = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if len(filtered) <= 1:
        return 0.0
    clusters = cluster_texts_by_semantics(
        filtered, chunk_embedding_cache, similarity_threshold, model_name=model_name
    )
    if not clusters:
        return 0.0
    total = float(len(filtered))
    probs = np.array([len(c) / total for c in clusters], dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def generate_chunk_summary(chunk_text: str) -> str:
    """
    Generate a 2-3 word summary of a chunk using OpenAI API.

    Args:
        chunk_text: The text content of the chunk

    Returns:
        A 2-4 word summary of what happens in the chunk
    """
    # Create a prompt to get a concise summary
    prompt = f"""Please provide a 2-4 word maximum summary of what specifically happens in this text chunk. Focus on the concrete action or calculation, not meta-descriptions like "planning" or "reasoning". 

    Examples:
    - "derive x=8" (for solving for a variable)
    - "suggest decimal conversion" (for recommending a calculation approach)
    - "calculate area=45" (for computing an area)
    - "check answer" (for verification)
    - "list possibilities" (for enumeration)
    
    The words should all be lowercase, with the exception of variable names, other proper nouns, or relevant math terms.
    
    Ideally, the summary should be a single sentence that captures the main action or calculation in the chunk.
    - If there is a variable involved, include the variable in the summary.
    - If there is a calculation involved, include the calculation in the summary.
    - If there is a number or value derived from a calculation, include the number or value in the summary.

    Text chunk:
    {chunk_text}

    Summary (2-4 words max):
    """

    try:
        summary = llm_chat_completion(
            prompt=prompt,
            model="mistralai/ministral-14b-2512",
            temperature=0.0,
            max_tokens=20,
        )

        # Ensure it's actually 2-3 words
        words = summary.split()
        if len(words) > 5:
            summary = " ".join(words[:5])

        return summary.replace('"', "")

    except Exception as e:
        print(f"Error generating chunk summary: {e}")
        return "unknown action"


def generate_problem_nickname(problem_text: str) -> str:
    """
    Generate a 2-3 word nickname for a problem using OpenAI API.

    Args:
        problem_text: The problem statement

    Returns:
        A 2-4 word nickname for the problem
    """
    # Create a prompt to get a concise nickname
    prompt = f"""Please provide a 2-4 word maximum nickname for this math problem that captures its essence. Focus on the main mathematical concept or scenario.

    Examples:
    - "Page counting" (for problems about counting digits in page numbers)
    - "Coin probability" (for probability problems with coins)
    - "Triangle area" (for geometry problems about triangles)
    - "Modular arithmetic" (for problems involving remainders)
    
    The first word should be capitalized and the rest should be lowercase.

    Problem:
    {problem_text}

    Nickname (2-4 words max):
    """

    try:
        nickname = llm_chat_completion(
            prompt=prompt,
            model="openai/gpt-4o-mini",
            temperature=0.0,
            max_tokens=20,
        )

        # Ensure it's actually 2-3 words
        words = nickname.split()
        if len(words) > 5:
            nickname = " ".join(words[:5])

        return nickname.replace('"', "")

    except Exception as e:
        print(f"Error generating problem nickname: {e}")
        return "math problem"


def classify_solution_error(
    problem_text: str,
    solution_text: str,
    ground_truth_answer: str,
    extracted_answer: str
) -> str:
    """
    Classify the type of math error in an incorrect solution using Gemini Flash.
    
    Args:
        problem_text: The original math problem
        solution_text: The model's reasoning/solution (full_cot)
        ground_truth_answer: The correct answer
        extracted_answer: The model's extracted (incorrect) answer
    
    Returns:
        One of the MATH_ERROR_CATEGORIES taxonomy codes, or empty string if no error
    """
    prompt = f"""You are an expert at analyzing mathematical reasoning errors. Given a math problem, the correct answer, and an incorrect solution, classify the PRIMARY error type.

## Error Taxonomy:
1. **Calculation Error (CA)** - Right formula/approach, but arithmetic mistake (e.g., 7+5=11)
2. **Counting Error (CO)** - Miscounting items, cases, or elements
3. **Operator Error (OP)** - Using wrong operator (+ instead of -, * instead of /)
4. **Context Value Error (CV)** - Hallucinating numbers not in the problem or using wrong values
5. **Unit Conversion Error (UC)** - Incorrect unit conversion (e.g., cm to m)
6. **Formula Confusion (FC)** - Using wrong formula for the situation
7. **Missing Step (MS)** - Skipping a required step in the solution
8. **Contradictory Step (CS)** - Step contradicts a previous step or given information
9. **Hallucination (HA)** - Adding fictitious constraints or information not in the problem

## Problem:
{problem_text}

## Correct Answer:
{ground_truth_answer}

## Model's Incorrect Answer:
{extracted_answer}

## Model's Solution:
{solution_text}

## Task:
Identify the PRIMARY error that led to the wrong answer. Respond with ONLY the error code in parentheses, e.g., "(CA)" or "(MS)".
If you cannot determine the error type, respond with "(HA)".

Error code:"""

    import re
    import time
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            result = llm_chat_completion(
                prompt=prompt,
                model="mistralai/ministral-14b-2512",
                temperature=0.0,
                max_tokens=10,
            )
            
            # Check if result is None (API returned empty content)
            if result is None:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  Retry {attempt + 1}/{max_retries}: Got None response, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  All {max_retries} retries failed (None response)")
                    return ""
            
            # Extract the error code from response
            # Look for pattern like "(CA)" or "CA"
            match = re.search(r'\(?(CA|CO|OP|CV|UC|FC|MS|CS|HA)\)?', result.upper())
            if match:
                code = match.group(1)
                # Map code to full category name
                code_to_category = {
                    "CA": "Calculation Error (CA)",
                    "CO": "Counting Error (CO)",
                    "OP": "Operator Error (OP)",
                    "CV": "Context Value Error (CV)",
                    "UC": "Unit Conversion Error (UC)",
                    "FC": "Formula Confusion (FC)",
                    "MS": "Missing Step (MS)",
                    "CS": "Contradictory Step (CS)",
                    "HA": "Hallucination (HA)"
                }
                return code_to_category.get(code, "")
            
            return ""
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{max_retries}: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error classifying solution error after {max_retries} attempts: {e}")
                return ""
    
    return ""


def label_chunk(problem_text: str, chunks: List[str], chunk_idx: int) -> Dict:
    """
    Label a chunk using OpenAI API with the DAG prompt.

    Args:
        problem_text: The problem text
        chunks: All chunks for context
        chunk_idx: The index of the chunk to label

    Returns:
        Dictionary with the label information
    """
    # Create the full chunked text with indices
    full_chunked_text = ""
    for i, chunk in enumerate(chunks):
        full_chunked_text += f"Chunk {i}:\n{chunk}\n\n"

    # Format the DAG prompt with the problem and chunks
    formatted_prompt = DAG_PROMPT.format(
        problem_text=problem_text, full_chunked_text=full_chunked_text
    )

    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            # Always use OpenRouter with Gemini for DAG labeling (handles long context better)
            response = openrouter_client.chat.completions.create(
                model="google/gemini-3-flash-preview",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_headers=OPENROUTER_HEADERS,
            )
            result_str = response.choices[0].message.content.strip()

            # Check for None or empty response
            if result_str is None or result_str.strip() == "":
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  Retry {attempt + 1}/{max_retries}: Empty response for chunk {chunk_idx}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ValueError("Empty response after all retries")

            result = json.loads(result_str)

            # Add the chunk and its index to the result
            result["chunk"] = chunks[chunk_idx]
            result["chunk_idx"] = chunk_idx

            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{max_retries}: Error labeling chunk {chunk_idx}: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error labeling chunk {chunk_idx} after {max_retries} attempts: {e}")
                return {
                    "categories": ["Unknown"],
                    "explanation": f"Error: {str(e)}",
                    "chunk": chunks[chunk_idx],
                    "chunk_idx": chunk_idx,
                }


def process_chunk_importance(
    chunk_idx,
    chunk_info,
    chunk_embedding_cache,
    chunk_accuracies,
    args,
    problem_dir=None,
    forced_answer_accuracies=None,
    chunk_answers=None,
):
    """Process importance metrics for a single chunk.

    Args:
        chunk_idx: Index of the chunk to process
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_accuracies: Dictionary of chunk accuracies
        args: Arguments containing processing parameters
        problem_dir: Path to the problem directory (needed for some metrics)
        forced_answer_accuracies: Dictionary of forced answer accuracies
        chunk_answers: Dictionary of chunk answers

    Returns:
        tuple: (chunk_idx, metrics_dict) where metrics_dict contains all calculated metrics
    """
    metrics = {}

    # Calculate counterfactual importance metrics
    cf_acc, different_trajectories_fraction, overdeterminedness = (
        calculate_counterfactual_importance_accuracy(
            chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, args
        )
    )
    cf_kl = calculate_counterfactual_importance_kl(
        chunk_idx,
        chunk_info,
        chunk_embedding_cache,
        chunk_accuracies,
        chunk_answers,
        args,
    )
    metrics.update(
        {
            "counterfactual_importance_accuracy": cf_acc,
            "counterfactual_importance_kl": cf_kl,
            "different_trajectories_fraction": different_trajectories_fraction,
            "overdeterminedness": overdeterminedness,
        }
    )

    # Calculate resampling importance metrics
    rs_acc = calculate_resampling_importance_accuracy(
        chunk_idx, chunk_accuracies, args
    )
    rs_kl = (
        calculate_resampling_importance_kl(
            chunk_idx, chunk_info, problem_dir, args
        )
        if problem_dir
        else 0.0
    )
    metrics.update(
        {
            "resampling_importance_accuracy": rs_acc,
            "resampling_importance_kl": rs_kl,
        }
    )

    # Fallback: if too few different trajectories for reliable CF, replace CF metrics with resampling metrics
    if (
        getattr(args, "cf_fraction_threshold", 0.0) is not None
        and args.cf_fraction_threshold > 0
        and different_trajectories_fraction < args.cf_fraction_threshold
    ):
        metrics["counterfactual_importance_accuracy"] = rs_acc
        metrics["counterfactual_importance_kl"] = rs_kl

    # Calculate forced importance metrics if forced answers available
    if (
        forced_answer_accuracies
        and hasattr(args, "forced_answer_dir")
        and args.forced_answer_dir
    ):
        forced_acc = calculate_forced_importance_accuracy(
            chunk_idx, forced_answer_accuracies, args
        )
        forced_kl = (
            calculate_forced_importance_kl(
                chunk_idx,
                forced_answer_accuracies,
                problem_dir,
                args.forced_answer_dir,
                args,
            )
            if problem_dir
            else 0.0
        )
        metrics.update(
            {
                "forced_importance_accuracy": forced_acc,
                "forced_importance_kl": forced_kl,
            }
        )

    return chunk_idx, metrics


def calculate_counterfactual_importance_accuracy(
    chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, args
):
    """
    Calculate counterfactual importance for a chunk based on accuracy differences.

    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_accuracies: Dictionary of chunk accuracies
        args: Arguments containing processing parameters

    Returns:
        float: Counterfactual importance score
    """
    if chunk_idx not in chunk_info:
        return 0.0, 0.0, 0.0

    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0, 0.0, 0.0

    next_chunk_idx = min(next_chunks)

    # Get current chunk solutions
    current_solutions = chunk_info[chunk_idx]

    # Get next chunk solutions (these will be the similar solutions)
    next_solutions = chunk_info.get(next_chunk_idx, [])

    # For dissimilar solutions, we'll use current solutions where resampled chunk is semantically different from removed chunk
    dissimilar_solutions = []
    different_trajectories_count = 0
    chunk_pairs = []

    for sol in current_solutions:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")

        if (
            isinstance(removed, str)
            and isinstance(resampled, str)
            and removed in chunk_embedding_cache
            and resampled in chunk_embedding_cache
        ):
            removed_embedding = chunk_embedding_cache[removed]
            resampled_embedding = chunk_embedding_cache[resampled]

            # Calculate similarity between removed and resampled chunks
            similarity = np.dot(removed_embedding, resampled_embedding) / (
                np.linalg.norm(removed_embedding)
                * np.linalg.norm(resampled_embedding)
            )

            # If similarity is below threshold, consider it a dissimilar solution
            if similarity < args.similarity_threshold:
                dissimilar_solutions.append(sol)
                different_trajectories_count += 1

            chunk_pairs.append((removed, resampled, sol))

    # Calculate different_trajectories_fraction
    different_trajectories_fraction = (
        different_trajectories_count / len(current_solutions)
        if current_solutions
        else 0.0
    )

    # Calculate overdeterminedness based on exact string matching of resampled chunks
    resampled_chunks_str = [
        pair[1] for pair in chunk_pairs
    ]  # Get all resampled chunks
    unique_resampled = set(resampled_chunks_str)  # Get unique resampled chunks

    # Calculate overdeterminedness as ratio of duplicates
    overdeterminedness = (
        1.0 - (len(unique_resampled) / len(resampled_chunks_str))
        if resampled_chunks_str
        else 0.0
    )

    # If we have no solutions in either group, return 0
    if not dissimilar_solutions or not next_solutions:
        return 0.0, different_trajectories_fraction, overdeterminedness

    # Calculate accuracy for dissimilar solutions
    dissimilar_correct = sum(
        1
        for sol in dissimilar_solutions
        if sol.get("is_correct", False) is True and sol.get("answer", "") != ""
    )
    dissimilar_total = len(
        [
            sol
            for sol in dissimilar_solutions
            if sol.get("is_correct", None) is not None
            and sol.get("answer", "") != ""
        ]
    )
    dissimilar_accuracy = (
        dissimilar_correct / dissimilar_total if dissimilar_total > 0 else 0.0
    )

    # Calculate accuracy for next chunk solutions (similar)
    next_correct = sum(
        1
        for sol in next_solutions
        if sol.get("is_correct", False) is True and sol.get("answer", "") != ""
    )
    next_total = len(
        [
            sol
            for sol in next_solutions
            if sol.get("is_correct", None) is not None
            and sol.get("answer", "") != ""
        ]
    )
    next_accuracy = next_correct / next_total if next_total > 0 else 0.0

    # Compare dissimilar solutions with next chunk solutions
    diff = dissimilar_accuracy - next_accuracy

    diff = (
        abs(diff)
        if hasattr(args, "use_abs_importance") and args.use_abs_importance
        else diff
    )
    return diff, different_trajectories_fraction, overdeterminedness


def calculate_counterfactual_importance_kl(
    chunk_idx,
    chunk_info,
    chunk_embedding_cache,
    chunk_accuracies,
    chunk_answers,
    args,
):
    """
    Calculate counterfactual importance for a chunk based on KL divergence between similar and dissimilar solutions.

    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        chunk_embedding_cache: Cache of chunk embeddings
        chunk_accuracies: Dictionary of chunk accuracies
        chunk_answers: Dictionary of chunk answers
        args: Arguments containing processing parameters

    Returns:
        float: Counterfactual importance score based on KL divergence
    """
    if chunk_idx not in chunk_info:
        return 0.0

    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)

    # Get current chunk solutions
    current_solutions = chunk_info[chunk_idx]

    # Get next chunk solutions (these will be the similar solutions)
    next_solutions = chunk_info.get(next_chunk_idx, [])

    # For dissimilar solutions, we'll use current solutions where resampled chunk is semantically different from removed chunk
    dissimilar_solutions = []
    similar_solutions = []

    for sol in current_solutions:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")

        if (
            isinstance(removed, str)
            and isinstance(resampled, str)
            and removed in chunk_embedding_cache
            and resampled in chunk_embedding_cache
        ):
            removed_embedding = chunk_embedding_cache[removed]
            resampled_embedding = chunk_embedding_cache[resampled]

            # Calculate similarity between removed and resampled chunks
            similarity = np.dot(removed_embedding, resampled_embedding) / (
                np.linalg.norm(removed_embedding)
                * np.linalg.norm(resampled_embedding)
            )

            # If similarity is below threshold, consider it a dissimilar solution
            if similarity < args.similarity_threshold:
                dissimilar_solutions.append(sol)
            else:
                similar_solutions.append(sol)

    # If we have no solutions in either group, return 0
    if not dissimilar_solutions or not next_solutions:
        return 0.0

    # Create answer distributions for similar solutions
    similar_answers = defaultdict(int)
    for sol in similar_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            similar_answers[answer] += 1

    # Create answer distributions for dissimilar solutions
    dissimilar_answers = defaultdict(int)
    for sol in dissimilar_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            dissimilar_answers[answer] += 1

    # Create answer distributions for next chunk solutions (similar)
    next_answers = defaultdict(int)
    for sol in next_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            next_answers[answer] += 1

    # Convert to lists of solutions format expected by calculate_kl_divergence
    similar_sols = []
    dissimilar_sols = []
    next_sols = []

    # Create a mapping of answers to their is_correct values
    answer_correctness = {}
    for sol in current_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            answer_correctness[answer] = sol.get("is_correct", False)

    for sol in next_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            answer_correctness[answer] = sol.get("is_correct", False)

    for answer, count in similar_answers.items():
        for _ in range(count):
            similar_sols.append(
                {
                    "answer": answer,
                    "is_correct": answer_correctness.get(answer, False),
                }
            )

    for answer, count in dissimilar_answers.items():
        for _ in range(count):
            dissimilar_sols.append(
                {
                    "answer": answer,
                    "is_correct": answer_correctness.get(answer, False),
                }
            )

    for answer, count in next_answers.items():
        for _ in range(count):
            next_sols.append(
                {
                    "answer": answer,
                    "is_correct": answer_correctness.get(answer, False),
                }
            )

    # Calculate KL divergence between dissimilar and next distributions
    # print(f"A: {args.laplace_alpha=}")
    kl_div = calculate_kl_divergence(
        dissimilar_sols,
        next_sols + similar_sols if args.use_similar_chunks else next_sols,
        use_prob_true=args.use_prob_true,
        global_vocab=args.global_vocab,
        alpha=args.laplace_alpha,
    )

    return kl_div


def calculate_resampling_importance_accuracy(
    chunk_idx, chunk_accuracies, args=None
):
    """
    Calculate resampling importance for a chunk based on accuracy differences.

    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_accuracies: Dictionary of chunk accuracies
        args: Arguments containing processing parameters

    Returns:
        float: Resampling importance score
    """
    if chunk_idx not in chunk_accuracies:
        return 0.0

    # Get accuracies of all other chunks
    current_accuracy = chunk_accuracies[chunk_idx]
    prev_accuracies = [
        acc for idx, acc in chunk_accuracies.items() if idx <= chunk_idx
    ]
    next_accuracies = [
        acc for idx, acc in chunk_accuracies.items() if idx == chunk_idx + 1
    ]

    if not prev_accuracies or not next_accuracies:
        return 0.0

    prev_avg_accuracy = sum(prev_accuracies) / len(prev_accuracies)
    next_avg_accuracy = sum(next_accuracies) / len(next_accuracies)
    diff = next_avg_accuracy - current_accuracy

    if args and hasattr(args, "use_abs_importance") and args.use_abs_importance:
        return abs(diff)
    return diff


def calculate_resampling_importance_kl(
    chunk_idx, chunk_info, problem_dir, args=None
):
    """
    Calculate resampling importance for a chunk based on KL divergence.

    Args:
        chunk_idx: Index of the chunk to calculate importance for
        chunk_info: Dictionary containing chunk information
        problem_dir: Directory containing the problem data

    Returns:
        float: Resampling importance score based on KL divergence
    """
    if chunk_idx not in chunk_info:
        return 0.0

    # Find next chunks
    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]

    if not next_chunks:
        return 0.0

    # Get chunk directories for the current and next chunk
    chunk_dir = problem_dir / f"chunk_{chunk_idx}"

    # Get next chunk
    next_chunk_idx = min(next_chunks)
    next_chunk_dir = problem_dir / f"chunk_{next_chunk_idx}"

    # Get solutions for both chunks
    chunk_sols1 = []
    chunk_sols2 = []

    # Load solutions for current chunk
    solutions_file = chunk_dir / "solutions.json"
    if solutions_file.exists():
        try:
            with open(solutions_file, "r", encoding="utf-8") as f:
                chunk_sols1 = json.load(f)
        except Exception as e:
            print(f"Error loading solutions from {solutions_file}: {e}")

    # Load solutions for next chunk
    next_solutions_file = next_chunk_dir / "solutions.json"
    if next_solutions_file.exists():
        try:
            with open(next_solutions_file, "r", encoding="utf-8") as f:
                chunk_sols2 = json.load(f)
        except Exception as e:
            print(f"Error loading solutions from {next_solutions_file}: {e}")

    if not chunk_sols1 or not chunk_sols2:
        return 0.0

    # Calculate KL divergence
    # print(f"B: {args.laplace_alpha=}")

    return calculate_kl_divergence(
        chunk_sols1,
        chunk_sols2,
        use_prob_true=args.use_prob_true,
        global_vocab=args.global_vocab,
        alpha=args.laplace_alpha,
    )


def calculate_forced_importance_accuracy(
    chunk_idx, forced_answer_accuracies, args=None
):
    """
    Calculate forced importance for a chunk based on accuracy differences.

    Args:
        chunk_idx: Index of the chunk to calculate importance for
        forced_answer_accuracies: Dictionary of forced answer accuracies
        args: Arguments containing processing parameters

    Returns:
        float: Forced importance score
    """
    if chunk_idx not in forced_answer_accuracies:
        return 0.0

    # Find next chunk with forced answer accuracy
    next_chunks = [
        idx for idx in forced_answer_accuracies.keys() if idx > chunk_idx
    ]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)

    # Get forced accuracies for current and next chunk
    current_forced_accuracy = forced_answer_accuracies[chunk_idx]
    next_forced_accuracy = forced_answer_accuracies[next_chunk_idx]

    # Calculate the difference
    diff = next_forced_accuracy - current_forced_accuracy

    if args and hasattr(args, "use_abs_importance") and args.use_abs_importance:
        return abs(diff)
    return diff


def calculate_forced_importance_kl(
    chunk_idx,
    forced_answer_accuracies,
    problem_dir,
    forced_answer_dir,
    args=None,
):
    """
    Calculate forced importance for a chunk based on KL divergence.

    Args:
        chunk_idx: Index of the chunk to calculate importance for
        forced_answer_accuracies: Dictionary of forced answer accuracies
        problem_dir: Directory containing the problem data
        forced_answer_dir: Directory containing forced answer data

    Returns:
        float: Forced importance score based on KL divergence
    """
    if chunk_idx not in forced_answer_accuracies:
        return 0.0

    # We need to find the answer distributions for the forced chunks
    # First, get forced answer distributions for current chunk
    current_answers = defaultdict(int)
    next_answers = defaultdict(int)

    # Find the forced problem directory
    if not forced_answer_dir:
        return 0.0

    forced_problem_dir = forced_answer_dir / problem_dir.name
    if not forced_problem_dir.exists():
        return 0.0

    # Get current chunk answers
    current_chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
    current_solutions = []
    if current_chunk_dir.exists():
        solutions_file = current_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, "r", encoding="utf-8") as f:
                    current_solutions = json.load(f)

                # Get answer distribution
                for sol in current_solutions:
                    answer = normalize_answer(sol.get("answer", ""))
                    if answer:
                        current_answers[answer] += 1
            except Exception:
                pass

    # Find next chunk to compare
    next_chunks = [
        idx for idx in forced_answer_accuracies.keys() if idx > chunk_idx
    ]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)

    # Get next chunk answers
    next_chunk_dir = forced_problem_dir / f"chunk_{next_chunk_idx}"
    next_solutions = []
    if next_chunk_dir.exists():
        solutions_file = next_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, "r", encoding="utf-8") as f:
                    next_solutions = json.load(f)

                # Get answer distribution
                for sol in next_solutions:
                    answer = normalize_answer(sol.get("answer", ""))
                    if answer:
                        next_answers[answer] += 1
            except Exception:
                pass

    # If either distribution is empty or we don't have solutions, return 0
    if (
        not current_answers
        or not next_answers
        or not current_solutions
        or not next_solutions
    ):
        return 0.0

    # Use the consistent KL divergence calculation
    # print(f"C: {args.laplace_alpha=}")

    return calculate_kl_divergence(
        current_solutions,
        next_solutions,
        use_prob_true=args.use_prob_true,
        global_vocab=args.global_vocab,
        alpha=args.laplace_alpha,
    )


def compute_global_vocabulary(
    problem_dirs: List[Path], use_prob_true: bool = True
) -> Optional[set]:
    """
    Compute the global vocabulary of all unique answers across all problems.

    Args:
        problem_dirs: List of problem directories to scan
        use_prob_true: If True, return None (not needed for P(true) distributions)
                      If False, compute vocabulary from answer distributions

    Returns:
        Set of all unique normalized answers, or None if use_prob_true is True
    """
    # For P(true) distributions, vocabulary is always {True, False}, so we don't need this
    if use_prob_true:
        return None

    print("Computing global vocabulary from all problems...")
    global_vocab = set()

    for problem_dir in tqdm(problem_dirs, desc="Scanning problem vocabularies"):
        # Find all chunk directories in this problem
        chunk_dirs = sorted(
            [
                d
                for d in problem_dir.iterdir()
                if d.is_dir() and d.name.startswith("chunk_")
            ]
        )

        for chunk_dir in chunk_dirs:
            solutions_file = chunk_dir / "solutions.json"
            if solutions_file.exists():
                try:
                    with open(solutions_file, "r", encoding="utf-8") as f:
                        solutions = json.load(f)

                    # Extract all unique answers
                    for sol in solutions:
                        answer = normalize_answer(sol.get("answer", ""))
                        if answer:
                            global_vocab.add(answer)
                except Exception as e:
                    # Silently continue on errors
                    pass

    print(f"Global vocabulary size: {len(global_vocab)} unique answers")
    return global_vocab


def calculate_kl_divergence(
    chunk_sols1,
    chunk_sols2,
    laplace_smooth=False,
    use_prob_true=False,
    global_vocab=None,
    alpha=1e-9,
):
    """Calculate KL divergence between answer distributions of two chunks.

    Args:
        chunk_sols1: First set of solutions
        chunk_sols2: Second set of solutions
        laplace_smooth: Whether to use Laplace smoothing (deprecated, use alpha parameter instead)
        use_prob_true: If True, calculate KL divergence between P(true) distributions
                      If False, calculate KL divergence between answer distributions
        global_vocab: Optional set of all possible answers. If provided, used for Laplace smoothing
                     to ensure consistent vocabulary size V across all KL divergence calculations.
                     If None, falls back to using union of answers in the two chunks.
        alpha: Smoothing parameter. Use 1e-9 for minimal smoothing (epsilon), 1.0 for Laplace smoothing.
    """

    # print(f"{args.laplace_alpha=}")
    # print(f"{alpha=}")
    # quit()

    if use_prob_true:
        # Calculate P(true) for each set
        correct1 = sum(
            1 for sol in chunk_sols1 if sol.get("is_correct", False) is True
        )
        total1 = sum(
            1 for sol in chunk_sols1 if sol.get("is_correct", None) is not None
        )
        correct2 = sum(
            1 for sol in chunk_sols2 if sol.get("is_correct", False) is True
        )
        total2 = sum(
            1 for sol in chunk_sols2 if sol.get("is_correct", None) is not None
        )

        # Early return if either set is empty
        if total1 == 0 or total2 == 0:
            return 0.0

        # Calculate probabilities with smoothing using alpha parameter
        p = (correct1 + alpha) / (
            total1 + 2 * alpha
        )  # Add alpha to both numerator and denominator
        q = (correct2 + alpha) / (total2 + 2 * alpha)

        # Calculate KL divergence for binary distribution
        kl_div = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        return max(0.0, kl_div)
    else:
        # Original implementation for answer distributions
        # Optimize by pre-allocating dictionaries with expected size
        answer_counts1 = defaultdict(int)
        answer_counts2 = defaultdict(int)

        # Process first batch of solutions
        for sol in chunk_sols1:
            answer = normalize_answer(sol.get("answer", ""))
            if answer:
                answer_counts1[answer] += 1

        # Process second batch of solutions
        for sol in chunk_sols2:
            answer = normalize_answer(sol.get("answer", ""))
            if answer:
                answer_counts2[answer] += 1

        # Early return if either set is empty
        if not answer_counts1 or not answer_counts2:
            return 0.0

        # Use global vocabulary if provided, otherwise fall back to union of answers
        if global_vocab is not None:
            # Use the global vocabulary for consistent V across all comparisons
            all_answers = global_vocab
            V = len(global_vocab)
        else:
            # Fall back to union of answers in the two chunks (original behavior)
            all_answers = set(answer_counts1.keys()) | set(
                answer_counts2.keys()
            )
            V = len(all_answers)

        # Pre-calculate totals once
        total1 = sum(answer_counts1.values())
        total2 = sum(answer_counts2.values())

        # Early return if either total is zero
        if total1 == 0 or total2 == 0:
            return 0.0

        # Use alpha parameter for smoothing: add alpha to counts, and alpha*V to totals
        smoothed_total1 = total1 + alpha * V
        smoothed_total2 = total2 + alpha * V

        # Calculate KL divergence in a single pass
        # Only iterate over answers that appear in at least one of the two chunks to save time
        # (answers not in either chunk contribute 0 to KL divergence)
        observed_answers = set(answer_counts1.keys()) | set(
            answer_counts2.keys()
        )
        kl_div = 0.0
        for answer in observed_answers:
            count1 = answer_counts1[answer]
            count2 = answer_counts2[answer]

            p = (count1 + alpha) / smoothed_total1
            q = (count2 + alpha) / smoothed_total2

            p_unsmoothed = count1 / total1

            # kl_div += p * math.log(p / q)
            kl_div += p_unsmoothed * math.log(p / q)

        return max(0.0, kl_div)


def analyze_problem(
    problem_dir: Path,
    use_absolute: bool = False,
    force_relabel: bool = False,
    forced_answer_dir: Optional[Path] = None,
    use_existing_metrics: bool = False,
    sentence_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    force_metadata: bool = False,
) -> Dict:
    """
    Analyze a single problem's solution.
    """
    # Check if required files exist
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    problem_file = problem_dir / "problem.json"

    if not (
        base_solution_file.exists()
        and chunks_file.exists()
        and problem_file.exists()
    ):
        print(f"Problem {problem_dir.name}: Missing required files")
        return None

    # Load problem
    with open(problem_file, "r", encoding="utf-8") as f:
        problem = json.load(f)

    # Generate problem nickname if it doesn't exist or if forced
    if (
        force_metadata
        or "nickname" not in problem
        or not problem.get("nickname")
    ):
        print(f"Problem {problem_dir.name}: Generating problem nickname...")
        try:
            nickname = generate_problem_nickname(problem["problem"])
            problem["nickname"] = nickname

            # Save the updated problem.json
            with open(problem_file, "w", encoding="utf-8") as f:
                json.dump(problem, f, indent=2)

        except Exception as e:
            print(
                f"Error generating nickname for problem {problem_dir.name}: {e}"
            )
            problem["nickname"] = "math problem"

    # Load base solution
    with open(base_solution_file, "r", encoding="utf-8") as f:
        base_solution = json.load(f)

    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]

    # Filter out chunks shorter than 4 characters
    valid_chunks = []
    valid_chunk_indices = []
    for i, chunk in enumerate(chunks):
        if len(chunk) >= 4:
            valid_chunks.append(chunk)
            valid_chunk_indices.append(i)

    if len(valid_chunks) < len(chunks):
        print(
            f"Problem {problem_dir.name}: Filtered out {len(chunks) - len(valid_chunks)} chunks shorter than 3 characters"
        )
        chunks = valid_chunks

    # Check if at least 25% of chunks have corresponding chunk folders
    chunk_folders = [problem_dir / f"chunk_{i}" for i in valid_chunk_indices]
    existing_chunk_folders = [
        folder for folder in chunk_folders if folder.exists()
    ]

    if len(existing_chunk_folders) < 0.1 * len(chunks):
        print(
            f"Problem {problem_dir.name}: Only {len(existing_chunk_folders)}/{len(chunks)} chunk folders exist"
        )
        return None

    # Calculate token counts for each chunk's full_cot
    token_counts = []

    # Pre-calculate accuracies for all chunks - do this once instead of repeatedly
    print(f"Problem {problem_dir.name}: Pre-calculating chunk accuracies")
    chunk_accuracies = {}
    chunk_answers = {}
    chunk_info = {}  # Store resampled chunks and answers
    chunk_response_texts = {}  # Store resampled chunk texts for semantic entropy
    chunk_category_rates = {}  # Store multiclass error category rates per chunk

    # Load forced answer accuracies if available
    forced_answer_accuracies = {}
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            for chunk_idx in valid_chunk_indices:
                chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"

                if chunk_dir.exists():
                    # Load solutions.json file to calculate accuracy
                    solutions_file = chunk_dir / "solutions.json"
                    if solutions_file.exists():
                        try:
                            with open(
                                solutions_file, "r", encoding="utf-8"
                            ) as f:
                                solutions = json.load(f)

                            # Calculate accuracy from solutions
                            correct_count = sum(
                                1
                                for sol in solutions
                                if sol.get("is_correct", False) is True
                                and sol.get("answer", "") != ""
                            )
                            total_count = sum(
                                1
                                for sol in solutions
                                if sol.get("is_correct", None) is not None
                                and sol.get("answer", "") != ""
                            )
                            accuracy = (
                                correct_count / total_count
                                if total_count > 0
                                else 0.0
                            )
                            forced_answer_accuracies[chunk_idx] = accuracy

                        except Exception as e:
                            print(
                                f"Error loading solutions from {solutions_file}: {e}"
                            )
                            forced_answer_accuracies[chunk_idx] = 0.0
                    else:
                        forced_answer_accuracies[chunk_idx] = 0.0
                else:
                    forced_answer_accuracies[chunk_idx] = 0.0

    # First pass: Load all solutions and collect basic stats (no classification yet)
    all_chunk_solutions = {}  # chunk_idx -> (solutions, solutions_file)
    ground_truth = problem.get("gt_answer", problem.get("answer", ""))
    problem_text = problem.get("problem", "")
    
    for chunk_idx in valid_chunk_indices:
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        solutions_file = chunk_dir / "solutions.json"

        if solutions_file.exists():
            with open(solutions_file, "r", encoding="utf-8") as f:
                solutions = json.load(f)
            
            all_chunk_solutions[chunk_idx] = (solutions, solutions_file)

            # Calculate accuracy
            correct = sum(
                1
                for sol in solutions
                if sol.get("is_correct", False) is True
                and sol.get("answer", "") != ""
            )
            total = sum(
                1
                for sol in solutions
                if sol.get("is_correct", None) is not None
                and sol.get("answer", "") != ""
            )
            chunk_accuracies[chunk_idx] = correct / total if total > 0 else 0.0

            # Calculate average token count
            if solutions:
                avg_tokens = np.mean(
                    [count_tokens(sol.get("full_cot", "")) for sol in solutions]
                )
                token_counts.append((chunk_idx, avg_tokens))

            if solutions:
                # Store answer distributions
                chunk_answers[chunk_idx] = defaultdict(int)
                for sol in solutions:
                    if (
                        sol.get("answer", "") != ""
                        and sol.get("answer", "") != "None"
                    ):
                        chunk_answers[chunk_idx][
                            normalize_answer(sol.get("answer", ""))
                        ] += 1

                # Store resampled chunks and answers for absolute metrics
                chunk_info[chunk_idx] = []
                for sol in solutions:
                    if (
                        sol.get("answer", "") != ""
                        and sol.get("answer", "") != "None"
                    ):
                        info = {
                            "chunk_removed": sol.get("chunk_removed", ""),
                            "chunk_resampled": sol.get("chunk_resampled", ""),
                            "full_cot": sol.get("full_cot", ""),
                            "is_correct": sol.get("is_correct", False),
                            "answer": normalize_answer(sol.get("answer", "")),
                        }
                        chunk_info[chunk_idx].append(info)

            # Collect resampled chunk texts for semantic entropy
            response_texts = []
            for sol in solutions:
                text = sol.get("chunk_resampled", "")
                if isinstance(text, str) and text.strip():
                    response_texts.append(text.strip())
            chunk_response_texts[chunk_idx] = response_texts

    # Second pass: Collect ALL solutions that need classification across ALL chunks
    all_solutions_to_classify = []  # List of (chunk_idx, solution_idx, sol)
    for chunk_idx, (solutions, solutions_file) in all_chunk_solutions.items():
        for i, sol in enumerate(solutions):
            if (sol.get("is_correct", True) is False  # Only classify incorrect solutions
                and sol.get("answer", "") != ""
                and (args.force_classify_error or not sol.get("multiclass_primary_category"))):  # Force or not already classified
                all_solutions_to_classify.append((chunk_idx, i, sol))
    
    # Classify ALL solutions in a single parallel batch across all chunks
    if all_solutions_to_classify:
        print(f"  Classifying {len(all_solutions_to_classify)} incorrect solutions across {len(all_chunk_solutions)} chunks...")
        
        def classify_single(item):
            chunk_idx, idx, sol = item
            category = classify_solution_error(
                problem_text=problem_text,
                solution_text=sol.get("full_cot", ""),
                ground_truth_answer=ground_truth,
                extracted_answer=sol.get("answer", "")
            )
            return chunk_idx, idx, category
        
        # Process ALL classifications in parallel with higher worker count
        with ThreadPoolExecutor(max_workers=1000) as executor:
            futures = [executor.submit(classify_single, item) for item in all_solutions_to_classify]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying solutions"):
                results.append(future.result())
        
        # Group results by chunk and update solutions
        chunks_updated = set()
        for chunk_idx, idx, category in results:
            if category:
                solutions, _ = all_chunk_solutions[chunk_idx]
                solutions[idx]["multiclass_primary_category"] = category
                chunks_updated.add(chunk_idx)
        
        # Save updated solutions back to files
        for chunk_idx in chunks_updated:
            solutions, solutions_file = all_chunk_solutions[chunk_idx]
            with open(solutions_file, "w", encoding="utf-8") as f:
                json.dump(solutions, f, indent=2)
        
        print(f"  Updated {len(chunks_updated)} chunk files with classifications")

    # Third pass: Calculate category rates for all chunks
    for chunk_idx, (solutions, solutions_file) in all_chunk_solutions.items():
        # Calculate multiclass error category rates for this chunk
        # Use total_solutions as denominator (not just classified ones)
        total_solutions = len(solutions)
        category_counts = {cat: 0 for cat in MATH_ERROR_CATEGORIES}
        for sol in solutions:
            cat = sol.get("multiclass_primary_category", "")
            if cat in category_counts:
                category_counts[cat] += 1
        
        # Calculate rates relative to ALL solutions (frequency of error)
        if total_solutions > 0:
            chunk_category_rates[chunk_idx] = {
                cat: count / total_solutions
                for cat, count in category_counts.items()
            }
        else:
            chunk_category_rates[chunk_idx] = {cat: 0.0 for cat in MATH_ERROR_CATEGORIES}

    # Create problem-level embedding cache (no local model needed - using OpenRouter API)
    chunk_embedding_cache = {}

    # Collect all unique chunks that need embedding
    all_chunks_to_embed = set()

    # Add chunks from chunk_info
    for solutions in chunk_info.values():
        for sol in solutions:
            # Add removed and resampled chunks
            if isinstance(sol.get("chunk_removed", ""), str):
                text = sol["chunk_removed"]
                all_chunks_to_embed.add(text)
                if text.strip():
                    all_chunks_to_embed.add(text.strip())
            if isinstance(sol.get("chunk_resampled", ""), str):
                text = sol["chunk_resampled"]
                all_chunks_to_embed.add(text)
                if text.strip():
                    all_chunks_to_embed.add(text.strip())

    # Add all texts from chunk_response_texts to ensure coverage
    for texts in chunk_response_texts.values():
        for text in texts:
             if isinstance(text, str):
                 all_chunks_to_embed.add(text)
                 if text.strip():
                     all_chunks_to_embed.add(text.strip())

    # Convert set to list and compute embeddings via OpenRouter API
    all_chunks_list = list(all_chunks_to_embed)
    batch_size = args.batch_size

    # Load cached embeddings from disk if available
    import pickle
    embedding_cache_file = problem_dir / "embedding_cache.pkl"
    if embedding_cache_file.exists():
        try:
            with open(embedding_cache_file, "rb") as f:
                chunk_embedding_cache = pickle.load(f)
            print(f"Loaded {len(chunk_embedding_cache)} cached embeddings from disk")
        except Exception as e:
            print(f"Warning: Could not load embedding cache: {e}")
            chunk_embedding_cache = {}
    
    # Filter out chunks that are already cached
    chunks_to_embed = [c for c in all_chunks_list if c not in chunk_embedding_cache]
    
    if chunks_to_embed:
        print(f"Computing embeddings for {len(chunks_to_embed)} new chunks via OpenRouter API ({len(all_chunks_list) - len(chunks_to_embed)} cached)...")
        embeddings = get_openrouter_embeddings(chunks_to_embed, model_name=sentence_model, batch_size=batch_size)
        for chunk, embedding in zip(chunks_to_embed, embeddings):
            chunk_embedding_cache[chunk] = embedding
        
        # Save updated cache to disk
        try:
            with open(embedding_cache_file, "wb") as f:
                pickle.dump(chunk_embedding_cache, f)
            print(f"Saved {len(chunk_embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
    else:
        print(f"All {len(all_chunks_list)} embeddings loaded from cache - no API calls needed")

    # Compute semantic entropy per chunk using sampled responses
    semantic_entropy_by_chunk_idx = {}
    for c_idx in valid_chunk_indices:
        texts = chunk_response_texts.get(c_idx, [])
        try:
            se_val = compute_semantic_entropy_for_texts(
                texts=texts,
                chunk_embedding_cache=chunk_embedding_cache,
                similarity_threshold=similarity_threshold,
                model_name=sentence_model,
            )
        except Exception:
            se_val = 0.0
        semantic_entropy_by_chunk_idx[c_idx] = se_val

    labeled_chunks = [{"chunk_idx": i} for i in valid_chunk_indices]

    # Save labeled chunks
    labeled_chunks_file = problem_dir / "chunks_labeled.json"

    # If labeled chunks exist and we're not forcing relabeling, load them
    if labeled_chunks_file.exists() and not force_relabel:
        with open(labeled_chunks_file, "r", encoding="utf-8") as f:
            labeled_chunks = json.load(f)

        # Filter out chunks shorter than 3 characters
        labeled_chunks = [
            chunk
            for chunk in labeled_chunks
            if chunk.get("chunk_idx") in valid_chunk_indices
        ]

        # Load summary cache from disk if available
        summary_cache_file = problem_dir / "summary_cache.json"
        summary_cache = {}
        if summary_cache_file.exists():
            try:
                with open(summary_cache_file, "r", encoding="utf-8") as f:
                    summary_cache = json.load(f)
                print(f"Loaded {len(summary_cache)} cached summaries from disk")
            except Exception as e:
                print(f"Warning: Could not load summary cache: {e}")
        
        # Generate summaries for chunks that don't have them or if forced
        chunks_need_summary = []
        for chunk in labeled_chunks:
            chunk_text = chunk.get("chunk", "")
            # Check cache first (unless force_metadata is set)
            if not force_metadata and chunk_text in summary_cache:
                chunk["summary"] = summary_cache[chunk_text]
            elif (
                force_metadata
                or "summary" not in chunk
                or not chunk.get("summary")
            ):
                chunks_need_summary.append(chunk)

        # Always add or update semantic entropy for each chunk
        added_semantic_entropy = False
        for chunk in labeled_chunks:
            se_val = semantic_entropy_by_chunk_idx.get(chunk.get("chunk_idx"), 0.0)
            if chunk.get("semantic_entropy") != se_val:
                chunk["semantic_entropy"] = se_val
                added_semantic_entropy = True

        if chunks_need_summary:
            cached_count = len(labeled_chunks) - len(chunks_need_summary)
            print(f"Generating summaries for {len(chunks_need_summary)} chunks ({cached_count} cached)...")
            
            def process_summary(chunk):
                chunk_text = chunk.get("chunk", "")
                if chunk_text:
                    try:
                        # Simple retry logic for summary generation
                        for attempt in range(3):
                            try:
                                summary = generate_chunk_summary(chunk_text)
                                chunk["summary"] = summary
                                summary_cache[chunk_text] = summary  # Add to cache
                                break
                            except Exception as e:
                                if attempt == 2:
                                    print(f"Error generating summary for chunk {chunk.get('chunk_idx')}: {e}")
                                    chunk["summary"] = "unknown action"
                                time.sleep(1)
                    except Exception as e:
                        chunk["summary"] = "unknown action"
                return chunk

            with ThreadPoolExecutor(max_workers=100) as executor:
                list(tqdm(executor.map(process_summary, chunks_need_summary), total=len(chunks_need_summary), desc="Processing chunks"))
            
            # Save updated summary cache to disk
            try:
                with open(summary_cache_file, "w", encoding="utf-8") as f:
                    json.dump(summary_cache, f, indent=2)
                print(f"Saved {len(summary_cache)} summaries to cache")
            except Exception as e:
                print(f"Warning: Could not save summary cache: {e}")
        else:
            print(f"All {len(labeled_chunks)} summaries loaded from cache - no API calls needed")

        # Only recalculate metrics if not using existing values
        if not use_existing_metrics:
            print(
                f"Problem {problem_dir.name}: Recalculating importance for {len(labeled_chunks)} chunks"
            )

            # Prepare arguments for parallel processing
            chunk_indices = [chunk["chunk_idx"] for chunk in labeled_chunks]

            # Create args object for process_chunk_importance
            args_obj = ImportanceArgs(
                use_absolute=use_absolute,
                forced_answer_dir=forced_answer_dir,
                similarity_threshold=similarity_threshold,
                use_similar_chunks=args.use_similar_chunks,
                use_abs_importance=args.absolute,
                top_chunks=args.top_chunks,
                use_prob_true=args.use_prob_true,
                laplace_alpha=args.laplace_alpha,
            )

            # Create a pool of workers
            # with mp.Pool(processes=args.num_processes) as pool:
            #     # Create partial function with fixed arguments
            #     process_func = partial(
            #         process_chunk_importance,
            #         chunk_info=chunk_info,
            #         chunk_embedding_cache=chunk_embedding_cache,
            #         chunk_accuracies=chunk_accuracies,
            #         args=args_obj,
            #         problem_dir=problem_dir,
            #         forced_answer_accuracies=forced_answer_accuracies,
            #         chunk_answers=chunk_answers,
            #     )

            #     # Process chunks in parallel
            #     results = list(
            #         tqdm(
            #             pool.imap(process_func, chunk_indices),
            #             total=len(chunk_indices),
            #             desc="Processing chunks",
            #         )
            #     )

            process_func = partial(
                process_chunk_importance,
                chunk_info=chunk_info,
                chunk_embedding_cache=chunk_embedding_cache,
                chunk_accuracies=chunk_accuracies,
                args=args_obj,
                problem_dir=problem_dir,
                forced_answer_accuracies=forced_answer_accuracies,
                chunk_answers=chunk_answers,
            )

            results = map(process_func, chunk_indices)

            # Update labeled_chunks with results
            for chunk_idx, metrics in results:
                for chunk in labeled_chunks:
                    if chunk["chunk_idx"] == chunk_idx:
                        # Remove old absolute importance metrics
                        chunk.pop("absolute_importance_accuracy", None)
                        chunk.pop("absolute_importance_kl", None)

                        # Update with new metrics
                        chunk.update(metrics)

                        # Reorder dictionary keys for consistent output
                        key_order = [
                            "chunk",
                            "chunk_idx",
                            "function_tags",
                            "depends_on",
                            "accuracy",
                            "resampling_importance_accuracy",
                            "resampling_importance_kl",
                            "counterfactual_importance_accuracy",
                            "counterfactual_importance_kl",
                            "forced_importance_accuracy",
                            "forced_importance_kl",
                            "different_trajectories_fraction",
                            "overdeterminedness",
                            "semantic_entropy",
                            "category_rates",
                            "summary",
                        ]

                        # Create new ordered dictionary
                        ordered_chunk = {}
                        for key in key_order:
                            if key in chunk:
                                ordered_chunk[key] = chunk[key]

                        # Add any remaining keys that weren't in the predefined order
                        for key, value in chunk.items():
                            if key not in ordered_chunk:
                                ordered_chunk[key] = value

                        # Replace the original chunk with the ordered version
                        chunk.clear()
                        chunk.update(ordered_chunk)
                        break

            for chunk in labeled_chunks:
                chunk.update({"accuracy": chunk_accuracies[chunk["chunk_idx"]]})
                chunk.update(
                    {
                        "semantic_entropy": semantic_entropy_by_chunk_idx.get(
                            chunk["chunk_idx"], 0.0
                        )
                    }
                )
                # Add multiclass error category rates
                if chunk["chunk_idx"] in chunk_category_rates:
                    chunk.update({"category_rates": chunk_category_rates[chunk["chunk_idx"]]})

            # Save updated labeled chunks
            with open(labeled_chunks_file, "w", encoding="utf-8") as f:
                json.dump(labeled_chunks, f, indent=2)

        else:
            # Just use existing metrics without recalculation, but still save if summaries or semantic entropy were added
            if chunks_need_summary or added_semantic_entropy:
                with open(labeled_chunks_file, "w", encoding="utf-8") as f:
                    json.dump(labeled_chunks, f, indent=2)

            print(
                f"Problem {problem_dir.name}: Using existing metrics from {labeled_chunks_file}"
            )
    else:
        # Label each chunk
        print(f"Problem {problem_dir.name}: Labeling {len(chunks)} chunks")

        # Use the DAG prompt to label all chunks at once
        try:
            labeled_chunks_result = label_chunk(problem["problem"], chunks, 0)

            # Process the result into the expected format
            labeled_chunks = []

            # Prepare arguments for parallel processing
            chunk_indices = list(range(len(valid_chunk_indices)))

            # Create args object for process_chunk_importance
            args_obj = ImportanceArgs(
                use_absolute=use_absolute,
                forced_answer_dir=forced_answer_dir,
                similarity_threshold=similarity_threshold,
                use_similar_chunks=args.use_similar_chunks,
                use_abs_importance=args.absolute,
                top_chunks=args.top_chunks,
                use_prob_true=args.use_prob_true,
            )

            # Create a pool of workers
            with mp.Pool(processes=args.num_processes) as pool:
                # Create partial function with fixed arguments
                process_func = partial(
                    process_chunk_importance,
                    chunk_info=chunk_info,
                    chunk_embedding_cache=chunk_embedding_cache,
                    chunk_accuracies=chunk_accuracies,
                    args=args_obj,
                    problem_dir=problem_dir,
                    forced_answer_accuracies=forced_answer_accuracies,
                    chunk_answers=chunk_answers,
                )

                # Process chunks in parallel
                results = list(
                    tqdm(
                        pool.imap(process_func, chunk_indices),
                        total=len(chunk_indices),
                        desc="Processing chunks",
                    )
                )

            # Load summary cache from disk if available
            summary_cache_file = problem_dir / "summary_cache.json"
            summary_cache = {}
            if summary_cache_file.exists():
                try:
                    with open(summary_cache_file, "r", encoding="utf-8") as f:
                        summary_cache = json.load(f)
                    print(f"Loaded {len(summary_cache)} cached summaries from disk")
                except Exception as e:
                    print(f"Warning: Could not load summary cache: {e}")
            
            # Create labeled chunks with results
            chunks_needing_summary = []
            for i, chunk_idx in enumerate(valid_chunk_indices):
                chunk = chunks[i]  # Use the filtered chunks list
                chunk_data = {"chunk": chunk, "chunk_idx": chunk_idx}

                # Check summary cache first
                if chunk in summary_cache:
                    chunk_data["summary"] = summary_cache[chunk]
                else:
                    chunks_needing_summary.append((chunk, chunk_data))

                # Extract function tags and dependencies for this chunk
                chunk_key = str(i)
                if chunk_key in labeled_chunks_result:
                    chunk_mapping = labeled_chunks_result[chunk_key]
                    chunk_data["function_tags"] = chunk_mapping.get(
                        "function_tags", ["unknown"]
                    )
                    chunk_data["depends_on"] = chunk_mapping.get(
                        "depends_on", []
                    )
                else:
                    chunk_data["function_tags"] = ["unknown"]
                    chunk_data["depends_on"] = []

                # Add metrics from parallel processing
                for idx, metrics in results:
                    if idx == i:
                        chunk_data.update(metrics)
                        break

                chunk_data.update({"accuracy": chunk_accuracies[chunk_idx]})
                chunk_data.update(
                    {
                        "semantic_entropy": semantic_entropy_by_chunk_idx.get(
                            chunk_idx, 0.0
                        )
                    }
                )
                # Add multiclass error category rates
                if chunk_idx in chunk_category_rates:
                    chunk_data.update({"category_rates": chunk_category_rates[chunk_idx]})
                labeled_chunks.append(chunk_data)

            # Generate summaries for chunks not in cache
            if chunks_needing_summary:
                cached_count = len(valid_chunk_indices) - len(chunks_needing_summary)
                print(f"Generating summaries for {len(chunks_needing_summary)} chunks ({cached_count} cached)...")
                
                def process_summary_item(item):
                    chunk_text, chunk_data = item
                    try:
                        for attempt in range(3):
                            try:
                                summary = generate_chunk_summary(chunk_text)
                                chunk_data["summary"] = summary
                                summary_cache[chunk_text] = summary
                                break
                            except Exception as e:
                                if attempt == 2:
                                    print(f"Error generating summary for chunk {chunk_data.get('chunk_idx')}: {e}")
                                    chunk_data["summary"] = "unknown action"
                                time.sleep(1)
                    except Exception as e:
                        chunk_data["summary"] = "unknown action"
                    return item
                
                with ThreadPoolExecutor(max_workers=100) as executor:
                    list(tqdm(executor.map(process_summary_item, chunks_needing_summary), total=len(chunks_needing_summary), desc="Generating summaries"))
                
                # Save updated summary cache to disk
                try:
                    with open(summary_cache_file, "w", encoding="utf-8") as f:
                        json.dump(summary_cache, f, indent=2)
                    print(f"Saved {len(summary_cache)} summaries to cache")
                except Exception as e:
                    print(f"Warning: Could not save summary cache: {e}")
            else:
                print(f"All {len(valid_chunk_indices)} summaries loaded from cache - no API calls needed")

            with open(labeled_chunks_file, "w", encoding="utf-8") as f:
                json.dump(labeled_chunks, f, indent=2)

        except Exception as e:
            print(f"Error using DAG prompt for problem {problem_dir.name}: {e}")
            return None

    # Load forced answer data if available
    forced_answer_accuracies_list = None
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            forced_answer_accuracies_list = []

            # Iterate through chunk directories
            for chunk_idx in valid_chunk_indices:
                chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"

                if chunk_dir.exists():
                    # Load solutions.json file to calculate accuracy
                    solutions_file = chunk_dir / "solutions.json"
                    if solutions_file.exists():
                        try:
                            with open(
                                solutions_file, "r", encoding="utf-8"
                            ) as f:
                                solutions = json.load(f)

                            # Calculate accuracy from solutions
                            correct_count = sum(
                                1
                                for sol in solutions
                                if sol.get("is_correct", False) is True
                            )
                            total_count = sum(
                                1
                                for sol in solutions
                                if sol.get("is_correct", None) is not None
                                and sol.get("answer", "") != ""
                            )

                            if total_count > 0:
                                accuracy = correct_count / total_count
                            else:
                                accuracy = 0.0

                            forced_answer_accuracies_list.append(accuracy)

                        except Exception as e:
                            print(
                                f"Error loading solutions from {solutions_file}: {e}"
                            )
                            forced_answer_accuracies_list.append(0.0)
                    else:
                        forced_answer_accuracies_list.append(0.0)
                else:
                    forced_answer_accuracies_list.append(0.0)

    # Return analysis results
    return {
        "problem_idx": problem_dir.name.split("_")[1],
        "problem_type": problem.get("type"),
        "problem_level": problem.get("level"),
        "base_accuracy": base_solution.get("is_correct", False),
        "num_chunks": len(chunks),
        "labeled_chunks": labeled_chunks,
        "token_counts": token_counts,
        "forced_answer_accuracies": forced_answer_accuracies_list,
    }


def generate_plots(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> pd.DataFrame:
    """
    Generate plots from the analysis results.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        importance_metric: Importance metric to use for plotting and analysis

    Returns:
        DataFrame with category importance rankings
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Prepare data for plots
    all_chunks = []
    all_chunks_with_forced = []  # New list for chunks with forced importance

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        problem_type = result.get("problem_type", "Unknown")
        problem_level = result.get("problem_level", "Unknown")

        for chunk in result.get("labeled_chunks", []):
            # Format function tags for better display
            raw_tags = chunk.get("function_tags", [])
            formatted_tags = []

            for tag in raw_tags:
                if tag.lower() == "unknown":
                    continue  # Skip unknown tags

                # Format tag for better display (e.g., "planning_step" -> "Planning Step")
                formatted_tag = " ".join(
                    word.capitalize() for word in tag.split("_")
                )
                formatted_tags.append(formatted_tag)

            # If no valid tags after filtering, skip this chunk
            if not formatted_tags:
                continue

            chunk_data = {
                "problem_idx": problem_idx,
                "problem_type": problem_type,
                "problem_level": problem_level,
                "chunk_idx": chunk.get("chunk_idx"),
                "function_tags": formatted_tags,
                "counterfactual_importance_accuracy": chunk.get(
                    "counterfactual_importance_accuracy", 0.0
                ),
                "counterfactual_importance_kl": chunk.get(
                    "counterfactual_importance_kl", 0.0
                ),
                "resampling_importance_accuracy": chunk.get(
                    "resampling_importance_accuracy", 0.0
                ),
                "resampling_importance_kl": chunk.get(
                    "resampling_importance_kl", 0.0
                ),
                "forced_importance_accuracy": chunk.get(
                    "forced_importance_accuracy", 0.0
                ),
                "forced_importance_kl": chunk.get("forced_importance_kl", 0.0),
                "chunk_length": len(chunk.get("chunk", "")),
            }
            all_chunks.append(chunk_data)

            # If forced importance is available, add to the forced importance list
            if "forced_importance_accuracy" in chunk:
                forced_chunk_data = chunk_data.copy()
                forced_chunk_data["forced_importance_accuracy"] = chunk.get(
                    "forced_importance_accuracy", 0.0
                )
                all_chunks_with_forced.append(forced_chunk_data)

    # Convert to DataFrame
    df_chunks = pd.DataFrame(all_chunks)

    # Create a DataFrame for chunks with forced importance if available
    df_chunks_forced = (
        pd.DataFrame(all_chunks_with_forced) if all_chunks_with_forced else None
    )

    # Explode function_tags to have one row per tag
    df_exploded = df_chunks.explode("function_tags")

    # If we have forced importance data, create special plots for it
    if df_chunks_forced is not None and not df_chunks_forced.empty:
        # Explode function_tags for forced importance DataFrame
        df_forced_exploded = df_chunks_forced.explode("function_tags")

        # Plot importance by function tag (category) using violin plot with means for forced importance
        plt.figure(figsize=(12, 8))
        # Calculate mean importance for each category to sort by
        # Convert to percentage for display
        df_forced_exploded["importance_pct"] = (
            df_forced_exploded["forced_importance_accuracy"] * 100
        )
        category_means = (
            df_forced_exploded.groupby("function_tags", observed=True)[
                "importance_pct"
            ]
            .mean()
            .sort_values(ascending=False)
        )
        # Reorder the data based on sorted categories
        df_forced_exploded_sorted = df_forced_exploded.copy()
        df_forced_exploded_sorted["function_tags"] = pd.Categorical(
            df_forced_exploded_sorted["function_tags"],
            categories=category_means.index,
            ordered=True,
        )
        # Create the sorted violin plot
        ax = sns.violinplot(
            x="function_tags",
            y="importance_pct",
            data=df_forced_exploded_sorted,
            inner="quartile",
            cut=0,
        )

        # Add mean markers
        means = df_forced_exploded_sorted.groupby(
            "function_tags", observed=True
        )["importance_pct"].mean()
        for i, mean_val in enumerate(means[means.index]):
            ax.plot([i], [mean_val], "o", color="red", markersize=8)

        # Add a legend for the mean marker
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Mean",
            )
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.title("Forced Importance by Category")
        plt.ylabel("Forced Importance (Accuracy Difference %)")
        plt.xlabel(None)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "forced_importance_by_category.png")
        plt.close()

        # Create a comparison plot that shows both normal importance and forced importance
        # Calculate category means for both metrics
        both_metrics = []

        # Metrics to compare
        metrics = [importance_metric, "forced_importance_accuracy"]
        metric_labels = ["Normal Importance", "Forced Importance"]

        # Find categories present in both datasets
        normal_categories = df_exploded["function_tags"].unique()
        forced_categories = df_forced_exploded["function_tags"].unique()
        common_categories = sorted(
            set(normal_categories) & set(forced_categories)
        )

        for category in common_categories:
            # Get mean for normal importance
            normal_mean = (
                df_exploded[df_exploded["function_tags"] == category][
                    importance_metric
                ].mean()
                * 100
            )

            # Get mean for forced importance
            forced_mean = (
                df_forced_exploded[
                    df_forced_exploded["function_tags"] == category
                ]["forced_importance_accuracy"].mean()
                * 100
            )

            # Add to list
            both_metrics.append(
                {
                    "Category": category,
                    "Normal Importance": normal_mean,
                    "Forced Importance": forced_mean,
                }
            )

        # Convert to DataFrame
        df_both = pd.DataFrame(both_metrics)

        # Sort by normal importance
        df_both = df_both.sort_values("Normal Importance", ascending=False)

        # Create bar plot
        plt.figure(figsize=(15, 10))

        # Set bar width
        bar_width = 0.35

        # Set positions for bars
        r1 = np.arange(len(df_both))
        r2 = [x + bar_width for x in r1]

        # Create grouped bar chart
        plt.bar(
            r1,
            df_both["Normal Importance"],
            width=bar_width,
            label="Normal Importance",
            color="skyblue",
        )
        plt.bar(
            r2,
            df_both["Forced Importance"],
            width=bar_width,
            label="Forced Importance",
            color="lightcoral",
        )

        # Add labels and title
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Importance (%)", fontsize=12)
        plt.title(
            "Comparison of Normal vs Forced Importance by Category", fontsize=14
        )

        # Set x-tick positions and labels
        plt.xticks(
            [r + bar_width / 2 for r in range(len(df_both))],
            df_both["Category"],
            rotation=45,
            ha="right",
        )

        # Add legend
        plt.legend()

        # Add grid
        # plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the comparison plot
        plt.savefig(plots_dir / "importance_comparison_by_category.png")
        plt.close()

        print(f"Forced importance plots saved to {plots_dir}")

    # 1. Plot importance by function tag (category) using violin plot with means
    plt.figure(figsize=(12, 8))
    # Calculate mean importance for each category to sort by
    # Convert to percentage for display
    df_exploded["importance_pct"] = df_exploded[importance_metric] * 100
    category_means = (
        df_exploded.groupby("function_tags", observed=True)["importance_pct"]
        .mean()
        .sort_values(ascending=False)
    )
    # Reorder the data based on sorted categories
    df_exploded_sorted = df_exploded.copy()
    df_exploded_sorted["function_tags"] = pd.Categorical(
        df_exploded_sorted["function_tags"],
        categories=category_means.index,
        ordered=True,
    )
    # Create the sorted violin plot
    ax = sns.violinplot(
        x="function_tags",
        y="importance_pct",
        data=df_exploded_sorted,
        inner="quartile",
        cut=0,
    )

    # Add mean markers
    means = df_exploded_sorted.groupby("function_tags", observed=True)[
        "importance_pct"
    ].mean()
    for i, mean_val in enumerate(means[means.index]):
        ax.plot([i], [mean_val], "o", color="red", markersize=8)

    # Add a legend for the mean marker
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Mean",
        )
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.title("Chunk Importance by Category")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.xlabel(None)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_category.png")
    plt.close()

    # 2. Plot average importance by problem level
    plt.figure(figsize=(10, 6))
    level_importance = (
        df_chunks.groupby("problem_level")[importance_metric]
        .mean()
        .reset_index()
    )
    sns.barplot(x="problem_level", y=importance_metric, data=level_importance)
    plt.title("Average Chunk Importance by Problem Level")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_level.png")
    plt.close()

    # 3. Plot average importance by problem type
    plt.figure(figsize=(12, 8))
    type_importance = (
        df_chunks.groupby("problem_type")[importance_metric]
        .mean()
        .reset_index()
    )
    type_importance = type_importance.sort_values(
        importance_metric, ascending=False
    )
    sns.barplot(x="problem_type", y=importance_metric, data=type_importance)
    plt.title("Average Chunk Importance by Problem Type")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_type.png")
    plt.close()

    # 4. Plot token counts by chunk index
    token_data = []
    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        for chunk_idx, token_count in result.get("token_counts", []):
            token_data.append(
                {
                    "problem_idx": problem_idx,
                    "chunk_idx": chunk_idx,
                    "token_count": token_count,
                }
            )

    df_tokens = pd.DataFrame(token_data)

    # Get function tag for each chunk
    chunk_tags = {}
    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")
            raw_tags = chunk.get("function_tags", [])

            # Format and filter tags
            formatted_tags = []
            for tag in raw_tags:
                if tag.lower() == "unknown":
                    continue
                formatted_tag = " ".join(
                    word.capitalize() for word in tag.split("_")
                )
                formatted_tags.append(formatted_tag)

            if formatted_tags:
                chunk_tags[(problem_idx, chunk_idx)] = formatted_tags[0]

    # Add function tag to token data
    df_tokens["function_tag"] = df_tokens.apply(
        lambda row: chunk_tags.get(
            (row["problem_idx"], row["chunk_idx"]), "Other"
        ),
        axis=1,
    )

    # Remove rows with no valid function tag
    df_tokens = df_tokens[df_tokens["function_tag"] != "Other"]

    if not df_tokens.empty:
        # Plot token counts by function tag (category)
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="function_tag", y="token_count", data=df_tokens)
        plt.title("Token Count by Category")
        plt.ylabel("Token Count")
        plt.xlabel(None)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "token_count_by_category.png")
        plt.close()

    # 5. Plot distribution of function tags (categories)
    plt.figure(figsize=(12, 8))
    tag_counts = df_exploded["function_tags"].value_counts()
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.title("Distribution of Categories")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "category_distribution.png")
    plt.close()

    # 6. Plot importance vs. chunk position
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="chunk_idx", y=importance_metric, data=df_chunks)
    plt.title("Chunk Importance by Position")
    plt.xlabel("Chunk Index")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_position.png")
    plt.close()

    # 7. Calculate and plot average importance by function tag (category) with error bars
    tag_importance = (
        df_exploded.groupby("function_tags")
        .agg({importance_metric: ["mean", "std", "count"]})
        .reset_index()
    )
    tag_importance.columns = ["categories", "mean", "std", "count"]
    tag_importance = tag_importance.sort_values("mean", ascending=False)

    # Convert to percentages for display
    tag_importance["mean_pct"] = tag_importance["mean"] * 100
    tag_importance["std_pct"] = tag_importance["std"] * 100

    # Calculate standard error (std/sqrt(n)) instead of using raw standard deviation
    tag_importance["se_pct"] = tag_importance["std_pct"] / np.sqrt(
        tag_importance["count"]
    )

    plt.figure(figsize=(12, 8))
    plt.errorbar(
        x=range(len(tag_importance)),
        y=tag_importance["mean_pct"],
        yerr=tag_importance["se_pct"],
        fmt="o",
        capsize=5,
    )
    plt.xticks(
        range(len(tag_importance)),
        tag_importance["categories"],
        rotation=45,
        ha="right",
    )
    plt.title("Average Importance by Category")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "avg_importance_by_category.png")
    plt.close()

    print(f"Generated plots in {plots_dir}")

    # Add the new analysis of top steps by category
    for top_n in [1, 3, 5, 10, 20, 30]:
        analyze_top_steps_by_category(
            results, output_dir, top_n=top_n, use_abs=False
        )

    # Add the new analysis of steps with high z-score by category
    for z_threshold in [1.5, 2, 2.5, 3]:
        analyze_high_zscore_steps_by_category(
            results, output_dir, z_threshold=z_threshold, use_abs=False
        )

    # Return the category importance ranking
    return tag_importance


def plot_category_distribution_flow(
    results: List[Dict], 
    output_dir: Path, 
    rollout_type: str = "correct", 
    max_chunks_to_show: Optional[int] = None,
    num_bins: int = 100
) -> None:
    """
    Plot math error category distribution as stacked area chart showing 
    how error types evolve across problem solving steps.
    
    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        rollout_type: Type of rollouts ("correct" or "incorrect")
        max_chunks_to_show: Maximum chunks to include
        num_bins: Number of bins to normalize X-axis (default 100 for 0-100%)
    """
    print("Creating math error category distribution flow visualization...")
    
    # Create explore directory
    explore_dir = output_dir / "explore"
    explore_dir.mkdir(exist_ok=True, parents=True)
    
    # Create problems directory for individual plots
    problems_dir = explore_dir / "problems"
    problems_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect data across all problems for aggregated plot
    all_problem_rates = []  # List of (normalized_position, category_rates) tuples
    
    # Create plots for each problem
    problem_indices = set()
    for result in results:
        if result:
            problem_indices.add(result["problem_idx"])
    
    for problem_idx in sorted(problem_indices):
        result = next((r for r in results if r and r["problem_idx"] == problem_idx), None)
        if not result:
            continue
            
        # Collect data for this problem
        chunk_data = []
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")
            
            # Only include the first max_chunks_to_show chunks
            if max_chunks_to_show is not None and chunk_idx > max_chunks_to_show:
                continue
                
            category_rates = chunk.get("category_rates", {})
            chunk_accuracy = chunk.get("accuracy", 0.0)  # Get actual accuracy
            if category_rates:
                chunk_data.append({
                    "chunk_idx": chunk_idx,
                    "accuracy": chunk_accuracy,
                    **category_rates
                })
        
        if not chunk_data:
            continue
            
        # Convert to DataFrame and sort by chunk index
        df = pd.DataFrame(chunk_data)
        df = df.sort_values("chunk_idx")
        
        total_chunks = len(df)
        
        # Normalize positions to 0-1 range for aggregation
        for i, row in df.iterrows():
            normalized_pos = row["chunk_idx"] / (total_chunks if total_chunks > 1 else 1)
            rates = {cat: row.get(cat, 0.0) for cat in MATH_ERROR_CATEGORIES}
            actual_accuracy = row.get("accuracy", 0.0)
            all_problem_rates.append((normalized_pos, rates, actual_accuracy))
        
        # Create individual problem plot
        x = df["chunk_idx"].values
        
        # Create a matrix where each row is a position and each column is a category
        y_data = []
        for category in MATH_ERROR_CATEGORIES:
            if category in df.columns:
                y_data.append(df[category].values)
            else:
                y_data.append(np.zeros(len(x)))
        
        y_data = np.array(y_data)
        
        # Create the stacked area plot
        plt.figure(figsize=(15, 10))
        
        # Stack the areas
        plt.stackplot(x, *y_data, 
                     labels=[cat for cat in MATH_ERROR_CATEGORIES],
                     colors=[MATH_ERROR_CATEGORY_COLORS[cat] for cat in MATH_ERROR_CATEGORIES],
                     alpha=0.8)
        
        # Add labels and title
        plt.xlabel("Chunk index", fontsize=24)
        plt.ylabel("Cumulative error probability", fontsize=24)
        plt.title(f"Problem {problem_idx}: Error Category Distribution ({rollout_type})", fontsize=32)
        
        # Set axis limits
        plt.xlim(x.min() - 1, x.max() + 1)
        plt.ylim(0, 1.05)
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(problems_dir / f"problem_{problem_idx}_error_category_flow.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    # Create aggregated plot across all problems (normalized to 0-100%)
    if all_problem_rates:
        # Bin the data into num_bins buckets
        bin_rates = {cat: [[] for _ in range(num_bins)] for cat in MATH_ERROR_CATEGORIES}
        bin_accuracies = [[] for _ in range(num_bins)]  # Track actual accuracies per bin
        
        for pos, rates, actual_accuracy in all_problem_rates:
            bin_idx = min(int(pos * num_bins), num_bins - 1)
            for cat in MATH_ERROR_CATEGORIES:
                bin_rates[cat][bin_idx].append(rates.get(cat, 0.0))
            bin_accuracies[bin_idx].append(actual_accuracy)
        
        # Calculate average rate per bin
        avg_bin_rates = {}
        for cat in MATH_ERROR_CATEGORIES:
            avg_bin_rates[cat] = []
            for bin_idx in range(num_bins):
                if bin_rates[cat][bin_idx]:
                    avg_bin_rates[cat].append(np.mean(bin_rates[cat][bin_idx]))
                else:
                    # Fill gaps with nearest neighbor or 0.0
                    avg_bin_rates[cat].append(0.0)
        
        # Calculate average accuracy per bin
        avg_bin_accuracy = []
        for bin_idx in range(num_bins):
            if bin_accuracies[bin_idx]:
                avg_bin_accuracy.append(np.mean(bin_accuracies[bin_idx]))
            else:
                avg_bin_accuracy.append(0.0)
        
        # Create aggregated stacked area plot
        plt.figure(figsize=(15, 10))
        
        x_progress = np.linspace(0, 100, num_bins)  # Progress from 0% to 100%
        
        # Prepare Y data
        y_data = [avg_bin_rates[cat] for cat in MATH_ERROR_CATEGORIES]
        
        # --- IMPROVEMENT: Apply Gaussian Smoothing ---
        # Sigma controls the smoothness. 
        # 1.0 is minimal smoothing, 2.0-3.0 gives a nice "flow" look.
        sigma = 2.0 
        y_data_smoothed = [gaussian_filter1d(y, sigma=sigma) for y in y_data]
        
        # --- NORMALIZE: Scale errors + correct to sum to 1.0 ---
        y_data_smoothed = np.array(y_data_smoothed)
        # Smooth the actual accuracy values
        accuracy_smoothed = gaussian_filter1d(avg_bin_accuracy, sigma=sigma)
        correct_rate = np.clip(accuracy_smoothed, 0.0, 1.0)
        
        # Error portion should be (1 - accuracy)
        error_portion = 1.0 - correct_rate
        
        # Scale error rates proportionally to fill exactly the error portion
        total_error_rates = np.sum(y_data_smoothed, axis=0)
        # Avoid division by zero
        total_error_rates[total_error_rates == 0] = 1.0
        # Scale each error category: (error_rate / total_errors) * error_portion
        y_data_scaled = y_data_smoothed * (error_portion / total_error_rates)
        
        # Build final data: errors first, then correct on top
        all_categories = list(MATH_ERROR_CATEGORIES) + ["Correct (CR)"]
        all_colors = [MATH_ERROR_CATEGORY_COLORS[cat] for cat in MATH_ERROR_CATEGORIES] + ["#4CAF50"]  # Green for correct
        y_final = list(y_data_scaled) + [correct_rate]
        
        # Use data for the stackplot
        plt.stackplot(x_progress, *y_final, 
                     labels=all_categories,
                     colors=all_colors,
                     alpha=0.85)
        
        plt.xlabel("Problem Progress (%)", fontsize=24)
        plt.ylabel("Outcome Distribution", fontsize=24)
        plt.title(f"Aggregated Error Category Distribution Flow ({rollout_type})", fontsize=32)
        
        # Fixed Y-Limit since data now sums to 1
        plt.ylim(0, 1.0)
        
        plt.xlim(0, 100)
        
        # Lighter, less intrusive grid
        plt.grid(True, alpha=0.15, linestyle='--')
        
        # Legend styling
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title="Error Types")
        
        plt.tight_layout()
        plt.savefig(explore_dir / f"aggregated_error_category_flow_{rollout_type}.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved aggregated error category flow plot to {explore_dir}")
    
    print(f"Created category flow plots for {len(problem_indices)} problems in {problems_dir}")


def analyze_chunk_variance(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze variance in chunk importance scores to identify potential reasoning forks.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print(
        "Analyzing chunk variance within problems to identify potential reasoning forks..."
    )

    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)

    # Collect all chunks across problems
    all_chunks = []
    problem_chunks = {}

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        problem_chunks[problem_idx] = []

        for chunk in result.get("labeled_chunks", []):
            chunk_data = {
                "problem_idx": problem_idx,
                "chunk_idx": chunk.get("chunk_idx"),
                "chunk_text": chunk.get("chunk", ""),
                "function_tags": chunk.get("function_tags", []),
                "counterfactual_importance_accuracy": chunk.get(
                    "counterfactual_importance_accuracy", 0.0
                ),
                "counterfactual_importance_kl": chunk.get(
                    "counterfactual_importance_kl", 0.0
                ),
                "resampling_importance_accuracy": chunk.get(
                    "resampling_importance_accuracy", 0.0
                ),
                "resampling_importance_kl": chunk.get(
                    "resampling_importance_kl", 0.0
                ),
                "forced_importance_accuracy": chunk.get(
                    "forced_importance_accuracy", 0.0
                ),
                "forced_importance_kl": chunk.get("forced_importance_kl", 0.0),
            }
            all_chunks.append(chunk_data)
            problem_chunks[problem_idx].append(chunk_data)

    # Calculate variance in importance within each problem
    problem_variances = {}
    high_variance_problems = []

    for problem_idx, chunks in problem_chunks.items():
        if len(chunks) < 3:  # Need at least 3 chunks for meaningful variance
            continue

        # Calculate importance variance
        importance_values = [chunk[importance_metric] for chunk in chunks]
        variance = np.var(importance_values)

        problem_variances[problem_idx] = {
            "variance": variance,
            "chunks": chunks,
            "importance_values": importance_values,
        }

        # Track problems with high variance
        high_variance_problems.append((problem_idx, variance))

    # Sort problems by variance
    high_variance_problems.sort(key=lambda x: x[1], reverse=True)

    # Save results
    with open(variance_dir / "chunk_variance.txt", "w", encoding="utf-8") as f:
        f.write(
            "Problems with highest variance in chunk importance (potential reasoning forks):\n\n"
        )

        for problem_idx, variance in high_variance_problems[
            :20
        ]:  # Top 20 problems
            f.write(f"Problem {problem_idx}: Variance = {variance:.6f}\n")

            # Get chunks for this problem
            chunks = problem_chunks[problem_idx]

            # Sort chunks by importance
            sorted_chunks = sorted(
                chunks, key=lambda x: x[importance_metric], reverse=True
            )

            # Write chunk information
            f.write("  Chunks by importance:\n")
            for i, chunk in enumerate(sorted_chunks):
                chunk_idx = chunk["chunk_idx"]
                importance = chunk[importance_metric]
                tags = ", ".join(chunk["function_tags"])

                # Truncate chunk text for display
                chunk_text = chunk["chunk_text"]
                if len(chunk_text) > 50:
                    chunk_text = chunk_text[:47] + "..."

                f.write(
                    f"    {i+1}. Chunk {chunk_idx}: {importance:.4f} - {tags} - '{chunk_text}'\n"
                )

            # Identify potential reasoning forks (clusters of important chunks)
            f.write("  Potential reasoning forks:\n")

            # Sort chunks by index to maintain sequence
            sequence_chunks = sorted(chunks, key=lambda x: x["chunk_idx"])

            # Find clusters of important chunks
            clusters = []
            current_cluster = []
            avg_importance = np.mean([c[importance_metric] for c in chunks])

            for chunk in sequence_chunks:
                if chunk[importance_metric] > avg_importance:
                    if (
                        not current_cluster
                        or chunk["chunk_idx"] - current_cluster[-1]["chunk_idx"]
                        <= 2
                    ):
                        current_cluster.append(chunk)
                    else:
                        if (
                            len(current_cluster) >= 2
                        ):  # At least 2 chunks in a cluster
                            clusters.append(current_cluster)
                        current_cluster = [chunk]

            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

            # Write clusters
            for i, cluster in enumerate(clusters):
                start_idx = cluster[0]["chunk_idx"]
                end_idx = cluster[-1]["chunk_idx"]
                f.write(f"    Fork {i+1}: Chunks {start_idx}-{end_idx}\n")

                # Combine chunk text
                combined_text = " ".join([c["chunk_text"] for c in cluster])
                if len(combined_text) > 100:
                    combined_text = combined_text[:97] + "..."

                f.write(f"      Text: '{combined_text}'\n")

                # List tags
                all_tags = set()
                for chunk in cluster:
                    all_tags.update(chunk["function_tags"])
                f.write(f"      Tags: {', '.join(all_tags)}\n")

            f.write("\n")

    # Create visualization of variance distribution
    plt.figure(figsize=(12, 8))
    variances = [v for _, v in high_variance_problems]
    plt.hist(variances, bins=20)
    plt.xlabel("Variance in Chunk Importance")
    plt.ylabel("Number of Problems")
    plt.title("Distribution of Variance in Chunk Importance Across Problems")
    plt.tight_layout()
    plt.savefig(variance_dir / "chunk_variance_distribution.png")
    plt.close()

    # Create visualization of top high-variance problems
    plt.figure(figsize=(15, 10))
    top_problems = high_variance_problems[:20]
    problem_ids = [str(p[0]) for p in top_problems]
    problem_variances = [p[1] for p in top_problems]

    plt.bar(range(len(problem_ids)), problem_variances)
    plt.xticks(range(len(problem_ids)), problem_ids, rotation=45)
    plt.xlabel("Problem ID")
    plt.ylabel("Variance in Chunk Importance")
    plt.title("Top 20 Problems with Highest Variance in Chunk Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "top_variance_problems.png")
    plt.close()

    print(f"Chunk variance analysis saved to {variance_dir}")


def analyze_function_tag_variance(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze variance in importance scores grouped by function tags.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print("Analyzing variance in importance across function tags...")

    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)

    # Collect chunks by function tag with all metrics
    tag_chunks = {}

    for result in results:
        if not result:
            continue

        for chunk in result.get("labeled_chunks", []):
            for tag in chunk.get("function_tags", []):
                if tag.lower() == "unknown":
                    continue

                # Format tag for better display
                formatted_tag = " ".join(
                    word.capitalize() for word in tag.split("_")
                )

                if formatted_tag not in tag_chunks:
                    tag_chunks[formatted_tag] = []

                tag_chunks[formatted_tag].append(
                    {
                        "problem_idx": result["problem_idx"],
                        "chunk_idx": chunk.get("chunk_idx"),
                        "counterfactual_importance_accuracy": chunk.get(
                            "counterfactual_importance_accuracy", 0.0
                        ),
                        "counterfactual_importance_kl": chunk.get(
                            "counterfactual_importance_kl", 0.0
                        ),
                        "resampling_importance_accuracy": chunk.get(
                            "resampling_importance_accuracy", 0.0
                        ),
                        "resampling_importance_kl": chunk.get(
                            "resampling_importance_kl", 0.0
                        ),
                        "forced_importance_accuracy": chunk.get(
                            "forced_importance_accuracy", 0.0
                        ),
                        "forced_importance_kl": chunk.get(
                            "forced_importance_kl", 0.0
                        ),
                    }
                )

    # Calculate variance for each tag using the BASE_IMPORTANCE_METRIC
    tag_variances = {}

    for tag, chunks in tag_chunks.items():
        if len(chunks) < 5:  # Need at least 5 chunks for meaningful variance
            continue

        importance_values = [chunk[importance_metric] for chunk in chunks]
        variance = np.var(importance_values)
        mean = np.mean(importance_values)
        count = len(chunks)

        tag_variances[tag] = {
            "variance": variance,
            "mean": mean,
            "count": count,
            "coefficient_of_variation": variance / mean if mean != 0 else 0,
        }

    # Sort tags by variance
    sorted_tags = sorted(
        tag_variances.items(), key=lambda x: x[1]["variance"], reverse=True
    )

    # Save results
    with open(
        variance_dir / "function_tag_variance.txt", "w", encoding="utf-8"
    ) as f:
        f.write("Function tags with highest variance in importance:\n\n")

        for tag, stats in sorted_tags:
            variance = stats["variance"]
            mean = stats["mean"]
            count = stats["count"]
            cv = stats["coefficient_of_variation"]

            f.write(f"{tag}:\n")
            f.write(f"  Variance: {variance:.6f}\n")
            f.write(f"  Mean: {mean:.6f}\n")
            f.write(f"  Count: {count}\n")
            f.write(f"  Coefficient of Variation: {cv:.6f}\n")
            f.write("\n")

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot top 15 tags by variance
    top_tags = sorted_tags[:15]
    tags = [t[0] for t in top_tags]
    variances = [t[1]["variance"] for t in top_tags]

    plt.bar(range(len(tags)), variances)
    plt.xticks(range(len(tags)), tags, rotation=45, ha="right")
    plt.xlabel("Function Tag")
    plt.ylabel("Variance in Importance")
    plt.title("Top 15 Function Tags by Variance in Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "function_tag_variance.png")
    plt.close()

    # Plot coefficient of variation (normalized variance)
    plt.figure(figsize=(15, 10))

    # Sort by coefficient of variation
    sorted_by_cv = sorted(
        tag_variances.items(),
        key=lambda x: x[1]["coefficient_of_variation"],
        reverse=True,
    )
    top_cv_tags = sorted_by_cv[:15]

    cv_tags = [t[0] for t in top_cv_tags]
    cvs = [t[1]["coefficient_of_variation"] for t in top_cv_tags]

    plt.bar(range(len(cv_tags)), cvs)
    plt.xticks(range(len(cv_tags)), cv_tags, rotation=45, ha="right")
    plt.xlabel("Function Tag")
    plt.ylabel("Coefficient of Variation (σ/μ)")
    plt.title("Top 15 Function Tags by Coefficient of Variation in Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "function_tag_cv.png")
    plt.close()

    print(f"Function tag variance analysis saved to {variance_dir}")


def analyze_within_problem_variance(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze variance in importance scores within problems to identify patterns across problem types and levels.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print(
        "Analyzing within-problem variance to identify potential reasoning forks..."
    )

    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)

    # Analyze each problem for high-variance chunks (potential reasoning forks)
    high_variance_problems = []

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        chunks = result.get("labeled_chunks", [])

        if len(chunks) < 3:  # Need at least 3 chunks for meaningful analysis
            continue

        # Calculate importance values and their variance
        importance_values = [
            chunk.get(importance_metric, 0.0) for chunk in chunks
        ]
        mean_importance = np.mean(importance_values)
        variance = np.var(importance_values)

        # Identify chunks with significantly higher or lower importance than average
        # These could represent "fork reasoning steps"
        potential_forks = []

        for i, chunk in enumerate(chunks):
            importance = chunk.get(importance_metric, 0.0)
            z_score = (importance - mean_importance) / (
                np.std(importance_values)
                if np.std(importance_values) > 0
                else 1
            )

            # Consider chunks with importance significantly different from mean as potential forks
            if abs(z_score) > 1.5:  # Threshold can be adjusted
                potential_forks.append(
                    {
                        "chunk_idx": chunk.get("chunk_idx"),
                        "chunk_text": chunk.get("chunk", ""),
                        "counterfactual_importance_accuracy": chunk.get(
                            "counterfactual_importance_accuracy", 0.0
                        ),
                        "counterfactual_importance_kl": chunk.get(
                            "counterfactual_importance_kl", 0.0
                        ),
                        "resampling_importance_accuracy": chunk.get(
                            "resampling_importance_accuracy", 0.0
                        ),
                        "resampling_importance_kl": chunk.get(
                            "resampling_importance_kl", 0.0
                        ),
                        "forced_importance_accuracy": chunk.get(
                            "forced_importance_accuracy", 0.0
                        ),
                        "forced_importance_kl": chunk.get(
                            "forced_importance_kl", 0.0
                        ),
                        "z_score": z_score,
                        "function_tags": chunk.get("function_tags", []),
                    }
                )

        if potential_forks:
            high_variance_problems.append(
                {
                    "problem_idx": problem_idx,
                    "variance": variance,
                    "mean_importance": mean_importance,
                    "potential_forks": potential_forks,
                }
            )

    # Sort problems by variance
    high_variance_problems.sort(key=lambda x: x["variance"], reverse=True)

    # Save results
    with open(
        variance_dir / "within_problem_variance.txt", "w", encoding="utf-8"
    ) as f:
        f.write(
            "Problems with high variance in chunk importance (potential reasoning forks):\n\n"
        )

        for problem in high_variance_problems:
            problem_idx = problem["problem_idx"]
            variance = problem["variance"]
            mean_importance = problem["mean_importance"]

            f.write(f"Problem {problem_idx}:\n")
            f.write(f"  Overall variance: {variance:.6f}\n")
            f.write(f"  Mean importance: {mean_importance:.6f}\n")
            f.write("  Potential fork reasoning steps:\n")

            # Sort potential forks by absolute z-score
            sorted_forks = sorted(
                problem["potential_forks"],
                key=lambda x: abs(x["z_score"]),
                reverse=True,
            )

            for fork in sorted_forks:
                chunk_idx = fork["chunk_idx"]
                importance = fork[importance_metric]
                z_score = fork["z_score"]
                tags = (
                    ", ".join(fork["function_tags"])
                    if fork["function_tags"]
                    else "No tags"
                )

                f.write(f"    Chunk {chunk_idx}:\n")
                f.write(
                    f"      Importance: {importance:.6f} (z-score: {z_score:.2f})\n"
                )
                f.write(f"      Function tags: {tags}\n")
                f.write(
                    f"      Text: {fork['chunk_text'][:100]}{'...' if len(fork['chunk_text']) > 100 else ''}\n\n"
                )

    # Create visualization of fork distribution
    if high_variance_problems:
        plt.figure(figsize=(12, 8))

        # Collect data for visualization
        problem_indices = [
            p["problem_idx"] for p in high_variance_problems[:15]
        ]  # Top 15 problems
        variances = [p["variance"] for p in high_variance_problems[:15]]
        fork_counts = [
            len(p["potential_forks"]) for p in high_variance_problems[:15]
        ]

        # Create bar chart
        plt.bar(range(len(problem_indices)), variances)

        # Add fork count as text on bars
        for i, count in enumerate(fork_counts):
            plt.text(
                i,
                variances[i] * 0.5,
                f"{count} forks",
                ha="center",
                color="white",
                fontweight="bold",
            )

        plt.xticks(
            range(len(problem_indices)),
            [f"Problem {idx}" for idx in problem_indices],
            rotation=45,
            ha="right",
        )
        plt.xlabel("Problem")
        plt.ylabel("Variance in Chunk Importance")
        plt.title(
            "Problems with Highest Variance in Chunk Importance (Potential Reasoning Forks)"
        )
        plt.tight_layout()
        plt.savefig(variance_dir / "within_problem_variance.png")
        plt.close()

    print(f"Within-problem variance analysis saved to {variance_dir}")


def plot_chunk_accuracy_by_position(
    results: List[Dict],
    output_dir: Path,
    rollout_type: str = "correct",
    max_chunks_to_show: Optional[int] = None,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Plot chunk accuracy by position to identify trends in where the model makes errors.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        rollout_type: Type of rollouts being analyzed
        max_chunks_to_show: Maximum number of chunks to include in the plots
        importance_metric: Importance metric to use for the analysis
    """
    print("Plotting chunk accuracy by position...")

    # Create explore directory
    explore_dir = output_dir / "explore"
    explore_dir.mkdir(exist_ok=True, parents=True)

    # Create problems directory for individual plots
    problems_dir = explore_dir / "problems"
    problems_dir.mkdir(exist_ok=True, parents=True)

    # Collect data for all chunks across problems
    chunk_data = []
    forced_chunk_data = []  # New list for forced answer data
    forced_importance_data = []  # For storing forced importance data

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]

        # Get the solutions for each chunk
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")

            # Only include the first 100 chunks
            if (
                max_chunks_to_show is not None
                and chunk_idx > max_chunks_to_show
            ):
                continue

            # Get the solutions for this chunk
            accuracy = chunk.get("accuracy", 0.0)

            # Get forced importance if available
            forced_importance = chunk.get("forced_importance_accuracy", None)

            # Get the first function tag if available
            function_tags = chunk.get("function_tags", [])
            first_tag = ""
            if (
                function_tags
                and isinstance(function_tags, list)
                and len(function_tags) > 0
            ):
                # Get first tag and convert to initials
                tag = function_tags[0]
                if isinstance(tag, str):
                    # Convert tag like "planning_step" to "PS"
                    words = tag.split("_")
                    first_tag = "".join(
                        word[0].upper() for word in words if word
                    )

            chunk_data.append(
                {
                    "problem_idx": problem_idx,
                    "chunk_idx": chunk_idx,
                    "accuracy": accuracy,
                    "tag": first_tag,
                }
            )

            # Add forced importance if available
            if forced_importance is not None:
                forced_importance_data.append(
                    {
                        "problem_idx": problem_idx,
                        "chunk_idx": chunk_idx,
                        "importance": forced_importance,
                        "tag": first_tag,
                    }
                )

            # Add forced answer accuracy if available
            if (
                "forced_answer_accuracies" in result
                and result["forced_answer_accuracies"] is not None
                and len(result["forced_answer_accuracies"]) > chunk_idx
            ):
                forced_accuracy = result["forced_answer_accuracies"][chunk_idx]
                forced_chunk_data.append(
                    {
                        "problem_idx": problem_idx,
                        "chunk_idx": chunk_idx,
                        "accuracy": forced_accuracy,
                        "tag": first_tag,
                    }
                )

    if not chunk_data:
        print("No chunk data available for plotting.")
        return

    # Convert to DataFrame
    df_chunks = pd.DataFrame(chunk_data)
    df_forced = pd.DataFrame(forced_chunk_data) if forced_chunk_data else None
    df_forced_importance = (
        pd.DataFrame(forced_importance_data) if forced_importance_data else None
    )

    # Get unique problem indices
    problem_indices = df_chunks["problem_idx"].unique()

    # Create a colormap for the problems (other options: plasma, inferno, magma, cividis)
    import matplotlib.cm as cm

    colors = cm.viridis(np.linspace(0, 0.75, len(problem_indices)))
    color_map = dict(zip(sorted(problem_indices), colors))

    # Create a figure for forced importance data if available
    if df_forced_importance is not None and not df_forced_importance.empty:
        plt.figure(figsize=(15, 10))

        # Plot each problem with a unique color
        for problem_idx in problem_indices:
            problem_data = df_forced_importance[
                df_forced_importance["problem_idx"] == problem_idx
            ]

            # Skip if no data for this problem
            if problem_data.empty:
                continue

            # Sort by chunk index
            problem_data = problem_data.sort_values("chunk_idx")

            # Convert to numpy arrays for plotting to avoid pandas indexing issues
            chunk_indices = problem_data["chunk_idx"].to_numpy()
            importances = problem_data["importance"].to_numpy()

            # Plot with clear label
            plt.plot(
                chunk_indices,
                importances,
                marker="o",
                linestyle="-",
                color=color_map[problem_idx],
                alpha=0.7,
                label=f"Problem {problem_idx}",
            )

        # Calculate and plot the average across all problems
        avg_by_chunk = (
            df_forced_importance.groupby("chunk_idx")["importance"]
            .agg(["mean"])
            .reset_index()
        )

        plt.plot(
            avg_by_chunk["chunk_idx"],
            avg_by_chunk["mean"],
            marker=".",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color="black",
            alpha=0.8,
            label="Average",
        )

        # Add labels and title
        plt.xlabel("Sentence Index")
        plt.ylabel("Forced Importance (Difference in Accuracy)")
        plt.title("Forced Importance by Sentence Index (First 100 Sentences)")

        # Set x-axis limits to focus on first 100 chunks
        plt.xlim(-3, 100 if max_chunks_to_show is None else max_chunks_to_show)

        # Add grid
        # plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(loc="upper right", ncol=2)

        # Save the forced importance plot
        plt.tight_layout()
        plt.savefig(explore_dir / "forced_importance_by_position.png")
        plt.close()
        print(
            f"Forced importance plot saved to {explore_dir / 'forced_importance_by_position.png'}"
        )

    # Create a single plot focusing on first 100 chunks
    plt.figure(figsize=(15, 10))

    # Plot each problem with a unique color
    for problem_idx in problem_indices:
        problem_data = df_chunks[df_chunks["problem_idx"] == problem_idx]

        # Sort by chunk index
        problem_data = problem_data.sort_values("chunk_idx")

        # Convert to numpy arrays for plotting to avoid pandas indexing issues
        chunk_indices = problem_data["chunk_idx"].to_numpy()
        accuracies = problem_data["accuracy"].to_numpy()
        tags = problem_data["tag"].to_numpy()

        # Plot with clear label
        line = plt.plot(
            chunk_indices,
            accuracies,
            marker="o",
            linestyle="-",
            color=color_map[problem_idx],
            alpha=0.7,
            label=f"Problem {problem_idx}",
        )[0]

        # Identify accuracy extrema (both minima and maxima)
        # Convert to numpy arrays for easier manipulation
        for i in range(1, len(chunk_indices) - 1):
            # For correct rollouts, annotate minima (lower than both neighbors)
            is_minimum = (
                accuracies[i] < accuracies[i - 1]
                and accuracies[i] < accuracies[i + 1]
            )
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = (
                accuracies[i] > accuracies[i - 1]
                and accuracies[i] > accuracies[i + 1]
            )

            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "correct" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = (
                        -15
                        if (rollout_type == "correct" and is_minimum)
                        else 7.5
                    )

                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], accuracies[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha="center",
                        fontsize=8,
                        color=line.get_color(),
                        alpha=0.9,
                        weight="bold",
                    )

        # Plot forced answer accuracy if available
        if df_forced is not None:
            forced_problem_data = df_forced[
                df_forced["problem_idx"] == problem_idx
            ]
            if not forced_problem_data.empty:
                forced_problem_data = forced_problem_data.sort_values(
                    "chunk_idx"
                )
                plt.plot(
                    forced_problem_data["chunk_idx"],
                    forced_problem_data["accuracy"],
                    marker="x",
                    linestyle="--",
                    color=color_map[problem_idx],
                    alpha=0.5,
                    label=f"Problem {problem_idx} Forced Answer",
                )

    # Calculate and plot the average across all problems
    avg_by_chunk = (
        df_chunks.groupby("chunk_idx")["accuracy"].agg(["mean"]).reset_index()
    )

    avg_by_chunk_idx = avg_by_chunk["chunk_idx"].to_numpy()
    avg_by_chunk_mean = avg_by_chunk["mean"].to_numpy()

    # Plot average without error bars, in gray and thinner
    plt.plot(
        avg_by_chunk_idx,
        avg_by_chunk_mean,
        marker=".",
        markersize=4,
        linestyle="-",
        linewidth=1,
        color="gray",
        alpha=0.5,
        label="Average",
    )

    # Plot average for forced answer if available
    if df_forced is not None:
        forced_avg_by_chunk = (
            df_forced.groupby("chunk_idx")["accuracy"]
            .agg(["mean"])
            .reset_index()
        )
        plt.plot(
            forced_avg_by_chunk["chunk_idx"],
            forced_avg_by_chunk["mean"],
            marker=".",
            markersize=4,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.5,
            label="Average Forced Answer",
        )

    # Add labels and title
    plt.xlabel("Sentence index")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by position (first 100 sentences)")

    # Set x-axis limits to focus on first 100 chunks
    plt.xlim(-3, 300 if max_chunks_to_show is None else max_chunks_to_show)

    # Set y-axis limits
    plt.ylim(-0.1, 1.1)

    # Add grid
    # plt.grid(True, alpha=0.3)

    # If not too many problems, include all in the main legend
    plt.legend(
        loc="lower right" if rollout_type == "correct" else "upper right",
        ncol=2,
    )

    # Save the main plot
    plt.tight_layout()
    plt.savefig(explore_dir / "chunk_accuracy_by_position.png")
    plt.close()

    # Create individual plots for each problem
    print("Creating individual problem plots...")
    for problem_idx in problem_indices:
        problem_data = df_chunks[df_chunks["problem_idx"] == problem_idx]

        # Sort by chunk index
        problem_data = problem_data.sort_values("chunk_idx")

        if len(problem_data) == 0:
            continue

        # Create a new figure for this problem
        plt.figure(figsize=(10, 6))

        # Get the color for this problem
        color = color_map[problem_idx]

        problem_data_idx = problem_data["chunk_idx"].to_numpy()
        problem_data_accuracy = problem_data["accuracy"].to_numpy()

        # Plot the problem data
        line = plt.plot(
            problem_data_idx,
            problem_data_accuracy,
            marker="o",
            linestyle="-",
            color=color,
            label=f"Resampling",
        )[0]

        # Identify accuracy extrema (minima for correct, maxima for incorrect)
        # Convert to numpy arrays for easier manipulation
        chunk_indices = problem_data["chunk_idx"].values
        accuracies = problem_data["accuracy"].values
        tags = problem_data["tag"].values
        # print(f"[DEBUG] Accuracies: {accuracies}")

        # Add function tag labels for both minima and maxima
        for i in range(1, len(chunk_indices) - 1):
            # For correct rollouts, annotate minima (lower than both neighbors)
            is_minimum = (
                accuracies[i] < accuracies[i - 1]
                and accuracies[i] < accuracies[i + 1]
            )
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = (
                accuracies[i] > accuracies[i - 1]
                and accuracies[i] > accuracies[i + 1]
            )

            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "correct" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = (
                        -15
                        if (rollout_type == "correct" and is_minimum)
                        else 7.5
                    )

                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], accuracies[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha="center",
                        fontsize=8,
                        color=color,
                        alpha=0.9,
                        weight="bold",
                    )

        # Plot forced answer accuracy if available
        if df_forced is not None:
            forced_problem_data = df_forced[
                df_forced["problem_idx"] == problem_idx
            ]
            if not forced_problem_data.empty:
                forced_problem_data = forced_problem_data.sort_values(
                    "chunk_idx"
                )
                forced_problem_data_idx = forced_problem_data[
                    "chunk_idx"
                ].to_numpy()
                forced_problem_data_accuracy = forced_problem_data[
                    "accuracy"
                ].to_numpy()

                plt.plot(
                    forced_problem_data_idx,
                    forced_problem_data_accuracy,
                    marker=".",
                    linestyle="--",
                    color=color,
                    alpha=0.7,
                    label=f"Forced answer",
                )

        # Remove the forced importance plot with twin axis

        # Add labels and title
        plt.xlabel("Sentence index")
        plt.ylabel("Accuracy")
        suffix = (
            f"\n(R1-Distill-Llama-8B)"
            if "llama-8b" in args.correct_rollouts_dir
            else ""
        )
        plt.title(
            f"Problem {problem_idx}: Sentence accuracy by position{suffix}"
        )

        # Set x-axis limits to focus on first 100 chunks
        plt.xlim(-3, 100 if max_chunks_to_show is None else max_chunks_to_show)

        # Set y-axis limits for accuracy
        plt.ylim(-0.1, 1.1)

        # Add grid
        # plt.grid(True, alpha=0.3)

        # Add legend in the correct position based on rollout type
        plt.legend(
            loc="lower right" if rollout_type == "correct" else "upper right"
        )

        # Save the plot
        plt.tight_layout()
        plt.savefig(problems_dir / f"problem_{problem_idx}_accuracy.png")
        plt.close()

    print(f"Chunk accuracy plots saved to {explore_dir}")


def process_rollouts(
    rollouts_dir: Path,
    output_dir: Path,
    problems: str = None,
    max_problems: int = None,
    absolute: bool = False,
    force_relabel: bool = False,
    rollout_type: str = "correct",
    dag_dir: Optional[str] = None,
    forced_answer_dir: Optional[str] = None,
    get_token_frequencies: bool = False,
    max_chunks_to_show: int = 100,
    use_existing_metrics: bool = False,
    importance_metric: str = "counterfactual_importance_accuracy",
    sentence_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    force_metadata: bool = False,
) -> None:
    """
    Process rollouts from a specific directory and save analysis results.

    Args:
        rollouts_dir: Directory containing rollout data
        output_dir: Directory to save analysis results
        problems: Comma-separated list of problem indices to analyze
        max_problems: Maximum number of problems to analyze
        absolute: Use absolute value for importance calculation
        force_relabel: Force relabeling of chunks
        rollout_type: Type of rollouts ("correct" or "incorrect")
        dag_dir: Directory containing DAG-improved chunks for token frequency analysis
        forced_answer_dir: Directory containing correct rollout data with forced answers
        get_token_frequencies: Whether to get token frequencies
        max_chunks_to_show: Maximum number of chunks to show in plots
        use_existing_metrics: Whether to use existing metrics
        importance_metric: Importance metric to use for plotting and analysis
        sentence_model: Sentence transformer model to use for embeddings
        similarity_threshold: Similarity threshold for determining different chunks
    """

    # Get problem directories
    problem_dirs = sorted(
        [
            d
            for d in rollouts_dir.iterdir()
            if d.is_dir() and d.name.startswith("problem_")
        ]
    )

    # Filter problems if specified
    if problems:
        problem_indices = [int(idx) for idx in problems.split(",")]
        problem_dirs = [
            d
            for d in problem_dirs
            if int(d.name.split("_")[1]) in problem_indices
        ]

    # Limit number of problems if specified
    if max_problems:
        problem_dirs = problem_dirs[:max_problems]

    # Count problems with complete chunk folders
    total_problems = len(problem_dirs)
    problems_with_complete_chunks = 0
    problems_with_partial_chunks = 0
    problems_with_no_chunks = 0

    for problem_dir in problem_dirs:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            problems_with_no_chunks += 1
            continue

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = chunks_data.get("chunks", [])
        if not chunks:
            problems_with_no_chunks += 1
            continue

        # Check if all chunk folders exist
        chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
        existing_chunk_folders = [
            folder for folder in chunk_folders if folder.exists()
        ]

        if len(existing_chunk_folders) == len(chunks):
            problems_with_complete_chunks += 1
        elif len(existing_chunk_folders) > 0:
            problems_with_partial_chunks += 1
        else:
            problems_with_no_chunks += 1

    print(f"\n=== {rollout_type.capitalize()} Rollouts Summary ===")
    print(f"Total problems found: {total_problems}")
    print(
        f"Problems with complete chunk folders: {problems_with_complete_chunks} ({problems_with_complete_chunks/total_problems*100:.1f}%)"
    )
    print(
        f"Problems with partial chunk folders: {problems_with_partial_chunks} ({problems_with_partial_chunks/total_problems*100:.1f}%)"
    )
    print(
        f"Problems with no chunk folders: {problems_with_no_chunks} ({problems_with_no_chunks/total_problems*100:.1f}%)"
    )

    # Only analyze problems with at least some chunk folders
    analyzable_problem_dirs = []
    for problem_dir in problem_dirs:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            continue

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = chunks_data.get("chunks", [])
        if not chunks:
            continue

        # Check if at least one chunk folder exists
        chunk_folders = [problem_dir / f"chunk_{i}" for i in range(len(chunks))]
        existing_chunk_folders = [
            folder for folder in chunk_folders if folder.exists()
        ]

        if existing_chunk_folders:
            analyzable_problem_dirs.append(problem_dir)

    print(
        f"Analyzing {len(analyzable_problem_dirs)} problems with at least some chunk folders"
    )

    # Analyze each problem
    results = []
    for problem_dir in tqdm(
        analyzable_problem_dirs, desc=f"Analyzing {rollout_type} problems"
    ):
        result = analyze_problem(
            problem_dir,
            absolute,
            force_relabel,
            forced_answer_dir,
            use_existing_metrics,
            sentence_model,
            similarity_threshold,
            force_metadata,
        )
        if result:
            results.append(result)

    # Generate plots
    category_importance = generate_plots(results, output_dir, importance_metric)

    # Generate error category distribution flow visualization
    plot_category_distribution_flow(
        results, output_dir, rollout_type=rollout_type, max_chunks_to_show=max_chunks_to_show
    )

    # Plot chunk accuracy by position
    plot_chunk_accuracy_by_position(
        results, output_dir, rollout_type, max_chunks_to_show, importance_metric
    )

    # Analyze within-problem variance
    analyze_within_problem_variance(results, output_dir, importance_metric)

    # Analyze chunk variance
    analyze_chunk_variance(results, output_dir, importance_metric)

    # Analyze function tag variance
    analyze_function_tag_variance(results, output_dir, importance_metric)

    # Analyze top steps by category
    analyze_top_steps_by_category(
        results,
        output_dir,
        top_n=20,
        use_abs=True,
        importance_metric=importance_metric,
    )

    # Analyze high z-score steps by category
    analyze_high_zscore_steps_by_category(
        results,
        output_dir,
        z_threshold=1.5,
        use_abs=True,
        importance_metric=importance_metric,
    )

    # Print category importance ranking with percentages
    if category_importance is not None and not category_importance.empty:
        print(f"\n{rollout_type.capitalize()} Category Importance Ranking:")
        for idx, row in category_importance.iterrows():
            print(
                f"{idx+1}. {row['categories']}: {row['mean_pct']:.2f}% ± {row['se_pct']:.2f}% (n={int(row['count'])})"
            )

    # Analyze token frequencies if requested
    if get_token_frequencies:
        if dag_dir and dag_dir != "None":
            print("\nAnalyzing token frequencies from DAG-improved chunks")
            dag_dir_path = Path(dag_dir)
            analyze_dag_token_frequencies(dag_dir_path, output_dir)
        else:
            print("\nAnalyzing token frequencies from rollout results")
            analyze_token_frequencies(results, output_dir, importance_metric)

    # Save overall results
    results_file = output_dir / "analysis_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = []
        for result in results:
            if not result:
                continue

            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)

        json.dump(serializable_results, f, indent=2)

    print(
        f"{rollout_type.capitalize()} analysis complete. Results saved to {output_dir}"
    )


def analyze_dag_token_frequencies(dag_dir: Path, output_dir: Path) -> None:
    """
    Analyze token frequencies from DAG-improved chunks.

    Args:
        dag_dir: Directory containing DAG-improved chunks
        output_dir: Directory to save analysis results
    """
    print("Analyzing token frequencies from DAG-improved chunks...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Collect all chunks by category
    category_chunks = {}

    # Find all problem directories
    problem_dirs = sorted(
        [
            d
            for d in dag_dir.iterdir()
            if d.is_dir() and d.name.startswith("problem_")
        ]
    )

    for problem_dir in problem_dirs:
        # Find seed directories
        seed_dirs = sorted(
            [
                d
                for d in problem_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            ]
        )

        for seed_dir in seed_dirs:
            # Look for chunks_dag_improved.json
            chunks_file = seed_dir / "chunks_dag_improved.json"

            if not chunks_file.exists():
                continue

            # Load chunks
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            # Process each chunk
            for chunk in chunks_data:
                # Get function tags (categories)
                function_tags = chunk.get("function_tags", [])

                # Skip chunks with no tags
                if not function_tags:
                    function_tags = chunk.get("categories", [])
                    if not function_tags:
                        continue

                # Get chunk text
                chunk_text = chunk.get("chunk", "")

                # Skip empty chunks
                if not chunk_text:
                    chunk_text = chunk.get("text", "")
                    if not chunk_text:
                        continue

                # Add chunk to each of its categories
                for tag in function_tags:
                    # Format tag for better display
                    if isinstance(tag, str):
                        formatted_tag = " ".join(
                            word.capitalize() for word in tag.split("_")
                        )
                    else:
                        continue  # Skip non-string tags

                    # Skip unknown category
                    if formatted_tag.lower() == "unknown":
                        continue

                    if formatted_tag not in category_chunks:
                        category_chunks[formatted_tag] = []

                    category_chunks[formatted_tag].append(chunk_text)

    # Skip if no categories found
    if not category_chunks:
        print("No categories found for token frequency analysis")
        return

    print(
        f"Found {len(category_chunks)} categories with {sum(len(chunks) for chunks in category_chunks.values())} total chunks"
    )

    # Generate plots for unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        print(f"Analyzing {n}-gram frequencies...")

        # Tokenize chunks and count frequencies
        category_ngram_frequencies = {}

        for category, chunks in category_chunks.items():
            # Tokenize all chunks
            all_tokens = []
            for chunk in chunks:
                # Simple tokenization by splitting on whitespace and punctuation
                tokens = re.findall(r"\b\w+\b", chunk.lower())

                # Filter out stopwords and numbers
                filtered_tokens = [
                    token
                    for token in tokens
                    if token not in stopwords
                    and not token.isdigit()
                    and len(token) > 1
                ]

                # Generate n-grams
                ngrams = []
                for i in range(len(filtered_tokens) - (n - 1)):
                    ngram = " ".join(filtered_tokens[i : i + n])
                    ngrams.append(ngram)

                all_tokens.extend(ngrams)

            # Count token frequencies
            token_counts = Counter(all_tokens)

            # Calculate percentages
            total_chunks = len(chunks)
            token_percentages = {}

            for token, count in token_counts.items():
                # Count in how many chunks this n-gram appears
                chunks_with_token = sum(
                    1
                    for chunk in chunks
                    if re.search(
                        r"\b" + re.escape(token) + r"\b", chunk.lower()
                    )
                )
                percentage = (chunks_with_token / total_chunks) * 100
                token_percentages[token] = percentage

            # Store results
            category_ngram_frequencies[category] = token_percentages

        # Create a master plot with subplots for each category
        num_categories = len(category_ngram_frequencies)

        # Calculate grid dimensions
        cols = min(3, num_categories)
        rows = (num_categories + cols - 1) // cols  # Ceiling division

        # Create figure
        fig = plt.figure(figsize=(20, 5 * rows))

        # Create subplots
        for i, (category, token_percentages) in enumerate(
            category_ngram_frequencies.items()
        ):
            # Sort tokens by percentage (descending)
            sorted_tokens = sorted(
                token_percentages.items(), key=lambda x: x[1], reverse=True
            )

            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]

            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)

            # Extract token names and percentages
            token_names = [token for token, _ in top_tokens]
            percentages = [percentage for _, percentage in top_tokens]

            # Create horizontal bar plot
            y_pos = range(len(token_names))
            ax.barh(y_pos, percentages, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(token_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel("Percentage of Chunks (%)")
            ax.set_title(f"Top 10 {n}-grams in {category} Chunks")

            # Add percentage labels
            for j, percentage in enumerate(percentages):
                ax.text(percentage + 1, j, f"{percentage:.1f}%", va="center")

        # Adjust layout
        plt.tight_layout()

        # Save plot
        ngram_name = "unigrams" if n == 1 else f"{n}-grams"
        plt.savefig(
            plots_dir / f"dag_token_{ngram_name}_by_category.png", dpi=300
        )
        plt.close()

        print(
            f"{n}-gram frequency analysis complete. Plot saved to {plots_dir / f'dag_token_{ngram_name}_by_category.png'}"
        )

        # Save token frequencies to JSON
        token_frequencies_file = (
            output_dir / f"dag_token_{ngram_name}_by_category.json"
        )
        with open(token_frequencies_file, "w", encoding="utf-8") as f:
            json.dump(category_ngram_frequencies, f, indent=2)


def analyze_token_frequencies(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze token frequencies in rollout chunks.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print("Analyzing token frequencies by category...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Collect all chunks by category
    category_chunks = {}

    for result in results:
        if not result or "labeled_chunks" not in result:
            continue

        for chunk in result.get("labeled_chunks", []):
            # Get function tags (categories)
            function_tags = chunk.get("function_tags", [])

            # Skip chunks with no tags
            if not function_tags:
                continue

            # Get chunk text
            chunk_text = chunk.get("chunk", "")

            # Skip empty chunks
            if not chunk_text:
                continue

            # Add chunk to each of its categories
            for tag in function_tags:
                # Format tag for better display
                if isinstance(tag, str):
                    formatted_tag = " ".join(
                        word.capitalize() for word in tag.split("_")
                    )
                else:
                    continue  # Skip non-string tags

                # Skip unknown category
                if formatted_tag.lower() == "unknown":
                    continue

                if formatted_tag not in category_chunks:
                    category_chunks[formatted_tag] = []

                category_chunks[formatted_tag].append(chunk_text)

    # Skip if no categories found
    if not category_chunks:
        print("No categories found for token frequency analysis")
        return

    # Generate plots for unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        print(f"Analyzing {n}-gram frequencies...")

        # Tokenize chunks and count frequencies
        category_ngram_frequencies = {}

        for category, chunks in category_chunks.items():
            # Tokenize all chunks
            all_tokens = []
            for chunk in chunks:
                # Simple tokenization by splitting on whitespace and punctuation
                tokens = re.findall(r"\b\w+\b", chunk.lower())

                # Filter out stopwords and numbers
                filtered_tokens = [
                    token
                    for token in tokens
                    if token not in stopwords
                    and not token.isdigit()
                    and len(token) > 1
                ]

                # Generate n-grams
                ngrams = []
                for i in range(len(filtered_tokens) - (n - 1)):
                    ngram = " ".join(filtered_tokens[i : i + n])
                    ngrams.append(ngram)

                all_tokens.extend(ngrams)

            # Count token frequencies
            token_counts = Counter(all_tokens)

            # Calculate percentages
            total_chunks = len(chunks)
            token_percentages = {}

            for token, count in token_counts.items():
                # Count in how many chunks this n-gram appears
                chunks_with_token = sum(
                    1
                    for chunk in chunks
                    if re.search(
                        r"\b" + re.escape(token) + r"\b", chunk.lower()
                    )
                )
                percentage = (chunks_with_token / total_chunks) * 100
                token_percentages[token] = percentage

            # Store results
            category_ngram_frequencies[category] = token_percentages

        # Create a master plot with subplots for each category
        num_categories = len(category_ngram_frequencies)

        # Calculate grid dimensions
        cols = min(3, num_categories)
        rows = (num_categories + cols - 1) // cols  # Ceiling division

        # Create figure
        fig = plt.figure(figsize=(20, 5 * rows))

        # Create subplots
        for i, (category, token_percentages) in enumerate(
            category_ngram_frequencies.items()
        ):
            # Sort tokens by percentage (descending)
            sorted_tokens = sorted(
                token_percentages.items(), key=lambda x: x[1], reverse=True
            )

            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]

            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)

            # Extract token names and percentages
            token_names = [token for token, _ in top_tokens]
            percentages = [percentage for _, percentage in top_tokens]

            # Create horizontal bar plot
            y_pos = range(len(token_names))
            ax.barh(y_pos, percentages, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(token_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel("Percentage of Chunks (%)")
            ax.set_title(f"Top 10 {n}-grams in {category} Chunks")

            # Add percentage labels
            for j, percentage in enumerate(percentages):
                ax.text(percentage + 1, j, f"{percentage:.1f}%", va="center")

        # Adjust layout
        plt.tight_layout()

        # Save plot
        ngram_name = "unigrams" if n == 1 else f"{n}-grams"
        plt.savefig(plots_dir / f"token_{ngram_name}_by_category.png", dpi=300)
        plt.close()

        print(
            f"{n}-gram frequency analysis complete. Plot saved to {plots_dir / f'token_{ngram_name}_by_category.png'}"
        )

        # Save token frequencies to JSON
        token_frequencies_file = (
            output_dir / f"token_{ngram_name}_by_category.json"
        )
        with open(token_frequencies_file, "w", encoding="utf-8") as f:
            json.dump(category_ngram_frequencies, f, indent=2)


def analyze_top_steps_by_category(
    results: List[Dict],
    output_dir: Path,
    top_n: int = 20,
    use_abs: bool = True,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze the top N steps by category based on importance scores.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        top_n: Number of top steps to analyze
        use_abs: Whether to use absolute values for importance scores
        importance_metric: Importance metric to use for the analysis
    """
    print(f"Analyzing top {top_n} steps by category")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create a dictionary to store z-scores by category
    category_zscores = {}

    # Process each problem
    for result in results:
        if not result:
            continue

        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue

        # Extract importance scores and convert to z-scores
        importance_scores = [
            (
                chunk.get(importance_metric, 0.0)
                if not use_abs
                else abs(chunk.get(importance_metric, 0.0))
            )
            for chunk in labeled_chunks
        ]

        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue

        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)

        if std_importance == 0:
            continue

        z_scores = [
            (score - mean_importance) / std_importance
            for score in importance_scores
        ]

        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            # Use absolute or raw z-score based on parameter
            score_for_ranking = z_score
            chunk_data.append((i, z_score, score_for_ranking, function_tags))

        # Sort by z-score (absolute or raw) and get top N
        top_chunks = sorted(chunk_data, key=lambda x: x[2], reverse=True)[
            :top_n
        ]

        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, _, function_tags in top_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score

            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(
                    word.capitalize() for word in tag.split("_")
                )
                if formatted_tag.lower() == "unknown":
                    continue

                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)

    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}

    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)

    # Sort categories by average z-score
    sorted_categories = sorted(
        category_avg_zscores.keys(),
        key=lambda x: category_avg_zscores[x],
        reverse=True,
    )

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]

    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [
        std / np.sqrt(count) for std, count in zip(std_zscores, counts)
    ]

    # Create bar plot with standard error bars
    bars = plt.bar(
        range(len(sorted_categories)),
        avg_zscores,
        yerr=standard_errors,
        capsize=5,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Set labels and title
    plt.xlabel("Function Tag Category", fontsize=12)
    plt.ylabel(f"Average Z-Score (Top {top_n} Steps) ± SE", fontsize=12)
    plt.title(f"Average Z-Score of Top {top_n} Steps by Category", fontsize=14)

    # Set x-tick labels
    plt.xticks(
        range(len(sorted_categories)),
        sorted_categories,
        rotation=45,
        ha="right",
    )

    # Add grid
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend explaining the error bars
    plt.figtext(
        0.91,
        0.01,
        "Error bars: Standard Error (SE)",
        ha="right",
        fontsize=10,
        style="italic",
    )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_path = plots_dir / f"top_{top_n}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")

    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append(
            {
                "category": category,
                "avg_zscore": category_avg_zscores[category],
                "std_zscore": category_std_zscores[category],
                "standard_error": category_std_zscores[category]
                / np.sqrt(category_counts[category]),
                "count": category_counts[category],
            }
        )

    csv_path = plots_dir / f"top_{top_n}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def analyze_high_zscore_steps_by_category(
    results: List[Dict],
    output_dir: Path,
    z_threshold: float = 1.5,
    use_abs: bool = True,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze steps with high z-scores by category to identify outlier steps.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        z_threshold: Threshold for z-scores to consider
        use_abs: Whether to use absolute values for z-scores
        importance_metric: Importance metric to use for the analysis
    """
    print(f"Analyzing steps with z-score > {z_threshold} by category...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create a dictionary to store z-scores by category
    category_zscores = {}
    total_high_zscore_steps = 0
    total_steps_analyzed = 0

    # Process each problem
    for result in results:
        if not result:
            continue

        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue

        # Extract importance scores and convert to z-scores
        importance_scores = [
            (
                chunk.get(importance_metric, 0.0)
                if not use_abs
                else abs(chunk.get(importance_metric, 0.0))
            )
            for chunk in labeled_chunks
        ]

        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue

        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)

        if std_importance == 0:
            continue

        z_scores = [
            (score - mean_importance) / std_importance
            for score in importance_scores
        ]
        total_steps_analyzed += len(z_scores)

        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            chunk_data.append((i, z_score, function_tags))

        # Filter chunks by z-score threshold
        high_zscore_chunks = [
            chunk for chunk in chunk_data if abs(chunk[1]) > z_threshold
        ]
        total_high_zscore_steps += len(high_zscore_chunks)

        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, function_tags in high_zscore_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score

            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(
                    word.capitalize() for word in tag.split("_")
                )
                if formatted_tag.lower() == "unknown":
                    continue

                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)

    print(
        f"Found {total_high_zscore_steps} steps with z-score > {z_threshold} out of {total_steps_analyzed} total steps ({total_high_zscore_steps/total_steps_analyzed:.1%})"
    )

    # Skip if no categories found
    if not category_zscores:
        print(f"No steps with z-score > {z_threshold} found")
        return

    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}

    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)

    # Sort categories by average z-score
    sorted_categories = sorted(
        category_avg_zscores.keys(),
        key=lambda x: category_avg_zscores[x],
        reverse=True,
    )

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]

    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [
        std / np.sqrt(count) for std, count in zip(std_zscores, counts)
    ]

    # Create bar plot with standard error bars
    bars = plt.bar(
        range(len(sorted_categories)),
        avg_zscores,
        yerr=standard_errors,
        capsize=5,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Set labels and title
    plt.xlabel("Function Tag Category", fontsize=12)
    plt.ylabel(
        f"Average Z-Score (Steps with |Z| > {z_threshold}) ± SE", fontsize=12
    )
    plt.title(
        f"Average Z-Score of Steps with |Z| > {z_threshold} by Category",
        fontsize=14,
    )

    # Set x-tick labels
    plt.xticks(
        range(len(sorted_categories)),
        sorted_categories,
        rotation=45,
        ha="right",
    )

    # Add grid
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend explaining the error bars
    plt.figtext(
        0.91,
        0.01,
        "Error bars: Standard Error (SE)",
        ha="right",
        fontsize=10,
        style="italic",
    )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")

    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append(
            {
                "category": category,
                "avg_zscore": category_avg_zscores[category],
                "std_zscore": category_std_zscores[category],
                "standard_error": category_std_zscores[category]
                / np.sqrt(category_counts[category]),
                "count": category_counts[category],
            }
        )

    csv_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def analyze_response_length_statistics(
    correct_rollouts_dir: Path = None,
    incorrect_rollouts_dir: Path = None,
    output_dir: Path = None,
) -> None:
    """
    Analyze response length statistics in sentences and tokens with 95% confidence intervals.
    Combines data from both correct and incorrect rollouts for aggregate statistics.

    Args:
        correct_rollouts_dir: Directory containing correct rollout data
        incorrect_rollouts_dir: Directory containing incorrect rollout data
        output_dir: Directory to save analysis results
    """
    print("Analyzing response length statistics across all rollouts...")

    # Create analysis directory
    analysis_dir = output_dir / "response_length_analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    # Collect response length data from both correct and incorrect rollouts
    sentence_lengths = []
    token_lengths = []

    # Process correct rollouts if provided
    if correct_rollouts_dir and correct_rollouts_dir.exists():
        problem_dirs = sorted(
            [
                d
                for d in correct_rollouts_dir.iterdir()
                if d.is_dir() and d.name.startswith("problem_")
            ]
        )

        for problem_dir in tqdm(
            problem_dirs, desc="Processing correct rollouts for length analysis"
        ):
            base_solution_file = problem_dir / "base_solution.json"
            if not base_solution_file.exists():
                continue

            try:
                with open(base_solution_file, "r", encoding="utf-8") as f:
                    base_solution = json.load(f)

                full_cot = base_solution.get("full_cot", "")
                if not full_cot:
                    continue

                # Count sentences
                sentences = split_solution_into_chunks(full_cot)
                sentence_lengths.append(len(sentences))

                # Count tokens
                num_tokens = count_tokens(full_cot, approximate=False)
                token_lengths.append(num_tokens)

            except Exception as e:
                print(
                    f"Error processing correct rollout {problem_dir.name}: {e}"
                )
                continue

    # Process incorrect rollouts if provided
    if incorrect_rollouts_dir and incorrect_rollouts_dir.exists():
        problem_dirs = sorted(
            [
                d
                for d in incorrect_rollouts_dir.iterdir()
                if d.is_dir() and d.name.startswith("problem_")
            ]
        )

        for problem_dir in tqdm(
            problem_dirs,
            desc="Processing incorrect rollouts for length analysis",
        ):
            base_solution_file = problem_dir / "base_solution.json"
            if not base_solution_file.exists():
                continue

            try:
                with open(base_solution_file, "r", encoding="utf-8") as f:
                    base_solution = json.load(f)

                full_cot = base_solution.get("full_cot", "")
                if not full_cot:
                    continue

                # Count sentences
                sentences = split_solution_into_chunks(full_cot)
                sentence_lengths.append(len(sentences))

                # Count tokens
                num_tokens = count_tokens(full_cot, approximate=False)
                token_lengths.append(num_tokens)

            except Exception as e:
                print(
                    f"Error processing incorrect rollout {problem_dir.name}: {e}"
                )
                continue

    # Skip if no data collected
    if not sentence_lengths or not token_lengths:
        print("No response length data collected")
        return

    # Calculate statistics
    sentence_lengths = np.array(sentence_lengths)
    token_lengths = np.array(token_lengths)

    # Calculate means
    mean_sentences = np.mean(sentence_lengths)
    mean_tokens = np.mean(token_lengths)

    # Calculate 95% confidence intervals using t-distribution

    # For sentences
    sentence_sem = stats.sem(sentence_lengths)
    sentence_ci = stats.t.interval(
        0.95, len(sentence_lengths) - 1, loc=mean_sentences, scale=sentence_sem
    )

    # For tokens
    token_sem = stats.sem(token_lengths)
    token_ci = stats.t.interval(
        0.95, len(token_lengths) - 1, loc=mean_tokens, scale=token_sem
    )

    # Create the summary string
    summary_text = (
        f"The average response is {mean_sentences:.1f} sentences long "
        f"(95% CI: [{sentence_ci[0]:.1f}, {sentence_ci[1]:.1f}]; "
        f"this corresponds to {mean_tokens:.0f} tokens "
        f"[95% CI: {token_ci[0]:.0f}, {token_ci[1]:.0f}])."
    )

    # Print the result
    print(f"\n{summary_text}")

    # Save detailed statistics
    stats_data = {
        "num_responses": len(sentence_lengths),
        "sentences": {
            "mean": float(mean_sentences),
            "std": float(np.std(sentence_lengths)),
            "median": float(np.median(sentence_lengths)),
            "min": int(np.min(sentence_lengths)),
            "max": int(np.max(sentence_lengths)),
            "ci_95_lower": float(sentence_ci[0]),
            "ci_95_upper": float(sentence_ci[1]),
            "sem": float(sentence_sem),
        },
        "tokens": {
            "mean": float(mean_tokens),
            "std": float(np.std(token_lengths)),
            "median": float(np.median(token_lengths)),
            "min": int(np.min(token_lengths)),
            "max": int(np.max(token_lengths)),
            "ci_95_lower": float(token_ci[0]),
            "ci_95_upper": float(token_ci[1]),
            "sem": float(token_sem),
        },
        "summary": summary_text,
    }

    # Save to JSON file
    stats_file = analysis_dir / "response_length_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    print(f"Response length analysis saved to {analysis_dir}")
    print(f"Statistics saved to {stats_file}")


def main():
    # Set up directories
    correct_rollouts_dir = (
        Path(args.correct_rollouts_dir)
        if args.correct_rollouts_dir and len(args.correct_rollouts_dir) > 0
        else None
    )
    incorrect_rollouts_dir = (
        Path(args.incorrect_rollouts_dir)
        if args.incorrect_rollouts_dir and len(args.incorrect_rollouts_dir) > 0
        else None
    )
    correct_forced_answer_dir = (
        Path(args.correct_forced_answer_rollouts_dir)
        if args.correct_forced_answer_rollouts_dir
        and len(args.correct_forced_answer_rollouts_dir) > 0
        else None
    )
    incorrect_forced_answer_dir = (
        Path(args.incorrect_forced_answer_rollouts_dir)
        if args.incorrect_forced_answer_rollouts_dir
        and len(args.incorrect_forced_answer_rollouts_dir) > 0
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check if at least one rollouts directory is provided
    if not correct_rollouts_dir and not incorrect_rollouts_dir:
        print(
            "Error: At least one of --correct_rollouts_dir or --incorrect_rollouts_dir must be provided"
        )
        return

    # Analyze response length statistics across both correct and incorrect rollouts
    if correct_rollouts_dir and incorrect_rollouts_dir:
        print("\n=== Analyzing Response Length Statistics ===\n")
        analyze_response_length_statistics(
            correct_rollouts_dir, incorrect_rollouts_dir, output_dir
        )

    # Process each rollout type if provided
    if correct_rollouts_dir:
        print(
            f"\n=== Processing CORRECT rollouts from {correct_rollouts_dir} ===\n"
        )
        correct_output_dir = output_dir / "correct_base_solution"
        correct_output_dir.mkdir(exist_ok=True, parents=True)
        process_rollouts(
            rollouts_dir=correct_rollouts_dir,
            output_dir=correct_output_dir,
            problems=args.problems,
            max_problems=args.max_problems,
            absolute=args.absolute,
            force_relabel=args.force_relabel,
            rollout_type="correct",
            dag_dir=(
                args.dag_dir if args.token_analysis_source == "dag" else None
            ),
            forced_answer_dir=correct_forced_answer_dir,
            get_token_frequencies=args.get_token_frequencies,
            max_chunks_to_show=args.max_chunks_to_show,
            use_existing_metrics=args.use_existing_metrics,
            importance_metric=args.importance_metric,
            sentence_model=args.sentence_model,
            similarity_threshold=args.similarity_threshold,
            force_metadata=args.force_metadata,
        )

        # If forced answer data is available, run additional analysis using forced_importance_accuracy
        if correct_forced_answer_dir:
            print(
                "\n=== Running additional analysis with forced importance metric ===\n"
            )
            forced_output_dir = output_dir / "forced_importance_analysis"
            forced_output_dir.mkdir(exist_ok=True, parents=True)

            # Check if the results have the forced importance metric
            forced_importance_exists = False

            for result_file in correct_output_dir.glob(
                "**/chunks_labeled.json"
            ):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        labeled_chunks = json.load(f)

                    # Check if at least one chunk has the forced importance metric
                    for chunk in labeled_chunks:
                        if (
                            "forced_importance_accuracy" in chunk
                            or "forced_importance_kl" in chunk
                        ):
                            forced_importance_exists = True
                            break

                    if forced_importance_exists:
                        break

                except Exception as e:
                    print(f"Error checking for forced importance metric: {e}")

            if forced_importance_exists:
                print(
                    "Found forced importance metric in results, running specialized analysis"
                )

                # Copy the results to the forced output directory to run separate analysis
                # Just use the existing files with symlinks
                import shutil

                try:
                    # Create a symlink to the analysis_results.json file
                    src_file = correct_output_dir / "analysis_results.json"
                    dst_file = forced_output_dir / "analysis_results.json"
                    if src_file.exists() and not dst_file.exists():
                        os.symlink(src_file, dst_file)

                    # Run top steps and high z-score analyses specifically for forced importance
                    print(
                        "Running top steps analysis with forced importance metrics"
                    )
                    results = []
                    with open(src_file, "r", encoding="utf-8") as f:
                        results = json.load(f)

                    analyze_top_steps_by_category(
                        results,
                        forced_output_dir,
                        top_n=20,
                        use_abs=True,
                        importance_metric="forced_importance_accuracy",
                    )
                    analyze_high_zscore_steps_by_category(
                        results,
                        forced_output_dir,
                        z_threshold=1.5,
                        use_abs=True,
                        importance_metric="forced_importance_accuracy",
                    )

                    # Also run analyses for forced_importance_kl
                    print(
                        "Running top steps analysis with forced importance KL metric"
                    )
                    analyze_top_steps_by_category(
                        results,
                        forced_output_dir,
                        top_n=20,
                        use_abs=True,
                        importance_metric="forced_importance_kl",
                    )
                    analyze_high_zscore_steps_by_category(
                        results,
                        forced_output_dir,
                        z_threshold=1.5,
                        use_abs=True,
                        importance_metric="forced_importance_kl",
                    )

                except Exception as e:
                    print(
                        f"Error during forced importance specialized analysis: {e}"
                    )
            else:
                print(
                    "No forced importance metric found in results, skipping specialized analysis"
                )

    if incorrect_rollouts_dir:
        if not incorrect_rollouts_dir.exists():
            print(f"\n=== Skipping INCORRECT rollouts - directory does not exist: {incorrect_rollouts_dir} ===\n")
        else:
            print(
                f"\n=== Processing INCORRECT rollouts from {incorrect_rollouts_dir} ===\n"
            )
            incorrect_output_dir = output_dir / "incorrect_base_solution"
            incorrect_output_dir.mkdir(exist_ok=True, parents=True)
            process_rollouts(
                rollouts_dir=incorrect_rollouts_dir,
                output_dir=incorrect_output_dir,
                problems=args.problems,
                max_problems=args.max_problems,
                absolute=args.absolute,
                force_relabel=args.force_relabel,
                rollout_type="incorrect",
                dag_dir=(
                    args.dag_dir if args.token_analysis_source == "dag" else None
                ),
                forced_answer_dir=incorrect_forced_answer_dir,
                get_token_frequencies=args.get_token_frequencies,
                max_chunks_to_show=args.max_chunks_to_show,
                use_existing_metrics=args.use_existing_metrics,
                importance_metric=args.importance_metric,
                sentence_model=args.sentence_model,
                similarity_threshold=args.similarity_threshold,
                force_metadata=args.force_metadata,
            )


if __name__ == "__main__":
    main()
