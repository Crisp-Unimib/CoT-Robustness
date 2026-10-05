###
### Start example of generate_rollouts.py
### source env/bin/activate && python generate_math/generate_rollouts.py -p OpenRouter -m "qwen/qwen3-14b" -orp "chutes/bf16" -t 0.6 -tp 0.95 -nr 100 -ip "330,1591,2050,2137,2189,2236,2238,2870,3360,3448,3550,3916,3935,4019,4164,4605,4682,6481,6596,6998" -l "Level 5"
###

import os
import json
import random
import numpy as np
import torch
import asyncio
import httpx
import sys

# Fix Windows "too many file descriptors in select()" error
# The default SelectorEventLoop on Windows has a 512 fd limit
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from utils import extract_boxed_answers, check_answer, split_solution_into_chunks, load_math_problems
from transformers import TextStreamer

# Load environment variables
load_dotenv()

# Get API keys
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
VLLM_BASE_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")

# Global counter for round-robin load balancing
vllm_request_counter = 0

def get_model_family(model_name: str) -> str:
    """Determine model family based on name string."""
    lower_name = model_name.lower()
    if "gpt-oss" in lower_name:
        return "gpt-oss"
    elif "qwen" in lower_name:
        return "qwen"
    return "default"

# Set up argument parser
import argparse
import argparse
import sys
parser = argparse.ArgumentParser(description='Generate chain-of-thought solutions with rollouts')
parser.add_argument('-m', '--model', type=str, default="deepseek/deepseek-r1-distill-qwen-14b", help='Model to use') # "deepseek/deepseek-r1-distill-llama-8b"
parser.add_argument('-b', '--base_solution_type', type=str, default='correct', choices=['correct', 'incorrect'], help='Type of base solution to generate')
parser.add_argument('-r', '--rollout_type', type=str, default='default', choices=['default', 'forced_answer'], help='Type of rollout to generate')
parser.add_argument('-o', '--output_dir', type=str, default='data/raw/math_rollouts', help='Directory to save results')
parser.add_argument('-np', '--num_problems', type=int, default=100, help='Number of problems to sample')
parser.add_argument('-nr', '--num_rollouts', type=int, default=100, help='Number of rollouts per chunk')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.95, help='Top-p sampling parameter')
parser.add_argument('-mt', '--max_tokens', type=int, default=16384, help='Maximum number of tokens for generation')
parser.add_argument('-mc', '--max_chunks', type=int, default=350, help='Maximum number of chunks to process')
parser.add_argument('-s', '--seed', type=int, default=44, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-ep', '--exclude_problems', type=str, default=None, help='Comma-separated list of problem IDs to exclude')
parser.add_argument('-ip', '--include_problems', type=str, default=None, help='Comma-separated list of problem IDs to include')
parser.add_argument('-ic', '--include_chunks', type=str, default=None, help='Comma-separated list of chunk IDs to include')
parser.add_argument('-ty', '--type', type=str, default=None, help='Problem type filter')
parser.add_argument('-l', '--level', type=str, default="Level 5", help='Problem level filter')
parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
parser.add_argument('-p', '--provider', type=str, default="Novita", choices=['Novita', 'Together', 'Fireworks', 'Local', 'vLLM', 'OpenRouter'], help='Provider to use') # "Together"
parser.add_argument('-or', '--use_openrouter', default=False, action='store_true', help='Use OpenRouter API')
parser.add_argument('-orp', '--openrouter_provider', type=str, default=None, help='Specific OpenRouter provider (e.g. deepinfra/turbo)')
parser.add_argument('-ore', '--openrouter_reasoning_effort', type=str, default=None, help='Reasoning effort for OpenRouter (high, medium, low, none)')
parser.add_argument('-fp', '--frequency_penalty', type=float, default=None, help='Frequency penalty parameter')
parser.add_argument('-pp', '--presence_penalty', type=float, default=None, help='Presence penalty parameter')
parser.add_argument('-rp', '--repetition_penalty', type=float, default=None, help='Repetition penalty parameter')
parser.add_argument('-tk', '--top_k', type=int, default=None, help='Top-k parameter')
parser.add_argument('-mp', '--min_p', type=float, default=None, help='Min-p parameter')
parser.add_argument('-sr', '--skip_recalculate', default=False, action='store_true', help='Skip recalculating accuracy for existing rollouts')
parser.add_argument('-q', '--quantize', default=False, action='store_true', help='Use quantization for local model')
parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for local model')
parser.add_argument('-mr', '--max_retries', type=int, default=1, help='Maximum number of retries for API requests')
parser.add_argument('-os', '--output_suffix', type=str, default=None, help='Suffix to add to the output directory')
parser.add_argument('-mcr', '--max_concurrent_requests', type=int, default=100, help='Maximum number of concurrent API requests')
parser.add_argument('-ns', '--num_servers', type=int, default=1, help='Number of vLLM servers for load balancing (ports 8000, 8001, ...)')
args = parser.parse_args()

# Generate vLLM server URLs based on num_servers
VLLM_API_URLS = []
if args.num_servers > 1:
    for i in range(args.num_servers):
        VLLM_API_URLS.append(f"http://localhost:{8000 + i}/v1")
else:
    VLLM_API_URLS = [VLLM_BASE_URL]

# Global semaphore for limiting concurrency
request_semaphore = None

# Create output directory
base_output_dir = Path(args.output_dir) / args.model.split("/")[-1] / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
if args.rollout_type == 'forced_answer':
    # NOTE: For forced answer rollouts, we use the correct base solution (we copy the files from the correct base solution directory before running this script)
    output_dir = base_output_dir / f"{args.base_solution_type}_base_solution_{args.rollout_type}_{args.output_suffix}" if args.output_suffix else base_output_dir / f"{args.base_solution_type}_base_solution_{args.rollout_type}"
else:
    output_dir = base_output_dir / f"{args.base_solution_type}_base_solution_{args.output_suffix}" if args.output_suffix else base_output_dir / f"{args.base_solution_type}_base_solution"
output_dir.mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Load local model if using Local provider
local_model = None
local_tokenizer = None

if args.provider == "Local":
    try:
        print(f"Loading local model: {args.model}")
        model = args.model.replace("deepseek/", "deepseek-ai/") # Slight adjustment we need to make
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        local_tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Load model with quantization if specified
        if args.quantize and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            local_model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
            )
        else:
            local_model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None
            )
        
        print("Local model loaded successfully")
        local_model.eval()
    except Exception as e:
        print(f"Error loading local model: {e}")
        exit(1)

def generate_with_local_model(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Generate text using a local model."""
    try:
        # Tokenize the prompt
        inputs = local_tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Create a streamer that shows progress
        streamer = TextStreamer(local_tokenizer, skip_special_tokens=True)
        
        # Set up generation parameters
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "use_cache": True,
            "pad_token_id": local_tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        # Add optional parameters
        if args.top_k is not None:
            generation_config["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            generation_config["repetition_penalty"] = args.repetition_penalty
        
        # Generate
        with torch.no_grad():
            outputs = local_model.generate(**inputs, **generation_config)
        
        # Decode the generated text
        generated_text = local_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "finish_reason": "stop",  # Simplified
            "usage": {"total_tokens": len(outputs[0])}
        }
    except Exception as e:
        print(f"Error in local generation: {e}")
        return {"error": str(e)}

def generate_with_local_model_batch(prompts: List[str], temperature: float, top_p: float, max_tokens: int) -> List[Dict]:
    """Generate text using a local model in batch mode for multiple prompts."""
    try:
        results = []
        batch_size = args.batch_size
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            # Tokenize all prompts in the batch
            batch_inputs = local_tokenizer(batch_prompts, padding=True, return_tensors="pt")
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}
            
            # Set up generation parameters
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "use_cache": True,
                "pad_token_id": local_tokenizer.eos_token_id
            }
            
            # Add optional parameters
            if args.top_k is not None:
                generation_config["top_k"] = args.top_k
            if args.repetition_penalty is not None:
                generation_config["repetition_penalty"] = args.repetition_penalty
            
            # Generate
            with torch.no_grad():
                batch_outputs = local_model.generate(**batch_inputs, **generation_config)
            
            # Process each output in the batch
            for j, (input_ids, output_ids) in enumerate(zip(batch_inputs["input_ids"], batch_outputs)):
                # Find where the generated text starts (after the prompt)
                input_length = len(input_ids)
                
                # Decode the generated text
                generated_text = local_tokenizer.decode(output_ids[input_length:], skip_special_tokens=True)
                
                results.append({
                    "text": generated_text,
                    "finish_reason": "stop",  # Simplified
                    "usage": {"total_tokens": len(output_ids)}
                })
        
        return results
    except Exception as e:
        print(f"Error in batch generation: {e}")
        return [{"error": str(e)} for _ in range(len(prompts))]

async def make_api_request(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Make an API request to either Novita, Together, Fireworks, or use a local model based on provider setting."""
    # If using local model, use synchronous generation
    if args.provider == "Local":
        return generate_with_local_model(prompt, temperature, top_p, max_tokens)
    
    # Otherwise, use API-based generation
    
    # Acquire semaphore if it exists
    if request_semaphore:
        async with request_semaphore:
            return await _make_api_request_impl(prompt, temperature, top_p, max_tokens)
    else:
        return await _make_api_request_impl(prompt, temperature, top_p, max_tokens)

# Global client for reusing connections
global_client = None

async def get_client():
    global global_client
    if global_client is None:
        # With ProactorEventLoop (set in __main__), we can use higher connection limits
        # The 512 fd limit only applies to SelectorEventLoop
        limits = httpx.Limits(max_keepalive_connections=500, max_connections=1000)
        global_client = httpx.AsyncClient(limits=limits, timeout=None)
    return global_client


async def _make_api_request_impl(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Internal implementation of API request."""
    if args.provider == "Novita":
        # Novita API request
        headers = {
            "Authorization": f"Bearer {NOVITA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": args.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False
        }
        
        api_url = "https://api.novita.ai/v3/openai/completions"
        
    elif args.provider == "Together":
        # Together API request
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        payload = {
            "model": "deepseek-ai/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        api_url = "https://api.together.xyz/v1/completions"
        
    elif args.provider == "Fireworks":
        # Fireworks API request
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "accounts/fireworks/models/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": True
        }
        
        api_url = "https://api.fireworks.ai/inference/v1/completions"
        
    elif args.provider == "vLLM":
        # vLLM API request (OpenAI-compatible)
        headers = {
            "Content-Type": "application/json"
        }
        
        # Determine model family for GPT-OSS specific handling
        model_family = get_model_family(args.model)
        
        # Round-robin load balancing across vLLM servers
        global vllm_request_counter
        server_idx = vllm_request_counter % len(VLLM_API_URLS)
        vllm_request_counter += 1
        
        if model_family == "gpt-oss":
            # GPT-OSS uses Harmony format with chat/completions endpoint
            # We need to inject a system message for reasoning mode
            messages = [
                {"role": "system", "content": "Reasoning: high"}
            ]
            
            # Parse prompt to extract user content and assistant prefill
            assistant_prefill = None
            if "Solution: \n" in prompt:
                parts = prompt.split("Solution: \n")
                user_content = parts[0] + "Solution: \n"
                if len(parts) > 1 and parts[1].strip():
                    assistant_prefill = parts[1]
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": prompt})
            
            # Add assistant prefill if present (for rollouts)
            if assistant_prefill:
                messages.append({"role": "assistant", "content": assistant_prefill})
            
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # If we have assistant prefill, use continue_final_message
            if assistant_prefill:
                payload["continue_final_message"] = True
            
            api_url = f"{VLLM_API_URLS[server_idx]}/chat/completions"
        else:
            # Qwen and default models use standard completions endpoint
            payload = {
                "model": args.model,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": 1,
                "stream": False
            }
            
            api_url = f"{VLLM_API_URLS[server_idx]}/completions"

    elif args.provider == "OpenRouter":
        # OpenRouter API request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Split prompt into messages for pre-filling
        messages = []
        is_prefilling_reasoning = False
        if "Solution: \n" in prompt:
            parts = prompt.split("Solution: \n")
            user_content = parts[0] + "Solution: \n"
            assistant_content = parts[1]
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
            if "<think>" in assistant_content and "</think>" not in assistant_content:
                is_prefilling_reasoning = True
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Add provider preference if specified
        if args.openrouter_provider:
            payload["provider"] = {
                "order": [args.openrouter_provider],
                "allow_fallbacks": False
            }
            
        # Add reasoning configuration
        if args.openrouter_reasoning_effort:
            payload["reasoning"] = {
                "effort": args.openrouter_reasoning_effort
            }
        
        api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Add optional parameters for all APIs
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None and args.provider != "Fireworks" and args.provider != "vLLM" and args.provider != "OpenRouter":  # Fireworks, vLLM, OpenRouter don't support min_p in this generic way
        payload["min_p"] = args.min_p
    if args.seed is not None and False: # NOTE: We don't use seeds for rollouts
        payload["seed"] = args.seed
    
    # Implement exponential backoff for retries
    max_retries = args.max_retries
    if args.provider == "vLLM":
        max_retries = 5  # Increase retries for vLLM to handle transient errors
        
    retry_delay = 2 if max_retries > 0 else None
    
    for attempt in range(max_retries):
        try:
            # Handle streaming responses for Together and Fireworks
            if (args.provider == "Together" or args.provider == "Fireworks") and payload.get("stream", False):
                return await handle_streaming_response(api_url, headers, payload)
            
            # For non-streaming responses
            # For non-streaming responses
            timeout = None if args.provider == "OpenRouter" else 3600
            
            client = await get_client()
            response = await client.post(api_url, headers=headers, json=payload, timeout=timeout)
            
            # Handle different error codes
            if response.status_code == 500:
                print(f"Server error (500) on attempt {attempt+1}/{max_retries}. Retrying...")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
                
            elif response.status_code == 429:
                print(f"Rate limit (429) on attempt {attempt+1}/{max_retries}. Retrying...")
                await asyncio.sleep(retry_delay * (2 ** attempt) + random.uniform(1, 3))  # Add jitter
                continue
                
            elif response.status_code != 200:
                print(f"Error from API: {response.status_code} - {response.text}")
                
                # If it's the last attempt, return the error
                if attempt == max_retries - 1:
                    return {"error": f"API error: {response.status_code}", "details": response.text}
                
                # Otherwise retry
                await asyncio.sleep(retry_delay * (2 ** attempt))
                continue
            
            # Success case
            result = response.json()
            
            if args.provider == "Novita" or args.provider == "Together":
                return {
                    "text": result["choices"][0]["text"],
                    "finish_reason": result["choices"][0].get("finish_reason", ""),
                    "usage": result.get("usage", {})
                }
            elif args.provider == "Fireworks":
                return {
                    "text": result["choices"][0]["text"],
                    "finish_reason": result["choices"][0].get("finish_reason", ""),
                    "usage": result.get("usage", {})
                }
            elif args.provider == "vLLM":
                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"Unexpected vLLM response format: {result}")
                    return {"error": "Invalid response format"}
                
                choice = result["choices"][0]
                model_family = get_model_family(args.model)
                
                if model_family == "gpt-oss":
                    # GPT-OSS uses chat/completions format
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        # Check for both 'reasoning' and 'reasoning_content' fields
                        reasoning = choice["message"].get("reasoning", "") or choice["message"].get("reasoning_content", "")
                        
                        # GPT-OSS behavior: The prompt already ends with "<think>\n" so we should NOT
                        # add another opening <think> tag. We only need to ensure the content is
                        # properly closed with </think> for the chunking logic.
                        # The full_cot will be: prompt + content = "...Solution: \n<think>\n" + content
                        if "<think>" not in content and "</think>" not in content:
                            # If we have substantial separate reasoning, prepend it then add closing tag
                            if reasoning and len(reasoning) > 100:
                                content = f"{reasoning}\n</think>\n{content}"
                            else:
                                # Content is the reasoning, just add closing tag at the end
                                # Find where the actual answer starts (after reasoning)
                                content = f"{content}\n</think>"
                        
                        return {
                            "text": content,
                            "finish_reason": choice.get("finish_reason", ""),
                            "usage": result.get("usage", {})
                        }
                    else:
                        print(f"Missing 'message.content' in GPT-OSS vLLM choice: {choice}")
                        return {"error": "Missing message content in response"}
                else:
                    # Qwen and default models use completions format
                    if "text" not in choice:
                        print(f"Missing 'text' in vLLM choice: {choice}")
                        # Fallback: check for 'message' content if it's a chat response
                        if "message" in choice and "content" in choice["message"]:
                             return {
                                "text": choice["message"]["content"],
                                "finish_reason": choice.get("finish_reason", ""),
                                "usage": result.get("usage", {})
                            }
                        return {"error": "Missing text in response"}

                    return {
                        "text": choice["text"],
                        "finish_reason": choice.get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
            elif args.provider == "OpenRouter":
                if "choices" not in result or len(result["choices"]) == 0:
                    print(f"Unexpected OpenRouter response format: {result}")
                    return {"error": "Invalid response format"}
                
                choice = result["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                reasoning = message.get("reasoning", "")
                
                # If reasoning is provided separately, prepend it to content wrapped in <think> tags
                # But only if it's not already there (some models might include it in content)
                if reasoning and "<think>" not in content:
                    # Clean up content if it starts with </think> to avoid duplication
                    cleaned_content = content.lstrip()
                    if cleaned_content.startswith("</think>"):
                        content = cleaned_content[8:] # Remove </think>
                    
                    if is_prefilling_reasoning:
                        # If we are pre-filling reasoning, we don't need the opening <think> tag
                        content = f"{reasoning}\n</think>\n{content}"
                    else:
                        content = f"<think>\n{reasoning}\n</think>\n{content}"
                
                return {
                    "text": content,
                    "finish_reason": choice.get("finish_reason", ""),
                    "usage": result.get("usage", {})
                }
                
        except Exception as e:
            print(f"Exception during API request (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {repr(e)}")
            
            # If it's the last attempt, return the error
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            
            # Otherwise retry
            await asyncio.sleep(retry_delay * (2 ** attempt))
    
    # If we get here, all retries failed
    return {"error": "All API request attempts failed"}

async def handle_streaming_response(api_url: str, headers: Dict, payload: Dict) -> Dict:
    """Handle streaming responses from Together or Fireworks API."""
    try:
        # Initialize variables to collect the response
        collected_text = ""
        finish_reason = None
        usage = None
        
        # Make the streaming request
        client = await get_client()
        async with client.stream("POST", api_url, headers=headers, json=payload, timeout=3600) as response:
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "details": await response.aread()}
            
            # Process the streaming response
            async for chunk in response.aiter_lines():
                # Skip empty lines
                if not chunk.strip():
                    continue
                
                # Check for the end of the stream
                if chunk == "data: [DONE]":
                    break
                
                # Parse the chunk
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])  # Remove "data: " prefix
                        
                        # Extract text from the chunk
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            
                            # Get text content
                            if "text" in choice and choice["text"]:
                                collected_text += choice["text"]
                            elif "delta" in choice and "content" in choice["delta"]:
                                collected_text += choice["delta"]["content"]
                            
                            # Check for finish reason
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                        
                        # Get usage information from the last chunk
                        if "usage" in data and data["usage"]:
                            usage = data["usage"]
                            
                    except json.JSONDecodeError:
                        print(f"Failed to parse chunk: {chunk}")
        
        # For Together API, we need to handle the <think> token and the newline after it
        if args.provider == "Together":
            if collected_text.startswith("<think>\n"):
                # Remove the <think> token and the newline after it
                collected_text = collected_text[len("<think>\n"):]
            elif collected_text.startswith("<think>"):
                # Remove just the <think> token if there's no newline
                collected_text = collected_text[len("<think>"):]
        
        return {
            "text": collected_text,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {}
        }
        
    except Exception as e:
        print(f"Exception during streaming: {e}")
        return {"error": f"Streaming exception: {str(e)}"}

async def generate_base_solution(problem: Dict, temperature: float = 0.6) -> Dict:
    """
    Generate a base solution for a problem using parallel requests.
    Sends 100 parallel requests and returns the first valid solution based on base_solution_type.
    If too many chunk-overflow failures occur, cancels and retries the batch.
    
    Args:
        problem: Problem dictionary
        temperature: Temperature for generation
        
    Returns:
        Dictionary with the generated solution
    """
    # Create prompt similar to generate_cots_math.py
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n"
    
    num_parallel_requests = args.max_concurrent_requests
    max_batch_retries = 3
    chunk_overflow_threshold = 5  # Cancel batch after this many consecutive chunk overflows
    
    async def single_request(request_id: int) -> Dict:
        """Make a single API request and process the result."""
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
            if "error" in response:
                return {"error": response["error"], "request_id": request_id}
            
            solution_text = response['text']
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(solution_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            # Check chunk count is within boundary
            full_cot = prompt + solution_text
            # Handle multiple <think> tags (e.g., when model outputs its own <think> but prompt already has one)
            if "<think>" in full_cot:
                thinking_text = full_cot
                while "<think>" in thinking_text:
                    thinking_text = thinking_text.split("<think>", 1)[1].strip()
                if "</think>" in thinking_text:
                    thinking_text = thinking_text.split("</think>")[0].strip()
            else:
                thinking_text = solution_text
            chunks = split_solution_into_chunks(thinking_text)
            num_chunks = len(chunks)
            
            return {
                "prompt": prompt,
                "solution": solution_text,
                "full_cot": full_cot,
                "answer": answer,
                "is_correct": is_correct,
                "num_chunks": num_chunks,
                "request_id": request_id
            }
        except Exception as e:
            return {"error": str(e), "request_id": request_id}
    
    for batch_attempt in range(max_batch_retries):
        if batch_attempt > 0:
            print(f"  Retrying batch (attempt {batch_attempt + 1}/{max_batch_retries})...")
        
        # Create all tasks
        tasks = [asyncio.create_task(single_request(i)) for i in range(num_parallel_requests)]
        
        # Wait for the first valid solution based on base_solution_type
        pending = set(tasks)
        result = None
        error_count = 0
        invalid_count = 0
        consecutive_chunk_overflow = 0
        should_retry_batch = False
        
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                try:
                    task_result = task.result()
                    
                    # Skip if there's an error
                    if "error" in task_result:
                        error_count += 1
                        print(f"  Request failed (errors: {error_count}, invalid: {invalid_count}, pending: {len(pending)})")
                        consecutive_chunk_overflow = 0  # Reset on non-chunk error
                        continue
                    
                    # Check correctness criteria
                    correctness_ok = (
                        (args.base_solution_type == "correct" and task_result.get("is_correct")) or
                        (args.base_solution_type == "incorrect" and not task_result.get("is_correct"))
                    )
                    
                    # Check chunk boundary
                    num_chunks = task_result.get("num_chunks", 0)
                    chunks_ok = num_chunks <= args.max_chunks
                    
                    # Check if this result matches ALL criteria
                    if correctness_ok and chunks_ok:
                        result = task_result
                        print(f"  Found valid solution with {num_chunks} chunks")
                        break
                    else:
                        # Valid response but doesn't meet criteria
                        invalid_count += 1
                        reason = []
                        if not correctness_ok:
                            reason.append(f"wrong correctness")
                            consecutive_chunk_overflow = 0  # Reset on correctness error
                        if not chunks_ok:
                            reason.append(f"too many chunks ({num_chunks} > {args.max_chunks})")
                            consecutive_chunk_overflow += 1
                        else:
                            consecutive_chunk_overflow = 0
                        
                        print(f"  Request invalid: {', '.join(reason)} (errors: {error_count}, invalid: {invalid_count}, pending: {len(pending)})")
                        
                        # Check if we should abort and retry the batch
                        if consecutive_chunk_overflow >= chunk_overflow_threshold:
                            print(f"  Detected {consecutive_chunk_overflow} consecutive chunk overflows - aborting batch to retry")
                            should_retry_batch = True
                            break
                            
                except Exception:
                    error_count += 1
                    print(f"  Request exception (errors: {error_count}, invalid: {invalid_count}, pending: {len(pending)})")
                    consecutive_chunk_overflow = 0
                    continue
            
            if result or should_retry_batch:
                break
        
        # Cancel all remaining tasks
        for task in pending:
            task.cancel()
        
        # Wait for cancellation to complete (suppress CancelledError)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        
        if result:
            print(f"Found valid {args.base_solution_type} solution from request {result.get('request_id', 'unknown')}")
            # Remove request_id from result before returning
            result.pop('request_id', None)
            return result
        
        if not should_retry_batch:
            # All requests completed without finding a valid solution and no retry needed
            break
    
    # If no valid solution found after all retries
    return {
        "prompt": prompt,
        "solution": "Error: No valid solution found after parallel requests",
        "error": "No valid solution found"
    }

async def generate_rollout(problem: Dict, chunk_text: str, full_cot_prefix: str, temperature: float = 0.7, rollout_type: str = 'default') -> Dict:
    """
    Generate a rollout by removing a specific chunk and regenerating from that point.
    
    Args:
        problem: Problem dictionary
        chunk_text: Text of the current chunk to remove
        full_cot_prefix: Full CoT text up to and including the current chunk
        temperature: Temperature for generation
        rollout_type: Type of rollout to generate
    Returns:
        Dictionary with the rollout result
    """
    # Remove the current chunk from the prefix to see how it gets regenerated
    prefix_without_chunk = full_cot_prefix.replace(chunk_text, "").strip()
    
    # Create prompt with the prefix without the current chunk
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
    
    if rollout_type == 'forced_answer':
        prompt += "\n</think>\n\nTherefore, the final answers is \\boxed{"
    
    max_retries = args.max_retries
    retry_delay = 2 if max_retries > 0 else None
    
    for attempt in range(max_retries):
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
            if "error" in response:
                raise Exception(response["error"])
            rollout_text = response['text']
            chunks_from_rollout = split_solution_into_chunks(rollout_text)
            chunk_resampled = chunks_from_rollout[0] if chunks_from_rollout else ""
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(f"{prompt}{rollout_text}" if rollout_type == 'forced_answer' else rollout_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "chunk_removed": chunk_text,
                "prefix_without_chunk": prefix_without_chunk,
                "chunk_resampled": chunk_resampled,
                "rollout": rollout_text,
                "full_cot": f"{prompt}{rollout_text}",
                "answer": answer,
                "is_correct": is_correct
            }
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "chunk_removed": chunk_text,
                    "prefix_without_chunk": prefix_without_chunk,
                    "error": str(e)
                }

async def process_problem(problem_idx: int, problem: Dict) -> None:
    """
    Process a single problem: generate base solution and rollouts.
    
    Args:
        problem_idx: Index of the problem
        problem: Problem dictionary
    """
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)
    
    # Save problem
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, 'w', encoding='utf-8') as f:
            json.dump(problem, f, indent=2)
    
    # Check if base solution already exists
    base_solution_file = problem_dir / "base_solution.json"
    base_solution = None
    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, 'r', encoding='utf-8') as f:
            base_solution = json.load(f)
            print(f"Problem {problem_idx}: Loaded existing base solution")
            
            # Recalculate accuracy for base solution if needed
            if not args.skip_recalculate and 'solution' in base_solution:
                extracted_answers = extract_boxed_answers(base_solution['solution'])
                answer = extracted_answers[0] if extracted_answers else ""
                is_correct = False
                
                if problem.get('gt_answer') and answer:
                    is_correct = check_answer(answer, problem['gt_answer'])
                
                # Update if different
                if base_solution.get('answer') != answer or base_solution.get('is_correct') != is_correct:
                    print(f"Problem {problem_idx}: Updating base solution accuracy")
                    base_solution['answer'] = answer
                    base_solution['is_correct'] = is_correct
                    
                    # Save updated base solution
                    with open(base_solution_file, 'w', encoding='utf-8') as f:
                        json.dump(base_solution, f, indent=2)
    
    # Generate base solution if needed
    if base_solution is None:
        print(f"Problem {problem_idx}: Generating {args.base_solution_type} base solution (100 parallel requests)")
        base_solution = await generate_base_solution(problem, args.temperature)
        
        # Check if we got a valid solution or an error
        if "error" in base_solution:
            print(f"Problem {problem_idx}: Failed to find a valid {args.base_solution_type} solution after 100 parallel requests")
            print(f"Error: {base_solution.get('error')}")
            return
        
        # Save base solution
        with open(base_solution_file, 'w', encoding='utf-8') as f:
            json.dump(base_solution, f, indent=2)
    
    # Get the source text for chunking
    source_text = base_solution["full_cot"]
    print(f"Problem {problem_idx}: Using full CoT for chunking")
    
    # Extract the solution part for chunking
    # Handle multiple <think> tags (e.g., when model outputs its own <think> but prompt already has one)
    if "<think>" in source_text:
        solution_text = source_text
        while "<think>" in solution_text:
            solution_text = solution_text.split("<think>", 1)[1].strip()
        if "</think>" in solution_text:
            solution_text = solution_text.split("</think>")[0].strip()
    else:
        solution_text = source_text
    
    # Save chunks to a separate file
    chunks_file = problem_dir / "chunks.json"
    
    if not chunks_file.exists() or args.force:
        chunks = split_solution_into_chunks(solution_text)
        print(f"Problem {problem_idx}: Split into {len(chunks)} chunks")
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({"source_text": source_text, "solution_text": solution_text, "chunks": chunks}, f, indent=2)
        
        print(f"Problem {problem_idx}: Saved chunks to {chunks_file}")
    else:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)['chunks']
        print(f"Problem {problem_idx}: Loaded {len(chunks)} existing chunks")
        
    if len(chunks) > args.max_chunks:
        print(f"Problem {problem_idx}: Too many chunks. Will not generate rollouts.")
        return
    
    # Build cumulative chunks for proper continuation
    cumulative_chunks = []
    current_cumulative = ""
    for chunk in chunks:
        current_cumulative += chunk + " "
        cumulative_chunks.append(current_cumulative.strip())
    
    # Collect chunk work items
    chunk_work_items = []
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        if args.include_chunks and str(chunk_idx) not in args.include_chunks.split(","):
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Skipping (not in include_chunks)")
            continue
        
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if solutions already exist
        solutions_file = chunk_dir / "solutions.json"
        existing_solutions = []
        valid_existing_solutions = []
        
        if solutions_file.exists() and not args.force:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                existing_solutions = json.load(f)
                
                # Recalculate accuracy for existing rollouts if needed
                if not args.skip_recalculate:
                    updated_count = 0
                    for rollout in existing_solutions:
                        if 'rollout' in rollout and 'error' not in rollout:
                            extracted_answers = extract_boxed_answers(rollout['full_cot'] if args.rollout_type == 'forced_answer' else rollout['rollout'])
                            answer = extracted_answers[0] if extracted_answers else ""
                            is_correct = False
                            
                            if problem.get('gt_answer') and answer:
                                is_correct = check_answer(answer, problem['gt_answer'])
                            
                            # Update if different
                            if rollout.get('answer') != answer or rollout.get('is_correct') != is_correct:
                                updated_count += 1
                                rollout['answer'] = answer
                                rollout['is_correct'] = is_correct
                    
                    if updated_count > 0:
                        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Updated accuracy for {updated_count} rollouts")
                        # Save updated rollouts
                        with open(solutions_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_solutions, f, indent=2)
                
                # Filter for valid solutions (has answer and no error)
                valid_existing_solutions = [s for s in existing_solutions if 'answer' in s and 'error' not in s]
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Found {len(valid_existing_solutions)} valid solutions")
        
        # Check if rollouts are needed
        num_rollouts_needed = args.num_rollouts - len(valid_existing_solutions)
        
        if num_rollouts_needed > 0:
            chunk_work_items.append({
                'chunk_idx': chunk_idx,
                'chunk': chunk,
                'full_prefix': full_prefix,
                'chunk_dir': chunk_dir,
                'solutions_file': solutions_file,
                'existing_solutions': existing_solutions,
                'num_rollouts_needed': num_rollouts_needed
            })
        else:
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Already have {len(valid_existing_solutions)} valid solutions")
    
    if not chunk_work_items:
        return
    
    # Calculate how many chunks can be processed in parallel
    # Each chunk needs num_rollouts concurrent slots
    max_parallel_chunks = max(1, args.max_concurrent_requests // args.num_rollouts)
    
    print(f"Problem {problem_idx}: Processing {len(chunk_work_items)} chunks with max {max_parallel_chunks} chunks in parallel")
    
    # Process chunks in parallel batches
    for batch_start in range(0, len(chunk_work_items), max_parallel_chunks):
        batch_end = min(batch_start + max_parallel_chunks, len(chunk_work_items))
        batch_items = chunk_work_items[batch_start:batch_end]
        
        if len(batch_items) > 1:
            print(f"Problem {problem_idx}: Processing chunks {[item['chunk_idx'] for item in batch_items]} in parallel")
        
        # For Local provider, process each chunk sequentially (batch mode doesn't support multi-chunk parallel)
        if args.provider == "Local":
            for work_item in batch_items:
                chunk_idx = work_item['chunk_idx']
                chunk = work_item['chunk']
                full_prefix = work_item['full_prefix']
                solutions_file = work_item['solutions_file']
                existing_solutions = work_item['existing_solutions']
                num_rollouts_needed = work_item['num_rollouts_needed']
                
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {num_rollouts_needed} rollouts")
                
                # Create prompts for all rollouts
                prompts = []
                for _ in tqdm(range(num_rollouts_needed), desc="Generating rollouts"):
                    # Remove the current chunk from the prefix to see how it gets regenerated
                    prefix_without_chunk = full_prefix.replace(chunk, "").strip()
                    
                    # Create prompt with the prefix without the current chunk
                    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
                    
                    if args.rollout_type == 'forced_answer':
                        prompt += "\n</think>\n\nTherefore, the final answers is \\boxed{"
                    
                    prompts.append(prompt)
                
                # Generate all rollouts in batch
                batch_results = generate_with_local_model_batch(prompts, args.temperature, args.top_p, args.max_tokens)
                
                # Process results
                new_solutions = []
                for i, result in enumerate(batch_results):
                    rollout_text = result.get('text', '')
                    
                    # Skip if there was an error
                    if 'error' in result:
                        new_solutions.append({"error": result['error']})
                        continue
                    
                    # Create the rollout object
                    prefix_without_chunk = full_prefix.replace(chunk, "").strip()
                    chunk_resampled = split_solution_into_chunks(rollout_text)[0] if rollout_text else ""
                    
                    # Extract answer and check correctness
                    prompt = prompts[i]
                    extracted_answers = extract_boxed_answers(f"{prompt}{rollout_text}" if args.rollout_type == 'forced_answer' else rollout_text)
                    answer = extracted_answers[0] if extracted_answers else ""
                    is_correct = False
                    
                    if problem.get('gt_answer') and answer:
                        is_correct = check_answer(answer, problem['gt_answer'])
                    
                    new_solutions.append({
                        "chunk_removed": chunk,
                        "prefix_without_chunk": prefix_without_chunk,
                        "chunk_resampled": chunk_resampled,
                        "rollout": rollout_text,
                        "full_cot": f"{prompt}{rollout_text}",
                        "answer": answer,
                        "is_correct": is_correct
                    })
                
                # Combine with existing solutions and save
                all_solutions = existing_solutions + new_solutions
                with open(solutions_file, 'w', encoding='utf-8') as f:
                    json.dump(all_solutions, f, indent=2)
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Saved {len(all_solutions)} solutions")
        else:
            # For API providers, process multiple chunks in parallel with separate progress bars
            async def process_chunk_batch(work_item: Dict) -> None:
                """Process a single chunk's rollouts with its own progress bar."""
                chunk_idx = work_item['chunk_idx']
                chunk = work_item['chunk']
                full_prefix = work_item['full_prefix']
                solutions_file = work_item['solutions_file']
                existing_solutions = work_item['existing_solutions']
                num_rollouts_needed = work_item['num_rollouts_needed']
                
                # Create tasks for all rollouts in this chunk
                tasks = [generate_rollout(problem, chunk, full_prefix, args.temperature, args.rollout_type) 
                        for _ in range(num_rollouts_needed)]
                
                # Create a progress bar for this chunk
                new_solutions = []
                pbar = tqdm(asyncio.as_completed(tasks), 
                           total=len(tasks), 
                           desc=f"Chunk {chunk_idx}", 
                           leave=False,
                           position=batch_items.index(work_item))
                
                for f in pbar:
                    new_solutions.append(await f)
                
                # Combine with existing solutions and save
                all_solutions = existing_solutions + new_solutions
                with open(solutions_file, 'w', encoding='utf-8') as f:
                    json.dump(all_solutions, f, indent=2)
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Saved {len(all_solutions)} solutions")
            
            # Process all chunks in this batch concurrently
            await asyncio.gather(*[process_chunk_batch(item) for item in batch_items])

async def main():
    """Main function to run the script."""
    # Check API keys based on provider
    if args.provider == "Novita" and not NOVITA_API_KEY:
        raise ValueError("NOVITA_API_KEY not found in environment variables")
    elif args.provider == "Together" and not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in environment variables")
    elif args.provider == "Fireworks" and not FIREWORKS_API_KEY:
        raise ValueError("FIREWORKS_API_KEY not found in environment variables")
    elif args.provider == "vLLM":
        if len(VLLM_API_URLS) > 1:
            print(f"Using {len(VLLM_API_URLS)} vLLM servers: {VLLM_API_URLS}")
        else:
            print(f"Using vLLM server at {VLLM_API_URLS[0]}")
    elif args.provider == "OpenRouter" and not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Initialize semaphore
    global request_semaphore
    request_semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    
    # Load problems
    problems = load_math_problems(problem_type=args.type, level=args.level, num_problems=args.num_problems, split=args.split, include_problems=args.include_problems)
    
    if args.exclude_problems:
        exclude_problems = [int(id) for id in args.exclude_problems.split(",")]
        problems = [problem for problem in problems if problem[0] not in exclude_problems]
        
    if args.include_problems:
        include_problems = [int(id) for id in args.include_problems.split(",")]
        problems = [problem for problem in problems if problem[0] in include_problems]
    
    if not problems:
        print(f"No problems loaded. Exiting.")
        exit(1)

    print(f"Loaded {len(problems)} problems.")
    
    # Process problems
    try:
        # Process problems
        for problem_idx, problem in tqdm(problems, desc="Processing problems"):
            await process_problem(problem_idx, problem)
    finally:
        # Close the global client if it exists
        global global_client
        if global_client:
            await global_client.aclose()

if __name__ == "__main__":
    if sys.platform == 'win32':
        # Use ProactorEventLoop instead of SelectorEventLoop to avoid 512 fd limit
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())


