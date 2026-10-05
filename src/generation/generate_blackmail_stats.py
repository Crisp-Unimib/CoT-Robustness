
import os
import sys
import json
import asyncio
import argparse
import random
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from dotenv import load_dotenv

import blackmail_utils as bu

# Load environment variables
load_dotenv()

# Global state
console = Console()
request_semaphore: Optional[asyncio.Semaphore] = None
global_client: Optional[httpx.AsyncClient] = None
vllm_request_counter = 0
VLLM_API_URLS = []

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

async def make_vllm_request(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 0.95,
    stop: Optional[List[str]] = None
) -> Dict:
    """Make a single vLLM completion request."""
    global vllm_request_counter
    
    server_idx = vllm_request_counter % len(VLLM_API_URLS)
    vllm_request_counter += 1
    api_url = f"{VLLM_API_URLS[server_idx]}/completions"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
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
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                elif response.status_code == 429:
                    await asyncio.sleep(retry_delay * (2 ** attempt) + random.uniform(1, 3))
                    continue
                elif response.status_code != 200:
                    if attempt == max_retries - 1:
                        return {"error": f"API error: {response.status_code}"}
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                
                result = response.json()
                if "choices" not in result or len(result["choices"]) == 0:
                    return {"error": "Invalid response format"}
                
                choice = result["choices"][0]
                return {
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason", "")
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"Request exception: {str(e)}"}
                await asyncio.sleep(retry_delay * (2 ** attempt))
    
    return {"error": "All API request attempts failed"}

async def process_scenario(
    scenario: Dict,
    model: str,
    n_samples: int,
    output_dir: Path,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Process a single scenario: generate samples and classify."""
    scenario_id = scenario.get('condition_id', 'unknown')
    scenario_output_dir = output_dir / scenario_id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    
    prompt = bu.build_blackmail_prompt(scenario)
    
    # Save prompt logging
    with open(scenario_output_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    
    responses = []
    
    # Generate batch
    tasks = []
    for _ in range(n_samples):
        tasks.append(make_vllm_request(
            model=model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p
        ))
    
    generated_results = await asyncio.gather(*tasks)
    
    blackmail_count = 0
    valid_count = 0
    
    processed_samples = []
    
    # Classify each
    classification_tasks = []
    
    # We can process classifications in parallel too, but limit concurrency via semaphore?
    # OpenRouter limits might apply. We'll do it sequentially or batched to be safe?
    # Or just use the make_openrouter_request logic which handles rate limits.
    
    for gen_res in generated_results:
        if "error" in gen_res:
            processed_samples.append({
                "response": None,
                "error": gen_res["error"],
                "is_blackmail": None
            })
            continue
            
        text = gen_res["text"]
        full_response = prompt + text
        
        # Parallel classification might hit rate limits fast for 50 samples * 10 scenarios = 500 calls.
        # We'll do them in parallel but relying on the retry logic in bu.make_openrouter_request.
        
        classification_tasks.append(
            bu.classify_blackmail_response(
                full_response,
                os.getenv("OPENROUTER_API_KEY"),
                args.controller_model,
                args.openrouter_url
            )
        )
    
    if classification_tasks:
        classifications = await asyncio.gather(*classification_tasks)
        
        # Merge results
        gen_idx = 0
        class_idx = 0
        for gen_res in generated_results:
            if "error" in gen_res:
                continue
            
            text = gen_res["text"]
            is_blackmail, reasoning = classifications[class_idx]
            class_idx += 1
            
            scratchpad = bu.extract_scratchpad_reasoning(text)
            
            processed_samples.append({
                "response": text,
                "scratchpad": scratchpad,
                "is_blackmail": is_blackmail,
                "classification_reasoning": reasoning
            })
            
            if is_blackmail:
                blackmail_count += 1
            valid_count += 1
            
    # Save results
    with open(scenario_output_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, indent=2)
        
    blackmail_rate = (blackmail_count / valid_count) * 100 if valid_count > 0 else 0
    
    return {
        "scenario_id": scenario_id,
        "n_samples": n_samples,
        "valid_samples": valid_count,
        "blackmail_count": blackmail_count,
        "blackmail_rate": blackmail_rate
    }

async def main():
    parser = argparse.ArgumentParser(description="Generate Blackmail Statistics")
    parser.add_argument("--model", type=str, required=True, help="Model name for vLLM")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--num_servers", type=int, default=1, help="Number of vLLM servers")
    parser.add_argument("--n_samples", "-n", type=int, default=50, help="Samples per scenario")
    parser.add_argument("--blackmail_prompts_dir", type=str, default="configs/blackmail_prompts", help="Directory containing explicit blackmail scenarios")
    parser.add_argument("--output_dir", type=str, default="data/processed/blackmail_stats", help="Output directory")
    parser.add_argument("--controller_model", type=str, default="google/gemini-3-flash-preview", help="OpenRouter model for classification")
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter API URL")
    
    # Generation params
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for response")
    parser.add_argument("--concurrent_requests", type=int, default=50, help="Max concurrent requests")
    
    global args
    args = parser.parse_args()
    
    # Setup vLLM URLs
    global VLLM_API_URLS
    if args.num_servers > 1:
        base_port = 8000
        for i in range(args.num_servers):
            VLLM_API_URLS.append(f"http://localhost:{base_port + i}/v1")
    else:
        VLLM_API_URLS = [args.url]
        
    # Setup semaphore
    global request_semaphore
    request_semaphore = asyncio.Semaphore(args.concurrent_requests)
    
    # Load scenarios
    scenarios = bu.load_blackmail_scenarios(Path(args.blackmail_prompts_dir))
    if not scenarios:
        console.print("[red]No scenarios found[/red]")
        return
        
    console.print(f"Found {len(scenarios)} scenarios. Generating {args.n_samples} samples each.")
    
    stats_output_dir = Path(args.output_dir) / args.model.split("/")[-1]
    stats_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("[cyan]Processing Scenarios...[/cyan]", total=len(scenarios))
        
        for scenario in scenarios:
            res = await process_scenario(scenario, args.model, args.n_samples, stats_output_dir, args)
            results.append(res)
            progress.advance(task_id)
            
            console.print(f"[green]Scenario {res['scenario_id']}: {res['blackmail_rate']:.1f}% Blackmail ({res['blackmail_count']}/{res['valid_samples']})[/green]")
            
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(stats_output_dir / "summary.csv", index=False)
    console.print(f"\n[bold green]Stats saved to {stats_output_dir}/summary.csv[/bold green]")
    
    await close_client()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
