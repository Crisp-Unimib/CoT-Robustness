"""Sample answers from LLMs on QA task via vLLM OpenAI-compatible API.

Targets K_HEAT uncertainty only — no token log-probs, no embeddings,
no p_true, no training split generation.

python generate_answers.py --model_name openai/gpt-oss-20b --dataset trivia_qa --num_samples 500 --num_test_samples 400 --num_generations 10 --temperature 1.0 --metric llm --vllm_base_url http://127.0.0.1:8000/v1 --model_max_new_tokens 16000 --reasoning_effort high --experiment_lot gpt_oss_20b_v1 --entity <your-wandb-entity> --compute_uncertainties --skip_generation --eval_wandb_runid 8amyqt0h --restore_entity_eval <your-wandb-entity>

python generate_answers.py --model_name Qwen/Qwen3-4B-Thinking-2507 --dataset trivia_qa --num_samples 500 --num_test_samples 400 --num_generations 10 --temperature 1.0 --metric llm --vllm_base_url http://127.0.0.1:8000/v1 --model_max_new_tokens 16000 --reasoning_effort high --experiment_lot qwen3_4b_thinking_v1 --entity <your-wandb-entity> --compute_uncertainties

"""
import os
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

import numpy as np
import wandb
import time

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from compute_uncertainty_measures import main as main_compute
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generation.utils import split_solution_into_chunks
from compute_uncertainty_measures import get_entailment_graph, heat_only_graph_entropies
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta


utils.setup_logger()


# ---------------------------------------------------------------------------
# Chat template registry
# model_name -> Callable[[str], Tuple[List[Dict], Dict]]
# ---------------------------------------------------------------------------
CHAT_TEMPLATES = {}

_CACHED_TOKENIZERS = {}

def get_cached_tokenizer(model_name: str):
    if model_name not in _CACHED_TOKENIZERS:
        from transformers import AutoTokenizer
        _CACHED_TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _CACHED_TOKENIZERS[model_name]


def _make_gpt_oss_template(reasoning_effort: str):
    """Factory for gpt-oss-20b/120b harmony format templates."""
    def _template(prompt_text: str) -> Tuple[List[Dict], Dict]:
        messages = [{"role": "user", "content": prompt_text}]
        extra_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {
                    "reasoning_effort": reasoning_effort
                }
            }
        }
        return messages, extra_kwargs
    return _template


def build_request_params(model_name: str, prompt_text: str) -> Tuple[List[Dict], Dict]:
    """Return (messages, extra_kwargs) for the given model.

    extra_kwargs are forwarded as **kwargs to client.chat.completions.create().
    Default: single user message, no extra kwargs.
    """
    if model_name in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[model_name](prompt_text)
    return [{"role": "user", "content": prompt_text}], {}


# ---------------------------------------------------------------------------
# Reasoning-model answer extraction
# ---------------------------------------------------------------------------
def extract_final_answer(raw_output: str, model_name: str = "") -> str:
    """Strip reasoning preamble; return final answer only.

    Handles two formats (checked in order):
    1. gpt-oss harmony: <|channel|>final<|message|>...<|return|>
    2. DeepSeek-R1 style: </think>...
    Falls back to the full output if neither tag is found.
    """
    # --- Harmony format (gpt-oss) ---
    HARMONY_FINAL = "<|channel|>final<|message|>"
    if HARMONY_FINAL in raw_output:
        idx = raw_output.rfind(HARMONY_FINAL)
        after = raw_output[idx + len(HARMONY_FINAL):]
        for stop in ("<|return|>", "<|end|>"):
            if stop in after:
                after = after[:after.index(stop)]
        return after.strip()

    # --- DeepSeek-R1 / Qwen3 / </think> style ---
    tag = '</think>'
    idx = raw_output.rfind(tag)
    if idx != -1:
        answer = raw_output[idx + len(tag):]
    else:
        answer = raw_output

    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in answer:
            answer = answer[:answer.index(stop)]
            
    answer = answer.strip()

    # Strip few-shot prompt artifact: "Answer: foo" → "foo"
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].strip()
    return answer


# ---------------------------------------------------------------------------
# LLM-based accuracy judge (via OpenRouter)
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def llm_accuracy(
    judge_client: OpenAI,
    judge_model: str,
    question: str,
    predicted_answer: str,
    correct_answers: list,
    max_tokens: int = 16,
) -> float:
    """Use an LLM judge to evaluate semantic correctness of an answer.

    Returns 1.0 if the judge says correct, 0.0 otherwise.
    On unexpected errors (e.g. BadRequestError), logs a warning and returns 0.0.
    """
    if not correct_answers:
        return 0.0

    ref_list = "\n".join(f"- {a}" for a in correct_answers)
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answers (any is valid):\n{ref_list}\n\n"
        f"Predicted answer: {predicted_answer}\n\n"
        f"Is the predicted answer correct? Paraphrases, abbreviations, "
        f"and equivalent formulations count as correct. "
        f"Respond with only \"yes\" or \"no\"."
    )
    for attempt in range(5):
        try:
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": 'You are an answer-correctness judge. Respond with a single word: "yes" or "no".'},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if not response.choices:
                raise ValueError("API returned no choices")
            
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("API returned empty content")

            verdict = extract_final_answer(raw).lower().strip()
            if verdict.startswith("yes"):
                return 1.0
            elif verdict.startswith("no"):
                return 0.0
            else:
                logging.warning("Judge returned unexpected response: %r", raw)
                # Don't retry on valid-but-weird responses, just treat as wrong.
                return 0.0

        except (openai.APITimeoutError, openai.APIConnectionError):
            # Network errors: re-raise to let tenacity handle them (if we want to keep tenacity).
            # But since we have a loop here, we can just handle them here too.
            logging.warning(f"Judge network error (attempt {attempt+1}/5). Retrying...")
            time.sleep(1)
        except Exception as exc:
            logging.warning(f"Judge call failed: {exc} (attempt {attempt+1}/5). Retrying...")
            time.sleep(1)

    logging.error("Judge failed after 5 attempts. Treating as wrong.")
    return 0.0


# ---------------------------------------------------------------------------
# vLLM API call with retry
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_vllm(client: OpenAI, model_name: str, prompt_text: str,
              temperature: float, max_tokens: int) -> Tuple[str, str]:
    """Call vLLM chat completions endpoint with retry on transient errors.

    Returns:
        (content, reasoning_trace): strings.
        reasoning_trace is populated if vLLM server-side parser is active.
    """
    messages, extra_kwargs = build_request_params(model_name, prompt_text)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_kwargs,
    )
    msg = response.choices[0].message
    content = msg.content

    # Check for reasoning field (varies by vLLM version/config)
    reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None) or ""

    if content is None:
        logging.warning('Content is None, but captured reasoning length: ' + str(len(reasoning)))
        return "", reasoning
    return content, reasoning


@retry(
    retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_vllm_branch(
    client: OpenAI,
    model_name: str,
    question_prompt: str,
    reasoning_prefix: str,
    temperature: float,
    max_tokens: int = 128,
    stop_sequences: list = None,
    n: int = 1,
) -> List[str]:
    """Sample continuations of reasoning_prefix from the model."""
    stop_sequences = ['\n', '.']
    
    if "gpt-oss" in model_name.lower():
        # Use the cached tokenizer to render the chat template into a raw prompt,
        # then continue mid-reasoning via the raw completions API.
        tok = get_cached_tokenizer('openai/gpt-oss-20b')
        messages = [{"role": "user", "content": question_prompt}]
        raw_prompt_prefix = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_effort="high"
        ).rstrip()  # Remove trailing newline after <|start|>assistant
        raw_prompt = f"{raw_prompt_prefix}<|channel|>analysis<|message|>{reasoning_prefix}"
    else:
        # Fallback to ChatML / Qwen format
        raw_prompt = f"<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba Cloud. You should think step-by-step.<|im_end|>\n<|im_start|>user\n{question_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning_prefix}"

    response = client.completions.create(
        model=model_name,
        prompt=raw_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequences,
        n=n,
    )

    branches = []
    for choice in response.choices:
        branches.append(choice.text)
            
    return branches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args, parser):

    # ---- Deprecated argument warnings ----
    _DEPRECATED_ARGS = [
        'compute_p_true',
        'p_true_num_fewshot',
        'p_true_hint',
        'ood_train_dataset',
        'get_training_set_generations',
        'get_training_set_generations_most_likely_only',
        'compute_accuracy_at_all_temps',
    ]
    for arg_name in _DEPRECATED_ARGS:
        default_val = parser.get_default(arg_name)
        if getattr(args, arg_name, default_val) != default_val:
            logging.warning('Argument --%s is deprecated and ignored.', arg_name)

    # ---- LLM judge setup ----
    judge_client = None
    if args.metric == 'llm':
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found in environment. "
                "Set it in .env or export it before running with --metric=llm.")
        judge_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60,
        )
        logging.info('LLM judge initialised: %s', args.judge_model)

    # ---- Register model-specific chat templates ----
    _gpt_oss_tmpl = _make_gpt_oss_template(args.reasoning_effort)
    CHAT_TEMPLATES["openai/gpt-oss-20b"]  = _gpt_oss_tmpl
    CHAT_TEMPLATES["openai/gpt-oss-120b"] = _gpt_oss_tmpl

    # ---- Entailment model setup ----
    logging.info("Loading entailment model (DeBERTa)")
    # Defaulting to DeBERTa for NLI as requested. Assumes caching is disabled/not required here.
    entailment_model = EntailmentDeberta(None, False)
    logging.info("Entailment model initialized.")

    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ.get('USER') or os.environ.get('USERNAME') or 'user'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")
    for i in range(3):
        try:
            wandb.init(
                entity=args.entity,
                project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
                dir=f"{scratch_dir}/{user}/uncertainty",
                config=args,
                notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
                tags=[args.model_name, args.dataset, args.brief_prompt, args.experiment_lot]
            )
            break
        except Exception:
            time.sleep(10 * 60)
    logging.info('Finished wandb init.')

    # Get accuracy metric (None for LLM judge — handled separately).
    metric = utils.get_metric(args.metric) if args.metric != 'llm' else None

    # Load dataset.
    wandb.log({"dataset": args.dataset})
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, _ = utils.split_dataset(train_dataset)

    if args.answerable_only:
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize vLLM client (replaces local HuggingFace model).
    wandb.log({"model": args.model_name})
    client = OpenAI(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        timeout=args.vllm_timeout,
    )

    # Start answer generation (validation split only).
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')

    dataset_split = 'validation'
    logging.info(80 * 'x')
    logging.info('Starting with dataset_split %s.', dataset_split)
    logging.info(80 * 'x')

    # This will store all input data and model predictions.
    accuracies, generations, results_dict = [], {}, {}
    reasoning_lengths = []
    
    # Structures for inline AUROC computation
    all_step_metrics = {} # metric_name -> list of values
    all_is_false = []     # list of 1 - accuracy

    dataset = validation_dataset
    possible_indices = range(0, len(dataset))

    # Evaluate over random subset of the datasets.
    indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
    experiment_details[dataset_split] = {'indices': indices}

    if args.num_samples > len(dataset):
        logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

    it = 0
    for index in tqdm(indices):
        it += 1

        # Grab example at index.
        example = dataset[index]
        question, context = example["question"], example['context']
        generations[example['id']] = {'question': question, 'context': context}
        correct_answer = example['answers']['text']

        current_input = make_prompt(
            context, question, None, BRIEF, args.brief_always and args.enable_brief)
        local_prompt = prompt + current_input

        logging.info('Current input: '.ljust(15) + current_input)

        full_responses = []

        # ── Generation 0: low-T (greedy) with trace length limits ──────────────
        current_max_steps = getattr(args, 'max_reasoning_steps', 20)
        attempts = 0
        
        while True:
            while True:
                raw_output, reasoning_trace = call_vllm(
                    client, args.model_name, local_prompt,
                    1, args.model_max_new_tokens)
                
                # Break if we have an answer OR if we have a reasoning trace
                if (raw_output and raw_output.strip()) or (reasoning_trace and reasoning_trace.strip()):
                    break
                
                logging.warning("Received empty response and reasoning from vLLM (low-T). Retrying in 1s...")
                time.sleep(1)

            predicted_answer = extract_final_answer(raw_output, model_name=args.model_name)

            # Check reasoning presence.
            has_inline_tags = '</think>' in raw_output or '<|channel|>' in raw_output
            has_server_reasoning = bool(reasoning_trace)
            
            # Determine logical reasoning string for split
            if has_server_reasoning:
                logic_str = reasoning_trace
            elif has_inline_tags:
                logic_str = raw_output[:len(raw_output)-len(predicted_answer)]
            else:
                logic_str = ""
                if not getattr(args, 'suppress_reasoning_warning', False):
                     logging.warning(
                         'Low-T response has no reasoning trace (no </think> tag, no harmony '
                         'channel, and reasoning_content is empty). If using a reasoning model, '
                         'check that --reasoning-parser is set on the vLLM server.')
                     args.suppress_reasoning_warning = True
            
            if current_max_steps <= 0 or not logic_str:
                break
                
            # Check length of the logic trace
            steps = split_solution_into_chunks(logic_str)
            if len(steps) > current_max_steps:
                attempts += 1
                logging.warning(f"Reasoning trace too long ({len(steps)} > {current_max_steps} steps). Discarding and resampling (Attempt {attempts}/5).")
                if attempts >= 5:
                    current_max_steps += 10
                    attempts = 0
                    logging.warning(f"Failed 5 times to find a trace within limit. Increasing max steps to {current_max_steps}.")
                continue
            
            break # Valid trace length found

        if has_server_reasoning:
            reasoning_lengths.append(len(reasoning_trace))
        elif has_inline_tags:
            reasoning_lengths.append(len(logic_str))

        # Accuracy (only for low-T).
        if correct_answer:
            if args.metric == 'llm':
                acc = llm_accuracy(
                    judge_client, args.judge_model,
                    question, predicted_answer, correct_answer,
                    max_tokens=args.judge_max_tokens,
                )
            else:
                acc = metric(predicted_answer, example, None)
        else:
            acc = 0.0

        logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
        if args.use_context:
            logging.info('context: '.ljust(15) + str(context))
        logging.info('question: '.ljust(15) + question)
        logging.info('low-t prediction: '.ljust(15) + predicted_answer)
        logging.info('correct answer: '.ljust(15) + str(correct_answer))
        logging.info('accuracy: '.ljust(15) + str(acc))

        accuracies.append(acc)
        most_likely_answer_dict = {
            'response': predicted_answer,
            'token_log_likelihoods': [],
            'embedding': None,
            'accuracy': acc}
        generations[example['id']].update({
            'most_likely_answer': most_likely_answer_dict,
            'reference': utils.get_reference(example)})

        # ── Generations 1..N: Reasoning-Step KLE ─────────────
        reasoning_trajectory = []
        reasoning_features = {}
        
        if not has_server_reasoning and not has_inline_tags:
            logging.warning("No reasoning trace found, setting NaN trajectory")
            reasoning_features = {
                'mean': float('nan'), 'max': float('nan'), 'last': float('nan'),
                'trend': float('nan'), 'auc': float('nan'), 'sum': float('nan'),
                'max_diff': float('nan')
            }
        else:
            steps = split_solution_into_chunks(logic_str)
            num_steps = len(steps)
            PARALLEL_BATCH_SIZE = 100  # Process up to 100 steps concurrently

            # Build all reasoning prefixes upfront
            prefixes = []
            for k in range(1, num_steps + 1):
                reasoning_prefix = " ".join(steps[:k])
                if not reasoning_prefix.endswith(" "):
                    reasoning_prefix += " "
                prefixes.append(reasoning_prefix)

            # ── Phase 1: Parallel branch generation ────────────────────
            def _generate_branches_for_step(step_idx):
                """Generate branches for a single step index (0-based). Returns (step_idx, valid_branches)."""
                reasoning_prefix = prefixes[step_idx]
                valid_branches = []
                seen_answers = set()
                try:
                    num_to_sample = args.num_generations
                    max_retries = 5
                    retries = 0
                    while len(valid_branches) < num_to_sample and retries < max_retries:
                        raw_branches = call_vllm_branch(
                            client, args.model_name, local_prompt, reasoning_prefix,
                            args.temperature, args.branch_max_tokens,
                            stop_sequences=['\n', '.'],
                            n=num_to_sample
                        )
                        for raw in raw_branches:
                            ans = raw.strip()
                            if ans and ans not in seen_answers:
                                seen_answers.add(ans)
                                valid_branches.append(ans)
                                if len(valid_branches) == num_to_sample:
                                    break
                        retries += 1
                except Exception as e:
                    logging.warning(f"Branch generation failed for step {step_idx + 1}: {e}")
                return step_idx, valid_branches

            # Fire all branch generation requests in batches of PARALLEL_BATCH_SIZE
            all_branches = [None] * num_steps  # indexed by step_idx
            for batch_start in range(0, num_steps, PARALLEL_BATCH_SIZE):
                batch_end = min(batch_start + PARALLEL_BATCH_SIZE, num_steps)
                batch_indices = list(range(batch_start, batch_end))
                logging.info(f"Generating branches for steps {batch_start + 1}-{batch_end} / {num_steps} in parallel...")

                with ThreadPoolExecutor(max_workers=min(len(batch_indices), PARALLEL_BATCH_SIZE)) as executor:
                    futures = {executor.submit(_generate_branches_for_step, si): si for si in batch_indices}
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating branches ({batch_start + 1}-{batch_end})", leave=False):
                        si, branches = future.result()
                        all_branches[si] = branches

            # ── Phase 2: Parallel entailment + heat metrics (GPU-semaphored) ──
            GPU_SEMAPHORE = threading.Semaphore(2)  # limit concurrent DeBERTa GPU calls

            def _compute_step_metrics(step_idx):
                """Compute entailment graph and heat metrics for a step. Returns (step_idx, step_metrics)."""
                valid_branches = all_branches[step_idx]
                step_metrics = {}
                k = step_idx + 1
                if valid_branches is None or len(valid_branches) < 2:
                    if valid_branches:
                        logging.warning(f"Step {k}: <2 valid branches")
                        for idx, res in enumerate(valid_branches):
                            logging.debug(f"  Branch {idx}: {repr(res)}")
                    return step_idx, step_metrics

                logging.debug(f"Computing metrics for step {k} with {len(valid_branches)} branches")
                with GPU_SEMAPHORE:
                    try:
                        graph = get_entailment_graph(valid_branches, model=entailment_model, example=example, is_weighted=False)
                        all_heat_metrics = heat_only_graph_entropies(graph)
                        for metric_name, value in all_heat_metrics:
                            step_metrics[metric_name] = value
                    except Exception as e:
                        logging.warning(f"Step {k}: KLE computation failed: {e}")
                return step_idx, step_metrics

            all_step_results = [{} for _ in range(num_steps)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_compute_step_metrics, si): si for si in range(num_steps)}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Computing metrics (parallel)", leave=False):
                    si, metrics = future.result()
                    all_step_results[si] = metrics

            # Build ordered trajectory from parallel results
            for k_idx in range(num_steps):
                reasoning_trajectory.append(all_step_results[k_idx])
                logging.info(f"Step {k_idx + 1}/{num_steps} metrics keys: {list(all_step_results[k_idx].keys())[:3]}... total: {len(all_step_results[k_idx])}")

            # Aggregate trajectory per metric
            if not reasoning_trajectory:
                 pass
            else:
                metric_names = set()
                for step_dict in reasoning_trajectory:
                    metric_names.update(step_dict.keys())
                
                for m_name in metric_names:
                    traj = [step.get(m_name, float('nan')) for step in reasoning_trajectory]
                    valid_vals = [v for v in traj if not np.isnan(v)]
                    
                    if not valid_vals:
                        reasoning_features[f'{m_name}_mean'] = float('nan')
                        reasoning_features[f'{m_name}_max'] = float('nan')
                        reasoning_features[f'{m_name}_last'] = float('nan')
                        reasoning_features[f'{m_name}_sum'] = float('nan')
                        reasoning_features[f'{m_name}_max_diff'] = float('nan')
                        reasoning_features[f'{m_name}_trend'] = float('nan')
                        reasoning_features[f'{m_name}_auc'] = float('nan')
                    else:
                        reasoning_features[f'{m_name}_mean'] = float(np.mean(valid_vals))
                        reasoning_features[f'{m_name}_max'] = float(np.max(valid_vals))
                        reasoning_features[f'{m_name}_last'] = float(valid_vals[-1])
                        reasoning_features[f'{m_name}_sum'] = float(np.sum(valid_vals))
                        reasoning_features[f'{m_name}_max_diff'] = float(np.max(valid_vals) - np.min(valid_vals))
                        
                        if len(valid_vals) > 1:
                            reasoning_features[f'{m_name}_trend'] = float(valid_vals[-1] - valid_vals[0])
                            reasoning_features[f'{m_name}_auc'] = float(np.trapz(valid_vals))
                        else:
                            reasoning_features[f'{m_name}_trend'] = 0.0
                            reasoning_features[f'{m_name}_auc'] = 0.0

        generations[example['id']]['reasoning_trajectory'] = reasoning_trajectory
        generations[example['id']]['reasoning_features'] = reasoning_features
        

    # Save generations for validation split.
    utils.save(generations, f'{dataset_split}_generations.pkl')

    # Log overall accuracy.
    accuracy = np.mean(accuracies) if accuracies else float('nan')
    print(f"Overall {dataset_split} split accuracy: {accuracy}")
    wandb.log({
        f"{dataset_split}_accuracy": accuracy,
        "mean_reasoning_chars": np.mean(reasoning_lengths) if reasoning_lengths else 0,
    })

    # Compute AUROC for all metrics
    logging.info(80 * '=')
    logging.info('Computing AUROC for Reasoning-Step KLE')
    logging.info(80 * '=')
    
    # First, collect all possible feature names across all generation items
    all_feature_names = set()
    for eid, gen in generations.items():
        all_feature_names.update(gen.get('reasoning_features', {}).keys())
    
    # Import evaluation utilities used by analyze_results.py
    from uncertainty.utils.eval_utils import (
        bootstrap as bs_func, compatible_bootstrap, auroc as auroc_func,
        area_under_thresholded_accuracy
    )

    uncertainty_results = {}

    def _compute_feature_auroc(feature_name, generations, rng_seed):
        """Compute AUROC + AUARC for a single feature. Returns (feature_name, result_dict | None)."""
        local_rng = np.random.default_rng(rng_seed)
        scores, labels, accs = [], [], []
        for eid, gen in generations.items():
            feat_val = gen.get('reasoning_features', {}).get(feature_name, float('nan'))
            acc = gen['most_likely_answer']['accuracy']
            labels.append(1.0 - acc)
            scores.append(feat_val)
            accs.append(acc)

        valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
        if len(valid_indices) < 2:
            logging.warning('Skipped %s: <2 valid items', feature_name)
            return feature_name, None

        v_scores = np.array([scores[i] for i in valid_indices])
        v_labels = np.array([labels[i] for i in valid_indices])
        v_accs   = np.array([accs[i] for i in valid_indices])

        if len(set(v_labels.tolist())) < 2:
            logging.warning('Skipped %s AUROC: only one class present in labels', feature_name)
            return feature_name, None

        try:
            auroc_val = roc_auc_score(v_labels, v_scores)
            auroc_bs  = compatible_bootstrap(auroc_func, local_rng)(v_labels, v_scores)
            auarc_val = area_under_thresholded_accuracy(v_accs, v_scores)
            auarc_bs  = compatible_bootstrap(
                lambda y, s: area_under_thresholded_accuracy(y, s), local_rng
            )(v_accs, v_scores)
            return feature_name, {
                'AUROC': {'mean': float(auroc_val), 'bootstrap': auroc_bs},
                'area_under_thresholded_accuracy': {'mean': float(auarc_val), 'bootstrap': auarc_bs},
            }
        except Exception as e:
            logging.warning('Failed metrics for %s: %s', feature_name, e)
            return feature_name, None

    with ThreadPoolExecutor(max_workers=max(min(len(all_feature_names), 8), 1)) as executor:
        futures = {
            executor.submit(_compute_feature_auroc, fname, generations, args.random_seed + i): fname
            for i, fname in enumerate(all_feature_names)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing AUROC (parallel)"):
            fname, result = future.result()
            if result is not None:
                uncertainty_results[fname] = result
                auroc_val = result['AUROC']['mean']
                auroc_bs  = result['AUROC']['bootstrap']
                auarc_val = result['area_under_thresholded_accuracy']['mean']
                logging.info('Reasoning Feature %s AUROC: %.4f [%.4f, %.4f]',
                             fname, auroc_val,
                             auroc_bs.get('low', float('nan')),
                             auroc_bs.get('high', float('nan')))
                logging.info('Reasoning Feature %s AUARC: %.4f', fname, auarc_val)
                wandb.log({f'reasoning_auroc_{fname}': auroc_val})

    # Store the nested uncertainty dict in results_dict so print_metrics.py works
    results_dict['uncertainty'] = uncertainty_results
    wandb.log({'uncertainty': uncertainty_results})

    # Save entitlement dict
    try:
        entailment_model.save_prediction_cache()
    except Exception as e:
        logging.warning("Failed to save entailment cache: %s", e)
        
    utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if not args.skip_generation:
        # First sample generations from LLM.
        logging.info('STARTING `generate_answers`!')
        main(args, parser)
        logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        logging.info("Skipping compute_uncertainties post-processing since AUROC is computed inline.")
