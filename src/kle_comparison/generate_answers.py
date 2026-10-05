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
from tqdm import tqdm

import numpy as np
import wandb
import time

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from compute_uncertainty_measures import main as main_compute


utils.setup_logger()


# ---------------------------------------------------------------------------
# Chat template registry
# model_name -> Callable[[str], Tuple[List[Dict], Dict]]
# ---------------------------------------------------------------------------
CHAT_TEMPLATES = {}


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
        logging.warning('Empty response from vLLM (content is None). Using empty string.')
        return "", reasoning
    return content, reasoning


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

        # ── Generation 0: low-T (greedy) — sequential ──────────────
        while True:
            raw_output, reasoning_trace = call_vllm(
                client, args.model_name, local_prompt,
                0.6, args.model_max_new_tokens)
            
            if raw_output and raw_output.strip():
                break
            
            logging.warning("Received empty response from vLLM (low-T). Retrying in 1s...")
            time.sleep(1)

        predicted_answer = extract_final_answer(raw_output, model_name=args.model_name)

        # Check reasoning presence.
        has_inline_tags = '</think>' in raw_output or '<|channel|>' in raw_output
        has_server_reasoning = bool(reasoning_trace)
        if not has_inline_tags and not has_server_reasoning:
            logging.warning(
                'Low-T response has no reasoning trace (no </think> tag, no harmony '
                'channel, and reasoning_content is empty). If using a reasoning model, '
                'check that --reasoning-parser is set on the vLLM server.')
        if has_server_reasoning:
            reasoning_lengths.append(len(reasoning_trace))
        elif has_inline_tags:
            reasoning_lengths.append(len(raw_output) - len(predicted_answer))

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

        # ── Generations 1..N: high-T — all in parallel ─────────────
        def _generate_one(gen_idx):
            """Worker for a single high-T generation."""
            raw, _ = call_vllm(
                client, args.model_name, local_prompt,
                args.temperature, args.model_max_new_tokens)
            pred = extract_final_answer(raw, model_name=args.model_name)
            return gen_idx, pred

        with ThreadPoolExecutor(max_workers=args.num_generations) as pool:
            futures = [
                pool.submit(_generate_one, i)
                for i in range(1, args.num_generations + 1)
            ]
            # Collect results, preserving original order.
            results_by_idx = {}
            for fut in as_completed(futures):
                idx, pred = fut.result()
                results_by_idx[idx] = pred

        for i in range(1, args.num_generations + 1):
            pred = results_by_idx[i]
            logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + pred)
            full_responses.append((pred, [], None, 0.0))

        # Append all predictions for this example to `generations`.
        generations[example['id']]['responses'] = full_responses

    # Save generations for validation split.
    utils.save(generations, f'{dataset_split}_generations.pkl')

    # Log overall accuracy.
    accuracy = np.mean(accuracies) if accuracies else float('nan')
    print(f"Overall {dataset_split} split accuracy: {accuracy}")
    wandb.log({
        f"{dataset_split}_accuracy": accuracy,
        "mean_reasoning_chars": np.mean(reasoning_lengths) if reasoning_lengths else 0,
    })

    # Save (empty) uncertainty measures dict — p_true no longer computed here.
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
        # Disable SE/entropy variants — token log probs are not collected in vLLM mode.
        if getattr(args, 'compute_predictive_entropy', False):
            logging.warning(
                'compute_predictive_entropy forced to False: token log probs '
                'are not collected in vLLM mode (not needed for K_HEAT).')
            args.compute_predictive_entropy = False
        # Only compute heat-kernel entropies — disable everything else.
        args.kle_heat_only = True
        args.compute_p_ik = False
        args.compute_p_ik_answerable = False
        args.compute_p_true_in_compute_stage = False
        args.recompute_accuracy = False
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
