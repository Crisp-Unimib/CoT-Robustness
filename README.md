# CoT-Robustness

Code and data for **Diagnosing LLM Fragility: An Empirical Study on Reasoning Entropy and Malicious Outputs**.

The repository holds the full pipeline (H_BE scoring of every reasoning step, adversarial probing of the selected steps, outcome classification) together with the processed data it produced, so every analysis in the paper can be rerun on CPU without querying a model.

## Layout

```
src/
  generation/        H_BE scoring of base responses (generate_bifurcation.py) and helpers
  pipeline.py        Probing pipeline: pick steps, inject, roll out, classify
  pipeline_utils/    Config, LLM client, controller, dataset loader
  analysis/          Phase classification, figures, breakage tables
  kle_comparison/    Step-level vs. final-answer KLE framework
analysis/
  tables/            Probing efficiency table
  robustness/        Quartile break-rate figure and the appendix sweeps
  nli/               Kernel cost and stability, NLI backbone swap, yield@k, classifier checks
configs/
  blackmail_prompts/ System, user and email prompts for each goal x urgency scenario
  prompts/           Controller and classifier prompts
infra/leonardo/      SLURM and vLLM launch scripts used to serve the target models
data/
  processed/blackmail_bifurcation/  Base response and per-step H_BE for each model and scenario
  results/blackmail/                Probe outcomes for the H_BE arm and the random arm
  validation/                       Outcome classifier agreement sample
outputs/classifications/            Reasoning-phase label for each step
```

The seven model directories under `data/processed` are the five reported models, `Apriel-1.6-15b-Thinker`, and the held-out `Qwen3-VL-32B-Thinking_evil` variant; the last two enter only the stability protocol and the NLI safety corpus.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # only needed to rerun the pipeline
```

## Reproducing the paper from the shipped data

Every command runs from the repository root and writes its output there or to `outputs/plots/`.

| Paper element | Command | Needs |
|---|---|---|
| H_BE distribution (fig:distribution) | `python src/analysis/plot_bifurcation_distribution.py` | CPU |
| Break rate by H_BE quartile, headline ratios (6.2x, 5.5x, 2.6x) | `python analysis/robustness/recreate_figure4.py` | CPU |
| Probing efficiency by model (tab:bif_vs_rand) | `python analysis/tables/table1_probing_efficiency.py` | CPU |
| H_BE by urgency level (fig:urgency_analysis) | `python src/analysis/plot_merged_urgency.py` | CPU |
| Structural localization (fig:structural_localization) | `python src/analysis/plot_fragility_by_phase.py` | CPU |
| Breakage tables | `python src/analysis/produce_breakage_table.py` | CPU |
| Kernel cost and subset stability (tab:cost, tab:subset_stability) | `python analysis/nli/validate_subset_timing.py`, `python analysis/nli/validate_subset_stability.py` | GPU, DeBERTa-large-MNLI |
| Ablation yield@3 (tab:ablation_yield) | `python analysis/nli/yield_at_k.py` | CPU |
| Classifier errors by arm (tab:clf_condition) | `python analysis/nli/classifier_condition_independence.py` | CPU |
| NLI backbone swap (tab:sweep_nli_backbone) | `python analysis/nli/nli_swap.py` | GPU, DeBERTa-v2-xlarge-MNLI |
| Appendix sweeps (k, segmentation, N, alpha/beta) | `python analysis/robustness/run_all.py` | CPU |

Timings in tab:cost depend on the GPU; the Spearman column does not.

## Rerunning the pipeline

Target models are served with vLLM behind an OpenAI-compatible endpoint (see `infra/leonardo/`), and the controller and outcome classifier (`google/gemini-3-flash-preview`) are called through OpenRouter.

**1. Base responses and H_BE.** For each scenario in `configs/blackmail_prompts/`, sample a base response, then 100 continuations per step, and score each step:

```bash
python src/generation/generate_bifurcation.py --blackmail -m <model> -u http://localhost:8000/v1 -n 100
# add --evil for the held-out variant
```

Output goes to `data/processed/blackmail_bifurcation/<model>/<scenario>/`.

**2. Probing.** Run once per arm, with these settings in `.env` (they produce the `google-gemini-3-flash-preview_n100_T1.0_p0.95_e1_k10[_random]` result directories):

```bash
IS_BLACKMAIL=true
RUN_ALL_PROBLEMS=true
ANCHOR_SELECTION_METHOD=bifurcation
BIFURCATION_METRIC=bifurcation_entropy_kle
CONTROLLER_MODEL=google/gemini-3-flash-preview
NUM_CANDIDATES=100
VLLM_TEMPERATURE=1.0
VLLM_TOP_P=0.95
MAX_EDITS=1
TOP_K_ANCHORS=10
RESULTS_DIR=data/results
RANDOM_ANCHOR=false      # true for the random arm
```

```bash
python src/pipeline.py
```

**3. Phase labels.** `python src/analysis/classify_entropy_sentences.py` writes `outputs/classifications/`.

`src/generation/update_bifurcation_kle_values.py` copies the H_BE values in `data/processed` into the `z_score` field of the `Qwen3-Next-80B` result files.

## License

See [LICENSE](LICENSE).
