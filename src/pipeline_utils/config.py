import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ExperimentConfig:
    """
    Configuration for the experiment.
    """
    def __init__(self):
        # Local directory paths
        self.math_rollouts_path = os.getenv("MATH_ROLLOUTS_PATH", "math_rollouts")
        self.analysis_path = os.getenv("ANALYSIS_PATH", "analysis/basic")
        self.problem_id = os.getenv("PROBLEM_ID")
        self.model_name = os.getenv("MODEL_NAME", "Qwen3-Next-80B-A3B-Thinking")
        self.solution_type = os.getenv("SOLUTION_TYPE", "correct_base_solution")
        
        # LLM configuration
        # vllm_model_name is the full model path for vLLM API (e.g., "Qwen/Qwen3-Next-80B-A3B-Thinking")
        # Falls back to model_name if not specified
        self.vllm_model_name = os.getenv("VLLM_MODEL_NAME", self.model_name)
        self.controller_model = os.getenv("CONTROLLER_MODEL", "google/gemini-3-flash-preview")
        self.vllm_temperature = float(os.getenv("VLLM_TEMPERATURE", "1"))
        self.vllm_top_p = float(os.getenv("VLLM_TOP_P", "0.95"))
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.num_servers = int(os.getenv("NUM_SERVERS", "1"))
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.num_candidates = int(os.getenv("NUM_CANDIDATES", "20"))
        self.max_edits = int(os.getenv("MAX_EDITS", "5"))
        self.top_k_anchors = int(os.getenv("TOP_K_ANCHORS", "3"))
        self.results_dir = os.getenv("RESULTS_DIR", "results")
        self.anchor_selection_method = os.getenv("ANCHOR_SELECTION_METHOD", "importance").lower()
        self.bifurcation_metric = os.getenv("BIFURCATION_METRIC", "bifurcation_entropy")
        self.max_tokens = 32768
        self.run_all_problems = os.getenv("RUN_ALL_PROBLEMS", "false").lower() in ("true", "1", "yes", "all")
        self.dry_run = False  # Set via CLI

        # Blackmail specific config
        self.is_blackmail = os.getenv("IS_BLACKMAIL", "false").lower() in ("true", "1", "yes")
        self.scenario_id = os.getenv("SCENARIO_ID", self.problem_id) # Alias/Fallback
        
        # Parallel execution config
        self.max_parallel_anchors = int(os.getenv("MAX_PARALLEL_ANCHORS", "5"))
        self.parallel_verbose_logs = os.getenv("PARALLEL_VERBOSE_LOGS", "true").lower() in ("true", "1", "yes")
        
        # Random anchor selection (control group - excludes standard method anchors)
        self.random_anchor = os.getenv("RANDOM_ANCHOR", "false").lower() in ("true", "1", "yes")
        
    @property
    def experiment_signature(self) -> str:
        """
        Generates a signature string for the current experiment configuration.
        Format: {controller_short}_n{num_candidates}_T{temp}_p{top_p}_e{max_edits}_k{top_k}
        """
        # Sanitize controller model name (e.g., "google/gemini-pro" -> "google-gemini-pro")
        controller_short = self.controller_model.replace("/", "-")
        
        return (
            f"{controller_short}_"
            f"n{self.num_candidates}_"
            f"T{self.vllm_temperature}_"
            f"p{self.vllm_top_p}_"
            f"e{self.max_edits}_"
            f"k{self.top_k_anchors}"
            f"{'_random' if self.random_anchor else ''}"
        )
