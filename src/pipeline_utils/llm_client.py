import json
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from openai import OpenAI
from pipeline_utils.config import ExperimentConfig

if TYPE_CHECKING:
    from pipeline_utils.anchor_logger import AnchorLogger

class LLMClient:
    """
    Abstracts interactions with LLM providers (vLLM and OpenRouter).
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._request_counter = 0
        
        # Build list of vLLM server URLs for round-robin
        self._vllm_urls = []
        if config.num_servers > 1:
            from urllib.parse import urlparse
            parsed = urlparse(config.vllm_base_url)
            hostname = parsed.hostname or 'localhost'
            scheme = parsed.scheme or 'http'
            base_port = parsed.port or 8000
            path = parsed.path
            
            for i in range(config.num_servers):
                self._vllm_urls.append(f"{scheme}://{hostname}:{base_port + i}{path}")
        else:
            self._vllm_urls = [config.vllm_base_url]
        
        # Initialize client with first URL (will be rotated per request)
        self.client = OpenAI(
            base_url=self._vllm_urls[0],
            api_key="EMPTY",
        )

    def query_vllm(self, prompt_or_messages: Union[str, List[Dict[str, str]]], stop: Optional[List[str]] = None, n: int = 1, extra_body: Optional[Dict[str, Any]] = None, max_tokens: Optional[int] = None, logger: Optional["AnchorLogger"] = None) -> Union[str, List[str]]:
        """
        Queries the local vLLM server with dynamic max_tokens fallback.
        Supports both Chat API (list of messages) and Completions API (string prompt).
        If logger is provided, prints are routed there for thread-safe parallel execution.
        """
        
        def _log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)
        
        if self.config.dry_run:
            _log("[DRY RUN] Skipping vLLM query. Returning mock response.")
            mock_response = "This is a DRY RUN mock response content for vLLM query."
            if n > 1:
                return [f"{mock_response} candidate {i}" for i in range(n)]
            return mock_response
        
        # Round-robin server selection
        server_idx = self._request_counter % len(self._vllm_urls)
        self._request_counter += 1
        self.client = OpenAI(
            base_url=self._vllm_urls[server_idx],
            api_key="EMPTY",
        )
        
        # Use provided max_tokens or fall back to config default
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        def make_request(token_limit: int):
            # Determine if we are doing Chat or Completion
            is_chat = isinstance(prompt_or_messages, list)
            
            # Filter extra_body for completion requests to avoid vLLM warnings
            request_extra_body = extra_body.copy() if extra_body else {}
            if not is_chat:
                request_extra_body.pop("add_generation_prompt", None)
                request_extra_body.pop("continue_final_message", None)
            
            common_params = {
                "model": self.config.vllm_model_name,
                "max_tokens": token_limit,
                "temperature": self.config.vllm_temperature,
                "top_p": self.config.vllm_top_p,
                "n": n,
                "extra_body": request_extra_body,
            }
            
            if stop:
                common_params["stop"] = stop

            if is_chat:
                return self.client.chat.completions.create(
                    messages=prompt_or_messages,
                    **common_params
                )
            else:
                return self.client.completions.create(
                    prompt=prompt_or_messages,
                    **common_params
                )
        
        # Retry parameters
        max_retries = 3
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                response = make_request(tokens)
                
                if n > 1:
                    # Filter out None/empty content - vLLM can return None for failed generations
                    # Also handle reasoning models that put content in 'reasoning' or 'reasoning_content' fields
                    results = []
                    none_count = 0
                    empty_count = 0
                    reasoning_fallback_count = 0
                    
                    for choice in response.choices:
                        # Handle both ChatCompletion and Completion choice objects
                        if hasattr(choice, 'message'):
                            content = choice.message.content
                            obj_for_reasoning = choice.message
                        else:
                            content = choice.text
                            obj_for_reasoning = choice
                        
                        # For reasoning models (QwQ, Qwen3), content may be None but reasoning has the output
                        if content is None:
                            # Try to get content from reasoning fields
                            reasoning = getattr(obj_for_reasoning, 'reasoning', None) or getattr(obj_for_reasoning, 'reasoning_content', None)
                            if reasoning and reasoning.strip():
                                content = reasoning
                                reasoning_fallback_count += 1
                        
                        if content is None:
                            none_count += 1
                        elif not content.strip():
                            empty_count += 1
                        else:
                            results.append(content)
                    
                    # Diagnostic logging if we got choices but no valid content
                    if len(response.choices) > 0 and not results:
                        _log(f"[vLLM Debug] Received {len(response.choices)} choices but 0 valid. None: {none_count}, Empty: {empty_count}")
                        # Sample the first choice for debugging
                        if response.choices:
                            first_choice = response.choices[0]
                            if hasattr(first_choice, 'message'):
                                 _log(f"[vLLM Debug] First choice message: {first_choice.message}")
                            else:
                                 _log(f"[vLLM Debug] First choice text: {first_choice.text!r}")
                            _log(f"[vLLM Debug] First choice finish_reason: {first_choice.finish_reason}")
                    elif reasoning_fallback_count > 0:
                        _log(f"[vLLM Debug] Used reasoning field fallback for {reasoning_fallback_count}/{len(results)} candidates")
                    
                    return results if results else []
                
                # Single choice extraction
                choice = response.choices[0]
                if hasattr(choice, 'message'):
                    return choice.message.content or ""
                return choice.text or ""

            except Exception as e:
                error_str = str(e)
                # Check if it's a context length error and extract available tokens
                if "max_tokens" in error_str and "too large" in error_str:
                    import re
                    # Parse: "maximum context length is X tokens and your request has Y input tokens"
                    # Different models/servers produce slightly different error messages, trying generic match
                    # OpenRouter: "Input validation error: inputs tokens + max_new_tokens must be <= max_position_embeddings"
                    # vLLM: "This model's max seq len is 32768. Req id ... has ... tokens"
                    
                     # Simple retry with reduced tokens logic
                    if attempt == 0:
                        _log(f"Possible context length error: {error_str}. Retrying with halving max_tokens.")
                        tokens = max(tokens // 2, 128)
                        continue

                # For other errors, retry with backoff
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * backoff_factor
                    _log(f"Error querying vLLM (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time}s...")
                    import time
                    time.sleep(sleep_time)
                else:
                    _log(f"Error querying vLLM after {max_retries} attempts: {e}")
                    import traceback
                    traceback.print_exc()
                    return [] if n > 1 else ""
        
        return [] if n > 1 else ""

    def query_openrouter(self, messages: List[Dict[str, str]], temperature: float = 0.0, response_format: Optional[Dict[str, Any]] = None, model: Optional[str] = None, logger: Optional["AnchorLogger"] = None) -> Union[str, Dict[str, Any]]:
        """
        Queries OpenRouter for controller decisions.
        """
        if self.config.dry_run:
            print("[DRY RUN] Skipping OpenRouter query.")
            return "Robust"
            
        client = OpenAI(
             base_url="https://openrouter.ai/api/v1",
             api_key=self.config.openrouter_api_key,
        )
        
        try:
            params = {
                "model": model or self.config.controller_model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                params["response_format"] = response_format

            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content
            
            if response_format and response_format.get("type") == "json_object":
                try:
                    # Clean up markdown code blocks if present
                    if "```" in content:
                        import re
                        content = re.sub(r"```json\s*", "", content)
                        content = re.sub(r"```\s*", "", content)
                        content = content.strip()
                    return json.loads(content)
                except json.JSONDecodeError:
                    if logger:
                        logger.error(f"Failed to parse JSON from OpenRouter. Raw content: {content}")
                    else:
                        print(f"Failed to parse JSON from OpenRouter. Raw content: {content}")
                    return {"error": "json_parse_error", "raw_content": content}
            
            return content
        except Exception as e:
            if logger:
                logger.error(f"OpenRouter Error: {e}")
            else:
                # Fallback to console print if console is available, otherwise just print
                print(f"OpenRouter Error: {e}")
            return "Error"
