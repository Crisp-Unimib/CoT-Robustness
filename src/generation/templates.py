import abc
import re
from typing import Dict, List, Optional, Tuple


class ChatTemplate(abc.ABC):
    """Abstract base class for chat templates."""

    @abc.abstractmethod
    def format_math_prompt(self, problem: str) -> str:
        """Format the prompt for a math problem."""
        pass

    @abc.abstractmethod
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        """Format the prompt for a blackmail scenario."""
        pass

    @abc.abstractmethod
    def format_context(self, prompt: str, chunks: List[str]) -> str:
        """Format the context for continuation sampling."""
        pass
    
    @abc.abstractmethod
    def parse_reasoning(self, response_text: str) -> str:
        """Extract the reasoning/thinking content from the response."""
        pass

    @abc.abstractmethod
    def get_reasoning_start_tag(self) -> str:
        """Return the start tag for reasoning."""
        pass

    def injects_reasoning_tag(self) -> bool:
        """Return True if format_*_prompt methods inject the reasoning start tag.
        Return False if the model is expected to generate the tag on its own (e.g., GLM).
        Default is True for most templates."""
        return True


class LegacyTemplate(ChatTemplate):
    """Original hardcoded behavior with multiple tag support."""
    
    def format_math_prompt(self, problem: str) -> str:
        return f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n"

    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "<think>")
        return f"{system}\n\n{user}\n{email}\n\n<think>\n"

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        # Ensure reasoning start tag is present before chunks
        start_tag = self.get_reasoning_start_tag()
        if start_tag not in context:
            # Add it if missing (was stripped by loader or not present)
            if not context.endswith("\n"): 
                context += "\n"
            context += f"{start_tag}\n"
        
        if chunks:
            # Ensure proper spacing before chunks
            if not context.endswith(" ") and not context.endswith("\n"):
                context += " "
            context += " ".join(chunks) + " "
        return context

    def parse_reasoning(self, response_text: str) -> str:
        if not response_text:
            return ""
        tag_patterns = [
            (r'<SCRATCHPAD_REASONING>(.*?)</SCRATCHPAD_REASONING>', '<SCRATCHPAD_REASONING>'),
            (r'<think>(.*?)</think>', '<think>'),
            (r'<reasoning>(.*?)</reasoning>', '<reasoning>'),
        ]
        for pattern, opening_tag in tag_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            if opening_tag in response_text:
                content = response_text.split(opening_tag, 1)[1]
                closing_tag = opening_tag.replace('<', '</')
                if closing_tag in content:
                    content = content.split(closing_tag)[0]
                return content.strip()
            closing_tag = opening_tag.replace('<', '</')
            if closing_tag in response_text and opening_tag not in response_text:
                return response_text.split(closing_tag)[0].strip()
        return response_text.strip()

    def get_reasoning_start_tag(self) -> str:
        return "<think>"


class Qwen3NextTemplate(ChatTemplate):
    """Qwen3-Next-80B-A3B-Thinking ChatML specific template."""

    def format_math_prompt(self, problem: str) -> str:
        return (
            f"<|im_start|>system\nSolve this math problem step by step. You MUST put your final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>user\nProblem: {problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
    
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "").replace("<think>", "").strip()
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}\n{email}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        # Ensure reasoning start tag is present before chunks
        start_tag = self.get_reasoning_start_tag()
        if start_tag not in context:
             if not context.endswith("\n"): context += "\n"
             context += f"{start_tag}\n"
             
        if chunks:
            if not context.endswith(" ") and not context.endswith("\n"):
                context += " "
            context += " ".join(chunks)
        return context

    def parse_reasoning(self, response_text: str) -> str:
        if not response_text:
            return ""
        match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "</think>" in response_text:
            return response_text.split("</think>")[0].strip()
        if response_text.strip().startswith("<think>"):
            return response_text.strip()[7:].strip()
        return response_text.strip()

    def get_reasoning_start_tag(self) -> str:
        return "<think>"


class Qwen3VLTemplate(ChatTemplate):
    """Qwen3-VL-32B-Thinking ChatML template.
    
    Uses same format as Qwen3-Next for text-only scenarios:
    - ChatML tags: <|im_start|>, <|im_end|>
    - Thinking: <think>, </think>
    
    Vision tokens (<|vision_start|>, <|image_pad|>, <|video_pad|>, <|vision_end|>)
    are handled by the processor/tokenizer when actual images are passed.
    
    The enable_thinking parameter (default=True) controls whether the model
    generates reasoning. When disabled, an empty <think>\n\n</think> block
    is injected to skip reasoning - but we always enable thinking for
    bifurcation analysis.
    """

    def format_math_prompt(self, problem: str) -> str:
        return (
            f"<|im_start|>system\nSolve this math problem step by step. You MUST put your final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>user\nProblem: {problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
    
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "").replace("<think>", "").strip()
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}\n{email}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        # Ensure reasoning start tag is present before chunks
        start_tag = self.get_reasoning_start_tag()
        if start_tag not in context:
             # Add it if missing (was stripped by loader)
             if not context.endswith("\n"): context += "\n"
             context += f"{start_tag}\n"
             
        if chunks:
            # Ensure we don't double space if not needed
            if not context.endswith(" ") and not context.endswith("\n"):
                context += " "
            context += " ".join(chunks)
        return context

    def parse_reasoning(self, response_text: str) -> str:
        if not response_text:
            return ""
        match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "</think>" in response_text:
            return response_text.split("</think>")[0].strip()
        if response_text.strip().startswith("<think>"):
            return response_text.strip()[7:].strip()
        return response_text.strip()

    def get_reasoning_start_tag(self) -> str:
        return "<think>"


class AprielThinkerTemplate(ChatTemplate):
    """ServiceNow-AI/Apriel-1.6-15b-Thinker chat template.
    
    Uses custom delimiters:
    - <|begin_system|>, <|begin_user|>, <|begin_assistant|>, <|end|>
    - Reasoning starts with "Here are my reasoning steps:\n"
    - Final answer after "[BEGIN FINAL RESPONSE]"
    - Tool calls in <tool_calls></tool_calls> XML tags
    
    The model has a built-in reasoning prompt that gets prepended to system.
    """
    
    # Default reasoning prompt from the model
    REASONING_PROMPT = (
        "You are a thoughtful, systematic AI assistant from ServiceNow Language Models (SLAM) lab. "
        "Analyze each question carefully, present your reasoning step-by-step, then provide the final "
        "response after the marker [BEGIN FINAL RESPONSE]."
    )
    
    REASONING_START = "Here are my reasoning steps:\n"
    FINAL_RESPONSE_MARKER = "[BEGIN FINAL RESPONSE]"

    def format_math_prompt(self, problem: str) -> str:
        system_msg = "Solve this math problem step by step. Put your final answer in \\boxed{}."
        return (
            f"<|begin_system|>\n{self.REASONING_PROMPT}\n{system_msg}\n"
            f"<|begin_user|>\nProblem: {problem}"
            f"\n<|begin_assistant|>\n{self.REASONING_START}"
        )
    
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "").replace("<think>", "").strip()
        return (
            f"<|begin_system|>\n{self.REASONING_PROMPT}\n{system}\n"
            f"<|begin_user|>\n{user}\n{email}"
            f"\n<|begin_assistant|>\n{self.REASONING_START}"
        )

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        # Ensure reasoning start tag is present before chunks
        start_tag = self.get_reasoning_start_tag()
        if start_tag not in context:
            # Add it if missing (was stripped by loader or not present)
            if not context.endswith("\n"): 
                context += "\n"
            context += start_tag
        
        if chunks:
            # Ensure proper spacing before chunks
            if not context.endswith(" ") and not context.endswith("\n"):
                context += " "
            context += " ".join(chunks)
        return context

    def parse_reasoning(self, response_text: str) -> str:
        """Extract reasoning content before [BEGIN FINAL RESPONSE]."""
        if not response_text:
            return ""
        
        # Strip the reasoning start prefix if present
        text = response_text
        if text.startswith(self.REASONING_START):
            text = text[len(self.REASONING_START):]
        
        # Extract content before the final response marker
        if self.FINAL_RESPONSE_MARKER in text:
            reasoning = text.split(self.FINAL_RESPONSE_MARKER)[0]
            return reasoning.strip()
        
        # If no marker, check for <|end|> tag
        if "<|end|>" in text:
            return text.split("<|end|>")[0].strip()
        
        return text.strip()

    def get_reasoning_start_tag(self) -> str:
        return self.REASONING_START


class NemotronNanoTemplate(ChatTemplate):
    """NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ChatML template.
    
    Uses ChatML format with thinking enabled by default:
    - Tags: <|im_start|>, <|im_end|>
    - Thinking: <think>, </think>
    - Tool calls: <tool_call>, </tool_call> with <function=name> format
    
    For text-only bifurcation analysis, functionally identical to Qwen3-Next.
    The model uses truncate_history_thinking to strip reasoning from older turns,
    but we always work with full reasoning for analysis.
    """

    def format_math_prompt(self, problem: str) -> str:
        return (
            f"<|im_start|>system\nSolve this math problem step by step. You MUST put your final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>user\nProblem: {problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
    
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "").replace("<think>", "").strip()
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}\n{email}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        # Ensure reasoning start tag is present before chunks
        start_tag = self.get_reasoning_start_tag()
        if start_tag not in context:
            if not context.endswith("\n"): context += "\n"
            context += f"{start_tag}\n"
            
        if chunks:
            # Strip any existing <think> tags from chunks to prevent duplicates
            cleaned_chunks = []
            for chunk in chunks:
                cleaned = chunk.replace("<think>", "").replace("</think>", "")
                cleaned_chunks.append(cleaned)
            if not context.endswith(" ") and not context.endswith("\n"):
                context += " "
            context += " ".join(cleaned_chunks)
        return context

    def parse_reasoning(self, response_text: str) -> str:
        if not response_text:
            return ""
        match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "</think>" in response_text:
            return response_text.split("</think>")[0].strip()
        if response_text.strip().startswith("<think>"):
            return response_text.strip()[7:].strip()
        return response_text.strip()

    def get_reasoning_start_tag(self) -> str:
        return "<think>"


class GPTOSSTemplate(ChatTemplate):
    """GPT-OSS 120B Harmony chat template with channel-based routing."""
    
    START = "<|start|>"
    END = "<|end|>"
    CHANNEL = "<|channel|>"
    MESSAGE = "<|message|>"
    
    def format_math_prompt(self, problem: str) -> str:
        system_msg = "You are a helpful assistant. Solve math problems step by step. Put your final answer in \\boxed{}."
        return (
            f"{self.START}system{self.MESSAGE}{system_msg}{self.END}"
            f"{self.START}user{self.MESSAGE}Problem: {problem}{self.END}"
            f"{self.START}assistant{self.CHANNEL}analysis{self.MESSAGE}"
        )
    
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "").replace("<think>", "").strip()
        return (
            f"{self.START}system{self.MESSAGE}{system}{self.END}"
            f"{self.START}user{self.MESSAGE}{user}\n{email}{self.END}"
            f"{self.START}assistant{self.CHANNEL}analysis{self.MESSAGE}"
        )
    
    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        # GPT-OSS uses channel-based format, not simple tags
        # The reasoning start should already be in the prompt from format_blackmail_prompt
        # (ending with <|start|>assistant<|channel|>analysis<|message|>)
        # We just need to append chunks after it
        
        if chunks:
            # Ensure proper spacing before chunks
            if not context.endswith(" ") and not context.endswith("\n"):
                context += " "
            context += " ".join(chunks) + " "
        return context
    
    def parse_reasoning(self, response_text: str) -> str:
        if not response_text:
            return ""
        analysis_pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)'
        match = re.search(analysis_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "<|end|>" in response_text:
            return response_text.split("<|end|>")[0].strip()
        return response_text.strip()
    
    def get_reasoning_start_tag(self) -> str:
        return f"{self.START}assistant{self.CHANNEL}analysis{self.MESSAGE}"


class GLM45AirTemplate(ChatTemplate):
    """
    GLM-4.5-Air chat template from zai-org/GLM-4.5-Air.
    Uses [gMASK] and sop initialization tokens with role markers user, assistant, system, observation.
    Supports think tags for hybrid reasoning mode.
    """

    # Prefix tokens
    GMASK = "[gMASK]"
    SOP = "<" + "sop>"
    
    # Role markers
    USER = "<" + "|user|>"
    ASSISTANT = "<" + "|assistant|>"
    SYSTEM = "<" + "|system|>"
    OBSERVATION = "<" + "|observation|>"
    
    # Thinking tags - GLM-4.5-Air uses <think></think>
    # NOTE: The model generates <think> on its own - we don't inject it in the prompt
    THINK_START = "<" + "think>"
    THINK_END = "<" + "/think>"
    
    def format_math_prompt(self, problem: str) -> str:
        """Format prompt WITHOUT injecting <think> - model generates it."""
        system_msg = "Solve this math problem step by step. You MUST put your final answer in \\boxed{}."
        return (
            f"{self.GMASK}{self.SOP}"
            f"{self.SYSTEM}\n{system_msg}"
            f"{self.USER}\nProblem: {problem}"
            f"{self.ASSISTANT}\n"  # No <think> tag - model generates it
        )
    
    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        """Format prompt WITHOUT injecting <think> - model generates it."""
        system = system.replace("<SCRATCHPAD_REASONING>", "").replace("<think>", "").strip()
        return (
            f"{self.GMASK}{self.SOP}"
            f"{self.SYSTEM}\n{system}"
            f"{self.USER}\n{user}\n{email}"
            f"{self.ASSISTANT}\n"  # No <think> tag - model generates it
        )

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        if chunks:
            context += " ".join(chunks)
        return context

    def parse_reasoning(self, response_text: str) -> str:
        if not response_text:
            return ""
        # Look for content between think tags
        match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no closing tag, take everything after opening
        if "</think>" in response_text:
            return response_text.split("</think>")[0].strip()
        if response_text.strip().startswith("<think>"):
            return response_text.strip()[7:].strip()  # len("<think>") = 7
        return response_text.strip()

    def get_reasoning_start_tag(self) -> str:
        return self.THINK_START

    def injects_reasoning_tag(self) -> bool:
        """GLM model generates <think> on its own - we don't inject it."""
        return False


def get_template_for_model(model_name: str) -> ChatTemplate:
    """
    Factory to return the appropriate template based on model name.
    """
    # Check for GLM-4.5-Air models
    if "GLM-4.5-Air" in model_name:
        return GLM45AirTemplate()
    
    # Check for GPT-OSS models
    if "gpt-oss" in model_name.lower():
        return GPTOSSTemplate()
    
    # Check for ServiceNow Apriel Thinker models
    if "Apriel" in model_name and "Thinker" in model_name:
        return AprielThinkerTemplate()
    
    # Check for NVIDIA Nemotron Nano models
    if "Nemotron" in model_name and "Nano" in model_name:
        return NemotronNanoTemplate()
    
    # Check for Qwen3-VL models (vision-language with thinking)
    if "Qwen3-VL" in model_name:
        return Qwen3VLTemplate()
    
    # Check for Qwen3-Next
    if "Qwen/Qwen3-Next-80B-A3B-Thinking" in model_name:
        return Qwen3NextTemplate()
    
    # Default to Legacy for everything else to maintain backward compatibility
    return LegacyTemplate()
