"""Script to write the complete templates.py file."""

content = r'''import abc
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


class LegacyTemplate(ChatTemplate):
    """Original hardcoded behavior with multiple tag support."""
    
    def format_math_prompt(self, problem: str) -> str:
        return f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n"

    def format_blackmail_prompt(self, system: str, user: str, email: str) -> str:
        system = system.replace("<SCRATCHPAD_REASONING>", "<think>")
        return f"{system}\n\n{user}\n{email}\n\n<think>\n"

    def format_context(self, prompt: str, chunks: List[str]) -> str:
        context = prompt
        if chunks:
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
        if chunks:
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
        if chunks:
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
    Uses [gMASK]<sop> initialization tokens and role markers.
    Supports think tags for hybrid reasoning mode.
    """
    
    # Prefix tokens
    GMASK = "[gMASK]"
    SOP = "<sop>"
    
    # Role markers
    USER = "
