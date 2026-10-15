"""Wispv1: a ~0.25B decoder-only LLM with pair-wise shared SwiGLU FFNs."""

from .config import WispConfig
from .model import (
    CausalLMOutput,
    WispForCausalLM,
    WispModel,
    count_parameters,
    estimate_parameter_breakdown,
)
from .tokenizer import WispTokenizer

__all__ = [
    "WispConfig",
    "WispModel",
    "WispForCausalLM",
    "WispTokenizer",
    "CausalLMOutput",
    "count_parameters",
    "estimate_parameter_breakdown",
]

__version__ = "1.0.0"
