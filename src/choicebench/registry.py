# src/choicebench/registry.py
#
# Single registration point for built-in methods and API clients.
# To add a built-in method: import its runner class and add an entry to METHOD_REGISTRY.
# To add a provider client: import its client class and add an entry to CLIENT_REGISTRY.
# External methods and metrics can be registered without touching this file —
# use "module.path:ClassName" syntax in the YAML config instead.

from choicebench.clients import AnthropicClient, GeminiClient, GroqClient, OpenAIClient, OpenRouterClient, TogetherAIClient, VLLMClient
from choicebench.methods import (
    CyclicLogprobRunner,
    DirectLogprobRunner,
    DirectMCQRunner,
    PermutationRunner,
    PriDeRunner,
    ShuffledBaselineRunner,
    TwoStageRunner,
)
# Paper-specific (eacl-2026-revision): imported directly from its library
# module, bypassing methods/__init__.py's re-export list, so this stays a
# paper-branch-only registration rather than promoting it into main's
# permanent default method library.
from choicebench.methods.library.independent_hypothesis import IndependentHypothesisRunner

METHOD_REGISTRY: dict[str, type] = {
    "direct_mcq": DirectMCQRunner,
    "cyclic_permutation": PermutationRunner,
    "cyclic_logprob": CyclicLogprobRunner,
    "two_stage": TwoStageRunner,
    "pride": PriDeRunner,
    "shuffled_baseline": ShuffledBaselineRunner,
    "direct_logprob": DirectLogprobRunner,
    # Paper-specific (eacl-2026-revision): reasoning-enabled variants reuse the
    # existing runner classes unchanged. The only difference from direct_mcq/
    # two_stage is prompt content (prompt_version: v1_reasoning) + generation
    # settings in config -- no new runner behavior.
    "reasoning_mcq": DirectMCQRunner,
    "reasoning_two_stage": TwoStageRunner,
    "independent_hypothesis": IndependentHypothesisRunner,
}

CLIENT_REGISTRY: dict[str, type] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
    "together": TogetherAIClient,
    "vllm": VLLMClient,
    "openrouter": OpenRouterClient,
}
