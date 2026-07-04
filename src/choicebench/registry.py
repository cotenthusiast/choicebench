# src/choicebench/registry.py
#
# Single registration point for built-in methods and API clients.
# To add a built-in method: import its runner class and add an entry to METHOD_REGISTRY.
# To add a provider client: import its client class and add an entry to CLIENT_REGISTRY.
# External methods and metrics can be registered without touching this file —
# use "module.path:ClassName" syntax in the YAML config instead.

from choicebench.clients import AnthropicClient, GeminiClient, GroqClient, OpenAIClient, TogetherAIClient, VLLMClient
from choicebench.methods import (
    CyclicLogprobRunner,
    DirectLogprobRunner,
    DirectMCQRunner,
    PermutationRunner,
    PriDeRunner,
    ShuffledBaselineRunner,
    TwoStageRunner,
)

METHOD_REGISTRY: dict[str, type] = {
    "direct_mcq": DirectMCQRunner,
    "cyclic_permutation": PermutationRunner,
    "cyclic_logprob": CyclicLogprobRunner,
    "two_stage": TwoStageRunner,
    "pride": PriDeRunner,
    "shuffled_baseline": ShuffledBaselineRunner,
    "direct_logprob": DirectLogprobRunner,
}

CLIENT_REGISTRY: dict[str, type] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
    "together": TogetherAIClient,
    "vllm": VLLMClient,
}
