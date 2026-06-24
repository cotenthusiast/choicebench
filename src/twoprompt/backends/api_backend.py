# src/twoprompt/backends/api_backend.py
# Placeholder — to be completed by researcher.

from twoprompt.backends.base import BaseBackend


class APIBackend(BaseBackend):
    """Wraps one of the provider clients in src/twoprompt/clients/
    (OpenAIClient, GeminiClient, GroqClient, TogetherAIClient) behind the
    BaseBackend interface, so runners depend only on BaseBackend and never
    import a provider client directly.

    # TODO: implement __init__(self, client: BaseClient), model_name,
    # provider, generate(), and score_options() (Together-only, via
    # request_logprobs + ModelResponse.logprobs) by delegating to the
    # wrapped client's generate(ModelRequest) -> ModelResponse.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("APIBackend is a stub — not yet implemented.")
