# tests/backends/test_hf_backend.py

from choicebench.backends.hf_backend import HuggingFaceBackend


class _FakeTensor:
    shape = (1, 3)

    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        self.prompt = prompt
        self.return_tensors = return_tensors
        return {"input_ids": _FakeTensor()}

    def decode(self, generated_ids, skip_special_tokens):
        self.generated_ids = generated_ids
        self.skip_special_tokens = skip_special_tokens
        return "decoded"


class _FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[101, 102, 103, 201]]


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTorch:
    def __init__(self):
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed

    def no_grad(self):
        return _NoGrad()


def test_generate_uses_constructor_generation_kwargs():
    backend = HuggingFaceBackend(
        "fake-model",
        "cpu",
        max_new_tokens=17,
        temperature=0.25,
        do_sample=True,
    )
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    torch = _FakeTorch()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = model
    backend._torch = torch

    text = backend.generate("prompt", seed=123)

    assert text == "decoded"
    assert torch.seed == 123
    assert model.generate_kwargs["max_new_tokens"] == 17
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == 0.25
