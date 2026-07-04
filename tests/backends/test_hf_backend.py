# tests/backends/test_hf_backend.py

from choicebench.backends.hf_backend import HuggingFaceBackend


class _FakeTensor:
    shape = (1, 3)

    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors, add_special_tokens=True):
        self.prompt = prompt
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        return {"input_ids": _FakeTensor()}

    def encode(self, text, add_special_tokens=False):
        return [ord(text)]  # options are single-char labels, e.g. "A" -> 65

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


class _FakeScalar:
    """Wraps a plain float so `.item()` (as real torch tensors expose) works."""

    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeLogitsRow(dict):
    """token_id -> logit, wrapping each lookup in a _FakeScalar for `.item()`."""

    def __getitem__(self, key):
        return _FakeScalar(dict.__getitem__(self, key))


class _FakeLogitsTensor:
    """Stands in for outputs.logits; supports the exact outputs.logits[0, -1, :] call."""

    def __init__(self, row: _FakeLogitsRow):
        self._row = row

    def __getitem__(self, index):
        assert index == (0, -1, slice(None, None, None))
        return self._row


class _FakeScoreOutputs:
    def __init__(self, row: _FakeLogitsRow):
        self.logits = _FakeLogitsTensor(row)


class _FakeScoringModel:
    def __init__(self):
        self._row = _FakeLogitsRow({ord("A"): -0.1, ord("B"): -1.5})

    def __call__(self, **kwargs):
        return _FakeScoreOutputs(self._row)


class _FakeF:
    @staticmethod
    def log_softmax(x, dim):
        return x  # identity — these tests only assert on the tokenizer kwarg


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


def test_score_options_passes_add_special_tokens_true_by_default():
    backend = HuggingFaceBackend("fake-model", "cpu")
    tokenizer = _FakeTokenizer()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = _FakeScoringModel()
    backend._torch = _FakeTorch()
    backend._F = _FakeF()

    backend.score_options("prompt", ["A", "B"])

    assert tokenizer.add_special_tokens == True  # noqa: E712


def test_score_options_passes_add_special_tokens_false_when_bos_disabled():
    backend = HuggingFaceBackend("fake-model", "cpu", add_bos_token=False)
    tokenizer = _FakeTokenizer()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = _FakeScoringModel()
    backend._torch = _FakeTorch()
    backend._F = _FakeF()

    backend.score_options("prompt", ["A", "B"])

    assert tokenizer.add_special_tokens == False  # noqa: E712


def test_generate_passes_add_special_tokens_false_when_bos_disabled():
    backend = HuggingFaceBackend("fake-model", "cpu", add_bos_token=False)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    torch = _FakeTorch()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = model
    backend._torch = torch

    backend.generate("prompt")

    assert tokenizer.add_special_tokens == False  # noqa: E712
