# tests/backends/test_hf_fast_fp8.py
#
# install_fast_fp8_qdq() swaps compressed-tensors' eager per-token FP8 activation
# QDQ for an arithmetic-identical fast forward, and proves bit-exactness against
# the original forward at install time.

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from choicebench.backends.hf_fast_fp8 import install_fast_fp8_qdq

FP8 = torch.float8_e4m3fn


def _scheme(**overrides):
    act = dict(dynamic=True, strategy="token", symmetric=True, type="float", num_bits=8, scale_dtype=None)
    act.update(overrides)
    return SimpleNamespace(input_activations=SimpleNamespace(**act), output_activations=None)


class _QuantLinear(torch.nn.Linear):
    """Reference per-token FP8 activation QDQ followed by a linear, written plainly."""

    def __init__(self, n_in, n_out, scheme=None):
        super().__init__(n_in, n_out, bias=True)
        self.to(torch.bfloat16)
        self.quantization_scheme = scheme or _scheme()

    def forward(self, x):
        info = torch.finfo(FP8)
        scale = x.abs().amax(dim=-1, keepdim=True) / ((info.max - info.min) / 2)
        scale = torch.where(scale == 0, torch.tensor(torch.finfo(x.dtype).eps, dtype=x.dtype), scale)
        q = (x / scale).clamp(info.min, info.max).to(FP8).to(x.dtype)
        return torch.nn.functional.linear(q * scale, self.weight, self.bias)


def _model(n=3, scheme=None):
    torch.manual_seed(0)
    return torch.nn.Sequential(*[_QuantLinear(16, 16, scheme) for _ in range(n)])


def test_patched_forward_is_bit_identical_to_original():
    model = _model()
    x = torch.randn(2, 7, 16, dtype=torch.bfloat16) * 5
    x[0, 0] = 0  # all-zero token row exercises the eps-scale branch
    before = [m(x) for m in model]
    assert install_fast_fp8_qdq(model) == 3
    assert all(torch.equal(m(x), b) for m, b in zip(model, before))


def test_non_3d_input_falls_back_to_original_forward():
    model = _model(1)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    before = model[0](x)
    install_fast_fp8_qdq(model)
    assert torch.equal(model[0](x), before)


def test_unsupported_scheme_is_rejected():
    with pytest.raises(AssertionError):
        install_fast_fp8_qdq(_model(scheme=_scheme(dynamic=False)))


def test_no_quantized_modules_fails_loudly():
    with pytest.raises(RuntimeError, match="no quantized"):
        install_fast_fp8_qdq(torch.nn.Sequential(torch.nn.Linear(4, 4)))


def test_mismatch_against_original_forward_raises_and_restores():
    model = _model(2)
    base_forward = _QuantLinear.forward

    def off_by_one(self, x):
        return base_forward(self, x) + 1

    for m in model:
        m.forward = off_by_one.__get__(m)
    with pytest.raises(RuntimeError, match="not bit-identical"):
        install_fast_fp8_qdq(model)
    assert all(m.forward.__func__ is off_by_one for m in model)
