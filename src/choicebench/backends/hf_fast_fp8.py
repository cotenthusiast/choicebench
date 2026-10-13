# src/choicebench/backends/hf_fast_fp8.py
#
# Fast, arithmetic-identical replacement for compressed-tensors' eager per-token
# FP8 activation QDQ (the dominant cost of the W8A8 FP8-dynamic HF path on GPUs
# without FP8 tensor cores). Same operations in the same order on the same
# dtype, minus the library's Python/observer overhead -- and proven bit-exact
# against the original forward at install time, so a mismatch fails loudly.

from __future__ import annotations

import types


def install_fast_fp8_qdq(model, check_modules: int = 16) -> int:
    """Patch every activation-quantized Linear in `model`; return how many."""
    import torch
    import torch.nn.functional as F

    val = lambda e: getattr(e, "value", e)
    mods = [
        mod for _, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and getattr(mod, "quantization_scheme", None) is not None
    ]
    if not mods:
        raise RuntimeError("install_fast_fp8_qdq: no quantized Linear modules found.")

    fp8 = torch.float8_e4m3fn
    if any(mod.weight.dtype == fp8 for mod in mods):
        raise RuntimeError(
            "install_fast_fp8_qdq: weights are still compressed (float8); compressed-tensors only "
            "decompresses them on the first forward pass -- run one forward before installing."
        )
    qmax, qmin = float(torch.finfo(fp8).max), float(torch.finfo(fp8).min)
    half_range = (qmax - qmin) / 2
    eps_cache: dict = {}

    def make(mod, orig):
        a = mod.quantization_scheme.input_activations
        assert a is not None and a.dynamic is True and val(a.strategy) == "token" and a.symmetric \
            and val(a.type) == "float" and a.num_bits == 8 and a.scale_dtype is None, a
        assert mod.quantization_scheme.output_activations is None

        def fast_forward(self, x):
            if x.ndim != 3:
                return orig(x)
            dt = x.dtype
            s = torch.amax(x.abs(), dim=2, keepdim=True) / half_range
            e = eps_cache.get((dt, x.device))
            if e is None:
                e = eps_cache[(dt, x.device)] = torch.tensor(torch.finfo(dt).eps, dtype=dt, device=x.device)
            scale = torch.where(s == 0, e, s)
            scaled = (x / scale).add_(0)
            q = torch.clamp(scaled, qmin, qmax).to(fp8).to(dt)
            return F.linear(q * scale, self.weight, self.bias)

        return fast_forward

    originals = [(mod, mod.forward) for mod in mods]
    for mod, orig in originals:
        mod.forward = types.MethodType(make(mod, orig), mod)

    step = max(1, len(mods) // check_modules)
    for mod, orig in originals[::step]:
        gen = torch.Generator(device=mod.weight.device).manual_seed(0)
        x = torch.randn(1, 5, mod.in_features, generator=gen, device=mod.weight.device,
                        dtype=mod.weight.dtype) * 4
        x[0, 0] = 0  # zero token row exercises the eps-scale branch
        if not torch.equal(mod.forward(x), orig(x)):
            for m, o in originals:
                m.forward = o
            raise RuntimeError("install_fast_fp8_qdq: fast QDQ is not bit-identical to the original forward.")
    return len(mods)
