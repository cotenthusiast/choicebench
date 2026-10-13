# tests/backends/test_hf_backend_batched.py
#
# Batched greedy generation for HuggingFaceBackend (opt-in via batch_size).
# Uses a real tiny random Llama + word-level tokenizer on CPU (no downloads), so
# "batched == one-at-a-time" is checked against actual HF generate(), padding included.

import asyncio

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from choicebench.backends.hf_backend import HuggingFaceBackend

PROMPTS = ["w5", "w6 w7 w8 w9 w10", "w11 w12", "w13 w14 w15 w16", "w17 w18 w19"]


def _tiny(eos_after=None):
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}": i for i in range(3, 40)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", bos_token="<bos>")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    return model, tokenizer


def _backend(model, tokenizer, batch_size, max_new_tokens=12):
    backend = HuggingFaceBackend(
        "tiny", device="cpu", batch_size=batch_size,
        max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False,
    )
    backend._model, backend._tokenizer, backend._torch = model, tokenizer, torch
    backend._loaded = True
    return backend


def _texts(responses):
    assert all(r.is_success() for r in responses)
    return [r.raw_text for r in responses]


def test_default_backend_is_not_async_capable():
    assert HuggingFaceBackend("tiny", device="cpu").is_async_capable() is False


def test_batch_size_makes_backend_async_capable():
    assert HuggingFaceBackend("tiny", device="cpu", batch_size=4).is_async_capable() is True


def test_generate_batch_matches_one_at_a_time_generation():
    model, tokenizer = _tiny()
    single = _backend(model, tokenizer, batch_size=None)
    expected = [single.generate(p) for p in PROMPTS]
    batched = _backend(model, tokenizer, batch_size=5)
    assert _texts(asyncio.run(batched.generate_batch(PROMPTS))) == expected


def test_generate_batch_trims_each_sequence_at_its_own_eos():
    model, tokenizer = _tiny()
    probe = _backend(model, tokenizer, batch_size=None)
    ids = tokenizer(PROMPTS[1], return_tensors="pt")["input_ids"]
    free_run = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=1)[0][ids.shape[1]:]
    model.generation_config.eos_token_id = int(free_run[2])  # stops prompt 1 after 3 tokens
    expected = [probe.generate(p) for p in PROMPTS]
    assert len(set(map(len, expected))) > 1  # sequences really do stop at different lengths
    batched = _backend(model, tokenizer, batch_size=5)
    assert _texts(asyncio.run(batched.generate_batch(PROMPTS))) == expected


def test_generate_batch_splits_into_fixed_size_chunks_in_input_order():
    model, tokenizer = _tiny()
    sizes = []
    real_generate = model.generate

    def spy(**kwargs):
        sizes.append(kwargs["input_ids"].shape[0])
        return real_generate(**kwargs)

    model.generate = spy
    batched = _backend(model, tokenizer, batch_size=2)
    out = _texts(asyncio.run(batched.generate_batch(PROMPTS)))
    assert sizes == [2, 2, 1]
    single = _backend(model, tokenizer, batch_size=None)
    assert out == [single.generate(p) for p in PROMPTS]


def test_generate_single_async_matches_generate():
    model, tokenizer = _tiny()
    backend = _backend(model, tokenizer, batch_size=4)
    response = asyncio.run(backend.generate_single_async(PROMPTS[1]))
    assert response.is_success()
    assert response.raw_text == backend.generate(PROMPTS[1])


def test_generate_batch_refuses_sampling():
    model, tokenizer = _tiny()
    backend = HuggingFaceBackend(
        "tiny", device="cpu", batch_size=2, max_new_tokens=4, temperature=0.7, do_sample=True,
    )
    backend._model, backend._tokenizer, backend._torch, backend._loaded = model, tokenizer, torch, True
    with pytest.raises(ValueError, match="greedy"):
        asyncio.run(backend.generate_batch(PROMPTS))


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        HuggingFaceBackend("tiny", device="cpu", batch_size=0)


def test_fast_qdq_load_runs_warmup_forward_before_patching(monkeypatch):
    """compressed-tensors only decompresses fp8 weights on the first forward, so load() must
    run one forward BEFORE installing the fast-QDQ patch (and its bit-exactness probe)."""
    import transformers
    import choicebench.backends.hf_fast_fp8 as fast
    model, tokenizer = _tiny()
    events = []
    real_forward = model.forward

    def spy(*args, **kwargs):
        events.append("forward")
        return real_forward(*args, **kwargs)

    model.forward = spy
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(fast, "install_fast_fp8_qdq", lambda m: events.append("install") or 3)
    backend = HuggingFaceBackend("tiny", device="cpu", fast_fp8_qdq=True)
    backend.load()
    assert events == ["forward", "install"]
