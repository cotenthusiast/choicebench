# tests/backends/test_backend_async_contract.py
"""The async half of the backend interface (generate_single_async / generate_batch) is part of
BaseBackend's declared contract, not an implicit convention between runners and two backends."""

import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.base import BaseBackend
from choicebench.backends.batch_api_backend import BatchAPIBackend
from choicebench.backends.dummy_backend import DummyBackend
from choicebench.backends.hf_backend import HuggingFaceBackend

ASYNC_METHODS = ("generate_single_async", "generate_batch")


def test_base_backend_declares_the_async_contract():
    for name in ASYNC_METHODS:
        assert callable(getattr(BaseBackend, name))


@pytest.mark.parametrize("backend_cls", [APIBackend, BatchAPIBackend, HuggingFaceBackend])
def test_async_capable_backends_implement_the_contract_themselves(backend_cls):
    # Not merely inherited from BaseBackend (which only raises NotImplementedError).
    for name in ASYNC_METHODS:
        assert any(name in vars(cls) for cls in backend_cls.__mro__ if cls is not BaseBackend), (
            f"{backend_cls.__name__} does not override {name}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args", [("generate_single_async", ("p",)), ("generate_batch", (["p"],))])
async def test_non_async_backend_fails_with_a_pointer_to_generate(method, args):
    backend = DummyBackend()
    assert backend.is_async_capable() is False
    with pytest.raises(NotImplementedError, match=r"is_async_capable\(\) is False.*generate\(\)"):
        await getattr(backend, method)(*args)
