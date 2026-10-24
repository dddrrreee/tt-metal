# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import weakref

import pytest
import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


@pytest.fixture
def isolated_program_key_caches(monkeypatch):
    module = importlib.import_module("models.experimental.ops.descriptors.op_descriptor")
    caches = []
    monkeypatch.setattr(module, "_PROGRAM_KEY_CACHE_MAX", 128)
    monkeypatch.setattr(module, "_PROGRAM_KEY_CACHE_REGISTRY", caches)
    try:
        yield
    finally:
        for cache in caches:
            cache.clear()
        caches.clear()


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("force_collision", [False, True])
def test_tensor_spec_program_key_cache(monkeypatch, isolated_program_key_caches, keyword, force_collision):
    tensors = {
        width: ttnn.Tensor([0.0] * width, [1, 1, 1, width], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) for width in (32, 64)
    }
    # Retain both wrappers: equal specs must hash equally without address reuse.
    first, second = tensors[32].spec, tensors[32].spec
    assert first == second
    if force_collision:
        # Exercise equality even when unequal specs have the same hash.
        monkeypatch.setattr(ttnn.TensorSpec, "__hash__", lambda self: 0)
        assert hash(tensors[32].spec) == hash(tensors[64].spec) == 0
    else:
        assert hash(first) == hash(second)
    factory_calls = []

    @OpDescriptor.create(name="tensor_spec_cache_regression")
    def factory(input_tensor):
        width = input_tensor.shape[-1]
        factory_calls.append(width)
        return OpDescriptor(
            factory_fn=lambda: None,
            input_tensors={"input_tensor": input_tensor},
            output_tensors=[],
            program_cache_key=width,
        )

    for call in range(512):
        width = (32, 64)[call % 2]
        tensor = tensors[width]
        result = factory(input_tensor=tensor) if keyword else factory(tensor)
        assert result.program_cache_key == width
        assert factory_calls == ([32] if call == 0 else [32, 64])


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("values", [(-1, -2), (True, 1, 1.0)], ids=["hash-collision", "equal-values-different-types"])
def test_content_argument_program_key_cache(isolated_program_key_caches, keyword, values):
    assert len({hash(value) for value in values}) == 1
    expected_keys = {(type(value), value): index + 1 for index, value in enumerate(values)}
    factory_calls = []

    @OpDescriptor.create(name="content_argument_cache_regression")
    def factory(value):
        key = expected_keys[type(value), value]
        factory_calls.append(key)
        return OpDescriptor(factory_fn=lambda: None, input_tensors=[], output_tensors=[], program_cache_key=key)

    for call in range(512):
        value = values[call % len(values)]
        result = factory(value=value) if keyword else factory(value)
        assert result.program_cache_key == expected_keys[type(value), value]
        assert factory_calls == list(range(1, min(call + 1, len(values)) + 1))


@pytest.mark.parametrize("keyword", [False, True])
def test_raw_tensor_spec_program_key_cache(monkeypatch, isolated_program_key_caches, keyword):
    tensors = {
        width: ttnn.Tensor([0.0] * width, [1, 1, 1, width], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) for width in (32, 64)
    }
    monkeypatch.setattr(ttnn.TensorSpec, "__hash__", lambda self: 0)
    assert hash(tensors[32].spec) == hash(tensors[64].spec) == 0
    factory_calls = []

    @OpDescriptor.create(name="raw_tensor_spec_cache_regression")
    def factory(spec):
        width = spec.shape[-1]
        factory_calls.append(width)
        return OpDescriptor(factory_fn=lambda: None, input_tensors=[], output_tensors=[], program_cache_key=width)

    for call in range(512):
        width = (32, 64)[call % 2]
        spec = tensors[width].spec
        result = factory(spec=spec) if keyword else factory(spec)
        assert result.program_cache_key == width
        assert factory_calls == ([32] if call == 0 else [32, 64])


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_tuple_argument_program_key_cache(isolated_program_key_caches, keyword, nested):
    values = [(1, 2), (True, 2.0)]
    if nested:
        values = [((value,),) for value in values]
    assert values[0] == values[1]
    assert hash(values[0]) == hash(values[1])
    factory_calls = []

    @OpDescriptor.create(name="tuple_argument_cache_regression")
    def factory(scale):
        pair = scale[0][0] if nested else scale
        key = 1 if type(pair[0]) is int else 2
        factory_calls.append(key)
        return OpDescriptor(factory_fn=lambda: None, input_tensors=[], output_tensors=[], program_cache_key=key)

    for call in range(512):
        value = values[call % 2]
        result = factory(scale=value) if keyword else factory(value)
        assert result.program_cache_key == call % 2 + 1
        assert factory_calls == ([1] if call == 0 else [1, 2])


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_tensor_tuple_program_key_cache(monkeypatch, isolated_program_key_caches, keyword, nested):
    # A Python subclass permits weakrefs without changing the Tensor binding.
    class WeakTensor(ttnn.Tensor):
        pass

    def no_tensor_equality(self, other):
        raise AssertionError("cache lookup must compare tensor specs, not tensors")

    monkeypatch.setattr(WeakTensor, "__hash__", lambda self: 0)
    monkeypatch.setattr(WeakTensor, "__eq__", no_tensor_equality)
    monkeypatch.setattr(ttnn.TensorSpec, "__hash__", lambda self: 0)
    factory_calls = []

    @OpDescriptor.create(name="tensor_tuple_cache_regression")
    def factory(value):
        tensor = value[0][1][0] if nested else value[0]
        width = tensor.shape[-1]
        factory_calls.append(width)
        return OpDescriptor(factory_fn=lambda: None, input_tensors=[], output_tensors=[], program_cache_key=width)

    for call, width in enumerate((32, 32, 64, 32)):
        tensor = WeakTensor([0.0] * width, [1, 1, 1, width], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        reference = weakref.ref(tensor)
        assert hash(tensor) == hash(tensor.spec) == 0
        value = (("nested", (tensor,)),) if nested else (tensor,)
        result = factory(value=value) if keyword else factory(value)
        assert result.program_cache_key == width
        assert factory_calls == ([32] if call < 2 else [32, 64])
        # Drop the result as well: a lazy returned descriptor owns its inputs,
        # whereas the program-key cache must not keep the tuple's tensor alive.
        del result, value, tensor
        gc.collect()
        assert reference() is None
