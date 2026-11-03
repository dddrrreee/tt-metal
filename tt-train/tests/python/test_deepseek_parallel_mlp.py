# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
import ttml
from ttml.models.deepseek.transformer import DeepSeekMLP
from ttml.models.deepseek.moe_sparse_ep import SparseMoEEP
from test_deepseek_sparse_ep import _is_blackhole

pytestmark = [pytest.mark.requires_device, pytest.mark.usefixtures("strict_ccl_topology")]


def _set(param, value, mapper, device):
    tensor = ttnn.from_torch(
        value.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=mapper
    )
    param.tensor.set_value(ttnn.to_layout(tensor, ttnn.TILE_LAYOUT))


def _parts(tensor):
    return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(tensor)]


def _close(got, expected, pcc=0.99, rtol=0.08, atol=0.08):
    got, expected = got.float(), expected.float()
    corr = torch.corrcoef(torch.stack((got.flatten(), expected.flatten())))[0, 1].item()
    error = (got - expected).abs().max().item()
    limit = atol + rtol * expected.abs().max().item()
    return error <= limit and (corr >= pcc or expected.abs().max().item() < 1e-7)


def _make_x(value, device, mapper):
    tensor = ttnn.from_torch(
        value.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
    )
    return ttml.autograd.create_tensor(tensor)


def _weight_set(mlp, weights, device, axis=None):
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
    for name, weight in weights.items():
        mapper = ttml.mesh().axis_mapper(axis, tdim=3 if name == "w2" else 2) if axis else replicate
        _set(getattr(mlp, name).weight, weight[None, None], mapper, device)


def _backward(output, upstream, device, mapper):
    output.set_grad(
        ttnn.from_torch(
            upstream.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mapper,
        )
    )
    output.backward(False)


def _check_mlp_forward_and_gradients(axis):
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.reset_graph()
    device = ctx.get_device()
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
    generator = torch.Generator().manual_seed(63965)
    weights = {
        name: torch.randn(shape, generator=generator) * 0.125
        for name, shape in [("w1", (128, 64)), ("w3", (128, 64)), ("w2", (64, 128))]
    }
    x = torch.randn((1, 1, 32, 64), generator=generator)
    upstream = torch.randn(x.shape, generator=generator)
    ref, tp = DeepSeekMLP(64, 128), DeepSeekMLP(64, 128, tp_axis_name=axis)
    _weight_set(ref, weights, device)
    _weight_set(tp, weights, device, axis)
    x_ref, x_tp = _make_x(x, device, replicate), _make_x(x, device, replicate)
    y_ref, y_tp = ref(x_ref), tp(x_tp)
    _backward(y_ref, upstream, device, replicate)
    _backward(y_tp, upstream, device, replicate)
    ttnn.synchronize_device(device)
    comparisons = []
    for index, (got, expected) in enumerate(zip(_parts(y_tp.get_value()), _parts(y_ref.get_value()))):
        comparisons.append((f"output chip {index}", got, expected))
    for index, (got, expected) in enumerate(zip(_parts(x_tp.get_grad()), _parts(x_ref.get_grad()))):
        comparisons.append((f"input gradient chip {index}", got, expected))
    for name in ("w1", "w2", "w3"):
        got = _parts(getattr(tp, name).weight.tensor.get_grad())
        expected = _parts(getattr(ref, name).weight.tensor.get_grad())[0]
        full_grad = torch.cat(got, dim=3 if name == "w2" else 2) if axis == "tp" else got[0]
        comparisons.append((f"{name} gradient", full_grad, expected))
    failed = [name for name, got, expected in comparisons if not _close(got, expected)]
    assert not failed, failed


def test_deepseek_mlp_tp_forward_and_gradients(tp_mesh):
    _check_mlp_forward_and_gradients("tp")


def test_deepseek_mlp_size_one_axis_forward_and_gradients(tp_mesh):
    _check_mlp_forward_and_gradients("dp")


class _Cfg:
    dim = 64
    moe_inter_dim = 64
    n_routed_experts = 4
    n_activated_experts = 2
    n_shared_experts = 1
    n_expert_groups = 1
    n_limited_groups = 1
    score_func = "sigmoid"
    route_scale = 1.0
    moe_type = "sparse_ep"
    moe_axis_name = "tp"


@pytest.mark.skipif(not _is_blackhole(), reason="sparse_ep MoE is supported/validated on Blackhole only")
def test_deepseek_shared_expert_ep_forward_and_gradients(tp_mesh):
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.reset_graph()
    device = ctx.get_device()
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
    ep_shard = ttml.mesh().axis_mapper("tp", tdim=0)
    generator = torch.Generator().manual_seed(63966)
    gate = torch.randn((1, 1, 4, 64), generator=generator) * 0.125
    weights = {
        name: [torch.randn((64, 64), generator=generator) * 0.125 for _ in range(4)]
        for name in ("w_gate", "w_up", "w_down")
    }
    shared = {name: torch.randn((64, 64), generator=generator) * 0.125 for name in ("w1", "w2", "w3")}
    ref, ep = SparseMoEEP(_Cfg(), axis_name="absent"), SparseMoEEP(_Cfg(), axis_name="tp")
    assert (ref.ep_size, ep.ep_size) == (1, 2)
    _set(ref.gate.weight, gate, replicate, device)
    _set(ep.gate.weight, gate, replicate, device)
    for name in weights:
        for i, weight in enumerate(weights[name]):
            _set(getattr(ref, name)[i], weight[None, None], replicate, device)
        for i in range(2):
            _set(getattr(ep, name)[i], torch.stack((weights[name][i], weights[name][i + 2]))[:, None], ep_shard, device)
    _weight_set(ref.shared_experts, shared, device)
    _weight_set(ep.shared_experts, shared, device, "tp")
    x = torch.randn((1, 1, 32, 64), generator=generator)
    upstream = torch.randn(x.shape, generator=generator)
    x_ref, x_ep = _make_x(x, device, replicate), _make_x(x, device, replicate)
    y_ref, y_ep = ref(x_ref), ep(x_ep)
    _backward(y_ref, upstream, device, replicate)
    _backward(y_ep, upstream, device, replicate)
    ttnn.synchronize_device(device)
    comparisons = []
    for index, (got, expected) in enumerate(zip(_parts(y_ep.get_value()), _parts(y_ref.get_value()))):
        comparisons.append((f"output chip {index}", got, expected))
    for index, (got, expected) in enumerate(zip(_parts(x_ep.get_grad()), _parts(x_ref.get_grad()))):
        comparisons.append((f"input gradient chip {index}", got, expected))
    for name in ("w1", "w2", "w3"):
        got = _parts(getattr(ep.shared_experts, name).weight.tensor.get_grad())
        expected = _parts(getattr(ref.shared_experts, name).weight.tensor.get_grad())[0]
        comparisons.append((f"shared {name} gradient", torch.cat(got, dim=3 if name == "w2" else 2), expected))
    for chip, (got, expected) in enumerate(
        zip(_parts(ep.gate.weight.tensor.get_grad()), _parts(ref.gate.weight.tensor.get_grad()))
    ):
        comparisons.append((f"gate gradient chip {chip}", got, expected))
    for name in ("w_gate", "w_up", "w_down"):
        for local in range(ep.e_local):
            for chip, got in enumerate(_parts(getattr(ep, name)[local].tensor.get_grad())):
                global_expert = chip * ep.e_local + local
                expected = _parts(getattr(ref, name)[global_expert].tensor.get_grad())[chip]
                comparisons.append((f"{name} expert {global_expert} gradient", got, expected))
    failed = [name for name, got, expected in comparisons if not _close(got, expected)]
    assert not failed, failed
