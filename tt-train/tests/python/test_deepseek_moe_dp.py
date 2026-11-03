# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import ttml

from ttml.models.deepseek.moe_sparse_ep import SparseMoEEP
from test_deepseek_parallel_mlp import _Cfg, _backward, _close, _make_x, _parts, _set, _weight_set
from test_deepseek_sparse_ep import _is_blackhole

pytestmark = [
    pytest.mark.requires_device,
    pytest.mark.usefixtures("strict_ccl_topology"),
    pytest.mark.skipif(not _is_blackhole(), reason="sparse_ep MoE is supported/validated on Blackhole only"),
]


@pytest.fixture(scope="module")
def dp_ep_mesh(fresh_device_mesh):
    with fresh_device_mesh((2, 1), ("dp", "tp"), what="dp expert parallel", require_mgd=True) as mesh:
        yield mesh


@pytest.mark.parametrize("mode", ["routed", "shared", "both", "no_shared"])
def test_sparse_ep_on_dp_output_and_gradients(dp_ep_mesh, mode):
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.reset_graph()
    device = ctx.get_device()
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
    ep_shard = ttml.mesh().axis_mapper("dp", tdim=0)
    generator = torch.Generator().manual_seed(639650)
    gate = torch.randn((1, 1, 4, 64), generator=generator) * 0.125
    weights = {
        name: [torch.randn((64, 64), generator=generator) * (0 if mode == "shared" else 0.125) for _ in range(4)]
        for name in ("w_gate", "w_up", "w_down")
    }
    shared = {
        name: torch.randn((64, 64), generator=generator) * (0 if mode == "routed" else 0.125)
        for name in ("w1", "w2", "w3")
    }
    config = _Cfg()
    if mode == "no_shared":
        config.n_shared_experts = 0
    ref, ep = SparseMoEEP(config, axis_name="absent"), SparseMoEEP(config, axis_name="dp")
    assert (ref.ep_size, ep.ep_size) == (1, 2)
    for module in (ref, ep):
        _set(module.gate.weight, gate, replicate, device)
    for name in weights:
        for i, weight in enumerate(weights[name]):
            _set(getattr(ref, name)[i], weight[None, None], replicate, device)
        for i in range(2):
            _set(getattr(ep, name)[i], torch.stack((weights[name][i], weights[name][i + 2]))[:, None], ep_shard, device)
    if ep.shared_experts is not None:
        _weight_set(ref.shared_experts, shared, device)
        _weight_set(ep.shared_experts, shared, device, "dp")
    x = torch.randn((2, 1, 32, 64), generator=generator)
    upstream = torch.randn(x.shape, generator=generator)
    x_ref, x_ep = _make_x(x, device, replicate), _make_x(x, device, ep_shard)
    y_ref, y_ep = ref(x_ref), ep(x_ep)
    _backward(y_ref, upstream, device, replicate)
    _backward(y_ep, upstream, device, ep_shard)
    ttnn.synchronize_device(device)
    comparisons = []
    for label, ep_t, ref_t in (
        ("output", y_ep.get_value(), y_ref.get_value()),
        ("input gradient", x_ep.get_grad(), x_ref.get_grad()),
    ):
        expected = _parts(ref_t)[0]
        for chip, got in enumerate(_parts(ep_t)):
            comparisons.append((f"{label} chip {chip}", got, expected[chip : chip + 1]))
    if ep.shared_experts is not None:
        for name in ("w1", "w2", "w3"):
            got = _parts(getattr(ep.shared_experts, name).weight.tensor.get_grad())
            expected = _parts(getattr(ref.shared_experts, name).weight.tensor.get_grad())[0]
            comparisons.append((f"shared {name} gradient", torch.cat(got, dim=3 if name == "w2" else 2), expected))
    for chip, got in enumerate(_parts(ep.gate.weight.tensor.get_grad())):
        comparisons.append((f"gate gradient chip {chip}", got, _parts(ref.gate.weight.tensor.get_grad())[0]))
    for name in weights:
        for local in range(ep.e_local):
            for chip, got in enumerate(_parts(getattr(ep, name)[local].tensor.get_grad())):
                expected = _parts(getattr(ref, name)[chip * ep.e_local + local].tensor.get_grad())[0]
                comparisons.append((f"{name} expert {chip * ep.e_local + local} gradient", got, expected))
    failed = [name for name, got, expected in comparisons if not _close(got, expected)]
    assert not failed, failed
