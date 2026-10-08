# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP gradient parity paths."""

import deepspeed
import deepspeed.comm as dist
import torch
import torch.nn as nn
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoEBlock,
    MockHFConfig,
    MockMoETransformer,
    engine_input_dtype as _engine_input_dtype,
    make_autoep_config,
    mixed_precision_config as _mixed_precision_config,
    run_cpu_gloo_test,
    seed_everything as _seed_everything,
)


def _make_model():
    return MockMoETransformer(num_layers=1, num_experts=4, hidden_size=128, intermediate_size=256)


def _make_zero2_config():
    return {
        **_mixed_precision_config(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        },
    }


def _make_autoep_zero2_config(ep_size):
    config = _make_zero2_config()
    config["expert_parallel"] = {
        "enabled": True,
        "autoep_size": ep_size,
        "preset_model": "mixtral",
        "load_balance_coeff": None,
        "use_grouped_mm": False,
    }
    return config


def _make_autoep_zero3_config(ep_size):
    config = _make_autoep_zero2_config(ep_size)
    config["zero_optimization"] = {
        "stage": 3,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
    }
    return config


def _make_local_batches(*, logical_dp_world_size, logical_dp_rank, grad_accum, seed, seq_len, micro_batch_size,
                        hidden_size, device, dtype):
    batches = []
    for accum_idx in range(grad_accum):
        batch_idx = accum_idx * logical_dp_world_size + logical_dp_rank
        generator = torch.Generator().manual_seed(seed + batch_idx)
        batches.append(
            torch.randn((micro_batch_size, seq_len, hidden_size), generator=generator, dtype=dtype).to(device))
    return batches


def _run_until_boundary(engine, *, logical_dp_world_size, logical_dp_rank, grad_accum, seed):
    batches = _make_local_batches(
        logical_dp_world_size=logical_dp_world_size,
        logical_dp_rank=logical_dp_rank,
        grad_accum=grad_accum,
        seed=seed,
        seq_len=16,
        micro_batch_size=1,
        hidden_size=128,
        device=engine.device,
        dtype=_engine_input_dtype(engine),
    )
    for batch_idx, batch in enumerate(batches):
        loss = engine(batch).mean()
        engine.backward(loss)
        if batch_idx + 1 < len(batches):
            engine.step()


def _gather_autoep_expert_grad(param, group):
    grad = safe_get_full_grad(param)
    assert grad is not None, "Expected full expert grad"
    group_size = dist.get_world_size(group=group)
    shards = [torch.zeros_like(grad) for _ in range(group_size)]
    dist.all_gather(shards, grad.detach(), group=group)
    # The gather reconstructs expert shards; gradient reduction has already
    # applied the data-parallel normalization, so do not average by EP size.
    return torch.cat([shard.float().cpu() for shard in shards], dim=0)


def _collect_autoep_expert_grads(engine):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    grads = {}
    for module_name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        prefix = f"{module_name}.experts"
        w1 = _gather_autoep_expert_grad(module.experts.w1, module.ep_group)
        w2 = _gather_autoep_expert_grad(module.experts.w2, module.ep_group)
        w3 = _gather_autoep_expert_grad(module.experts.w3, module.ep_group)
        grads[f"{prefix}.gate_up_proj"] = torch.cat([w1, w3], dim=1)
        grads[f"{prefix}.down_proj"] = w2
    return grads


def _collect_zero2_expert_grads(engine):
    grads = {}
    for name, param in engine.module.named_parameters():
        if name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
            grad = safe_get_full_grad(param)
            assert grad is not None, f"Expected full grad for {name}"
            grads[name] = grad.detach().float().cpu().clone()
    return grads


def _assert_grad_maps_close(actual, expected, *, lhs_name, rhs_name):
    for name in sorted(expected):
        assert name in actual, f"Missing {lhs_name} param snapshot for {name}"
        diff = (actual[name] - expected[name]).abs()
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   atol=1e-1,
                                   rtol=5e-3,
                                   msg=(f"Gradient mismatch for {name} between {lhs_name} and {rhs_name}; "
                                        f"max_diff={diff.max().item()} "
                                        f"actual_norm={actual[name].norm().item()} "
                                        f"expected_norm={expected[name].norm().item()}"))


class TestAutoEPGradParity(DistributedTest):
    world_size = 4

    def test_zero2_autoep_matches_zero2_after_one_update(self):
        ep_size = 2
        seed = 1234

        _seed_everything(seed)
        reference_state = _make_model().state_dict()

        autoep_model = _make_model()
        zero2_model = _make_model()
        autoep_model.load_state_dict(reference_state)
        zero2_model.load_state_dict(reference_state)

        autoep_engine, _, _, _ = deepspeed.initialize(model=autoep_model, config=_make_autoep_zero2_config(ep_size))
        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_zero2_config())

        autoep_rank = dist.get_rank() // ep_size
        _run_until_boundary(autoep_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)

        autoep_expert = _collect_autoep_expert_grads(autoep_engine)
        zero2_expert = _collect_zero2_expert_grads(zero2_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(autoep_expert, zero2_expert, lhs_name="AutoEP expert", rhs_name="ZeRO-2 expert")

    def test_zero3_autoep_expert_grads_match_zero2_autoep(self):
        ep_size = 2
        seed = 2345

        _seed_everything(seed)
        reference_state = _make_model().state_dict()

        zero2_model = _make_model()
        zero3_model = _make_model()
        zero2_model.load_state_dict(reference_state)
        zero3_model.load_state_dict(reference_state)

        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_autoep_zero2_config(ep_size))
        zero3_engine, _, _, _ = deepspeed.initialize(model=zero3_model, config=_make_autoep_zero3_config(ep_size))

        logical_rank = dist.get_rank() // ep_size
        logical_world_size = self.world_size // ep_size
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=logical_world_size,
                            logical_dp_rank=logical_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero3_engine,
                            logical_dp_world_size=logical_world_size,
                            logical_dp_rank=logical_rank,
                            grad_accum=2,
                            seed=seed)

        zero2_expert = _collect_autoep_expert_grads(zero2_engine)
        zero3_expert = _collect_autoep_expert_grads(zero3_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(zero3_expert,
                                zero2_expert,
                                lhs_name="ZeRO-3 AutoEP expert",
                                rhs_name="ZeRO-2 AutoEP expert")


class _DecoderLayer(nn.Module):
    """Named `mlp` + Mixtral-shaped experts so AutoEP can replace the block."""

    def __init__(self, hidden_size=32, num_experts=4, intermediate_size=64):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = MockMoEBlock(num_experts, intermediate_size, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        x = residual + self.dense(self.input_layernorm(hidden_states))
        residual = x
        mlp_out = self.mlp(self.post_attention_layernorm(x))
        if isinstance(mlp_out, tuple):
            extra = [item for item in mlp_out[1:] if item is not None]
            if extra:
                raise TypeError("decoder returned extra outputs")
            mlp_out = mlp_out[0]
        return residual + mlp_out


class _TinyDecoderMoE(nn.Module):

    def __init__(self, num_layers=2, hidden_size=32, num_experts=4, intermediate_size=64, vocab=16):
        super().__init__()
        self.config = MockHFConfig()
        self.config.num_local_experts = num_experts
        self.config.hidden_size = hidden_size
        self.config.intermediate_size = intermediate_size
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_DecoderLayer(hidden_size, num_experts, intermediate_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


def _tiled_zero2_autoep_config(ep_size, micro_batch):
    config = make_autoep_config(zero_stage=2, ep_size=ep_size, mixed_precision=False)
    config["train_micro_batch_size_per_gpu"] = micro_batch
    config["expert_parallel"]["load_balance_coeff"] = None
    return config


def _install_decoder_tiled_remat(model, shards):
    from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import tiled_rematerialize_compute

    for layer in model.model.layers:
        orig = layer.forward

        def _make(orig_fwd, layer_ref, nshards):

            def tiled_forward(hidden_states):

                def _fn(h):
                    return orig_fwd(h)

                params = [param for param in layer_ref.parameters() if param.requires_grad]
                return tiled_rematerialize_compute(_fn, hidden_states, nshards, params)

            return tiled_forward

        layer.forward = _make(orig, layer, shards)


def _collect_named_full_grads(engine):
    grads = {}
    for name, param in engine.module.named_parameters():
        if not param.requires_grad:
            continue
        grad = safe_get_full_grad(param)
        if grad is None:
            continue
        grads[name] = grad.detach().float().cpu().clone()
    return grads


def _autoep_token_counts(engine):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    return [
        module.tokens_per_expert.detach().float().cpu().clone() for module in engine.module.modules()
        if isinstance(module, AutoEPMoELayer)
    ]


def _patch_reduce_and_alltoall(engine):
    import deepspeed.module_inject.auto_ep_layer as auto_ep_layer

    stats = {"ready_ids": [], "a2a": 0}
    orig_reduce = engine.optimizer.reduce_independent_p_g_buckets_and_remove_grads
    orig_a2a = auto_ep_layer.dist.all_to_all_single

    def counting_reduce(param, i):
        if getattr(param, "ds_grad_is_ready", True):
            stats["ready_ids"].append(id(param))
        return orig_reduce(param, i)

    def counting_a2a(*args, **kwargs):
        stats["a2a"] += 1
        return orig_a2a(*args, **kwargs)

    engine.optimizer.reduce_independent_p_g_buckets_and_remove_grads = counting_reduce
    auto_ep_layer.dist.all_to_all_single = counting_a2a
    return stats, orig_reduce, orig_a2a


def _run_tiled_vs_untiled(shards, batch=4, seq_len=8, hidden=32, seed=7):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
    import deepspeed.module_inject.auto_ep_layer as auto_ep_layer

    _seed_everything(seed)
    reference = _TinyDecoderMoE(hidden_size=hidden).state_dict()
    x = torch.randn(batch, seq_len, hidden)

    def _one(nshards):
        import sys
        _seed_everything(seed)
        model = _TinyDecoderMoE(hidden_size=hidden)
        model.load_state_dict(reference)
        cfg = _tiled_zero2_autoep_config(ep_size=2, micro_batch=batch)
        engine, _, _, _ = deepspeed.initialize(model=model, config=cfg)
        autoep_layers = [m for m in engine.module.modules() if isinstance(m, AutoEPMoELayer)]
        assert autoep_layers, "AutoEP did not replace MoE blocks"
        if nshards > 1:
            _install_decoder_tiled_remat(engine.module, nshards)
        stats, orig_reduce, orig_a2a = _patch_reduce_and_alltoall(engine)
        dtype = _engine_input_dtype(engine)
        inp = x.to(device=engine.device, dtype=dtype).clone().detach().requires_grad_(True)
        loss = engine(inp).float().mean()
        engine.backward(loss)
        grads = _collect_named_full_grads(engine)
        counts = _autoep_token_counts(engine)
        ready_unique = len(set(stats["ready_ids"]))
        ready_total = len(stats["ready_ids"])
        a2a = stats["a2a"]
        engine.step()
        inp2 = x.to(device=engine.device, dtype=dtype).clone().detach().requires_grad_(True)
        loss2 = engine(inp2).float().mean()
        engine.backward(loss2)
        engine.optimizer.reduce_independent_p_g_buckets_and_remove_grads = orig_reduce
        auto_ep_layer.dist.all_to_all_single = orig_a2a
        result = {
            "loss": float(loss.detach()),
            "grads": grads,
            "counts": counts,
            "ready_unique": ready_unique,
            "ready_total": ready_total,
            "a2a": a2a,
            "loss2": float(loss2.detach()),
            "n_autoep": len(autoep_layers),
        }
        print(f"[compose] shards={nshards} loss={result['loss']:.5f} a2a={a2a} ready={ready_total}",
              file=sys.stderr,
              flush=True)
        engine.destroy()
        return result

    return _one(1), _one(shards)


def _cpu_gloo_zero2_autoep_tiled_remat_worker(rank, world_size, _shared_tmpdir):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
    from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import tiled_rematerialize_compute

    assert world_size == 2
    untiled, tiled = _run_tiled_vs_untiled(shards=2)
    assert tiled["n_autoep"] == untiled["n_autoep"] == 2
    assert abs(tiled["loss"] - untiled["loss"]) < 1e-4
    _assert_grad_maps_close(tiled["grads"], untiled["grads"], lhs_name="tiled AutoEP", rhs_name="untiled AutoEP")
    # Last-row reduce: each param that reduced did so once on the first backward.
    assert tiled["ready_unique"] == tiled["ready_total"]
    assert tiled["ready_unique"] > 0
    # Lock-step AllToAll: both ranks take the tiled path; count is higher than untiled.
    assert tiled["a2a"] > untiled["a2a"]
    for tiled_count, plain_count in zip(tiled["counts"], untiled["counts"]):
        torch.testing.assert_close(tiled_count, plain_count * 2, atol=0, rtol=0)

    # Router logits captured on the no_grad full forward cannot carry aux loss.
    _seed_everything(11)
    model = _TinyDecoderMoE()
    engine, _, _, _ = deepspeed.initialize(model=model, config=_tiled_zero2_autoep_config(ep_size=2, micro_batch=4))
    try:
        layer = [m for m in engine.module.modules() if isinstance(m, AutoEPMoELayer)][0]
        layer.return_router_logits = True
        captured = {}

        def _fn(h):
            out = layer(h)
            captured["logits"] = out[1]
            return out[0]

        hidden = torch.randn(4, 8, 32, device=engine.device, dtype=_engine_input_dtype(engine), requires_grad=True)
        tiled_rematerialize_compute(_fn, hidden, 2, [p for p in layer.parameters() if p.requires_grad])
        logits = captured["logits"]
        assert logits is not None
        assert logits.grad_fn is None, "tiled remat captured router logits on the no_grad forward; aux loss is unsupported"
    finally:
        engine.destroy()


def test_cpu_gloo_zero2_autoep_tiled_remat_composition(tmpdir):
    run_cpu_gloo_test(_cpu_gloo_zero2_autoep_tiled_remat_worker, tmpdir, world_size=2)
