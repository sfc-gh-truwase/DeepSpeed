# Agent instructions — reproduce saved-activation H200 results

Reproduce blog application (3): credit-bounded saved MLP activation offload, no rematerialization, Qwen3-8B, ZeRO-2 CPU AdamW, pin vs pageable. The reference run is `autorun/logs/job-20261008T131352Z.log` on torch `2.14.0+cu130`.

Do not modify DeepSpeed for this. The engine is `DeepSpeedExamples/benchmarks/cpu_offload/saved_activation_offload/` on branch `tjruwase/pin-memory-experiments` (`a7f80f4` on `sfc-gh-truwase/DeepSpeedExamples`). It calls accelerator `pin_memory` / `unpin_memory` and, with `--offload-optimizer`, existing ZeRO-2 CPU AdamW.

Do not start `blogs/pinned-offload/08-2026/README.md`. Do not publish the momentum-free SGD sweep. Do not treat Qwen3.5 as required.

## What “reproduced” means

Nine cells: Qwen3-8B, batch 1, seq `4096`, `2048`, `1024`, `--offload-tensors` `1`, `2`, `3`. Each cell must exit 0 with `pin_separated=true`, `optimizer=adamw_zero2_cpu`, `zero_stage=2`, and `torch=2.14.0+cu130`.

Reference medians from `job-20261008T131352Z` (20 measured steps, 5 warmup). Match the band, not the exact seconds. One-way bytes are exact for this model (35 layers, bf16).

| seq | K | Pageable → pinned | Speedup | One-way | Pinned `r` |
| --- | --- | --- | --- | --- | --- |
| 4096 | 1 | 7.127 → 3.672 s | 1.94× | 3.28 GiB | 1.05 |
| 4096 | 2 | 7.674 → 3.672 s | 2.09× | 6.56 GiB | 2.08 |
| 4096 | 3 | 8.352 → 3.751 s | 2.23× | 9.84 GiB | 3.12 |
| 2048 | 1 | 7.040 → 3.493 s | 2.02× | 1.64 GiB | 0.89 |
| 2048 | 2 | 7.672 → 3.446 s | 2.23× | 3.28 GiB | 1.47 |
| 2048 | 3 | 7.223 → 3.431 s | 2.11× | 4.92 GiB | 2.20 |
| 1024 | 1 | 6.796 → 3.340 s | 2.03× | 0.82 GiB | 0.57 |
| 1024 | 2 | 6.676 → 3.392 s | 1.97× | 1.64 GiB | 0.98 |
| 1024 | 3 | rerun | rerun | 2.46 GiB | rerun |

Seq 1024 K=3 in that log exited 1 because `common.py` lost `build_hf_causal_lm` while the tree was rebased under the running job. That is not an OOM and not the expected result. On commit `a7f80f4` the import exists; the cell should pass. Headline row is seq 4096 K=3.

`job-20261007T215537Z` is the same Adam recipe on torch `2.9.1+cu130`. Do not use it as the stack oracle.

## Hardware and stack

Run on the H200 pod `tunji-h200-n1g8-ds-pin-0`, one GPU (`WORLD_SIZE=1`). The node-local env is `/home/yak/miniconda3/envs/torch214`. The CPU box `dev` env is a different torch and is not this stack. Submit work by writing `autorun/queue.sh`. The runner on the pod executes it. Read `autorun/status.txt` and `autorun/logs/job-<id>.log`. Wait until `queue.sh` is gone before submitting another job.

Required stack:

- torch `2.14.0+cu130`
- transformers `5.16.0.dev0`
- flash-attn `2.8.3` built for C++20 and `sm_90`
- DeepSpeed from this checkout on `PYTHONPATH` (pin APIs already on `master`; no saved-activation patch)
- `ulimit -l unlimited`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Upstream flash-attn `2.8.3` hardcodes `-std=c++17` on the `nvcc` line. `CXXFLAGS=-std=c++20` does not override it, and torch 2.14 headers need `std::strong_ordering`. Install by cloning v2.8.3, replacing every `-std=c++17` in `setup.py` with `-std=c++20`, and building with `NVCC_APPEND_FLAGS=-std=c++20` and `TORCH_CUDA_ARCH_LIST=9.0`. Uninstall any existing `flash_attn` and delete `flash_attn_2_cuda*.so` first. Do not import `flash_attn` until the new extension is installed. `pip install --no-build-isolation --no-deps` from that patched tree.

Pinned arms set `DS_PIN_MEMORY_BACKEND=native` and `DS_PIN_MEMORY_REGISTER_DEVICE=1`. The bench does that itself. Pin covers activation host buffers and the ZeRO-2 optimizer host buffers.

## Sweep

From `DeepSpeedExamples`, with `torch214` active and `PYTHONPATH` pointing at this DeepSpeed checkout:

```bash
ulimit -l unlimited
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for SEQ in 4096 2048 1024; do
  for K in 1 2 3; do
    python benchmarks/cpu_offload/saved_activation_offload/bench.py \
      --model Qwen/Qwen3-8B --batch 1 --seq "$SEQ" --offload-tensors "$K" \
      --offload-optimizer \
      --manifest-out "/tmp/saved-activation-z2adam-torch214-seq${SEQ}-k${K}.json"
  done
done
```

Leave the other defaults: `--credits 2`, `--prefetch 2`, `--offload-modules '\.mlp$'`, `--skip-last-layers 1`, `--steps 20`, `--warmup 5`, `--mode offload`. The driver runs pageable, pinned, pinned, pageable. `--offload-optimizer` is ZeRO stage 2, parameters on GPU, CPU AdamW `lr=1e-4`, `offload_optimizer.pin_memory` following the pin arm.

A cell passes only when its `DRIVERRESULT` line has `pin_separated=true` and the checks above. Copy `seq`, `offload_tensors_k`, `pageable_median_s`, `pinned_median_s`, `pin_speedup`, `offloaded_bytes`, and pinned `r_d2h_over_mlp_span` into the report. Do not replace the reference table until all nine cells have those fields.

## Out of this repro

- Momentum-free SGD (omit `--offload-optimizer`). That is the activation-only diagnostic, including the seq 8192 K=3 3.56× row.
- Qwen3.5-9B. Optional. Seq 4096 OOMs. Do not plot it next to Qwen3-8B.
- `--mode remat`. That is checkpoint offload, application (2), not this pin comparison.
- `--n-sweep`. Credit-cap check, not the published pin figure.
- GeForce and Arc Pro B70. Not gates for this application.
