# Proposed layout — Pinned Host Memory for CPU Offload

**Article file:** [`README.md`](README.md) (drafted 2026-11-05; B60 rows and bound Ascend reruns pending).\
**Figures:** `media/`, one panel per machine, one chart per model family.\
**Source of the argument:** [`BLOG_PLAN.md`](BLOG_PLAN.md). This file is the heading layout only.

**Title:** Pinned Host Memory for CPU Offload: A Practitioner Guide

Headings stay accelerator-neutral. ZeRO and pull-request numbers stay in the body. Each machine’s wall-time figure is its own panel; Ascend 910B4 does not share a chart with H200 or B60. Every evaluated machine (Arc Pro B60, H200, Ascend 910B4) runs all three workloads and the H2D/D2H microbench. Wall time is measured bound to the accelerator’s NUMA node.

**Voice:** a technical blog, not a tutorial. Each evaluation section leads with its results table and the interpretation. Setup and method stay in their own sections. Benchmark commands go only in the closing appendix.

**Model families:** each workload section reports Qwen3 and Qwen3.5 in separate tables and separate charts, Qwen3 first. Qwen3.5 never shares a chart with Qwen3. There is no standalone Qwen3.5 section.

**Sequence lengths:** the activation checkpoint and saved-activation sections print only the longest sequence each machine ran, per family. Shorter rows stay in source comments, not in the article. Model-tensor prints every collected length. The headline table and the conclusion use the same longest-sequence numbers.

---

## Introduction

### CPU offload for cost-efficiency

Offload as a way to run on smaller or cheaper GPUs. Capacity claims need a no-offload baseline that does not fit. The activation rows here do not meet that bar: on H200, plain checkpointing fits at every length measured, up to 128K.

### Three applications: model tensors, checkpoint activations, and saved activations

Name all three before any DeepSpeed tour.

1. Model-tensor offload — parameters and optimizer states.
2. Activation checkpoint workload — checkpoint hidden states with rematerialization, together with ZeRO-2 CPU AdamW and `--act-mem-opt`.
3. Saved-activation offload — intermediates saved to host and reloaded, without rematerialization. A benchmark, not a shipping config field.

### Pin vs pageable as system efficiency

Device-visible host memory makes copies eligible for DMA and overlap. Wall time is measured afterward.

### What this post covers (evaluations, APIs, frameworks, best practices)

Headline table first: pin speedup per workload, one column per machine (Qwen3, longest sequence for the activation workloads). Then three or four findings in one sentence each. Close the introduction with the decision hint: pin on unless host RAM or memlock is tight; measure on a new host link.

## Background: why pin can change offload efficiency

Page-lock for the OS, register with the driver for the copy engine. `mlock` without register stays pageable-class for GPU copies. Pinning does not by itself guarantee overlap; show the H200 overlap probe (pageable copy blocks the issuing thread for 27.9 ms; pinned returns in 0.017 ms and hides behind a ~10.9 ms GEMM). Why pageable is slower depends on the driver. On CUDA (measured on H200), pageable copies stage through a driver-managed buffer, block the issuing thread, and need CPU threads near the device. On Ascend (reported by the driver team, not measured by us), pageable memory is pinned and page-tabled on every copy. Say which statement is measured.

## Real-world evaluation

Equal section weight for the three workloads. Wall time leads. H2D/D2H follows.

### Setup (Qwen3; host RAM)

Qwen3 and Qwen3.5 are both required families. Describe both here, once: Qwen3 is dense full attention; Qwen3.5 interleaves Gated DeltaNet linear-attention layers with full attention and is not a bigger Qwen3. Hardware table for measured SKUs only: SKU, host CPU, interconnect, Torch/CUDA or PyTorch XPU, DeepSpeed commit, `ulimit -l`, `kernel.numa_balancing`, the launch policy, and the models per SKU. Host RAM: H200 Qwen3-8B with `zero.Init` held 137 GB on the host; ~27 bytes/param applies only without `zero.Init`.

### Method: written buffers, bound placement, step counts

Host buffers are written before timing (a never-written pageable buffer reads the zero page and inflates pageable H2D). Every e2e row is bound to the accelerator's NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`) and runs at least 10 steps. Report pin speedup within one SKU, one page condition, and one placement. Show the unbound-versus-bound spread once, as the reason for the method: same-host, same-day H200 pairs, model-tensor 2.90× vs 2.82×, activation checkpoint 64K 1.28× vs 1.15×, saved-activation 64K 4.46× vs 2.83×. Say which rows are not yet bound (H200 Qwen3.5; all Ascend until the rerun).

Each workload section below has the same shape: a Qwen3 table (one row per machine), a Qwen3.5 table (one row per machine), then the interpretation. Tables give pageable step, pinned step, and speedup only; no peak device memory column. Memory figures that support a point (the 64K capability claim, the saved-activation versus checkpoint comparison) go in the prose.

### Host placement: bound versus unbound (the fair pageable baseline)

On the H200 host, automatic NUMA balancing (`kernel.numa_balancing=1`) moves the pageable arm and leaves the pinned arm nearly unchanged. Unbound pageable times are inflated by balancing and by CPU threads landing on the far socket. Bound (`--cpunodebind` and `--preferred` on the GPU's node) is the best pageable condition we measured, so it is the fair baseline for pinned-versus-pageable. Show the unbound numbers as the cost of leaving placement to the scheduler, not as the pinning result. The bound speedups are smaller, and the post should say so: saved-activation at 64K drops from 3.85× unbound to 2.74× bound, and model-tensor from 3.62× to 2.82×. This is one host with balancing on. Hosts with balancing off, or other GPU-to-CPU topologies, may show less of the gap; the post should record the setting for each machine.

### Model-tensor offload (Arc Pro B60, H200, Ascend 910B4)

Pin on/off wall time, every collected sequence length. Qwen3: H200 Qwen3-8B seq 4096; Ascend Qwen3-4B seq 1024 and 2048; B60 Qwen3-0.6B or 1.7B. Qwen3.5: H200 Qwen3.5-9B seq 4096; Ascend Qwen3.5-4B seq 512 and 1024; B60 Qwen3.5-0.8B or 2B.

### Activation checkpoint workload (same hardware and families)

Same machines. Async on. Pin on/off moves the activation pool and the optimizer host buffers together. Longest sequence only: H200 Qwen3-8B and Qwen3.5-9B at 64K; Ascend Qwen3-4B and Qwen3.5-4B at 8192. Lead the interpretation with capability (64K on one GPU in under 56 GB), then why the pin gain is modest at that length. Cite #8282 async-vs-blocking (both pinned) as a different axis. A B60 engine miss opens this section with the commit and the error.

### Saved-activation offload, no rematerialization (Arc Pro B60, H200, Ascend 910B4)

Same three machines, separate panels. Longest sequence only: H200 Qwen3-8B and Qwen3.5-9B at 64K, `N=1`, `M=2`, `--offload-tensors -1`. Ascend 910B4 Qwen3-4B, batch 1, sequence 2048 (Qwen3.5-4B not collected). B60 is a Qwen3 0.6B or 1.7B, batch 1, sequence 2048, not yet collected. Close with the same-job 64K comparison against the activation checkpoint workload: pinning makes skipping recomputation competitive, but recomputation is still faster and uses far less memory on H200. Do not copy the H200 ratio onto another machine.

### Supporting H2D/D2H microbench

After the wall-time sections. CUDA: pageable versus pinned, with Torch versus native-registered as a parity inset. B60: pageable versus Torch/XPU pin. Ascend 910B4: native-registered versus pageable, own panel. Include the 4K-versus-2M result for H200 as one line: with placement fixed, page size moves the copy speedup by at most 3% (5.43× vs 5.26× H2D) and every end-to-end pin speedup by at most 0.08×.

## APIs and popular frameworks

### PyTorch

`Tensor.pin_memory()`, and that `non_blocking` copies are asynchronous only from pinned memory. DataLoader `pin_memory` once, as the familiar hook, not as a benchmark. `torch.autograd.graph.save_on_cpu()` is stock saved-activation offload; its `pin_memory` defaults to `False`.

### Hugging Face checkpoint hooks

Non-reentrant checkpoint hooks used by the activation checkpoint workload: `CheckpointHiddenStatesOffload` / `get_checkpoint_hidden_states_offloading_ctx_manager`, with one short snippet. Backward must run inside the same context as forward.

### DeepSpeed (model-tensor CPU offload and cpu_checkpointing)

`offload_*.pin_memory`, activation `use_pin_memory`, and `cpu_checkpointing`. `cpu_checkpointing` without `partition_activations` uses the same pinned asynchronous pool and always pins; with `partition_activations` it uses blocking copies. Saved-activation has no config field.

### How to pin: Torch or native (parity once registered; when unpin matters)

`DS_PIN_MEMORY_BACKEND` defaults to Torch. `DS_PIN_MEMORY_REGISTER_DEVICE` is native only and defaults on.

Torch pin needs an accelerator. Its caching allocator rounds the request up to a power of two and keeps the block after the tensor is freed until `_host_emptyCache()`. A 5 GiB request is 8 GiB resident and stays 8 GiB after free.

DeepSpeed native pin is `posix_memalign` plus `mlock`, then a device register. The allocation is the size requested. `unpin` unregisters, then `munlock`s. Collection frees the pages if the caller never unpins. `mlock` without register does not make the copy asynchronous.

Once registered, native matches Torch on CUDA. Native is the path that returns locked pages, not a faster copy. The round-up and the cache-after-free matter when the locked footprint approaches memlock or host RAM.

## Best practices

### Independent knobs: offload pin_memory vs use_pin_memory

Model-tensor pin and activation-pool pin are separate switches. Both default on. Neither knob selects Torch versus native.

### Backend and device register

Default backend remains Torch. Device register is native CUDA and stays on. Ascend’s measured DMA arm is native-registered versus pageable and is not `cudaHostRegister`. B60 is Torch/XPU pin versus pageable. Use native when the locked footprint is large enough that power-of-two rounding or a cache that survives free will exhaust memlock.

### When to disable pin (memlock, host RAM)

Turn pin off when host RAM or memlock is tight. One machine’s results do not transfer to another. A flat large-copy GB/s chart is not by itself a reason to turn pin off: the issuing thread still waits on a pageable async copy.

### Host memory placement: NUMA node, automatic NUMA balancing, page size

Launch CPU-offload training bound to the accelerator’s NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`). Automatic NUMA balancing taxes every large host buffer each step, pinned or not, because CPU Adam state is pageable. Pageable arms are placement-sensitive; pinned arms much less so. Huge pages (2M) are not a lever on H200 once placement is fixed: end-to-end step times move ≤4% and pin speedups ≤0.08×. Ascend’s page-size result is separate and unverified.

## Conclusion

Use the same longest-sequence numbers as the headline table. Qwen3 first; Qwen3.5 as a separate clause, marked provisional while its H200 rows are unbound. End with the gaps: no XPU host registration yet; saved-activation offload is a benchmark, not a config option.

## Contributors

## Appendix: reproducing the results

Benchmark commands from DeepSpeedExamples `benchmarks/pin_memory` (link the [dev fork](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory) until it merges upstream), one per workload plus H2D/D2H, at the published shapes (activation workloads at 64K), each prefixed with the `numactl` binding. Pass `--credits 1 --prefetch 2 --offload-tensors -1` and `--steps 10` (plus `--warmup 1` for the activation workloads) explicitly, because the bench defaults differ. Mention that the `cuda/run.sh` and `npu/run.sh` launchers run the same rows with the binding applied. The saved-activation settings rationale (whole-layer scope, `--offload-tensors -1`, `N=1`, `M=2`) goes here, not in the evaluation.

---

## Figure files

Under `media/`, equal weight for the three wall-time workloads, microbench supporting. Each workload has a Qwen3 figure and a separate Qwen3.5 figure, one panel per machine (B60, H200, Ascend 910B4):

- `summary.png`: headline pin speedups, Qwen3, one panel per machine.
- `model_tensor_qwen3.png` / `model_tensor_qwen3_5.png`. H200 bound Qwen3-8B: 2.82×. H200 Qwen3.5-9B (unbound): 2.94×.
- `activation_checkpoint_qwen3.png` / `activation_checkpoint_qwen3_5.png`, longest sequence only. H200 at 256K (where offload starts to matter), bound: Qwen3-8B 1.15×, Qwen3.5-9B 1.16×.
- `saved_activation_qwen3.png` / `saved_activation_qwen3_5.png`, longest sequence only. H200 bound Qwen3-8B at 64K: 2.74×. H200 Qwen3.5-9B at 64K (unbound): 5.41×.
- Optional `activation_async_vs_blocking.png`: #8282 async versus blocking (both pinned), beside the activation checkpoint section, labeled as a different axis.
- `h2d_d2h.png`: supporting panels for CUDA (with the Torch versus native-registered parity inset), B60, and Ascend 910B4.
- `host_footprint.png`, beside How to pin: 5 GiB requested, 8 GiB Torch, 5 GiB native; after free, Torch still 8 GiB until the host cache is emptied, native at 0.
- `placement.png`, beside the placement section: faults per step and step time, unbound versus bound, pinned and pageable arms (H200).
- `page_size.png`, beside the same section: pin speedup at 4K versus 2M, bound, all H200 workloads.
