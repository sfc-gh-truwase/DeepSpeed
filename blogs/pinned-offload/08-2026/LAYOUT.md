# Proposed layout — Efficient CPU Offload

**Article files:** [`README.md`](README.md) is the short blog (2026-11-13). [`ARXIV_DRAFT.md`](ARXIV_DRAFT.md) is the long version, preserved as the arXiv draft; the blog links to it for methodology, per-row results, and reproduction. B60 rows, bound Ascend reruns, and Ascend/B60 multi-device results are pending in both.\
**Figures:** `media/`, one panel per machine, one chart per model family.\
**Source of the argument:** [`BLOG_PLAN.md`](BLOG_PLAN.md). This file is the heading layout only.

**Title (both):** Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads

## Blog layout (README.md)

Short and high level (user, 2026-11-13): results as speedups only, no step-time tables, no method section, no commands, no job IDs in the text. Every number comes from the arXiv draft.

```
# Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads
## Introduction
## Performance impact of tuning
### Workloads and hardware
### Single device: pinning
### Multiple devices: pinning, NUMA placement, and CPU threads
### What the results show
## Framework support in the PyTorch ecosystem
## Best practices
## Conclusion
## Contributors
```

- **Introduction:** CPU offload as a training scalability technique that reduces the accelerator memory a job needs (a cheaper accelerator, or longer sequences); performance depends on accelerator–host transfers and the host work around them, controlled by pinning, NUMA placement, and CPU threads. Then the three things the blog shows: (1) tuning impact on three workloads across three accelerator types, single and multi-device; (2) how PyTorch-ecosystem frameworks support the three settings; (3) best practices. Link the full report.
- **Workloads and hardware:** one bullet per workload, naming the DeepSpeed feature each ran on; one sentence that single-device activation runs also offload (and pin) the optimizer while multi-device activation runs keep it on the accelerator; the three machines, two model families, single-host scope, and model setup in one paragraph, with the rule to compare within a column.
- **Single device:** a Qwen3 table and a Qwen3.5 table, workload × machine, pin speedup only (sequence length in parentheses for every cell; list multiple lengths separately, never as a range). Unfilled cells say *pending*. One short paragraph of interpretation that defines the bound pageable baseline and quotes the full single-device range across both families.
- **Multiple devices:** a Qwen3 table and a Qwen3.5 table per machine, workload × setting (pinning, NUMA binding, CPU threads, all), from torchrun's defaults, with the ratio and **All** defined before the tables. Then: the dashes; the bound-baseline pin speedups, since the Pinning column also recovers what binding fixes; and the capacity gains, each against its own baseline (saved-activation offload still fits far fewer tokens than plain checkpointing).
- **What the results show:** five short findings. Scope each to the machine it was measured on ("On the H200") until the other machines confirm it.
- **Framework support in the PyTorch ecosystem:** scope to the four PyTorch training frameworks compared, named in the first sentence; then the claim that among them DeepSpeed gives the most control over the three settings (the only one that handles all three itself). One table, framework (DeepSpeed first; PyTorch including FSDP; Hugging Face Transformers and Accelerate; Megatron-LM with Transformer Engine) × setting (pinning, NUMA placement, CPU threads per rank). Report what each framework supports, not how: no flags, environment variables, or allocator internals (those stay in the arXiv draft's "Across the PyTorch ecosystem"). Then a short summary that DeepSpeed provides everything practitioners need to maximize offload performance: pinning on by default, binding and threads from one launcher option, and the measured cost of skipping that option (no per-setting bullets; the table already covers them), the NUMA-order caveat, and a link to the full report for the flags.
- **Best practices:** five short bullets saying what to do, not how: no flags, commands, or environment variables (those stay in the arXiv draft).
- **Conclusion:** one paragraph, then the benchmark and full-report links.

## arXiv draft layout (ARXIV_DRAFT.md)

The rest of this file is the layout of the long version. Its headings are the `##`–`####` headings below.

**Scope (2026-11-13):** the post covers three levers for efficient CPU offload. Pinned memory is the lead lever. All three levers are measured on every machine, on one accelerator and on several in one host; today only the H200 set is complete, and the Ascend 910B4 and Arc Pro B60 single- and multi-device results will be collected before publishing (user, 2026-11-13). The thesis: pinning is the biggest single lever when the step is copy-bound; when the CPU computes (CPU optimizer), threads and placement matter as much; the levers interact, so each is measured against a good setting of the others.

Headings stay accelerator-neutral. ZeRO and pull-request numbers stay in the body. Each machine’s wall-time figure is its own panel; Ascend 910B4 does not share a chart with H200 or B60. Every evaluated machine (Arc Pro B60, H200, Ascend 910B4) runs all three workloads and the H2D/D2H microbench on one accelerator, and the multi-device workloads on several accelerators in one host. Pinning comparisons are measured bound to the accelerator’s NUMA node.

**Evaluation hierarchy:** device count, then workload, then machine. `### Single accelerator` holds the three workloads and the microbench for every machine; `### Multiple accelerators` holds optimizer offload, activation checkpoint offload, and saved-activation offload with the capacity, placement, and thread results for every machine. In the single-accelerator sections, machines are rows within each workload's tables; in the multi-device sections, each machine gets its own Qwen3 and Qwen3.5 tables (labeled `**<machine>, Qwen3**`), because rows there are launches or placements.

**Voice:** a technical blog, not a tutorial. Each evaluation section leads with its results table and the interpretation. Setup and method stay in their own sections. Benchmark commands go only in the closing appendix.

**Model families:** each workload section reports Qwen3 and Qwen3.5 in separate tables and separate charts, Qwen3 first. Qwen3.5 never shares a chart with Qwen3. There is no standalone Qwen3.5 section. Capacity tables (one row per model, no pin comparison) may list both families.

**Sequence lengths:** the single-accelerator activation sections print only the longest sequence each machine ran, per family, except H200 activation checkpoint, which prints 256K (where offload starts to matter on one GPU). Shorter rows stay in source comments. Model-tensor prints every collected length. The multi-device sections print the capacity walls and one pin/placement length per workload and machine (H200: 256K per GPU for activation checkpoint, 16K per GPU for saved-activation, 4096 per GPU for optimizer offload; Ascend and B60 lengths chosen the same way, inside each capacity window).

**Tables** give pageable step, pinned step, and speedup; multi-device tables add a launch or placement column. No peak device memory column. Memory figures that support a point (capacity, the saved-activation versus checkpoint comparison) go in the prose.

---

## Introduction

### CPU offload for cost-efficiency and capability

Offload trades device memory for host memory: smaller or cheaper cards, longer sequences. The cost is host work: link traffic and, with a CPU optimizer, host compute. Capacity claims cite a no-offload baseline that does not fit (1×H200 256K, multi-device capacity tables).

### Three applications: model tensors, checkpoint activations, and saved activations

Name all three before any DeepSpeed tour.

1. Model-tensor offload — parameters and optimizer states. Optimizer-only offload (multi-device) is the same application.
2. Activation checkpoint offload — checkpoint hidden states with rematerialization. On one GPU, together with ZeRO-2 CPU AdamW and memory-saving kernels.
3. Saved-activation offload — intermediates saved to host and reloaded, without rematerialization. A benchmark, not a shipping config field.

### Three levers: pinned memory, NUMA placement, and CPU threads

Pinning makes copies eligible for DMA and overlap. Placement decides where host buffers and pageable staging threads live. Threads decide how fast host compute (CPU Adam) runs. Each lever acts on a different part of the step; its gain follows that part's share.

### What this post covers (evaluations, APIs, frameworks, best practices)

Two tables, then findings, then the decision hint.

1. Lever table: multi-device, Qwen3, one table or column group per machine (H200 today; Ascend and B60 to come), each lever alone and all together, from torchrun's defaults (unbound, pageable, one OpenMP thread). Rows: optimizer offload (Qwen3-32B), activation checkpoint (Qwen3-8B, 256K per GPU), saved-activation (Qwen3-8B, 16K per GPU). Dashes where a lever was not varied alone, explained under the table.
2. Pinning table: one accelerator per machine, bound pageable baseline, Qwen3, longest sequence for the activation workloads (H200 activation checkpoint at 256K).

Five or six one-sentence findings: pinning is the biggest single lever when copy-bound; threads and placement matter as much when the CPU computes; the levers interact; copy bandwidth does not predict step time; registration, not page-locking; offload buys capacity, the levers buy speed. Decision hint: keep pinning on, bind each rank with a memory policy, give a CPU optimizer its cores; disable pinning only under memlock or host-RAM pressure; measure on a new host.

## Background: why pinning, placement, and threads change offload efficiency

### Pinned memory: page-locked and registered

Page-lock for the OS, register with the driver for the copy engine. `mlock` without register stays pageable-class. Pinning does not by itself guarantee overlap; show the H200 overlap probe (pageable copy blocks the issuing thread for 27.9 ms; pinned returns in 0.017 ms and hides behind a ~10.9 ms GEMM). Why pageable is slower, per driver: CUDA measured on H200; Ascend reported by the driver team; XPU not characterized. Say which statement is measured.

### NUMA placement: where host memory and staging threads live

CUDA stages pageable copies on CPU threads, so they depend on thread and page placement; pinned copies are DMA. One number: single-GPU activation checkpoint, far-socket threads, pageable 7.2 → 10.5 s, pinned 4.2 → 5.5 s. Automatic NUMA balancing migrates pages of large host buffers; an explicit memory policy turns it off.

### CPU threads: host compute inside the step

CPU Adam runs on OpenMP threads; pinning cannot speed up that part. torchrun sets `OMP_NUM_THREADS=1` per rank by default.

## Real-world evaluation

Grouped by device count. Equal weight for the three applications. Wall time leads; H2D/D2H follows the single-accelerator workloads.

### Setup (Qwen3; host RAM)

Describe Qwen3 (dense) and Qwen3.5 (Gated DeltaNet hybrid, not a bigger Qwen3) once. Hardware table for measured SKUs only: devices used (1 of 8 and all 8 on H200), device memory, host CPU, link, software, DeepSpeed commit, `ulimit -l`, `kernel.numa_balancing`, launch policy, models (including Qwen3-32B and Qwen3.5-27B for 8×H200 optimizer offload). Host RAM: H200 Qwen3-8B with `zero.Init` held 137 GB; ~27 bytes/param without it.

### Method: written buffers, one lever at a time, step counts

Write host buffers before timing. Change one lever at a time. Measure pinning against a bound baseline (the fastest pageable placement measured). Compare within one machine, page size, and placement. At least 10 steps per arm, fresh process per arm. Show the unbound-versus-bound pin speedups once (H200, same-job pairs: model-tensor 3.62× vs 2.82×, activation checkpoint 256K 1.24× vs 1.15×, saved-activation 64K 3.85× vs 2.74×, 8×H200 activation checkpoint 1.24× vs 1.10×, 8×H200 saved-activation 5.98× vs 2.56×) as the reason for the bound baseline.

### Single accelerator (Arc Pro B60, H200, Ascend 910B4)

#### Model-tensor offload

Pin on/off, every collected sequence length. Qwen3: H200 Qwen3-8B 4096; Ascend Qwen3-4B 1024 and 2048; B60 Qwen3-0.6B or 1.7B. Qwen3.5: H200 Qwen3.5-9B 4096; Ascend Qwen3.5-4B 512 and 1024; B60 Qwen3.5-0.8B or 2B. One placement sentence per machine (unbound vs bound; on H200 also the far-socket result).

#### Activation checkpoint offload

Async on. Pin on/off moves the activation pool and the optimizer host buffers together. H200 Qwen3-8B and Qwen3.5-9B at 256K (`cpu_checkpointing`, packed 4096-token samples, ZeRO-2 CPU AdamW); Ascend at 8192. Lead with why 256K (plain checkpointing does not fit or barely fits), then why the pin gain is modest. One placement sentence per machine (unbound inflates pageable only). Cite #8282 async-vs-blocking (both pinned) as a different axis. A B60 engine miss opens this section with the commit and the error.

#### Saved-activation offload, no rematerialization

H200 Qwen3-8B and Qwen3.5-9B at 64K, `N=1`, `M=2`, `--offload-tensors -1`. Ascend Qwen3-4B, sequence 2048 (Qwen3.5-4B not collected). B60 Qwen3 0.6B or 1.7B, sequence 2048, not yet collected. One placement paragraph per machine (largest unbound inflation on H200). Close with the 64K comparison against checkpoint offload (bound, separate jobs): pinning makes skipping recomputation competitive, but recomputation is still faster and uses far less memory on H200. Do not copy the H200 ratio onto another machine.

#### Supporting H2D/D2H microbench

CUDA: pageable versus pinned, Torch versus native-registered parity inset. B60: pageable versus Torch/XPU pin. Ascend: native-registered versus pageable, own panel, plus the remote-node cost on pinned D2H (~26%, against ≤6% on H200). One line on 4K versus 2M pages on H200.

### Multiple accelerators (Arc Pro B60, H200, Ascend 910B4)

One paragraph on why the shared host makes placement and threads matter, and the four launches (unbound, torchrun `--numa-binding=node`, per-rank `numactl`, `deepspeed --bind_cores_to_rank`). Each workload below: H200 first, then Ascend 910B4 and Arc Pro B60 with the same matrix and their own models and lengths.

#### Optimizer offload

Why offload at the chosen size (H200: GPU AdamW OOMs on Qwen3-32B, fits Qwen3.5-27B at the HBM edge; Ascend and B60 sized the same way). Per machine, a Qwen3 table and a Qwen3.5 table by launch (DS launcher, numactl + 12 threads, torchrun 12 threads unbound, torchrun default). Lead the interpretation with the lever split (threads 1.51×, pinning 1.70×, threads + binding 1.94×, all 3.10× on Qwen3-32B), then the interaction (pin gain 1.70× → 2.04× → 1.60×), then the GPU-AdamW cost for 27B.

#### Activation checkpoint offload, optimizer on the GPU

Capacity table per machine (checkpointing only vs checkpoint offload). Per machine, Qwen3 and Qwen3.5 placement tables at a length where checkpointing alone does not fit (H200: 256K per GPU). Interpretation: binding moves only the pageable arm; pinning alone recovers the whole gain; torchrun's CPU-only binding is not a substitute; the DS launcher matches `numactl`.

#### Saved-activation offload, optimizer on the GPU

Capacity table per machine (no checkpointing, no offload vs saved-activation offload; host-memory bound on H200). Per machine, Qwen3 and Qwen3.5 placement tables inside the capacity window (H200: 16K per GPU). Interpretation: binding doubles the pageable copy rate; pinning is still ~2.55× bound and the larger lever.

## APIs and popular frameworks

### PyTorch

`Tensor.pin_memory()`, and that `non_blocking` copies are asynchronous only from pinned memory. DataLoader `pin_memory` once. `torch.autograd.graph.save_on_cpu()` with `pin_memory` defaulting to `False`.

### Hugging Face checkpoint hooks

`CheckpointHiddenStatesOffload` / `get_checkpoint_hidden_states_offloading_ctx_manager`, one short snippet. Backward inside the same context.

### DeepSpeed (model-tensor CPU offload and cpu_checkpointing)

`offload_*.pin_memory`, activation `use_pin_memory`, `cpu_checkpointing` (pinned async pool without `partition_activations`; blocking with it). Saved-activation has no config field. `get_accelerator()` pin APIs, including `pin_empty()` / `pin_empty_like()` and eager `unpin_memory()`.

### How to pin: Torch or native (parity once registered; when unpin matters)

`DS_PIN_MEMORY_BACKEND` (default Torch), `DS_PIN_MEMORY_REGISTER_DEVICE` (native only, default on). Torch rounds to a power of two and caches after free; native is exact-size, registers where a hook exists, unpins eagerly, and works on CPU-only hosts. Parity once registered on CUDA.

### Launchers: NUMA binding and CPU threads

`numactl --cpunodebind --preferred` (CPU and memory policy; opts out of NUMA balancing; set threads yourself). `torchrun --numa-binding=node` (CPU affinity only; `OMP_NUM_THREADS=1` default). `deepspeed --bind_cores_to_rank` (cores by local rank order, strict memory binding, sets `OMP_NUM_THREADS`), with its two caveats.

### Across the PyTorch ecosystem

One table: PyTorch (including FSDP), Hugging Face Transformers and Accelerate, Megatron-LM with Transformer Engine, DeepSpeed × pinning, NUMA placement, per-rank CPU threads. One paragraph on what only DeepSpeed does (native pinning off PyTorch's allocator; launcher binds memory and sets threads). Sources in a comment.

## Best practices

Open with a three-line summary of the levers (pin offload buffers; bind each rank's CPUs and memory even when pinning; give the CPU optimizer its threads) and that `deepspeed --bind_cores_to_rank` covers the last two.

### Pinned memory: independent knobs (offload pin_memory vs use_pin_memory)

Separate switches, both default on; neither selects the backend.

### Pinned memory: backend and device register

Default Torch; native under memlock pressure; keep registration on. Ascend parity unmeasured; XPU has no registration hook.

### Pinned memory: when to disable (memlock, host RAM)

Only when host RAM or memlock is tight. A flat result is not a reason to disable. Results do not transfer across machines.

### NUMA placement: bind each rank to its accelerator's node

Bind with a memory policy (per-rank `numactl` or the DS launcher); CPU affinity alone is not enough. Pinning makes placement matter less (state the Qwen3.5 saved-activation exception). Automatic NUMA balancing taxes pageable CPU Adam state even when offload buffers are pinned.

### CPU threads: give the CPU optimizer its cores

Set `OMP_NUM_THREADS` to the cores per rank, or use the DS launcher. torchrun's default made a pinned step slower than a well-launched pageable one.

### What did not help: huge pages and backend choice

2M pages are not a lever on H200 once placement is fixed (≤4% step time, ≤0.08× pin speedup); Ascend's microbench result is separate and unverified. Torch and native tie once registered; choose the backend for host footprint.

## Conclusion

Lead with the levers: pinning biggest when copy-bound (H200 single-GPU Qwen3 1.15×–2.82× bound; Qwen3.5 about the same except saved-activation 3.73×; Ascend range from the bound rerun), host levers as large when the CPU computes (8×H200 Qwen3-32B: threads 1.51×, pinning 1.70×, all 3.10×; add the Ascend and B60 multi-device results). Four portable findings, checked on every machine. Close with the gaps: no XPU host registration; saved-activation offload is a benchmark.

## Contributors

## Appendix: reproducing the results

Benchmark commands from DeepSpeedExamples `benchmarks/pin_memory` (link the [dev fork](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory) until it merges upstream), grouped as single accelerator then multiple accelerators (H200, Ascend `npu/`, B60), at the published shapes, under the NUMA binding. Explain how to rerun the placement and thread arms (`--numa_bind none|torchrun_node|node|ds_bind_cores`, `OMP_NUM_THREADS`; launcher changes for saved-activation). Pass `--credits 1 --prefetch 2 --offload-tensors -1` explicitly. Mention the `cuda/run.sh` and `npu/run.sh` launchers. The saved-activation settings rationale goes here, not in the evaluation.

---

## Figure files

Under `media/`. Each workload has a Qwen3 figure and a separate Qwen3.5 figure, one panel per machine:

- `levers.png`: multi-device, Qwen3, speedup from the worst launch for each lever alone and all together, one group per workload, one panel per machine.
- `summary.png`: single-accelerator pin speedups, Qwen3, one panel per machine.
- `model_tensor_qwen3.png` / `model_tensor_qwen3_5.png`. H200 bound: Qwen3-8B 2.82×, Qwen3.5-9B 2.77×.
- `activation_checkpoint_qwen3.png` / `activation_checkpoint_qwen3_5.png`. H200 at 256K, bound: Qwen3-8B 1.15×, Qwen3.5-9B 1.16×.
- `saved_activation_qwen3.png` / `saved_activation_qwen3_5.png`. H200 at 64K, bound: Qwen3-8B 2.74×, Qwen3.5-9B 3.73×.
- Optional `activation_async_vs_blocking.png`: #8282 async versus blocking (both pinned), labeled as a different axis.
- `h2d_d2h.png`: supporting panels for CUDA (with the parity inset), B60, and Ascend 910B4.
- `host_footprint.png`, beside How to pin: 5 GiB requested, 8 GiB Torch, 5 GiB native; after free, Torch still 8 GiB, native 0.
- `placement.png`, beside the placement best practice: faults per step and step time, unbound versus bound, pinned and pageable arms (H200).
- `page_size.png`, beside What did not help: pin speedup at 4K versus 2M, bound, all H200 workloads.
