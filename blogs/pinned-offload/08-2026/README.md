<!--
EDITOR NOTES (not rendered). Draft of 2026-11-13. Scope widened on 2026-11-13 from pinned memory alone to efficient CPU
offload: pinned memory, NUMA placement, and CPU threads. Headings follow LAYOUT.md. The evaluation is organized by device
count, then workload, then machine. Qwen3.5 sits in each workload section in its own table and chart; the single-accelerator
activation workloads print only the longest sequence per machine (shorter rows are kept in HTML comments); benchmark
commands are in the closing appendix.

Blocks publish:
- Intel Arc Pro B60 (confirmed SKU): model-tensor, activation checkpoint, saved-activation, H2D/D2H, and Qwen3.5 rows are not collected.
  If the activation engine does not run on stock PyTorch XPU, the activation checkpoint section must open with the commit and error.
- Ascend 910B4: every row (Qwen3 and Qwen3.5) will be replaced by a run bound to the NPU's NUMA node.
  The Ascend numbers below are the earlier unbound collection and are placeholders until then.
- Ascend H2D/D2H will be re-run with host buffers written before timing; the Ascend page-size sentence waits on the same rerun.
- H200: every printed pinning row is bound, Qwen3.5-9B included (its model-tensor 2.77x is the bound run in job-20261107T043012Z; 2.94x is the unbound run in the same job).
- H200 Qwen3.5-9B saved-activation 64K, bound (2026-11-07 rerun): 108.53 s pageable, 29.08 s pinned, 3.73×.
- Ascend Qwen3.5-4B saved-activation row is not collected.
- Placement and CPU-thread levers are measured on the H200 only (plus one Ascend placement microbench point). Decide whether
  Ascend and B60 need a placement row, or whether the post states the gap.
- Setup table: DeepSpeed commit per SKU, and Ascend ulimit -l / numa_balancing / node, are missing.
- media/ figures not generated; placeholders mark where each goes.
- Contributors list is a placeholder.

Unbound H200 history (do not print as current): MT 3.40x (job-20260928T140704Z), ACT 64K 1.19x (job-20260926T165140Z),
SAO 64K 3.80x (job-20261011T171729Z).
-->

# Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads

## Introduction

### CPU offload for cost-efficiency and capability

CPU offload trades accelerator memory for host memory. It lets a smaller or cheaper card train a model that would not otherwise fit, and lets a datacenter GPU train sequences far longer than its memory alone allows. The price is host work: every offloaded byte crosses the host–device link at least twice per step, and an offloaded optimizer runs its update on host cores.

### Three applications: model tensors, checkpoint activations, and saved activations

Offload is not one technique. We study three, which are independent and can be combined in one job:

1. **Model-tensor offload.** Parameters and optimizer states live in host memory and are copied to the device as each layer needs them. Offloading only the optimizer, as in our 8-GPU runs, is the same application with the parameters left on the device.
2. **Activation checkpoint offload.** With activation checkpointing, only each layer's input is kept for the backward pass and everything else is recomputed. Those inputs go to the host during the forward pass and come back for recomputation. Our single-GPU workload combines this with CPU optimizer offload and memory-saving kernels, a realistic recipe for long-sequence training.
3. **Saved-activation offload.** Every tensor autograd saves inside each decoder layer goes to the host and comes back, with nothing recomputed. It moves far more data than checkpoint offload, in exchange for skipping the second forward pass.

### Three levers: pinned memory, NUMA placement, and CPU threads

How fast an offloaded step runs depends on the host as much as on the link. We measured three levers:

1. **Pinned memory.** The host side of each copy is either ordinary **pageable** memory or **pinned** memory: page-locked and registered with the device driver. Pinning makes a copy *eligible* for direct memory access (DMA) and for overlap with compute.
2. **NUMA placement.** A multi-socket server has a near and a far memory node for each accelerator. Placement decides which node holds the host buffers and which cores run the threads that stage pageable copies.
3. **CPU threads.** With a CPU optimizer, the optimizer step itself is host compute, and its speed depends on how many cores each rank gets.

Each lever acts on a different part of the step: pinning on the copies, placement on pageable staging and host memory traffic, threads on host compute. How much a lever helps depends on how much of the step that part takes, which is what we measured.

### What this post covers (evaluations, APIs, frameworks, best practices)

We measured all three levers end to end on Hugging Face Qwen3 and Qwen3.5 models: pinning on every machine and every application, placement and CPU threads on NVIDIA H200 servers. The first table separates the levers on an 8×H200 server, starting from the slowest launch we measured: torchrun's defaults, which leave ranks unbound, memory pageable, and one OpenMP thread per rank.

<!-- FIGURE: media/levers.png — 8xH200, Qwen3: step-time speedup from the worst launch for each lever alone and all together, one group per workload. -->

| 8×H200, Qwen3 | Pinning alone | NUMA binding alone | CPU threads alone | All levers |
|---------------|---------------|--------------------|-------------------|------------|
| Optimizer offload, Qwen3-32B, 4K tokens per GPU | 1.70× | — | 1.51× | **3.10×** |
| Activation checkpoint offload, Qwen3-8B, 256K tokens per GPU | 1.24× | 1.13× | — | **1.24×** |
| Saved-activation offload, Qwen3-8B, 16K tokens per GPU | 5.98× | 2.45× | — | **6.28×** |

Each entry is the worst launch's step time divided by the step time with only that lever changed. Binding means each rank under `numactl` on its own GPU's node, with a memory policy. Threads means `OMP_NUM_THREADS=12`, the cores each rank owns. A dash means the lever was not varied on its own: the activation workloads keep the optimizer on the GPU, so there is no host compute to give threads to, and for optimizer offload we bound ranks only together with setting threads (the two together: 1.94×).

The second table isolates pinning on one accelerator of each type, against a pageable baseline already bound to the accelerator's NUMA node:

<!-- FIGURE: media/summary.png — pin speedup by workload, one panel per machine (pageable = 1.0). -->

| Application | NVIDIA H200, Qwen3-8B | Ascend 910B4, Qwen3-4B | Intel Arc Pro B60 |
|-------------|-----------------------|------------------------|-------------------|
| Model-tensor offload | **2.82×** at 4K tokens | **1.85×** at 1K, **1.47×** at 2K | <!-- TODO --> |
| Activation checkpoint offload | **1.15×** at 256K | **1.23×** at 8K | <!-- TODO --> |
| Saved-activation offload | **2.74×** at 64K | **1.14×** at 2K | <!-- TODO --> |
| Host-to-device copy bandwidth | **5.4×** | **4.0×** | <!-- TODO --> |

<!-- TODO: Ascend column from the bound, written-buffer rerun. TODO: B60 column. -->

Speedup is the pageable step time divided by the pinned step time, for the same model, batch, and sequence length on the same machine. Activation rows are at the longest sequence each machine ran, except H200 activation checkpoint, which is at 256K, where offload starts to matter on one GPU. Columns use different models and sequence lengths, so compare within a column, not across.

- **Pinning is the biggest single lever when the step is copy-bound.** On 8×H200 saved-activation offload, pinning alone made the step 5.98× faster than the worst launch, against 2.45× for NUMA binding alone; once memory was pinned, binding added about 5%. Against a bound pageable baseline, pinning sped up every Qwen3 workload we measured, by 1.10× to 2.82×.
- **When the CPU computes, threads and placement matter as much as pinning.** With the optimizer offloaded on 8×H200, giving CPU Adam its 12 cores per rank made the Qwen3-32B step 1.51× faster, and threads plus binding 1.94×, against 1.70× for pinning alone. All three together: 3.10×.
- **The levers interact, so measure each against a good setting of the others.** Binding speeds up mostly the pageable arm, so it shrinks the measured pin speedup: without it, the same comparison overstated pinning by up to 2.3× (5.98× unbound against 2.56× bound). More CPU threads shorten the host compute, so copies become a larger share and pinning gains more (1.70× to 2.04× on Qwen3-32B).
- **Copy bandwidth does not predict step time.** On the H200, a 5.4× faster copy became anywhere from a 1.15× to a 2.82× faster single-GPU Qwen3 step.
- **Registration, not page-locking, makes host memory fast.** Locked-but-unregistered memory copies at pageable speed. Huge pages and the choice of pinning backend did not change speed once placement was fixed and memory registered.
- **Offload buys the capacity; the levers buy the speed.** On 8×H200 with the optimizer on the GPU, checkpoint offload ran 2.25× longer sequences than plain checkpointing, and saved-activation offload 2.29× longer than training without checkpointing. Pinning changed neither footprint.

After the evaluation, we cover the pinning APIs in PyTorch, Hugging Face, and DeepSpeed, the launcher settings that control placement and threads, and the practices that follow from the measurements. The short version: **keep pinning on, bind every rank to its accelerator's NUMA node with a memory policy, and give a CPU optimizer the cores each rank owns. Turn pinning off only when host RAM or the locked-memory limit is tight, and measure again on a new host.**

## Background: why pinning, placement, and threads change offload efficiency

### Pinned memory: page-locked and registered

A device's copy engine can only read host memory that will not move. Making memory usable for DMA takes two steps: **page-locking** it so the OS cannot move or swap it, and **registering** it with the device driver so the copy engine can address it (`cudaHostRegister` on CUDA, `aclrtHostRegisterV2` on Ascend). Locking alone is not enough. On the H200, memory that DeepSpeed had locked but not registered copied at 10.2 GB/s, the same as pageable memory.

Pinning makes overlap possible; it does not guarantee it. To see the difference, we issued a 256 MiB host-to-device copy with `non_blocking=True` on the H200, alongside a ~10.9 ms matrix multiply on another stream:

| Host buffer | Time for the copy call to return | Copy + matrix multiply, wall time |
|-------------|----------------------------------|-----------------------------------|
| Pageable | 27.9 ms | 27.9 ms |
| Pinned | 0.017 ms | 11.25 ms |

The pageable copy blocked the calling thread for the entire transfer, even though it was requested as non-blocking, so nothing launched after it could start. The pinned copy returned at once and hid behind the compute.

Why pageable copies are slower depends on the driver:

- **CUDA (measured on the H200).** The driver stages pageable data through its own pinned buffer, using CPU threads, while the issuing thread waits. A system-call probe found no per-copy `mlock`, `ioctl`, or `mmap` difference between pageable and pinned copies, and 2 MiB host pages left pageable bandwidth within noise.
- **Ascend (reported by the driver team, not measured by us).** Pageable memory is pinned and its device page tables are built on every copy, and asynchronous copies from pageable memory become synchronous.
- **Intel XPU.** We have not characterized the pageable path.

### NUMA placement: where host memory and staging threads live

Because CUDA stages pageable copies on CPU threads, a pageable copy is only as fast as those threads and the memory they read. Threads on the far socket, or pages on the far node, slow it down. A pinned copy is DMA by the device's copy engine and needs no CPU threads, so it is much less sensitive. On the H200, moving the CPU threads of a single-GPU activation checkpoint run to the far socket slowed the pageable step from 7.2 s to 10.5 s, and the pinned step only from 4.2 s to 5.5 s.

The kernel can also move memory on its own. With automatic NUMA balancing (`kernel.numa_balancing=1`), it samples the pages of every large host buffer each step and migrates them toward the threads that touch them. Any explicit memory policy for the process, such as `numactl --preferred`, turns balancing off for that process.

### CPU threads: host compute inside the step

With optimizer offload, DeepSpeed's CPU Adam updates the fp32 master weights and Adam state on the host, using OpenMP threads. That part of the step involves no copies, so pinning cannot speed it up, and its time scales with the threads each rank gets. torchrun sets `OMP_NUM_THREADS=1` for every rank unless told otherwise.

## Real-world evaluation

The three applications get equal weight. Results are grouped by device count: one accelerator of each type first, then an 8×H200 server. Each section leads with end-to-end step time; the copy microbenchmark follows the single-accelerator results as supporting evidence.

### Setup (Qwen3; host RAM)

All end-to-end runs use Hugging Face models built from their configs with random weights, batch 1, bf16. Each workload section reports two architectures in separate tables. **Qwen3** is dense, with full attention in every layer. **Qwen3.5** interleaves linear-attention (Gated DeltaNet) layers with full-attention layers, so its compute and activation profile differs: it is not a bigger Qwen3. Model size and sequence length are chosen per machine to fit its device and host memory, so ratios compare within a machine and architecture, not across.

| | NVIDIA H200 | Ascend 910B4 | Intel Arc Pro B60 |
|--|-------------|--------------|-------------------|
| Devices used | 1 GPU of 8; all 8 in the 8×H200 sections | 1 NPU | 1 GPU <!-- TODO --> |
| Device memory | 141 GB HBM3e | 29.5 GiB usable (32 GiB physical) | 24 GB |
| Host CPU | 2× Xeon 8488C, 192 CPUs, 2 NUMA nodes | Kunpeng-920, 42-core container quota | <!-- TODO --> |
| Host–device link | PCIe Gen5 | Ascend host link | PCIe <!-- TODO: generation --> |
| Software | PyTorch 2.14.0+cu130, Transformers 5.16.0.dev0, flash-attn 2.8.3, Liger kernel 0.8.4 | PyTorch with `torch_npu` <!-- TODO: versions --> | Stock PyTorch `xpu` (2.8+) <!-- TODO: versions --> |
| DeepSpeed commit | upstream `581ac41` (fork branch `tjruwase/pinned-offload-blog`; its later commits change docs only) | <!-- TODO --> | <!-- TODO --> |
| `ulimit -l` | unlimited | <!-- TODO --> | <!-- TODO --> |
| `kernel.numa_balancing` | 1 | <!-- TODO --> | <!-- TODO --> |
| Launch | `numactl --cpunodebind=0 --preferred=0`; in 8-GPU runs, each rank on its own GPU's node unless a table says otherwise | `numactl --cpunodebind=<n> --preferred=<n>` <!-- TODO: node --> | <!-- TODO --> |
| Models | Qwen3-8B, Qwen3.5-9B; Qwen3-32B, Qwen3.5-27B for 8×H200 optimizer offload | Qwen3-4B, Qwen3.5-4B | <!-- TODO: Qwen3-0.6B or 1.7B; Qwen3.5-0.8B or 2B --> |

Host RAM, not device memory, is often the limit for model-tensor offload. On the H200, Qwen3-8B with `zero.Init` kept 137 GB of offloaded state in host memory. Without `zero.Init`, plan on about 27 bytes of host memory per parameter (18 bytes of state, times 1.5 for buffers), about 206 GiB for Qwen3-8B. Pinning that much also needs `ulimit -l` raised to cover it.

### Method: written buffers, one lever at a time, step counts

The pinned arm uses DeepSpeed's native backend with device registration on CUDA and Ascend, except the Ascend activation checkpoint rows, which use PyTorch's NPU pinned allocator. XPU also uses PyTorch's pinned allocator, because DeepSpeed has no XPU registration hook. The pageable arm sets `pin_memory` to `false`. Each arm runs in a fresh process, for at least 10 steps. Four rules keep the comparisons honest:

- **Write host buffers before timing.** A never-written buffer maps to the kernel's shared zero page, which inflates pageable bandwidth: 20.8 GB/s instead of 10.1 GB/s host to device on the H200.
- **Change one lever at a time.** Placement and thread comparisons run every arm in one job, with pinning on and off.
- **Measure pinning against a bound baseline.** Every pinned-versus-pageable table without a placement column is launched with `numactl --cpunodebind=<n> --preferred=<n>` on the accelerator's node, which also turns off automatic NUMA balancing for the process. Bound was the fastest pageable placement we measured, so it is the fair baseline.
- **Compare within one machine, one page size, and one placement.**

The bound baseline matters because the pageable arm is the one that placement moves. On the same H200 host, comparing unbound and bound runs from the same job, unbound runs inflated every pin speedup:

| Application | Unbound | Bound |
|-------------|---------|-------|
| Model-tensor, 4K | 3.62× | **2.82×** |
| Activation checkpoint, 256K | 1.24× | **1.15×** |
| Saved-activation, 64K | 3.85× | **2.74×** |
| Activation checkpoint, 8×H200, 256K per GPU | 1.24× | **1.10×** |
| Saved-activation, 8×H200, 16K per GPU | 5.98× | **2.56×** |

<!-- Each row pairs unbound and bound runs from the same job. Model-tensor: job-20261107T043012Z (unbound 14.82 -> 4.09 s; bound 10.53 -> 3.74 s); an earlier job, job-20261024T171244Z, measured 2.90x unbound vs 2.82x bound, which is itself evidence that unbound pageable runs do not reproduce. Activation checkpoint 256K: job-20261111T160827Z (unbound 57.03 -> 46.07 s; bound 51.05 -> 44.52 s). Saved-activation 64K: job-20261107T202125Z. 8xH200: activation checkpoint job-20261112T192249Z (53.46 -> 43.01 s; 47.33 -> 42.97 s), saved-activation job-20261110T223544Z (58.17 -> 9.73 s; 23.71 -> 9.26 s). -->

Across the pairs with both arms recorded, binding moved the pinned step by 0–9% and the pageable step by 10–59%. Unbound runs also failed to reproduce: the same 8K saved-activation command measured 3.08× in one unbound run and 3.60× in another, while bound run-to-run spread was at most 0.37 s.

### Single accelerator (Arc Pro B60, H200, Ascend 910B4)

One accelerator per run. The activation workloads offload the optimizer to the CPU as well. Pinning is measured on every machine; placement on the H200 only.

#### Model-tensor offload

<!-- FIGURE: media/model_tensor_qwen3.png — one panel per machine, pageable = 1.0. -->

**Qwen3**

| Machine | Model, sequence length | Pageable step | Pinned step | Speedup |
|---------|------------------------|---------------|-------------|---------|
| H200 | Qwen3-8B, 4096 | 10.53 s | 3.74 s | **2.82×** |
| Ascend 910B4 | Qwen3-4B, 1024 | 21.26 s | 11.48 s | **1.85×** |
| Ascend 910B4 | Qwen3-4B, 2048 | 19.32 s | 13.16 s | **1.47×** |
| Arc Pro B60 | <!-- TODO --> | | | |

<!-- H200: job-20261107T043012Z, bound to GPU0's node (unbound in the same job: 14.82 -> 4.09 s, 3.62x). Ascend: TODO bound rerun; in the unbound data, seq 2048 pageable was faster than seq 1024 pageable — check whether that persists. -->

<!-- FIGURE: media/model_tensor_qwen3_5.png — own chart, never beside Qwen3. -->

**Qwen3.5**

| Machine | Model, sequence length | Pageable step | Pinned step | Speedup |
|---------|------------------------|---------------|-------------|---------|
| H200 | Qwen3.5-9B, 4096 | 11.36 s | 4.10 s | **2.77×** |
| Ascend 910B4 | Qwen3.5-4B, 512 | 14.45 s | 10.95 s | **1.32×** |
| Ascend 910B4 | Qwen3.5-4B, 1024 | 17.95 s | 13.95 s | **1.29×** |
| Arc Pro B60 | <!-- TODO: Qwen3.5-0.8B or 2B --> | | | |

<!-- H200 Qwen3.5: job-20261107T043012Z, bound to GPU0's node (unbound in the same job: 13.39 -> 4.56 s, 2.94x); fla + causal-conv1d installed. Ascend Qwen3.5: TODO bound rerun. -->

With ZeRO-3 CPU offload, every layer's parameters cross the link in every forward and backward pass, whatever the sequence length. On the H200 that transfer dominates the pageable step, and pinning turns a transfer-bound step into one close to compute-bound: about 3× on both architectures, as expected when parameter traffic, not the attention design, sets the step time. The Ascend gains are smaller but still substantial.

Placement mattered less here than the pinning gain. Unbound, the Qwen3-8B pageable step took 14.82 s instead of 10.53 s, and the pinned step 4.09 s instead of 3.74 s. With the CPU threads moved to the far socket, the pinned step did not change (3.77 s against 3.76 s).

#### Activation checkpoint offload

<!-- If the B60 activation engine does not run on stock PyTorch XPU, open this section with the commit and the error. -->

The H200 rows are at 256K tokens, where offload starts to matter on one GPU; the other rows are the longest sequence we ran on that machine.

<!-- FIGURE: media/activation_checkpoint_qwen3.png — one panel per machine, pageable = 1.0. -->

**Qwen3**

| Machine | Model, sequence length | Pageable step | Pinned step | Speedup |
|---------|------------------------|---------------|-------------|---------|
| H200 | Qwen3-8B, 256K | 51.05 s | 44.52 s | **1.15×** |
| Ascend 910B4 | Qwen3-4B, 8K | 15.44 s | 12.60 s | **1.23×** |
| Arc Pro B60 | <!-- TODO --> | | | |

<!-- H200 256K: job-20261111T160827Z, bound to node 0, mean of 2 runs x 10 timed steps; unbound 57.03 -> 46.07 s (1.24x, 1 run). Packed 4096-token samples, cpu_checkpointing with async copies (ds_async), ZeRO-2 + CPU AdamW, bench_8gpu.py --nproc 1; peak device memory 66.1 GB. Capacity (job-20261111T144343Z): checkpointing alone 136.1 GB at 262,144, out of memory at 294,912. 320K (same job): 62.73 -> 55.03 s, 1.14x bound; 65.30 -> 56.48 s unbound. Superseded copy-cost row, a length where offload is not needed: 64K dense, HF checkpoint hook, 32.60 -> 28.97 s, 1.13x, job-20261107T043012Z, 50.6 GB peak. Ascend: TODO bound rerun. -->

<!-- FIGURE: media/activation_checkpoint_qwen3_5.png — own chart, never beside Qwen3. -->

**Qwen3.5**

| Machine | Model, sequence length | Pageable step | Pinned step | Speedup |
|---------|------------------------|---------------|-------------|---------|
| H200 | Qwen3.5-9B, 256K | 49.54 s | 42.87 s | **1.16×** |
| Ascend 910B4 | Qwen3.5-4B, 8K | 92.58 s | 85.85 s | **1.08×** |
| Arc Pro B60 | <!-- TODO: Qwen3.5-0.8B or 2B --> | | | |

<!-- H200 Qwen3.5 256K: job-20261111T171212Z, bound to node 0, mean of 2 runs x 10 timed steps; unbound 57.47 -> 44.72 s (1.29x, 1 run). Same setup as the Qwen3-8B row; fla kernel active. Checkpointing alone runs out of device memory at 256K; offload peaks at 84.8 GB. Superseded unbound copy-cost row: 64K, 20.81 -> 16.67 s, 1.25x, job-20261016T193445Z, 55.7 GB.
Ascend Qwen3.5-4B is ~6x slower per step than Qwen3-4B at the same length; check that fla + causal-conv1d kernels were active (the same missing-package fallback made H200 Qwen3.5 ~8x slower before job-20261016T193445Z). TODO bound rerun. -->

Copies stay asynchronous in both arms; pinning applies to the activation pool and to the CPU optimizer's host buffers together. The H200 rows use DeepSpeed's `cpu_checkpointing` on packed 4096-token samples, with the optimizer offloaded to the CPU. 256K is where offload starts to matter on one GPU: plain checkpointing runs out of device memory there on Qwen3.5-9B, and fits Qwen3-8B with only 3.7 GB to spare (136.1 of 139.8 GB) before running out at 288K. With offload, the same steps peak at 66.1 GB and 84.8 GB. The pinning gain is modest because checkpoint offload copies only one tensor per layer, so copies are a small part of the step: pinning saved 6.5 s (Qwen3-8B) and 6.7 s (Qwen3.5-9B) per step. At 320K, where checkpointing alone no longer fits Qwen3-8B, the bound gain was 1.14× (62.73 s against 55.03 s).

Unbound, the 256K pageable steps took 57.03 s and 57.47 s instead of 51.05 s and 49.54 s, while the pinned steps moved by under 2 s, so the pin gains read 1.24× and 1.29×.

This comparison is different from the asynchronous-versus-blocking one in [#8282](https://github.com/deepspeedai/DeepSpeed/pull/8282), where both arms are pinned. That measures the value of overlap; this section measures the value of pinning.

<!-- OPTIONAL FIGURE: media/activation_async_vs_blocking.png — #8282 async vs blocking, both pinned; label as a different axis. -->

#### Saved-activation offload, no rematerialization

Each row is the longest sequence we ran on that machine.

<!-- FIGURE: media/saved_activation_qwen3.png — one panel per machine, pageable = 1.0. -->

**Qwen3**

| Machine | Model, sequence length | Pageable step | Pinned step | Speedup |
|---------|------------------------|---------------|-------------|---------|
| H200 | Qwen3-8B, 64K | 91.22 s | 33.25 s | **2.74×** |
| Ascend 910B4 | Qwen3-4B, 2K | 9.48 s | 8.34 s | **1.14×** |
| Arc Pro B60 | <!-- TODO: Qwen3-0.6B or 1.7B, 2K --> | | | |

<!-- H200 64K: job-20261107T202125Z, bound to GPU0's node, mean of 2 run medians (91.62 / 90.81 -> 33.24 / 33.26 s); unbound in the same job 3.85x. Peak device memory 96.2 GB allocated / 125.1 GB reserved (same job). Shorter H200 rows (not printed, bound, same job): 8K/16K/32K 2.98x/3.26x/3.17x. Ascend: TODO bound rerun. -->

<!-- FIGURE: media/saved_activation_qwen3_5.png — own chart, never beside Qwen3. -->

**Qwen3.5**

| Machine | Model, sequence length | Pageable step | Pinned step | Speedup |
|---------|------------------------|---------------|-------------|---------|
| H200 | Qwen3.5-9B, 64K | 108.53 s | 29.08 s | **3.73×** |
| Ascend 910B4 | <!-- TODO: Qwen3.5-4B, 2K --> | | | |
| Arc Pro B60 | <!-- TODO: Qwen3.5-0.8B or 2B --> | | | |

<!-- H200 Qwen3.5: rerun job-20261107T202125Z, bound and unbound (unbound 64K: 162.43 -> 30.04 s, 5.41x). Peak device memory not re-extracted from the rerun. -->

At most one layer's saved tensors stay on the device during the forward pass, and two layers are prefetched for the backward pass. On the H200, saved-activation offload gains the most from pinning of the single-GPU workloads because it copies the most: every saved tensor in every layer, both ways, so its copy volume grows as fast as the activations themselves. On Qwen3.5 the gain at 64K is 3.73×. One plausible reason is that its linear-attention layers add compute only linearly with length, so copies remain most of the step. On the Ascend, at a 2K sequence, the Qwen3 gain was only 1.14×.

Placement moved this workload the most. Unbound, the 64K pin gains read 3.85× (Qwen3-8B) and 5.41× (Qwen3.5-9B). For Qwen3.5-9B, the unbound pageable step took 162.43 s against 108.53 s bound, while the pinned step took 30.04 s against 29.08 s.

Pinning is what makes skipping recomputation competitive at all. At 64K on the H200 (bound, separate jobs), a pageable Qwen3-8B saved-activation step took 91.22 s, against 32.60 s for pageable checkpoint offload. Pinned, the gap narrows to 33.25 s against 28.97 s. Checkpoint offload remained faster and used far less device memory (50.6 GB against 125.1 GB reserved). Qwen3.5-9B shows the same ordering (29.08 s against 16.67 s pinned; the checkpoint figure is from an unbound run, and binding barely moves a pinned step). On this machine recomputation is still the better trade; saved-activation offload is the technique whose viability depends most on pinning. At 98K tokens it runs out of device memory on the H200.

<!-- 64K comparison: saved-activation job-20261107T202125Z; checkpoint offload (HF hook, ZeRO-2 CPU AdamW, --act-mem-opt) job-20261107T043012Z, bound, 32.60 -> 28.97 s, 50.6 GB peak. Qwen3.5 checkpoint 64K: job-20261016T193445Z, unbound, 20.81 -> 16.67 s. -->

#### Supporting H2D/D2H microbench

<!-- FIGURE: media/h2d_d2h.png — separate panels per machine; CUDA panel has the Torch vs native parity inset. -->

| Machine | Direction | Pageable | Pinned | Ratio |
|---------|-----------|----------|--------|-------|
| H200 | Host to device | 10.2 GB/s | 55.4 GB/s | **5.4×** |
| H200 | Device to host | 17.3 GB/s | 54.9 GB/s | **3.2×** |
| Ascend 910B4 | Host to device | 6.48 GB/s | 25.97 GB/s | **4.0×** |
| Ascend 910B4 | Device to host | 6.71 GB/s | 26.49 GB/s | **3.9×** |
| Arc Pro B60 | Host to device | <!-- TODO --> | <!-- TODO --> | |
| Arc Pro B60 | Device to host | <!-- TODO --> | <!-- TODO --> | |

<!-- 256 MiB transfers. H200: job-20261024T171244Z, bound, median of 3. Ascend: native-registered vs pageable; TODO bound, written-buffer rerun. B60: Torch/XPU pin vs pageable. -->

These numbers explain the step-time results but do not predict them: the H200's 5.4× copy speedup became a 1.15× to 2.82× faster Qwen3 step, depending on how much of the step was copying. Each machine's ratio also stands alone, because the link, the driver, and the pageable path differ.

Placement reaches pinned copies on some machines. On the Ascend, a buffer on the remote NUMA node cost pinned device-to-host copies about 26%, against at most 6% on the H200.

On CUDA, how memory is pinned does not change bandwidth, as long as it is registered. At 256 MiB host to device on the H200 (unbound), PyTorch's pinned memory and DeepSpeed's native registered memory both reached 55.4 GB/s. Native memory locked without registration reached 10.2 GB/s, the same as pageable memory.

<!-- H200 parity inset: job-20261023T131838Z, unbound, 256 MiB H2D/D2H: pageable 10.13/16.38, torch 55.40/54.92, native-unregistered 10.15/16.07, native-registered 55.40/54.93 GB/s. -->

With placement fixed, switching the H200 host from 4 KiB to 2 MiB pages changed the copy speedup by at most 3% (5.43× against 5.26× host to device) and the end-to-end pin speedup of every application by at most 0.08×.

### Eight GPUs (8×H200)

With eight ranks on one server, the host is shared: eight sets of host buffers, eight ranks' staging threads, and, with optimizer offload, eight CPU optimizers on two sockets. This is where placement and threads matter most. Each workload below varies pinning and placement in one job; optimizer offload also varies the thread count. The placements come from four launches: unbound; torchrun's `--numa-binding=node`; `numactl --cpunodebind=<n> --preferred=<n>` per rank; and DeepSpeed's `deepspeed --bind_cores_to_rank`.

#### Optimizer offload (Qwen3-32B, Qwen3.5-27B)

With eight GPUs, ZeRO shards the optimizer state, so a 9B model trains comfortably with AdamW on the GPU. Offload starts to matter at 27B–32B. With ZeRO-2 and 4096 tokens per GPU, AdamW on the GPU does not fit Qwen3-32B at all, and fits Qwen3.5-27B with only about 14.5 GB of HBM to spare (125.3 of 139.8 GB reserved), close enough to the edge that a small change in the environment can push it over. Offloading the optimizer to the CPU (DeepSpeed's CPU Adam, fp32 master weights and Adam state on the host) brings the peak down to 70.7 GB and 65.3 GB.

**Qwen3**

| Model, 4096 tokens per GPU | Launch | Pageable step | Pinned step | Speedup |
|----------------------------|--------|---------------|-------------|---------|
| Qwen3-32B | `deepspeed --bind_cores_to_rank` | 15.41 s | 9.78 s | **1.58×** |
| Qwen3-32B | `numactl` per rank, `OMP_NUM_THREADS=12` | 15.53 s | 9.73 s | **1.60×** |
| Qwen3-32B | torchrun, unbound, `OMP_NUM_THREADS=12` | 20.02 s | 9.82 s | 2.04× |
| Qwen3-32B | torchrun, unbound, default (`OMP_NUM_THREADS=1`) | 30.16 s | 17.73 s | 1.70× |

**Qwen3.5**

| Model, 4096 tokens per GPU | Launch | Pageable step | Pinned step | Speedup |
|----------------------------|--------|---------------|-------------|---------|
| Qwen3.5-27B | `deepspeed --bind_cores_to_rank` | 12.63 s | 8.01 s | **1.58×** |
| Qwen3.5-27B | `numactl` per rank, `OMP_NUM_THREADS=12` | 12.60 s | 8.02 s | **1.57×** |
| Qwen3.5-27B | torchrun, unbound, `OMP_NUM_THREADS=12` | 14.67 s | 8.90 s | 1.65× |
| Qwen3.5-27B | torchrun, unbound, default (`OMP_NUM_THREADS=1`) | 22.71 s | 13.98 s | 1.62× |

<!-- 8xH200 optimizer offload: job-20261111T024327Z, bench_8gpu.py (DSE 1bd3ad4) --zero2_offload --cpu_optimizer AdamW, one packed 4096-token sample per GPU, gradient checkpointing, Liger fused linear CE + TiledMLP x4; arm = offload_optimizer.pin_memory, native pin backend; mean of 2 runs x 10 timed steps; kernel.numa_balancing=1. Controls (ZeRO-2, GPU AdamW): Qwen3-32B out of memory; Qwen3.5-27B 2.96 s, 112.8 GB allocated / 125.3 GB reserved. Host 579 GiB (32B) and 491 GiB (27B) pod-wide. -->

This is the one workload where pinning is not clearly the biggest lever, because CPU Adam is host compute, not copying. From torchrun's defaults, each lever alone sped up the Qwen3-32B step by:

- **Threads** (`OMP_NUM_THREADS=12`): 1.51×.
- **Pinning:** 1.70×.
- **Threads plus binding:** 1.94×.
- **All three:** 3.10×, from 30.16 s to 9.73 s.

Qwen3.5-27B followed the same pattern: 1.55× for threads, 1.62× for pinning, 1.80× for threads plus binding, and 2.83× for all three.

The levers interact. With one thread, CPU Adam dominates the step, so pinning gains only 1.70×. Give it 12 threads and copies become a larger share, so pinning gains 2.04×. Then bind the ranks, which mostly speeds up pageable copies, and pinning's measured gain settles at 1.60×. With a good launch, pinning makes the offloaded step about 1.6× faster on both models.

`deepspeed --bind_cores_to_rank` sets each rank's thread count to the cores it hands out (12 here) and binds memory, and matched hand-tuned `numactl` with `OMP_NUM_THREADS=12`. For Qwen3.5-27B, offload still trades speed for headroom: with AdamW on the GPU its step takes 2.96 s, about 2.7× faster than the best offloaded step.

#### Activation checkpoint offload, optimizer on the GPU

The single-GPU rows keep the optimizer on the CPU, which leaves plain checkpointing enough device memory for those lengths. Keep the optimizer on the GPU instead, as many multi-GPU jobs do, and the picture changes. On 8×H200 with ZeRO-2 and GPU AdamW (fp32 master weights and optimizer state, sharded across the eight GPUs), plain checkpointing runs out of device memory well before DeepSpeed's CPU checkpoint offload (`cpu_checkpointing`) does:

| Model | Checkpointing only: longest fit | Checkpoint offload: longest run | Capacity gain |
|-------|---------------------------------|---------------------------------|---------------|
| Qwen3-8B | 233,472 tokens per GPU (out of memory at 241,664) | 524,288 (host memory runs out above) | **2.25×** |
| Qwen3.5-9B | 208,896 (out of memory at 217,088) | 413,696 or more (limit not reached) | **≥ 1.98×** |

At 262,144 tokens per GPU (256K), where checkpointing alone does not fit, pinning still pays, and placement decides how much:

**Qwen3**

| Model, 256K tokens per GPU | Placement | Pageable step | Pinned step | Speedup |
|----------------------------|-----------|---------------|-------------|---------|
| Qwen3-8B | bound, `numactl` per rank | 47.33 s | 42.97 s | **1.10×** |
| Qwen3-8B | bound, `deepspeed --bind_cores_to_rank` | 47.24 s | 42.97 s | 1.10× |
| Qwen3-8B | torchrun `--numa-binding=node` | 52.77 s | 43.04 s | 1.23× |
| Qwen3-8B | unbound | 53.46 s | 43.01 s | 1.24× |

**Qwen3.5**

| Model, 256K tokens per GPU | Placement | Pageable step | Pinned step | Speedup |
|----------------------------|-----------|---------------|-------------|---------|
| Qwen3.5-9B | bound, `numactl` per rank | 44.90 s | 41.04 s | **1.09×** |
| Qwen3.5-9B | bound, `deepspeed --bind_cores_to_rank` | 44.85 s | 41.04 s | 1.09× |
| Qwen3.5-9B | torchrun `--numa-binding=node` | 49.00 s | 41.09 s | 1.19× |
| Qwen3.5-9B | unbound | 50.07 s | 41.15 s | 1.22× |

Binding each rank to its own GPU's NUMA node took 6.1 s (Qwen3-8B) and 5.2 s (Qwen3.5-9B) off the pageable step and left the pinned step unchanged within 0.11 s. Pinning alone recovers the whole gain wherever the ranks land; binding alone recovers 55–60% of it. Pinning changes neither device nor host memory use: it buys speed, and the capacity comes from offload. torchrun's built-in `--numa-binding=node` is not a substitute: it pins each rank's CPU threads to the right node but sets no memory policy, and the pageable step stayed within 1.1 s of unbound. The memory policy that `numactl --preferred` adds, which also opts the process out of automatic NUMA balancing, is what the pageable arm needs. DeepSpeed's launcher applies one: `deepspeed --bind_cores_to_rank` matched the per-rank `numactl` result on both models.

<!-- 8xH200 setup: ZeRO-2, GPU AdamW, bf16 with fp32 master + Adam state; one packed row of 4096-token samples per GPU (attention within each sample); Liger fused linear CE, TiledMLP x4; ds_async = cpu_checkpointing with async copies, keep_last 1, no prefetch; DeepSpeed 0f8f058c9 (= upstream 581ac41 for these files); harness DSE activation_offload/bench_8gpu.py (74daac2).
Walls: job-20261110T002045Z (Qwen3-8B), job-20261110T011111Z (Qwen3.5-9B); a wall is the last passing cell, verdict from the harness RAW ok field; Qwen3-8B 540,672 was SIGKILLed with the pod at 1,900 GiB host. Step time jumps once reserved device memory nears ~135 GB (Qwen3-8B >= 491,520, Qwen3.5-9B >= 364,544), so 256K was chosen for the pin rows.
256K: job-20261112T192249Z, all four placements in one job -- unbound; numactl --cpunodebind=N --preferred=N per rank (--numa_bind node); torchrun --numa-binding=node (--numa_bind torchrun_node; ranks got cpus 0-47,96-143 / 48-95,144-191, mems 0-1); deepspeed --bind_cores_to_rank (--numa_bind ds_bind_cores; numactl -m <node> -C 0-11 ... 84-95) -- native pin backend, mean of 2 runs x 10 timed steps after 1 warmup, placement verified per rank; kernel.numa_balancing=1 on the host. Supersedes the 5-step rows (jobs 20261110T114439Z / 143411Z / 161516Z / 175333Z), which agree within 0.5 s when bound; torch and native pin backends tied within 0.03 s there. -->

#### Saved-activation offload, optimizer on the GPU

Saved-activation offload has its own capacity case: it skips recomputation, so its fair comparison is training without checkpointing, which keeps every saved tensor on the device. On 8×H200 with ZeRO-2 and GPU AdamW (fp32 state), that baseline runs out of device memory early, and saved-activation offload roughly doubles the length that fits, until the eight ranks' host buffers fill the host:

| Model | No checkpointing, no offload: longest fit | Saved-activation offload: longest run | Capacity gain |
|-------|-------------------------------------------|---------------------------------------|---------------|
| Qwen3-8B | 14,336 tokens per GPU (out of memory at 16,384) | 32,768 (host memory runs out at 36,864) | **2.29×** |
| Qwen3.5-9B | 10,240 (out of memory at 12,288) | 24,576 (host memory runs out at 28,672) | **2.40×** |

Each rank holds 7.3 MB (Qwen3-8B) or 9.1 MB (Qwen3.5-9B) of host buffers per token, so at 32K the eight ranks hold about 1.8 TiB. Device memory is not the limit: 68.5 GB at 32K. Plain checkpointing fits far longer sequences on the same setup (233,472 tokens per GPU for Qwen3-8B), so this gain is over training without recomputation only.

At 16K tokens per GPU, inside both windows, placement matters more here than for any other workload we measured:

**Qwen3**

| Model, 16K tokens per GPU | Placement | Pageable step | Pinned step | Speedup |
|---------------------------|-----------|---------------|-------------|---------|
| Qwen3-8B | bound, `numactl` per rank | 23.71 s | 9.26 s | **2.56×** |
| Qwen3-8B | bound, `deepspeed --bind_cores_to_rank` | 23.36 s | 9.25 s | 2.53× |
| Qwen3-8B | torchrun `--numa-binding=node` | 38.05 s | 9.27 s | 4.11× |
| Qwen3-8B | unbound | 58.17 s | 9.73 s | 5.98× |

**Qwen3.5**

| Model, 16K tokens per GPU | Placement | Pageable step | Pinned step | Speedup |
|---------------------------|-----------|---------------|-------------|---------|
| Qwen3.5-9B | bound, `numactl` per rank | 29.01 s | 11.37 s | **2.55×** |
| Qwen3.5-9B | bound, `deepspeed --bind_cores_to_rank` | 29.00 s | 11.30 s | 2.57× |
| Qwen3.5-9B | torchrun `--numa-binding=node` | 59.00 s | 11.33 s | 5.21× |
| Qwen3.5-9B | unbound | 80.72 s | 13.94 s | 5.79× |

Binding with a memory policy cut the pageable step by 2.5× (Qwen3-8B) and 2.8× (Qwen3.5-9B), because it doubled the pageable copy rate: device-to-host went from about 7 to about 15 GB/s per rank, and host-to-device from about 5 to 8.5 GB/s. Pinned copies ran at about 27 GB/s each way in every bound placement, and the pinned step moved by under 0.1 s across them. torchrun's CPU-only binding recovered only part of the pageable loss. Even with the best placement, pinning still made the step about 2.55× faster, against 1.10× for checkpoint offload at 256K, because saved-activation offload moves about 25 times more bytes per token. Of the two levers, pinning is the larger one here: alone it took the Qwen3-8B step from 58.17 s to 9.73 s, while binding alone reached 23.71 s.

<!-- 8xH200 saved-activation setup: ZeRO-2 + GPU AdamW (--zero2-gpu, fp32 master + Adam state), torchrun 8 ranks, published settings credits 1 / prefetch 2 / offload-tensors -1 / native pin backend, calibrated discovery 4096/8192; DSE 53096d8 (per-rank manifest device fix included). Walls: no-remat job-20261110T200243Z, offload job-20261110T213152Z (rerun after the fix). 16K matrix: job-20261110T223544Z, 10 timed steps after 1 warmup, mean of 2 run medians; numactl via benchmarks/pin_memory/numa_bind_rank.sh; kernel.numa_balancing=1. -->

## APIs and popular frameworks

### PyTorch

`Tensor.pin_memory()` returns a pinned copy of a CPU tensor, and `torch.empty(..., pin_memory=True)` allocates one directly. A `copy_(..., non_blocking=True)` or `.to(device, non_blocking=True)` is asynchronous only when the host side is pinned; from pageable memory it blocks, as the probe above shows. Most practitioners first meet pinning as `DataLoader(pin_memory=True)`, which pins input batches; offload applies the same idea to model state and activations.

PyTorch also ships a simple saved-activation offload, `torch.autograd.graph.save_on_cpu()`. Its `pin_memory` argument defaults to `False`, so its copies are pageable unless you opt in.

### Hugging Face checkpoint hooks

With non-reentrant gradient checkpointing (`use_reentrant=False`), the tensors each checkpointed Hugging Face decoder layer saves are visible to `torch.autograd.graph.saved_tensors_hooks`. DeepSpeed's `CheckpointHiddenStatesOffload` is a hook context manager that uses this to offload only the checkpointed hidden-state inputs, asynchronously on a side stream, into a pool of pinned host buffers; every other saved tensor passes through:

```python
from deepspeed.runtime.activation_checkpointing.offload_activations import (
    get_checkpoint_hidden_states_offloading_ctx_manager)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
offload = get_checkpoint_hidden_states_offloading_ctx_manager(use_pin_memory=True)
with offload:
    loss = model(**batch).loss
    loss.backward()
```

The backward pass must run inside the same context as the forward pass. `use_pin_memory` (default `True`) is the pin switch for the activation pool.

### DeepSpeed (model-tensor CPU offload and cpu_checkpointing)

**Model-tensor offload** is ZeRO CPU offload, with a pin switch for each offloaded state. Both default to `true`:

```json
"zero_optimization": {
  "stage": 3,
  "offload_param": {"device": "cpu", "pin_memory": true},
  "offload_optimizer": {"device": "cpu", "pin_memory": true}
}
```

**Activation checkpointing** has two pinned paths. Hugging Face models use the context manager above. DeepSpeed's own checkpointing API offloads checkpoint inputs when `activation_checkpointing.cpu_checkpointing` is `true`. Without `partition_activations`, it uses the same asynchronous engine and pinned pool as the Hugging Face path, and always pins. With `partition_activations`, it falls back to blocking copies.

**Saved-activation offload** has no DeepSpeed configuration field. We measured it with a benchmark built on the same hooks, not a shipping feature.

Code that manages its own host buffers can call `get_accelerator().pin_memory()`, `pin_empty()`, `pin_empty_like()`, `unpin_memory()`, and `is_pinned()`, which route to whichever pinning backend is active. Two of them matter for offload buffers. `pin_empty()` and `pin_empty_like()` allocate a pinned buffer of the needed shape directly, so a copy destination is never first built in pageable memory and then copied, as `tensor.pin_memory()` does. `unpin_memory()` releases a pinned buffer eagerly, when the caller is done with it, instead of when it is garbage-collected.

### How to pin: Torch or native (parity once registered; when unpin matters)

DeepSpeed has two pinning backends, selected with the `DS_PIN_MEMORY_BACKEND` environment variable. The default is `torch`.

**Torch** uses PyTorch's pinned allocator. It needs an accelerator build of PyTorch. It rounds each request up to a power of two and keeps freed blocks locked in its cache until `_host_emptyCache()`: a 5 GiB request locks 8 GiB, and the 8 GiB stay locked after the tensor is freed.

**Native** (`DS_PIN_MEMORY_BACKEND=native`) allocates exactly the requested size with `posix_memalign`, locks it with `mlock`, then registers it with the device where a registration hook exists (CUDA, Ascend, and SUPA, but not yet XPU). `unpin_memory()` unregisters and unlocks the pages at once; buffers the caller never unpins are freed when garbage-collected. `DS_PIN_MEMORY_REGISTER_DEVICE` (default `1`) controls the registration step; setting it to `0` leaves memory locked but unregistered, which copies at pageable speed. Native buffers are not tracked by PyTorch's stream machinery, so the caller must make sure pending copies finish before unpinning. Native also works on CPU-only hosts, where it locks without registering. On a host with no GPU, where both `tensor.pin_memory()` and `torch.empty(..., pin_memory=True)` fail for lack of a driver, native `pin_empty_like()` locked exactly the 1 GiB requested and `unpin_memory()` released all of it at once. <!-- 2026-11-10, cpu-dev pod (no NVIDIA driver), torch 2.9.1+cu130, DeepSpeed 9e82f407f, DS_PIN_MEMORY_BACKEND=native: VmLck 0 -> 1048576 kB after pin_empty_like of a 256x1024x1024 fp32 tensor, 0 kB after unpin_memory; torch pinning raised 'Found no NVIDIA driver on your system'. -->

<!-- FIGURE: media/host_footprint.png — 5 GiB requested: Torch 8 GiB, native 5 GiB; after free: Torch 8 GiB until the host cache is emptied, native 0. -->

Once registered, native copies exactly as fast as Torch on CUDA. Native is not a faster copy path. Its value is that it locks only what you asked for and gives it back when you unpin, which matters when the locked footprint approaches the memlock limit or host RAM.

### Launchers: NUMA binding and CPU threads

Placement and threads are set when the job launches, not in a configuration file. Three launch paths behave differently:

- **`numactl --cpunodebind=<node> --preferred=<node>`**, per process or per rank, binds CPU threads to the accelerator's node and prefers that node's memory. The memory policy also opts the process out of automatic NUMA balancing. Set `OMP_NUM_THREADS` yourself.
- **`torchrun --numa-binding=node`** sets CPU affinity only. It sets no memory policy, so automatic NUMA balancing stays on. torchrun also sets `OMP_NUM_THREADS=1` for every rank unless you override it.
- **`deepspeed --bind_cores_to_rank`** gives each local rank its own block of cores, binds memory to their node (`numactl -m <node>`), and sets `OMP_NUM_THREADS` to the number of cores it gave the rank. It hands out cores by local rank order, not by GPU, so it binds correctly only when GPU numbering follows NUMA order, as on our H200 (GPUs 0–3 on node 0; check with `nvidia-smi topo -m`). Its memory binding is strict, so a rank cannot spill to the other node when its own runs out.

## Best practices

### Pinned memory: independent knobs (offload pin_memory vs use_pin_memory)

ZeRO's `offload_*.pin_memory` controls model-tensor buffers; `use_pin_memory` controls the activation checkpoint pool. Both default on, and you can turn one off without touching the other. Neither selects a backend: they decide *whether* to pin, and `DS_PIN_MEMORY_BACKEND` decides *how*. Treat the defaults as a starting point to confirm with a measurement.

### Pinned memory: backend and device register

Keep the default Torch backend unless locked memory is tight. When it is, switch to native for exact-size allocations and explicit release, and leave `DS_PIN_MEMORY_REGISTER_DEVICE` on. On CUDA, registered native memory copies as fast as Torch's. On Ascend, native also registers (`aclrtHostRegisterV2`), but we have not measured parity, and the Ascend activation checkpoint runs used Torch's pinning, which was faster on that host. On XPU, DeepSpeed has no registration hook yet, so use Torch: native would be locked but unregistered, at pageable speed. The H200 results use native with `cudaHostRegister`; the B60 results use Torch's XPU pinning.

### Pinned memory: when to disable (memlock, host RAM)

Pinned pages cannot be swapped out, and the locked total is capped by `ulimit -l`. Turn pinning off when the offloaded footprint approaches host RAM or that limit, after first trying the native backend to avoid Torch's power-of-two rounding. A small or flat measured speedup is not by itself a reason to disable pinning: a pageable copy still blocks its issuing thread. Results from one machine do not transfer to another.

### NUMA placement: bind each rank to its accelerator's node

**Bind with a memory policy.** Launch single-accelerator offload training under `numactl --cpunodebind=<node> --preferred=<node>`, and bind each rank to its own device's node in multi-rank jobs, either with per-rank `numactl` or with `deepspeed --bind_cores_to_rank`. CPU affinity alone is not enough. On 8×H200 at 256K tokens per GPU, binding cut the pageable activation checkpoint step from 53.5 s to 47.3 s, while torchrun's CPU-only `--numa-binding=node` left it at 52.8 s. Saved-activation offload is the most sensitive: at 16K tokens per GPU on Qwen3-8B, binding took the pageable step from 58.2 s to 23.7 s, and CPU-only binding got it only to 38.1 s.

**Pinning makes placement matter less.** Across the 8-GPU placements, the pageable step moved by up to 2.8×, while the pinned step moved by about 5% or less, except in Qwen3.5-9B saved-activation offload, where unbound cost the pinned step 23% (13.94 s against 11.37 s). Pageable copies are staged by CPU threads; pinned copies are DMA. Model-tensor offload on one GPU was insensitive to placement once pinned (3.76 s against 3.77 s). Placement can still reach pinned copies on other platforms: on the Ascend, a remote node cost pinned device-to-host copies about 26%, against at most 6% on the H200, so measure per platform.

**Watch automatic NUMA balancing.** With `kernel.numa_balancing=1`, the kernel samples and migrates pages of every large host buffer each step. Pinning the offload buffers does not prevent this, because DeepSpeed's CPU Adam state is ordinary pageable memory. On the H200 it cost 1.6 to 6.8 million page faults and 10 to 70 CPU-seconds per step, in pinned and pageable runs alike. Any explicit task memory policy, such as the binding above, turns it off for the process.

<!-- FIGURE: media/placement.png — faults per step and step time, unbound vs bound, pinned and pageable arms (H200). -->

### CPU threads: give the CPU optimizer its cores

With optimizer offload, set `OMP_NUM_THREADS` to the cores each rank owns, or launch with `deepspeed --bind_cores_to_rank`, which does it for you. torchrun's default of one thread per rank left DeepSpeed's CPU Adam on a single core: on 8×H200 with Qwen3-32B, that made the pinned step 17.73 s instead of 9.82 s, slower than a pageable step with 12 threads per rank and binding (15.53 s). Pinning does not make up for a starved optimizer; set threads, placement, and pinning together (9.73 s).

### What did not help: huge pages and backend choice

**Don't use huge pages as a substitute for pinning.** With placement fixed, 2 MiB pages changed any H200 step time by at most 4% and any end-to-end pin speedup by at most 0.08×. An earlier impression that huge pages helped turned out to be NUMA-balancing noise. The Ascend driver may behave differently: a microbenchmark there showed pageable bandwidth rising sharply on 2 MiB pages, which we have not confirmed end to end. <!-- TODO: keep the Ascend clause only if the written-buffer rerun reproduces it. -->

<!-- FIGURE: media/page_size.png — pin speedup at 4K vs 2M, bound, all H200 workloads (job-20261027T132751Z). -->

**Don't switch pinning backends for speed.** Once registered, Torch and native copy at the same rate on CUDA, and the two tied within 0.03 s in the 8-GPU activation checkpoint runs. Choose the backend for host footprint, as above.

## Conclusion

CPU offload buys capacity with host memory; pinning, placement, and threads decide how much speed that capacity costs. Pinning is the biggest single lever when the step is copy-bound. On one H200 with Qwen3-8B, against a bound pageable baseline, it made offloaded steps **1.15× to 2.82× faster**: 2.82× for model-tensor offload, 2.74× for saved-activation offload at 64K tokens, and 1.15× for activation checkpoint offload at 256K. Qwen3.5-9B gained about the same, except for saved-activation offload, where it gained 3.73×. On an Ascend 910B4, with smaller models and shorter sequences, pinning helped every workload too, by **1.08× to 1.85×**<!-- TODO: update from bound Ascend rerun; add B60 -->. When the CPU computes, the host levers matter as much: on 8×H200 with the optimizer offloaded, CPU threads alone sped up the Qwen3-32B step 1.51×, pinning alone 1.70×, and all three levers together 3.10×.

Four findings should hold beyond these machines:

- The benefit of each lever follows the share of the step it touches, not raw copy bandwidth or core count.
- Registration with the device, not page-locking, is what makes host memory fast.
- Pinned runs are far less sensitive to NUMA placement than pageable ones, so pin speedups need a bound pageable baseline.
- Launcher defaults leave speed on the table. With torchrun's single OpenMP thread, a pinned optimizer-offload step was slower than a pageable one launched with threads and binding, and torchrun's CPU-only NUMA binding barely helped the pageable arm.

Pinning is the right default, with binding and threads set at launch. The exception is a host where the locked pages do not fit.

Three gaps remain. DeepSpeed does not yet register host memory on Intel XPU, so the native backend there is lock-only. Saved-activation offload is a benchmark, not a configuration option, even though on the H200 it is the application with the most to gain from pinning. And we measured placement and threads on H200 servers only; other hosts need their own measurement.

## Contributors

<!-- TODO: contributor list (DeepSpeed, Snowflake, Intel, Huawei Ascend). -->

## Appendix: reproducing the results

<!-- TODO: switch links to deepspeedai/DeepSpeedExamples master once the dev-fork branch merges. -->

All benchmarks are in [DeepSpeedExamples `benchmarks/pin_memory`](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory). Each command runs the pinned and pageable arms in separate processes, or one arm per command where noted. Run them from that directory, bound to the accelerator's NUMA node:

```bash
N=$(cat /sys/bus/pci/devices/<gpu BDF>/numa_node)
BIND="numactl --cpunodebind=$N --preferred=$N"

# --- Single accelerator ---

# Model-tensor offload
$BIND python model_tensor_offload/bench.py --model Qwen/Qwen3-8B --batch 1 --seq 4096 --steps 10

# Activation checkpoint offload (one arm per command; --no-act_pin_memory --no-opt_pin_memory for pageable)
DS_PIN_MEMORY_BACKEND=native $BIND python activation_offload/bench_8gpu.py --nproc 1 --zero2_offload --cpu_optimizer AdamW \
  --fused_ce --tiled_mlp_shards 4 --model_name Qwen/Qwen3-8B --mbs 1 --gas 1 --modes ds_async --seqlens 262144 \
  --packed_seq --pack_lens $(python -c "print(','.join(['4096'] * 64))") --packed_tiled_remat_budget 0 \
  --warmup 1 --iters 10 --act_pin_memory --opt_pin_memory

# Saved-activation offload
$BIND python saved_activation_offload/bench.py --model Qwen/Qwen3-8B --offload-optimizer --act-mem-opt \
  --offload-tensors -1 --credits 1 --prefetch 2 --batch 1 --seq 65536 --steps 10 --warmup 1 \
  --discover-calib-seqs 4096 8192

# Copy bandwidth (add --skip-native-register on XPU)
$BIND python h2d_d2h/bench.py

# --- Eight GPUs ---

# Optimizer offload (Qwen3-32B, Qwen3.5-27B). One arm per command (--no-opt_pin_memory for pageable).
# --numa_bind ds_bind_cores launches with deepspeed --bind_cores_to_rank; --numa_bind node uses numactl per rank.
DS_PIN_MEMORY_BACKEND=native python activation_offload/bench_8gpu.py --nproc 8 --zero2_offload --cpu_optimizer AdamW \
  --fused_ce --tiled_mlp_shards 4 --model_name Qwen/Qwen3-32B --mbs 1 --gas 1 --modes gc --seqlens 4096 \
  --packed_seq --pack_lens 4096 --packed_tiled_remat_budget 0 --warmup 1 --iters 10 --opt_pin_memory \
  --numa_bind ds_bind_cores

# Activation checkpoint offload, optimizer on the GPU (capacity and 256K rows).
# One arm per command (use --no-act_pin_memory for pageable); --numa_bind node binds each rank itself.
python activation_offload/bench_8gpu.py --nproc 8 --zero2 --fused_ce --tiled_mlp_shards 4 \
  --model_name Qwen/Qwen3-8B --mbs 1 --gas 1 --modes ds_async --seqlens 262144 --packed_seq \
  --pack_lens $(python -c "print(','.join(['4096'] * 64))") --packed_tiled_remat_budget 0 \
  --warmup 1 --iters 10 --act_pin_memory --numa_bind node

# Saved-activation offload, optimizer on the GPU (capacity and 16K rows).
# 1) Prepare manifests once, single process (forward-only discovery):
python saved_activation_offload/bench.py --model Qwen/Qwen3-8B --act-mem-opt --offload-tensors -1 --credits 1 \
  --prefetch 2 --batch 1 --offload-optimizer --discover-calib-seqs 4096 8192 \
  --manifest-out man-{seq}.json --prepare-seqs 16384
# 2) One arm per command (--pin 0 for pageable), each rank bound to its GPU's node:
torchrun --standalone --nproc_per_node=8 --no-python bash numa_bind_rank.sh python \
  saved_activation_offload/bench.py --model Qwen/Qwen3-8B --act-mem-opt --offload-tensors -1 --credits 1 \
  --prefetch 2 --batch 1 --zero2-gpu --mode offload --seq 16384 --steps 10 --warmup 1 \
  --manifest-out man-16384.json --pin 1
```

For the placement and thread rows of optimizer offload and activation checkpoint offload, rerun the 8-GPU commands with `--numa_bind` set to `none` (unbound, the default), `torchrun_node` (torchrun's `--numa-binding=node`), `node` (`numactl` per rank), or `ds_bind_cores` (`deepspeed --bind_cores_to_rank`), and set `OMP_NUM_THREADS` explicitly except under `ds_bind_cores`. For saved-activation offload, change the launcher around the same command: drop `numa_bind_rank.sh` for unbound, add `--numa-binding=node` to torchrun for its binding, or launch with `deepspeed --bind_cores_to_rank`.

For Qwen3.5, substitute `--model Qwen/Qwen3.5-9B` (or `--model_name`); it needs a Transformers release that registers `qwen3_5`, plus the `flash-linear-attention` and `causal-conv1d` packages, without which the linear-attention layers silently fall back to a much slower path. The `cuda/run.sh` launchers for the activation checkpoint and saved-activation workloads apply the binding (`MODEL=` and `SEQ=` override the model and length), and the `npu/run.sh` launchers run the Ascend rows (set `NUMA_NODE=<n>`).

`--act-mem-opt` enables Liger fused linear cross-entropy and TiledMLP (Liger kernel 0.8.4 or later for Qwen3.5). Pass `--prefetch 2`, `--credits 1`, and `--offload-tensors -1` explicitly: the benchmark's defaults (`1`, `2`, and `3`) differ from the settings these results used. `--discover-calib-seqs` extrapolates the offload plan from two short calibration runs, which keeps discovery itself from running out of memory at 32K and longer. On the H200, the saved-activation settings were chosen by measurement. Offloading only the MLP left attention activations resident and ran out of memory at 16K. Capping tensors per layer raised peak memory. Holding one layer instead of two cut peak memory 8–10% at 32K–64K, with step time within ±2.3%. Prefetching one layer instead of two saved 22% more memory at 64K but slowed the pinned step by 12%.
