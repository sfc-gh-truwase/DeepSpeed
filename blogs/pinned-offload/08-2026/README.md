<!--
EDITOR NOTES (not rendered). Short blog, 2026-11-13. The long version with methodology, per-row step times, job IDs, and
reproduction commands is ARXIV_DRAFT.md; every number here comes from it. Headings follow the blog layout in LAYOUT.md.

Blocks publish:
- Intel Arc Pro B60: every result (single and multi-device) is not collected.
- Ascend 910B4: bound reruns of the single-device rows and the copy bandwidth (current numbers are unbound
  placeholders), the Qwen3.5-4B saved-activation row, and the multi-device results, including all-NPU copy bandwidth.
- 8×H200 torchrun --numa-binding=node with 12 threads for optimizer offload, to fill the launcher table's torchrun
  cells without numactl (EXPERIMENTS_HANDOFF.md section 2, item 8).
- Figures (media/) and the contributor list.
- Link to the arXiv version once it is posted.
-->

# Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads

## Introduction

CPU offload is an important technique for scaling training. By keeping model state or activations in host memory instead of accelerator memory, it reduces the accelerator memory a job needs, so a smaller or cheaper accelerator can train a model that would not otherwise fit, or a given accelerator can take on longer sequences. Its performance depends on efficient data transfers between accelerator and host memory, and on the host work around them. Three host-side settings control both: **pinning** the host buffers, **NUMA placement** of host memory and threads, and the number of **CPU threads** for host compute such as a CPU optimizer.

This blog shows:

1. The performance impact of tuning these settings on four workloads (host-device copies and three offload training workloads), across three accelerator types, on single-device and multi-device runs.
2. How PyTorch-ecosystem frameworks support pinning, NUMA placement, and CPU threads.
3. Best practices.

The [full technical report](ARXIV_DRAFT.md) has the methodology, every measurement, and the commands to reproduce them. <!-- TODO: link the arXiv version once posted. -->

## Performance impact of tuning

### Workloads and hardware

We evaluated four workloads. The first measures the copies themselves; the other three are offload training workloads, which can be combined in one job, and every training run used DeepSpeed:

- **Host-device copies.** A microbenchmark that copies 256 MiB host buffers to the accelerator and back, the transfer every offload workload depends on.
- **Model-tensor offload.** Parameters and optimizer states live in host memory and are copied to the accelerator as needed (DeepSpeed ZeRO CPU offload). The multi-device runs offload only the optimizer and keep the parameters on the accelerator.
- **Activation checkpoint offload.** The inputs saved by activation checkpointing go to host memory in the forward pass and come back for recomputation (DeepSpeed CPU checkpointing).
- **Saved-activation offload.** Every tensor saved for the backward pass goes to host memory and comes back, with no recomputation. This one is a benchmark built on DeepSpeed's pinned memory, not yet a DeepSpeed option.

On a single device, the two activation workloads also offload the optimizer, and pinning covers its host buffers too. The multi-device activation runs keep the optimizer on the accelerator.

We ran them on three accelerator types: NVIDIA H200, Huawei Ascend 910B4, and Intel Arc Pro B60, each within a single host (multi-node runs were not tested). Each training workload uses two Hugging Face model families in separate tables: Qwen3 (dense) and Qwen3.5 (hybrid linear and full attention), randomly initialized, in bf16, at batch size 1. Model size and sequence length differ by machine, so compare results within a column, not across columns. The full report lists hardware and software versions. Single-device runs and the copy benchmark are bound to the accelerator's NUMA node with `numactl`; multi-device runs use only each launcher's own binding.

### Single device: pinning

<!-- FIGURE: media/single_device.png — pin speedup per workload, one panel per machine, Qwen3 and Qwen3.5 as separate charts. -->

**Qwen3**

| Workload | H200, Qwen3-8B | Ascend 910B4, Qwen3-4B | Arc Pro B60 |
|----------|----------------|------------------------|-------------|
| Model-tensor offload | **2.82×** (4K tokens) | **1.85×** (1K), **1.47×** (2K) | *pending* |
| Activation checkpoint offload | **1.15×** (256K) | **1.23×** (8K) | *pending* |
| Saved-activation offload | **2.74×** (64K) | **1.14×** (2K) | *pending* |

**Qwen3.5**

| Workload | H200, Qwen3.5-9B | Ascend 910B4, Qwen3.5-4B | Arc Pro B60 |
|----------|------------------|--------------------------|-------------|
| Model-tensor offload | **2.77×** (4K tokens) | **1.32×** (512), **1.29×** (1K) | *pending* |
| Activation checkpoint offload | **1.16×** (256K) | **1.08×** (8K) | *pending* |
| Saved-activation offload | **3.73×** (64K) | *pending* | *pending* |

<!-- TODO: Ascend columns from the bound rerun; B60 columns. -->

Speedup is the pageable step time divided by the pinned step time, with both runs bound to the accelerator's NUMA node, the fairest pageable baseline we found. Pinning helped every workload on every machine we measured. The gain tracks how much of the step is copying: on the H200, model-tensor and saved-activation offload move the most data and gain the most, while checkpoint offload copies one tensor per layer and gains least. Qwen3.5 saved-activation offload gains most of all, plausibly because its linear-attention layers add little compute per token, so copies stay most of the step.

### Multiple devices: pinning, NUMA placement, and CPU threads

With several ranks on one host, the ranks share host memory, sockets, and cores, so placement and threads start to matter alongside pinning. The tables below start from torchrun's defaults (ranks unbound, memory pageable, one OpenMP thread per rank), on 8×H200. Each number is the default step time divided by the step time with the named setting changed; **All** changes every setting that applies to the workload. NUMA binding and **All** use DeepSpeed's launcher binding.

<!-- FIGURE: media/levers.png — speedup from the default launch for each setting alone and all together, one group per workload, one panel per machine. -->

**Qwen3, 8×H200**

| Workload | Pinning | NUMA binding | CPU threads | All |
|----------|---------|--------------|-------------|-----|
| Optimizer offload, Qwen3-32B | 1.70× | *pending* | 1.51× | **3.08×** |
| Activation checkpoint offload, Qwen3-8B, 256K tokens per GPU | 1.24× | 1.13× | — | **1.24×** |
| Saved-activation offload, Qwen3-8B, 16K tokens per GPU | 5.98× | 2.49× | — | **6.29×** |

**Qwen3.5, 8×H200**

| Workload | Pinning | NUMA binding | CPU threads | All |
|----------|---------|--------------|-------------|-----|
| Optimizer offload, Qwen3.5-27B | 1.62× | *pending* | 1.55× | **2.84×** |
| Activation checkpoint offload, Qwen3.5-9B, 256K tokens per GPU | 1.22× | 1.12× | — | **1.22×** |
| Saved-activation offload, Qwen3.5-9B, 16K tokens per GPU | 5.79× | 2.78× | — | **7.14×** |

*Ascend 910B4 and Arc Pro B60 multi-device results: pending.* <!-- TODO: Ascend 910B4 and Arc Pro B60 multi-device tables. -->

A dash means the setting does not apply: the activation workloads keep the optimizer on the GPU, so there is no host compute to thread. For optimizer offload, DeepSpeed's launcher always sets threads along with binding, so binding alone waits on a run with torchrun's own binding; binding and threads together gave 1.96× on Qwen3-32B and 1.80× on Qwen3.5-27B. The activation rows pack 4K-token samples into each step.

Because the starting point is unbound and pageable, the Pinning column also recovers some of the slowdown that binding fixes. Against a pageable run bound by DeepSpeed's launcher, pinning alone was 1.10× for Qwen3-8B checkpoint offload and 2.53× (Qwen3-8B) and 2.57× (Qwen3.5-9B) for saved-activation offload.

Offload is also what makes these runs fit. With the optimizer on the GPU, checkpoint offload fit 2.25× more tokens per GPU than plain checkpointing on Qwen3-8B. Saved-activation offload fit 2.29× more than training without checkpointing, but still far fewer than plain checkpointing, which remains the better choice for long sequences on this machine.

### Host-device copies

The step-time gains come from faster copies. The table gives the pinned speedup for 256 MiB copies in each direction, per device, bound to the accelerator's NUMA node, with one device copying and with every device in the host copying at once (eight on the H200 server).

<!-- FIGURE: media/h2d_d2h.png — pinned speedup per direction, one panel per machine, one device and all devices copying. -->

| Machine | Direction | One device copying | All devices copying |
|---------|-----------|--------------------|---------------------|
| H200 | Host to device | **5.4×** | **3.2×** |
| H200 | Device to host | **3.2×** | **1.7×** |
| Ascend 910B4 | Host to device | **4.0×** | *pending* |
| Ascend 910B4 | Device to host | **3.9×** | *pending* |
| Arc Pro B60 | Host to device | *pending* | *pending* |
| Arc Pro B60 | Device to host | *pending* | *pending* |

<!-- Sources: ARXIV_DRAFT.md single-device and all-GPUs-copying microbench tables. H200 one device: 10.2 -> 55.4 and 17.3 -> 54.9 GB/s; all eight, bound, per GPU: 8.6 -> 27.9 and 15.4 -> 26.4 GB/s. Ascend: 6.48 -> 25.97 and 6.71 -> 26.49 GB/s (unbound placeholder). -->

Copy speedups are larger than step speedups because only part of a step is copying: on one H200, a 5.4× faster host-to-device copy became a 1.15× to 3.73× faster training step. With all eight H200 GPUs copying, pinned bandwidth per GPU halves while pageable bandwidth barely moves, so the speedup shrinks. On our server, GPUs share a PCIe switch uplink in pairs, and pageable copies are limited by CPU staging rather than the link. Pinned copies in eight-GPU training ran at the same ceiling as this benchmark. Placement also starts to reach pinned copies: unbound, pinned device-to-host copies were 17% slower than bound.

### What the results show

- **Pinning is the biggest single lever when the step is copy-bound.** From torchrun's defaults, it alone made 8-GPU saved-activation offload 5.98× faster, against 2.49× for NUMA binding alone. Against a bound pageable run it was still 2.53×.
- **When the CPU computes, threads matter as much as pinning.** With the optimizer offloaded, CPU Adam is host compute that pinning cannot speed up. Giving it 12 threads per rank was worth 1.51×, close to pinning's 1.70×, and all three settings together gave 3.08×.
- **Pinning makes placement matter less, not zero.** On the H200, binding moved the pageable step by up to 2.8× and the pinned step by about 5% or less, except on Qwen3.5: 23% for saved-activation offload and 11% for optimizer offload. An unbound comparison can overstate the benefit of pinning by up to 2.4×. On the Ascend, a remote NUMA node still slowed pinned device-to-host copies by about 26%.
- **Shared links cap multi-device copies.** On our 8×H200 server, pinned copy bandwidth per GPU halved when all eight GPUs copied, while pageable bandwidth barely moved, so the per-GPU pin advantage shrank from 5.4× to 3.2× host to device.
- **Registration, not page-locking, makes host memory fast.** On the H200, memory that was locked but not registered with the device driver copied at pageable speed.
- **Some settings did not help.** On the H200, huge pages and the choice of pinning backend changed step time by at most 4% once placement was fixed and memory registered.

## Framework support in the PyTorch ecosystem

We compared four PyTorch training frameworks: PyTorch itself (including FSDP), Hugging Face Transformers and Accelerate, Megatron-LM with Transformer Engine, and DeepSpeed. Among them, DeepSpeed gives the most control over the three settings, and it is the only one that can set all three itself.

| Framework | Pinning | NUMA placement | CPU threads per rank |
|-----------|---------|----------------|----------------------|
| **DeepSpeed** | **On by default, adjustable per offloaded buffer, with a choice of pinning backend** | **Launcher option binds each rank's cores and memory** | **Same option matches threads to each rank's cores** |
| PyTorch, including FSDP | Through PyTorch's allocator; on by default for FSDP offload | Binds cores only, not memory | One by default; set by the user |
| Hugging Face Transformers and Accelerate | Through PyTorch's allocator; offload is delegated to DeepSpeed or FSDP | Binds cores only, not memory | One by default on GPUs; set by the user |
| Megatron-LM, with Transformer Engine | Through PyTorch's allocator; on by default for offload | Left to the user | Left to the user |

<!-- Sources and the exact flags per framework: ARXIV_DRAFT.md ("Across the PyTorch ecosystem"). DeepSpeed defaults verified in deepspeed/runtime/zero/offload_config.py, runtime/activation_checkpointing/offload_activations.py, launcher/launch.py, and utils/numa.py. -->

In short, DeepSpeed provides everything practitioners need to maximize offload performance: pinning is on by default, and one launcher option binds each rank's cores and memory and sets its threads. The table shows the best speedup each launcher reached on 8×H200 over torchrun's defaults (pageable, unbound, one thread per rank), each using only its own binding, with torchrun's thread count set by hand to each rank's share of cores.

| Workload, 8×H200 | torchrun | DeepSpeed launcher |
|------------------|----------|--------------------|
| Optimizer offload, Qwen3-32B | 3.07× | 3.08× |
| Optimizer offload, Qwen3.5-27B | 2.55× | **2.84×** |
| Activation checkpoint offload, Qwen3-8B | 1.24× | 1.24× |
| Activation checkpoint offload, Qwen3.5-9B | 1.22× | 1.22× |
| Saved-activation offload, Qwen3-8B | 6.28× | 6.29× |
| Saved-activation offload, Qwen3.5-9B | 7.12× | 7.14× |

<!-- From the ARXIV_DRAFT.md 8xH200 tables, DeepSpeed pinning on in both columns (pinned was the best arm for both launchers). Baseline = torchrun unbound pageable OMP_NUM_THREADS=1 (30.16, 22.71, 53.46, 50.07, 58.17, 80.72 s). torchrun = best of unbound and --numa-binding=node (activation workloads), and unbound with OMP_NUM_THREADS=12 (optimizer offload; --numa-binding was not run there): 9.82, 8.90, 43.01, 41.09, 9.27, 11.33 s. At torchrun's default one thread, optimizer offload was 17.73 s (1.70x) and 13.98 s (1.62x). DeepSpeed = deepspeed --bind_cores_to_rank: 9.78, 8.01, 42.97, 41.04, 9.25, 11.30 s. Pageable-only best, torchrun vs DeepSpeed: 1.00/1.96, 1.00/1.80, 1.01/1.13, 1.02/1.12, 1.53/2.49, 1.37/2.78. TODO: replace the torchrun optimizer-offload cells with --numa-binding=node plus OMP_NUM_THREADS=12 if better; that run is pending (EXPERIMENTS_HANDOFF.md section 2, item 8). -->

The launchers tie on the activation workloads, where pinning does most of the work, and on Qwen3-32B optimizer offload once torchrun's thread count is set by hand. On Qwen3.5-27B optimizer offload, DeepSpeed's launcher leads because it also binds each rank's memory. DeepSpeed's launcher sets the thread count itself; torchrun leaves CPU Adam on one thread unless the user changes it.

DeepSpeed's binding assumes accelerators are numbered in NUMA order; on hosts where they are not, bind ranks manually. The [full technical report](ARXIV_DRAFT.md) lists the flags and settings for each framework.

## Best practices

- **Keep pinning on.** It helped every workload we measured; turn it off only when locked host memory runs short.
- **Bind each rank's cores and memory to its accelerator's NUMA node.** Binding cores alone is not enough.
- **Give a CPU optimizer all the cores its rank owns.**
- **Measure on your own host, against a bound baseline.** Results do not transfer between machines.
- **Don't count on huge pages or a different pinning backend for speed.** On the H200, neither changed step time by more than 4%.

## Conclusion

CPU offload buys capacity with host memory, and three host-side settings decide how much speed that capacity costs. Pinning is the biggest single lever when the step is dominated by copies, giving up to 3.73× on one H200 and 2.57× on eight, both against a bound pageable baseline. When the CPU computes, as with an offloaded optimizer, CPU threads matter as much, and the three settings together made an 8-GPU optimizer-offload step 3.08× faster than torchrun's defaults. The defaults to change are the launcher's, not DeepSpeed's: keep pinning on, bind each rank with a memory policy, and give the CPU optimizer its cores.

The benchmarks are in [DeepSpeedExamples `benchmarks/pin_memory`](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory), and the [full technical report](ARXIV_DRAFT.md) has the details. <!-- TODO: switch to deepspeedai/DeepSpeedExamples master and the arXiv link. -->

## Contributors

<!-- TODO: contributor list (DeepSpeed, Snowflake, Intel, Huawei Ascend). -->
