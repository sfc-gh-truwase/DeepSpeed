<!--
EDITOR NOTES (not rendered). Short blog, 2026-11-13. There is no long report (dropped 2026-11-14): the how (flags per framework,
setup, method, reproduction commands) lives in the DeepSpeedExamples benchmarks/cpu_offload README. Per-row step times
and job IDs are in the hidden source comments below and in ARXIV_DRAFT.md as of commit 3d2fd3b (git history). Headings follow the blog layout in LAYOUT.md.

Blocks publish:
- Intel Arc Pro B60: every result (single and multi-device) is not collected.
- Ascend 910B4: bound reruns of the single-device rows and the copy bandwidth (the earlier unbound numbers are
  hidden as pending), the Qwen3.5-4B saved-activation row, and the multi-device results, including all-NPU copy bandwidth.
- Ascend and B60 single-device rows and one-device copies under launcher binding instead of numactl
  (EXPERIMENTS_HANDOFF.md section 2, item 10; H200 collected).
- Figures (media/) and the contributor list.
- Switch the DeepSpeedExamples links to deepspeedai/DeepSpeedExamples master once the branch merges.
-->

# Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads

## Introduction

CPU offload is an important technique for scaling training. By keeping model state or activations in host memory instead of accelerator memory, it reduces the accelerator memory a job needs, so a smaller or cheaper accelerator can train a model that would not otherwise fit, or a given accelerator can take on longer sequences. Its performance depends on efficient data transfers between accelerator and host memory, and on the host work around them. Three host-side settings control both: **pinning** the host buffers, **NUMA placement** of host memory and threads, and the number of **CPU threads** for host compute such as a CPU optimizer. Pinned host memory is locked in place and registered with the accelerator, so the accelerator can copy it directly; NUMA placement keeps a process's threads and host memory on the CPU socket nearest its accelerator.

This blog shows:

1. The performance impact of tuning these settings on four workloads (three offload training workloads and host-device copies), across three accelerator types, on single-device and multi-device runs.
2. How PyTorch-ecosystem frameworks support pinning, NUMA placement, and CPU threads.
3. Best practices.

The [benchmark README](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/cpu_offload) in DeepSpeedExamples has the setup, the method, the settings for each framework, and the commands to reproduce every result.

## Performance impact of tuning

### Workloads and hardware

We evaluated four workloads and measured the speedup of using pinned memory over pageable memory. The first three are offload training workloads, which can be combined in one job, and every training run used DeepSpeed; the last measures the copies themselves:

- **Model-tensor offload.** Parameters and optimizer states live in host memory and are copied to the accelerator as needed (DeepSpeed ZeRO CPU offload). The multi-device runs offload only the optimizer and keep the parameters on the accelerator.
- **Activation checkpoint offload.** The inputs saved by activation checkpointing go to host memory in the forward pass and come back for recomputation (DeepSpeed CPU checkpointing).
- **Saved-activation offload.** Every tensor saved for the backward pass inside each decoder layer goes to host memory and comes back, with no recomputation. This one is a DeepSpeedExamples recipe built on DeepSpeed's pinned memory, accelerator APIs, and ZeRO engine; the offload logic itself lives in the recipe, not in DeepSpeed.
- **Host-device copies.** A microbenchmark that copies 256 MiB host buffers to the accelerator and back, the transfer every offload workload depends on.

On a single device, the two activation workloads also offload the optimizer, and pinning covers its host buffers too. The multi-device activation runs keep the optimizer on the accelerator.

We ran the training workloads on three accelerator types with two Hugging Face model families:

| Accelerator | Qwen3 (dense) | Qwen3.5 (hybrid linear and full attention) |
|-------------|---------------|--------------------------------------------|
| NVIDIA H200 | 8B; 32B for multi-device optimizer offload | 9B; 27B for multi-device optimizer offload |
| Huawei Ascend 910B4 | 4B | 4B |
| Intel Arc Pro B60 | *pending* | *pending* |

<!-- TODO: B60 model sizes once collected (BLOG_PLAN.md: Qwen3-0.6B or 1.7B, Qwen3.5-0.8B or 2B, by host RAM); Ascend multi-device model sizes. -->

Every run is on a single host (multi-node runs were not tested), with randomly initialized models in bf16 at batch size 1. We picked a model size or sequence length that needs offload for stable training: without it, training either runs out of device memory or runs so close to the limit that a small change can push it over. Model size and sequence length differ by machine, so compare results within a column, not across columns. <!-- Fit evidence (H200, ARXIV_DRAFT.md at 3d2fd3b). Saved-activation lengths are measured against training without checkpointing, its no-recomputation counterpart; plain checkpointing still fits them. single-device checkpoint offload 256K: plain checkpointing runs out of device memory on Qwen3.5-9B and fits Qwen3-8B with 3.7 GB to spare. Single-device saved-activation 64K: offloaded activations are about 7.3 MB per token per rank (about 480 GB at 64K), far beyond 141 GB, so no-offload training cannot fit (arithmetic, not a measured wall). Single-device model-tensor 4K: Qwen3-8B with AdamW on the GPU needs about 128 GB for weights, gradients, and optimizer state before activations (arithmetic, not measured). 8xH200 optimizer offload 4K: GPU AdamW does not fit Qwen3-32B and fits Qwen3.5-27B with about 14.5 GB to spare. 8xH200 checkpoint offload 256K per GPU: past the checkpointing-only walls (233,472 and 208,896 tokens per GPU). 8xH200 saved-activation 16K per GPU: past the no-checkpointing walls (14,336 and 10,240). Ascend and B60 lengths should follow the same rule. --> The benchmark README lists hardware and software versions. Where a speedup comes from a bound run, the binding is a launcher's own, never an external tool; the benchmark README notes the few supporting results that were bound with numactl instead.

### Single device: pinning speedup

<!-- FIGURE: media/single_device.png — pin speedup per workload, one panel per machine, Qwen3 and Qwen3.5 as separate bars. -->

| Workload | Model | H200 (Tokens) | Ascend 910B4 (Tokens) | Arc Pro B60 (Tokens) |
|----------|-------|---------------|-----------------------|----------------------|
| Model-tensor offload | Qwen3 | **2.73×** (4K) | *pending* | *pending* |
| | Qwen3.5 | **2.78×** (4K) | *pending* | *pending* |
| Activation checkpoint offload | Qwen3 | **1.15×** (256K) | *pending* | *pending* |
| | Qwen3.5 | **1.15×** (256K) | *pending* | *pending* |
| Saved-activation offload | Qwen3 | **2.74×** (64K) | *pending* | *pending* |
| | Qwen3.5 | **3.79×** (64K) | *pending* | *pending* |

<!-- TODO: Ascend column from the bound rerun; B60 column (model sizes per BLOG_PLAN.md). Earlier unbound Ascend placeholders, not publishable under the launcher-binding rule: Qwen3-4B model-tensor 1.85x (1K), 1.47x (2K), checkpoint 1.23x (8K), saved-activation 1.14x (2K); Qwen3.5-4B model-tensor 1.32x (512), 1.29x (1K), checkpoint 1.08x (8K). -->

Speedup is the pageable step time divided by the pinned step time, with both runs under DeepSpeed's launcher binding (the process's cores and memory on the accelerator's NUMA node), the fairest pageable baseline we found. On the H200, the 256K checkpoint-offload steps pack 4K-token samples, while saved-activation offload trains one 64K-token sequence. torchrun's binding covers CPU cores only, so it left H200 pageable steps up to 1.6× slower; its pin ratios ran 1.31× to 5.94×, against 1.15× to 3.79× with memory bound as well, overstating what pinning adds to a well-placed job. Pinning helped every workload on every machine we measured. The gain tracks how much of the step is copying: on the H200, model-tensor and saved-activation offload move the most data and gain the most, while checkpoint offload copies one tensor per layer and gains least. Qwen3.5 saved-activation offload gains most of all, plausibly because its linear-attention layers add compute only linearly with sequence length, so copies stay most of the step.

### Multiple devices: pinning, NUMA placement, and CPU threads

With several ranks on one host, the ranks share host memory, sockets, and cores, so placement and threads start to matter alongside pinning. The tables below start from an untuned launch (ranks unbound, memory pageable, one OpenMP thread per rank), on 8×H200. Pinning is off in this launch, although DeepSpeed pins by default. Each number is the untuned step time divided by the step time with the named setting changed; **All** changes every setting that applies to the workload. NUMA binding and **All** use DeepSpeed's launcher binding, except where noted below.

<!-- FIGURE: media/levers.png — speedup from the untuned launch for each setting alone and all together, one group per workload, one panel per machine. -->

**Qwen3, 8×H200**

| Workload | Pinning | NUMA binding | CPU threads | All |
|----------|---------|--------------|-------------|-----|
| Optimizer offload, Qwen3-32B, 4K tokens per GPU | 1.68× | 0.89× (cores only) | 1.51× | **2.97×** |
| Activation checkpoint offload, Qwen3-8B, 256K tokens per GPU | 1.24× | 1.13× | — | **1.24×** |
| Saved-activation offload, Qwen3-8B, 16K tokens per GPU | 5.98× | 2.49× | — | **6.29×** |

**Qwen3.5, 8×H200**

| Workload | Pinning | NUMA binding | CPU threads | All |
|----------|---------|--------------|-------------|-----|
| Optimizer offload, Qwen3.5-27B, 4K tokens per GPU | 1.62× | 1.07× (cores only) | 1.55× | **2.86×** |
| Activation checkpoint offload, Qwen3.5-9B, 256K tokens per GPU | 1.22× | 1.12× | — | **1.22×** |
| Saved-activation offload, Qwen3.5-9B, 16K tokens per GPU | 5.79× | 2.78× | — | **7.14×** |

*Ascend 910B4 and Arc Pro B60 multi-device results: pending.* <!-- TODO: Ascend 910B4 and Arc Pro B60 multi-device tables. -->

<!-- Sources, optimizer offload: Pinning, NUMA binding, and All from job-20261113T151443Z, all against that job's untuned launch (torchrun unbound, one thread). Qwen3-32B untuned 28.83 s; pinned 17.18 s (1.68x); torchrun --numa-binding=node 32.52 s (0.89x); deepspeed --bind_cores_to_rank pageable 15.35 s (1.88x), pinned 9.71 s (2.97x). Qwen3.5-27B untuned 22.82 s; pinned 14.06 s (1.62x); torchrun binding 21.33 s (1.07x); DeepSpeed launcher pageable 12.61 s (1.81x), pinned 7.97 s (2.86x). CPU threads from job-20261111T024327Z (unbound, 12 threads, pageable): 30.16 -> 20.02 s (1.51x), 22.71 -> 14.67 s (1.55x). Earlier drafts divided the Nov 11 untuned step by later launcher times (3.08x, 1.96x, 2.84x, 1.80x); replaced 2026-11-14. Activation checkpoint: job-20261112T192249Z. Saved-activation: job-20261110T223544Z. -->

A dash means the setting does not apply: the activation workloads keep the optimizer on the GPU, so there is no host compute to thread. DeepSpeed's launcher always sets threads along with binding, so for optimizer offload the NUMA binding column uses torchrun's core-only binding with one thread. It did not help (0.89× and 1.07×), because torchrun binds cores but not memory. With pinning off, DeepSpeed's binding and threads together gave 1.88× on Qwen3-32B and 1.81× on Qwen3.5-27B; with pinning on, that launch is the **All** column. The optimizer-offload CPU threads column comes from an earlier job, against the untuned launch in that job. The checkpoint-offload rows pack 4K-token samples into each step; the saved-activation rows train one 16K-token sequence per GPU.

Because the starting point is unbound and pageable, the Pinning column also recovers some of the slowdown that binding fixes. Against a pageable run bound by DeepSpeed's launcher, pinning alone was 1.10× for Qwen3-8B checkpoint offload and 2.53× (Qwen3-8B) and 2.57× (Qwen3.5-9B) for saved-activation offload.

Offload is also what makes these runs fit. With the optimizer on the GPU, checkpoint offload fit 2.25× as many tokens per GPU as plain checkpointing on Qwen3-8B, and at least 1.98× as many on Qwen3.5-9B. Saved-activation offload fit 2.29× (Qwen3-8B) and 2.40× (Qwen3.5-9B) as many tokens as training without checkpointing, but still far fewer than checkpointing, so for long sequences on this machine checkpointing remains the better choice over saved-activation offload.

### Host-device copies

The step-time gains come from faster copies. The table gives the pinned speedup for 256 MiB copies in each direction, per device, bound to the accelerator's NUMA node, with one device copying and with every device in the host copying at once (eight on the H200 server, bound by DeepSpeed's launcher).

<!-- FIGURE: media/h2d_d2h.png — pinned speedup per direction, one panel per machine, one device and all devices copying. -->

| Machine | Direction | One device copying | All devices copying |
|---------|-----------|--------------------|---------------------|
| H200 | Host to device | **5.3×** | **3.3×** |
| H200 | Device to host | **3.1×** | **1.7×** |
| Ascend 910B4 | Host to device | *pending* | *pending* |
| Ascend 910B4 | Device to host | *pending* | *pending* |
| Arc Pro B60 | Host to device | *pending* | *pending* |
| Arc Pro B60 | Device to host | *pending* | *pending* |

<!-- Sources: ARXIV_DRAFT.md at 3d2fd3b single-device and all-GPUs-copying microbench tables. H200 one device under deepspeed --bind_cores_to_rank --bind_core_list=0-47: 10.36 -> 55.40 and 17.59 -> 54.95 GB/s (job-20261113T215205Z, median of 3); all eight under deepspeed --bind_cores_to_rank, per GPU: 8.57 -> 27.91 and 15.00 -> 26.18 GB/s (job-20261113T151443Z; torchrun --numa-binding=node 8.62 -> 28.00 and 15.03 -> 26.30; torchrun unbound pinned D2H 21.22 GB/s, 19% below DeepSpeed's launcher). Ascend one device, hidden as pending: 6.48 -> 25.97 and 6.71 -> 26.49 GB/s (4.0x / 3.9x, unbound placeholder). -->

Only part of a step is copying: on one H200, host-to-device copies were 5.3× faster and device-to-host copies 3.1× faster, while training steps were 1.15× to 3.79× faster. With all eight H200 GPUs copying, pinned bandwidth per GPU halves while pageable bandwidth falls only 15–17%, so the speedup shrinks. The halving is consistent with GPU pairs sharing PCIe switch uplinks on our server, though a cap on total host bandwidth would look the same; pageable copies are limited by CPU staging rather than the link. Pinned copies in eight-GPU training ran at the same ceiling as this benchmark. Placement also starts to reach pinned copies: with all eight GPUs copying, unbound pinned device-to-host copies were 19% slower than under either launcher's binding, while with one GPU they were unchanged.

### What the results show

- **Pinning is the biggest single lever when the step is copy-bound.** From an untuned start, it alone made 8-GPU Qwen3-8B saved-activation offload 5.98× faster, against 2.49× for NUMA binding alone. Against a bound pageable run it was still 2.53×.
- **When the CPU computes, threads matter as much as pinning.** With the optimizer offloaded, CPU Adam is host compute that pinning cannot speed up. On 8-GPU Qwen3-32B, giving it 12 threads per rank was worth 1.51×, close to pinning's 1.68×, and all three settings together gave 2.97×. The settings interact: a pinned step with one thread was still slower than a pageable step with binding and threads.
- **Pinning makes placement matter less, not zero.** On 8×H200, binding moved the pageable step by up to 2.8×, a gain binding cores alone did not capture. The pinned step moved by about 5% or less, except on Qwen3.5: 23% for saved-activation offload and 11% for optimizer offload with 12 threads; binding cores alone closed both. An unbound comparison can overstate the benefit of pinning by up to 2.4×.
- **Contention caps multi-device pinned copies.** On our 8×H200 server, pinned copy bandwidth per GPU halved when all eight GPUs copied, while pageable bandwidth fell only 15–17%, so the per-GPU pin advantage shrank from 5.3× to 3.3× host to device.
- **Registration, not page-locking, makes host memory fast.** On the H200, memory that was locked but not registered with the device driver copied at pageable speed.
- **Some settings did not help.** On the H200, huge pages changed step time by at most 4% once placement was fixed, and the two pinning backends tied once memory was registered.

## Framework support in the PyTorch ecosystem

We compared four PyTorch training frameworks: PyTorch itself (including FSDP), Hugging Face Transformers and Accelerate, Megatron-LM with Transformer Engine, and DeepSpeed. Among them, DeepSpeed is the only one whose launcher binds each rank's memory as well as its cores and sets the rank's threads to match.

| Framework | Pinning | NUMA placement | CPU threads per rank |
|-----------|---------|----------------|----------------------|
| **DeepSpeed** | **On by default, adjustable per offloaded buffer** | **Launcher option binds each rank's cores and memory** | **Same option matches threads to each rank's cores** |
| PyTorch, including FSDP | Through PyTorch's allocator; on by default for FSDP offload | Binds cores only, not memory | One by default; set by the user |
| Hugging Face Transformers and Accelerate | Through PyTorch's allocator; offload is delegated to DeepSpeed or FSDP | Binds cores near each GPU, not memory | One by default on GPUs; set by the user |
| Megatron-LM, with Transformer Engine | Through PyTorch's allocator; on by default for offload | Left to the user | Left to the user |

<!-- Sources and the exact flags per framework: ARXIV_DRAFT.md at 3d2fd3b ("Across the PyTorch ecosystem"). DeepSpeed defaults verified in deepspeed/runtime/zero/offload_config.py, runtime/activation_checkpointing/offload_activations.py, launcher/launch.py, and utils/numa.py. -->

In short, DeepSpeed covers all three settings with the least setup: pinning is on by default, and one launcher option binds each rank's cores and memory and sets its threads to match. Elsewhere, users reach the same settings only by configuring them separately; torchrun, for example, leaves a CPU optimizer on one thread per rank unless the user changes it.

<!-- Evidence (8xH200, ARXIV_DRAFT.md at 3d2fd3b, job-20261113T151443Z): with memory pinned and OMP_NUM_THREADS set by hand, torchrun --numa-binding=node ties deepspeed --bind_cores_to_rank on every workload (within 1%), so the blog claims ease of use, not speed. At torchrun's default one thread, pinned optimizer offload was 17.73 s vs 9.78 s (Qwen3-32B). -->

Can launcher binding replace binding each process by hand with a low-level tool such as numactl? On the H200, DeepSpeed's launcher matched it within 3% of step time on every training workload, single and multi-device. torchrun's binding matched it on pinned steps once the thread count was set to match. Some gaps remain, and we see them as areas for the frameworks to improve:

- **DeepSpeed's launcher** assigns cores by rank order, not by each accelerator's location, so it binds correctly only when that order lines up with the accelerators' NUMA nodes, as on our H200 host. Its memory binding is strict, so a rank cannot spill to the other node when its own fills up. On a multi-socket host, a single process binds memory only when told which cores to use.
- **torchrun** binds cores but not memory, which leaves pageable training steps slow, and it does not set threads to match the binding.

<!-- Evidence: ARXIV_DRAFT.md at 3d2fd3b. Single H200 (job-20261113T215205Z), DeepSpeed launcher vs per-process numactl --cpunodebind --preferred, pageable -> pinned: model-tensor 10.55 -> 3.86 vs 10.36 -> 3.75 s; checkpoint 256K 52.07 -> 45.44 vs 52.04 -> 45.42 s; saved-activation 64K 90.81 -> 33.11 vs 89.91 -> 33.06 s. 8xH200: saved-activation pin 2.53x vs 2.56x (Qwen3-8B), 2.57x vs 2.55x (Qwen3.5-9B); optimizer offload 15.41 -> 9.78 vs 15.53 -> 9.73 s; copies within 0.4 GB/s. torchrun pageable lags (single H200 saved-activation 124.67 s vs 90.81 s), pinned ties. Gaps from deepspeed/utils/numa.py (cores by local rank; -m only when a rank's cores fit one node; no core list = all physical cores, so no -m with one process) and torch/numa/binding.py (numactl --physcpubind only). Because the launcher matches numactl, the numactl-bound H200 results the blog still cites (registration, huge pages, backend tie) carry over; a launcher rerun is not required. -->

The [benchmark README](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/cpu_offload) lists the flags and settings for each framework.

## Best practices

- **Keep pinning on.** It helped every workload we measured; turn it off only when the job cannot lock enough host memory for its offloaded buffers.
- **Bind each rank to its accelerator's NUMA node.** On the H200, with pinned memory and threads set, binding cores was enough. If pinning is not an option, use DeepSpeed's launcher with core binding turned on, which binds each rank's memory as well as its cores; binding cores alone is not enough for pageable memory.
- **Give a CPU optimizer all the cores its rank owns.**
- **Measure on your own host, against a bound baseline.** Results do not transfer between machines.
- **Don't count on huge pages or a different pinning backend for speed.** On the H200, huge pages changed step time by at most 4%, and the two backends tied.

## Conclusion

CPU offload buys capacity with host memory, and three host-side settings decide how much speed that capacity costs. Pinning is the biggest single lever when the step is dominated by copies: against a bound pageable baseline, it made DeepSpeed ZeRO CPU offload 2.73× to 2.78× faster on one H200, and our saved-activation offload recipe up to 3.79× faster on one H200 and 2.57× on eight. When the CPU computes, as with an offloaded optimizer, CPU threads matter as much, and the three settings together made an 8-GPU optimizer-offload step 2.86× to 2.97× faster than an untuned launch. Keep pinning on, bind each rank to its accelerator's NUMA node, and give the CPU optimizer its cores.

As future work, we plan to extend this study to machines where the CPU and accelerator share a coherent chip-to-chip (C2C) link instead of PCIe, such as NVIDIA Grace Hopper and Grace Blackwell. There, the accelerator can access host memory over that link, including pageable memory, and accelerator memory can appear as its own NUMA node. Whether pinning, NUMA placement, and CPU threads play the same roles there is an open question.

The benchmarks, the settings for each framework, and the commands to reproduce every result are in [DeepSpeedExamples `benchmarks/cpu_offload`](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/cpu_offload). <!-- TODO: switch to deepspeedai/DeepSpeedExamples master. -->

## Contributors

<!-- TODO: contributor list (DeepSpeed, Snowflake, Intel, Huawei Ascend). -->
