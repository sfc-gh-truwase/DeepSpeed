# Proposed layout — Pinned Host Memory for CPU Offload

**Article file:** [`README.md`](README.md) (not drafted).  
**Figures:** `media/`, one panel per machine.  
**Source of the argument:** [`BLOG_PLAN.md`](BLOG_PLAN.md). This file is the heading layout only.

**Title:** Pinned Host Memory for CPU Offload: A Practitioner Guide

Headings stay accelerator-neutral. ZeRO and pull-request numbers stay in the body. Each machine’s wall-time figure is its own panel; Ascend 910B4 does not share a chart with H200 or B60. Qwen3.5, if shown, stays off the Qwen3 chart. Every evaluated machine (Arc Pro B60, H200, Ascend 910B4) runs all three workloads and the H2D/D2H microbench. Wall time is measured bound to the accelerator’s NUMA node.

---

## Introduction

### CPU offload for cost-efficiency and capability

Offload as a way to run on smaller or cheaper GPUs and to train workloads that would not fit otherwise.

### Three applications: model tensors, checkpoint activations, and saved activations

Name all three before any DeepSpeed tour.

1. Model-tensor offload — parameters and optimizer states.
2. Activation checkpoint workload — checkpoint hidden states with rematerialization, together with ZeRO-2 CPU AdamW and `--act-mem-opt`.
3. Saved-activation offload — intermediates saved to host and reloaded, without rematerialization. A benchmark, not a shipping config field.

### Pin vs pageable as system efficiency

Device-visible host memory makes copies eligible for DMA and overlap. Wall time is measured afterward.

### What this post covers (evaluations, APIs, frameworks, best practices)

Close the introduction with the decision hint: pin on unless host RAM or memlock is tight; measure on a new host link.

## Background: why pin can change offload efficiency

Page-lock for the OS, register with the driver for the copy engine. `mlock` without register stays pageable-class for GPU copies. Pinning does not by itself guarantee overlap. Why pageable is slower depends on the driver. On CUDA (measured on H200), pageable copies stage through a driver-managed buffer, block the issuing thread, and need CPU threads near the device. On Ascend (reported by the driver team, not measured by us), pageable memory is pinned and page-tabled on every copy. Say which statement is measured.

## Real-world evaluation

Equal section weight for the three workloads. Wall time leads. H2D/D2H follows.

### Setup (Qwen3; host RAM)

Qwen3 is the required family. Hardware table for measured SKUs only: SKU, host CPU, interconnect, Torch/CUDA or PyTorch XPU, DeepSpeed commit, `ulimit -l`, `kernel.numa_balancing`, and the launch policy.

### Method: written buffers, bound placement, step counts

Host buffers are written before timing (a never-written pageable buffer reads the zero page and inflates pageable H2D). Every e2e row is bound to the accelerator's NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`) and runs at least 10 steps. Report pin speedup within one SKU, one page condition, and one placement. Show the unbound-versus-bound spread once, as the reason for the method.

### Host placement: bound versus unbound (the fair pageable baseline)

On the H200 host, automatic NUMA balancing (`kernel.numa_balancing=1`) moves the pageable arm and leaves the pinned arm nearly unchanged. Unbound pageable times are inflated by balancing and by CPU threads landing on the far socket. Bound (`--cpunodebind` and `--preferred` on the GPU's node) is the best pageable condition we measured, so it is the fair baseline for pinned-versus-pageable. Show the unbound numbers as the cost of leaving placement to the scheduler, not as the pinning result. The bound speedups are smaller, and the post should say so: saved-activation at 64K drops from 4.46× unbound to 2.83× bound, and model-tensor from 2.90× to 2.82×. This is one host with balancing on. Hosts with balancing off, or other GPU-to-CPU topologies, may show less of the gap; the post should record the setting for each machine.

### Model-tensor offload (Arc Pro B60, H200, Ascend 910B4)

Pin on/off wall time. Ascend’s published model is Qwen3-4B (batch 1, sequence 1024 and 2048).

### Activation checkpoint workload (same hardware and families)

Same machines. Async on. Pin on/off moves the activation pool and the optimizer host buffers together. Ascend 910B4 is Qwen3-4B, batch 1, sequence 8192. A B60 engine miss opens this section with the commit and the error.

### Saved-activation offload, no rematerialization (Arc Pro B60, H200, Ascend 910B4)

Same three machines, separate panels. H200 is Qwen3-8B, batch 1, sequence 8K / 16K / 32K / 64K, `N=1`, `M=2`, `--offload-tensors -1`. Ascend 910B4 is Qwen3-4B, batch 1, sequence 2048. B60 is a Qwen3 0.6B or 1.7B, batch 1, sequence 2048, not yet collected. Do not copy the H200 ratio onto another machine.

### Qwen3.5 hybrid architecture (own chart)

Qwen3.5 rows for the same workloads, on their own chart, never beside Qwen3.

### Supporting H2D/D2H microbench

After the wall-time sections. CUDA: pageable versus pinned, with Torch versus native-registered as a parity inset. B60: pageable versus Torch/XPU pin. Ascend 910B4: native-registered versus pageable, own panel. Include the 4K-versus-2M result for H200 as one line: page size does not move the pin speedup when placement is fixed.

## APIs and popular frameworks

### PyTorch

`Tensor.pin_memory()`. DataLoader `pin_memory` once, as the familiar hook, not as a benchmark.

### Hugging Face checkpoint hooks

Non-reentrant checkpoint hooks used by the activation checkpoint workload.

### DeepSpeed (model-tensor CPU offload and cpu_checkpointing)

`offload_*.pin_memory`, activation `use_pin_memory`, and `cpu_checkpointing`. Saved-activation has no config field.

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

Launch CPU-offload training bound to the accelerator’s NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`). Automatic NUMA balancing taxes every large host buffer each step, pinned or not, because CPU Adam state is pageable. Pageable arms are placement-sensitive; pinned arms much less so. Huge pages (2M) are not a lever on H200 once placement is fixed: pin speedup moves ≤0.08×. Ascend’s page-size result is separate and unverified.

## Conclusion

## Contributors

---

## Figure files

Under `media/`, equal weight for the three wall-time figures, microbench supporting:

- Model-tensor pin versus pageable, one panel per machine (B60, H200, Ascend 910B4). H200 bound: 2.82×.
- Activation checkpoint pin versus pageable, one panel per machine. H200 bound at 64K: 1.15×.
- Saved-activation pin versus pageable, one panel per machine. H200 bound at 8K / 16K / 32K / 64K: 3.03× / 3.32× / 3.25× / 2.83×.
- H2D/D2H supporting panels: CUDA, B60, Ascend 910B4.
- Torch versus native host footprint, beside How to pin: 5 GiB requested, 8 GiB Torch, 5 GiB native; after free, Torch still 8 GiB until the host cache is emptied, native at 0.
- Placement, beside the placement section: faults per step and step time, unbound versus bound, pinned and pageable arms (H200).
- Page size, beside the same section: pin speedup at 4K versus 2M, bound, all H200 workloads.
