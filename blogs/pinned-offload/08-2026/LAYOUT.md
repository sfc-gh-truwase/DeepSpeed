# Proposed layout — Efficient CPU Offload

**Article files:** [`README.md`](README.md) is the short blog (2026-11-13). There is no long report (user, 2026-11-14): `ARXIV_DRAFT.md` was removed, and the how (flags and settings per framework, setup, method, reproduction commands) moved to the [DeepSpeedExamples `benchmarks/cpu_offload` README](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/cpu_offload), which the blog links to. Per-row step times and job IDs stay in the blog's hidden source comments; the removed draft is in git history at commit 3d2fd3b. B60 rows, bound Ascend reruns, and Ascend/B60 multi-device results are pending.\
**Figures:** `media/`, one panel per machine, one chart per model family.\
**Source of the argument:** [`BLOG_PLAN.md`](BLOG_PLAN.md). This file is the heading layout only.

**Title:** Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads

## Blog layout (README.md)

Short and high level (user, 2026-11-13): results as speedups only, no step-time tables, no method section, no commands, no job IDs in the text. Every number comes from a measured run cited in a hidden source comment.

```
# Efficient CPU Offload: Pinned Memory, NUMA Placement, and CPU Threads
## Introduction
## Performance impact of tuning
### Workloads and hardware
### Single device: pinning
### Multiple devices: pinning, NUMA placement, and CPU threads
### Host-device copies
### What the results show
## Framework support in the PyTorch ecosystem
## Best practices
## Conclusion
## Contributors
```

- **Introduction:** CPU offload as a training scalability technique that reduces the accelerator memory a job needs (a cheaper accelerator, or longer sequences); performance depends on accelerator–host transfers and the host work around them, controlled by pinning, NUMA placement, and CPU threads. Then the three things the blog shows: (1) tuning impact on four workloads (host-device copies plus three offload training workloads) across three accelerator types, single and multi-device; (2) how PyTorch-ecosystem frameworks support the three settings; (3) best practices. Link the DeepSpeedExamples benchmark README for setup, method, and reproduction.
- **Workloads and hardware:** one bullet per workload, host-device copies first, then the three training workloads, each naming the DeepSpeed feature it ran on; one sentence that single-device activation runs also offload (and pin) the optimizer while multi-device activation runs keep it on the accelerator; the three machines, two model families, single-host scope, and model setup in one paragraph, with the rule to compare within a column, and one sentence that every run, single and multi-device, is bound using only a launcher's own binding (no external `numactl`).
- **Single device:** one table, workload × machine, each cell "Qwen3 / Qwen3.5" with the model sizes in the machine's column header (user, 2026-11-14), pin speedup only (sequence length in parentheses for every cell; when the two families ran different lengths, give each its own, e.g. "1.85× (1K) / 1.32× (512)"; list multiple lengths separately, never as a range). Unfilled cells say *pending*. One short paragraph of interpretation that defines the bound pageable baseline and quotes the full single-device range across both families.
- **Multiple devices:** a Qwen3 table and a Qwen3.5 table per machine, workload × setting (pinning, NUMA binding, CPU threads, all), from an untuned launch (unbound, pageable, one thread; not "torchrun's defaults", since pageable is not a torchrun setting), with the ratio and **All** defined before the tables. Multi-device numbers use only each launcher's own binding, never external `numactl` (user, 2026-11-13): NUMA binding and **All** come from DeepSpeed's launcher; optimizer-offload binding alone uses torchrun's own binding with one thread, because DeepSpeed's launcher always sets threads with it. Then: the dashes; the bound-baseline pin speedups, since the Pinning column also recovers what binding fixes; and the capacity gains, each against its own baseline (saved-activation offload still fits far fewer tokens than plain checkpointing).
- **Host-device copies:** one table, machine × direction (host to device, device to host), pinned speedup only (no GB/s), for one device copying and all devices copying. The all-devices cells use each launcher's own binding (DeepSpeed's launcher in the table, torchrun's in the report), never per-rank `numactl`. One paragraph: copy speedups exceed step speedups because only part of the step copies; with all devices copying, pinned bandwidth per device drops (consistent with shared PCIe uplinks on the H200 server; a host-wide bandwidth cap would look the same) while pageable barely moves; training copies hit the same ceiling; placement starts to reach pinned copies.
- **What the results show:** six short findings, one per setting or effect (pinning, CPU threads, NUMA placement, multi-device copies, registration, huge pages). Once Ascend 910B4 and Arc Pro B60 single-device and multi-device results are in, write each finding across machines: state where it holds on every machine and name the machine where it differs, in the same sentence, rather than adding per-machine findings (user, 2026-11-14). Until a machine confirms a finding, scope it to the machines measured ("On the H200"). Huge pages are machine-dependent: if the Ascend or B60 end-to-end runs show a pageable gain at 2 MiB pages (EXPERIMENTS_HANDOFF.md section 2, item 12), huge pages leave "Some settings did not help" and get their own finding, with the H200 non-result as the contrast. Likely cross-machine contrast to check: the multi-device copy ceiling on each host's PCIe topology. Leave out the Ascend remote-node slowdown on pinned copies (about 26%): it is a technical detail for a long report, not the blog (user, 2026-11-14), and there is no long report now. Keep the blog short.
- **Framework support in the PyTorch ecosystem:** scope to the four PyTorch training frameworks compared, named in the first sentence; then the claim that among them DeepSpeed gives the most control over the three settings (the only one that handles all three itself). One table, framework (DeepSpeed first; PyTorch including FSDP; Hugging Face Transformers and Accelerate; Megatron-LM with Transformer Engine) × setting (pinning, NUMA placement, CPU threads per rank). Report what each framework supports, not how: no flags, environment variables, or allocator internals (those go in the DeepSpeedExamples benchmark README's "Across the PyTorch ecosystem"). Then a short summary that DeepSpeed provides everything practitioners need to maximize offload performance: pinning on by default, binding and threads from one launcher option (no per-setting bullets; the table already covers them). Keep the DeepSpeed claim to ease of use: it reaches the best settings from defaults and one launcher option, while other frameworks need each setting configured separately (torchrun leaves a CPU optimizer on one thread). No launcher speed comparison (user, 2026-11-13): with memory pinned and threads set, torchrun's own binding ties DeepSpeed's launcher. Then answer whether launcher binding can replace low-level per-process binding such as numactl (user, 2026-11-14): DeepSpeed's launcher matched it within 3% of step time on every H200 training workload, torchrun's only on pinned steps with threads set to match; list the remaining gaps as areas for the frameworks to improve (DeepSpeed: rank-order core assignment, so it needs NUMA-ordered accelerators; strict memory binding with no spill; single process needs a core list; torchrun: no memory binding, threads not set from the binding). Close with a pointer to the report for flags.
- **Best practices:** five short bullets saying what to do, not how: no flags, commands, or environment variables (those go in the DeepSpeedExamples benchmark README). The binding bullet says binding cores was enough on the H200 with pinned memory and threads set, and names DeepSpeed's launcher, which also binds memory, for when pinning is not an option (user, 2026-11-13).
- **Conclusion:** one paragraph, then the DeepSpeedExamples benchmark README link.

## DeepSpeedExamples benchmark README

The [benchmark README](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/cpu_offload) holds what the blog leaves out: how to apply each setting (pinning in ZeRO offload, activation checkpoint offload, and the pin APIs; launcher binding, numactl, and threads), the flags per framework, what did not help, the setup, the method rules (including the three numactl-bound supporting results), and the commands to reproduce every published row. Keep it in step with the blog when a published number or setting changes.

Technical details that the removed draft carried and the blog leaves out (for example, the Ascend remote-node cost on pinned device-to-host copies, ~26%) are not published anywhere now; recover them from commit 3d2fd3b if a later write-up needs them.

---

## Figure files

Under `media/`. Figures that only the removed draft used (`levers.png`, `host_footprint.png`, `placement.png`, `page_size.png`, the async-versus-blocking chart) are optional. Each workload has a Qwen3 figure and a separate Qwen3.5 figure, one panel per machine:

- `levers.png`: multi-device, Qwen3, speedup from the worst launch for each lever alone and all together, one group per workload, one panel per machine.
- `summary.png`: single-accelerator pin speedups, Qwen3, one panel per machine.
- `model_tensor_qwen3.png` / `model_tensor_qwen3_5.png`. H200, DeepSpeed launcher: Qwen3-8B 2.73×, Qwen3.5-9B 2.78×.
- `activation_checkpoint_qwen3.png` / `activation_checkpoint_qwen3_5.png`. H200 at 256K, DeepSpeed launcher: Qwen3-8B 1.15×, Qwen3.5-9B 1.15×.
- `saved_activation_qwen3.png` / `saved_activation_qwen3_5.png`. H200 at 64K, DeepSpeed launcher: Qwen3-8B 2.74×, Qwen3.5-9B 3.79×.
- Optional `activation_async_vs_blocking.png`: #8282 async versus blocking (both pinned), labeled as a different axis.
- `h2d_d2h.png`: supporting panels for CUDA (with the parity inset), B60, and Ascend 910B4.
- `host_footprint.png`, beside How to pin: 5 GiB requested, 8 GiB Torch, 5 GiB native; after free, Torch still 8 GiB, native 0.
- `placement.png`, beside the placement best practice: faults per step and step time, unbound versus bound, pinned and pageable arms (H200).
- `page_size.png`, beside What did not help: pin speedup at 4K versus 2M, bound, all H200 workloads.
