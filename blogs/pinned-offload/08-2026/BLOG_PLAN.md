# Blog Plan — Pinned Host Memory for CPU Offload

**Audience:** agent or writer drafting the blog and collecting benchmarks.\
**Venue:** [`blogs/pinned-offload/08-2026/README.md`](README.md) (drafted 2026-11-05; B60 rows and bound Ascend reruns pending).\
**Layout peer only:** [`blogs/deepspeed-offloadpp/README.md`](../../deepspeed-offloadpp/README.md) (README/media shape). **Do not** copy its ZeRO-product lede, hero speedup, or “we release feature X” voice.\
**Related:** experiment/DSE work is [`EXPERIMENTS_HANDOFF.md`](EXPERIMENTS_HANDOFF.md) (out of scope for the blog writer). Phase 2 routing is already on DeepSpeed `master` ([`PHASE2_PR_HANDOFF.md`](PHASE2_PR_HANDOFF.md) is historical).

**Out of scope:** NVMe, DeepNVMe, AIO, GDS, ZeRO-Inference disk offload, DataLoader pin as an experiment. Do not mention NVMe as a reason to pick native pin. DataLoader `pin_memory` may appear once as the PyTorch API practitioners already know — not as a benchmark.

---

## Working title

**Pinned Host Memory for CPU Offload: A Practitioner Guide**

Do **not** use “DeepSpeed Offload,” “Faster Transfers,” or “Explicit Lifetime Control” in the title. Lifetime/unpin belongs under APIs / best practices. Speedups belong in Evaluation after they are measured.

**Pitch:** CPU offload is a **cost-efficiency** tool for deep learning: run on smaller or cheaper GPUs. A capacity claim needs a no-offload baseline that does not fit; none of the activation rows below meets that bar. It has **more than one application**, presented as peers:

1. **Model-tensor offload** — parameters and optimizer states (DeepSpeed ZeRO CPU offload is the mechanism, not the headline). Gradients are the same idea; this post does not measure them.
2. **Activation checkpoint workload** — checkpoint inputs moved to host during forward, then rematerialized, so long sequences fit. The published workload is that offload together with ZeRO-2 CPU AdamW (`--offload-optimizer`; parameters stay on GPU) and `--act-mem-opt` (Qwen3: Liger fused linear cross-entropy, TiledMLP, and fused cross-entropy). Async copies stay on. Pin on/off moves the activation pool and the optimizer host buffers together. Do **not** call this generic “input offload.”
3. **Saved-activation offload** — every saved-for-backward tensor in each decoder layer saved to host and reloaded, **without** rematerialization. Credit-bounded: published H200 rows use `N=1` (`--credits 1`), `M=2` (`--prefetch 2`), no per-layer tensor cap (`--offload-tensors -1`). See “Saved-activation settings, as measured” under Benchmark matrix. Published rows use ZeRO-2 CPU AdamW (`--offload-optimizer`); pin toggles activation buffers and optimizer host buffers. A different algorithm from (2), not a config of `CheckpointHiddenStatesOffload`. No `ds_config` field; the blog reports the benchmark, not a shipping API.

Do **not** tell this as a ZeRO story with activations as a sequel. Name all three applications before any DeepSpeed tour.

**Pin vs non-pin** is the **system-efficiency** question on top of that: whether host buffers are device-visible so copies are **eligible** for DMA and overlap. Pinning does **not** guarantee end-to-end overlap or wall-time wins; that depends on workload and interconnect. Pageable copies often bounce; do not write that they always “stall the bus.”

This blog is a **practitioner guide**. It should deliver:

- **Real-world evaluations** — pin on/off wall time for **all three** applications on **Qwen3** (dense). Qwen3.5 is required alongside Qwen3: each workload section carries a Qwen3 table and a separate Qwen3.5 table, each with its own chart. The activation checkpoint and saved-activation sections print only the longest sequence per machine and family. All three applications, (1) model-tensor, (2) activation checkpoint, and (3) saved-activation, run on every evaluated accelerator: Intel Arc Pro B60, high-end PCIe (H200), and **Ascend 910B4** (Qwen3-4B), through DeepSpeed’s accelerator abstraction. Synthetic MLP is smoke, not a published substitute. H2D/D2H is supporting DMA evidence, not the lede figure. Keep each machine on its own panel. Wall time is measured bound to the accelerator’s NUMA node (see Best practices).
- **Available APIs** — PyTorch `Tensor.pin_memory()` (and, in passing, `DataLoader(pin_memory=True)` as the familiar hook); DeepSpeed `get_accelerator().pin_memory()` / unpin / `is_pinned`; ZeRO `offload_*.pin_memory`; activation `use_pin_memory`; `DS_PIN_MEMORY_BACKEND` and `DS_PIN_MEMORY_REGISTER_DEVICE`.
- **Popular frameworks** — three sibling sections: PyTorch, Hugging Face non-reentrant checkpoint hooks, DeepSpeed (model-tensor CPU offload and `cpu_checkpointing`). Do not invent APIs other stacks do not have.
- **Best practices** — shipping defaults are pin-on; confirm with the e2e numbers. When to turn pin off (memlock / host RAM). Torch vs native (how to pin). Host memory placement (bind to the accelerator’s NUMA node; page size is not a lever on H200). Do not copy one machine’s numbers onto another. A **flat** pin-on/off result on PCIe is publishable.

Cost is **framing**. Pin on/off numbers do not prove “ran on a cheaper GPU.” Activation capacity is not claimed: on H200, plain checkpointing without offload runs Qwen3-8B at 64K (44.8 GB peak), 96K (57.6 GB) and 128K (70.6 GB) with no OOM (job `20260929T152934Z`), so offload does not extend the sequence length. You may cite that no-offload arm and #8282’s peak/seq arms as the baseline; do not relabel them as pin vs pageable.

Native vs Torch is a how-to-pin detail, not the lead. Do not imply native is faster than Torch on CUDA; #8283 is **parity** once registered.

---

## Readers (evaluated classes)

Write for **practitioners** who already offload (or are about to) and need to know whether **pin vs pageable** is worth it as a **system-efficiency** choice. They move **model tensors**, **checkpoint activations**, **saved activations**, or a combination. ZeRO and `use_pin_memory` are knobs, not the lead. Saved-activation pin is a benchmark arm, not a config field.

| Class | Evaluated hardware (this blog) | CPU–GPU path | Why they care |
|-------|------------------------------|--------------|---------------|
| **Workstation PCIe** | **Intel Arc Pro B60** (workstation/prosumer XPU, 24 GB; not consumer-class Arc B580). Required. | PCIe | Offload is how a single card is cost-efficient and capable. Measure pin vs non-pin for model-tensor and activation checkpoint. B60 saved-activation is required (runs via the accelerator abstraction). |
| **High-end discrete** | One NVIDIA PCIe datacenter GPU (H200 already measured in #8283 / #8282). Other PCIe vendors only if actually run | PCIe | All three applications at datacenter PCIe rates. H200 is a required SKU for saved-activation. Pin makes copies *eligible* for overlap. Measure wall time. |
| **Ascend NPU** | **Ascend 910B4** (single device). 29.5 GiB HBM usable (32 GiB physical, about 2.5 GiB firmware-reserved). Required. | Host Link (Ascend-specific). Do not describe it as PCIe NVIDIA or as Intel XPU | Model-tensor and the activation checkpoint workload on Qwen3-4B, plus H2D/D2H. Saved-activation: Qwen3-4B, batch 1, seq 2048, 1.14× (9.48→8.34 s). |

**Coverage:** every class above runs all three applications and the H2D/D2H microbench. The only permitted gap is a named engine failure on B60 activation (see below).

**Voice:** (1) offload as cost, with capability only where a no-offload run does not fit, three peer applications; (2) pin vs non-pin as system efficiency (eligibility, then measured wall time); (3) evaluations, then APIs / frameworks, then best practices. **Article order is not collection order.** Do not open as “ZeRO offload, plus a note on activations.”

**Register:** a technical blog about results, insights, and conclusions, not a tutorial. Each evaluation section leads with its tables and interpretation. Setup and method get their own sections; benchmark commands and the saved-activation settings rationale go only in the closing appendix. API sections explain, with at most one short snippet each.

**XPU:** not a separate class. **Intel Arc Pro B60** is the required non-NVIDIA PCIe SKU. DeepSpeed XPU is **stock PyTorch** (`xpu`, PyTorch 2.8+), not IPEX ([#7877](https://github.com/deepspeedai/DeepSpeed/pull/7877)). `cudaHostRegister` in #8283 is CUDA-only; B60 pin vs non-pin is Torch/XPU pin vs pageable. Do not advertise a native-registered XPU arm until `register_host_memory` exists.

**Ascend:** required class, not an XPU and not a PCIe NVIDIA SKU. Publish **Qwen3-4B** (usable HBM is 29.5 GiB, so this is not the B60 model-id anchor and not the H200 8B row). Model-tensor: batch 1, seq 1024 and seq 2048. Activation checkpoint workload: batch 1, seq 8192. H2D/D2H: native-registered vs pageable at 256 MiB. Do not relabel that DMA arm as Torch pin, and do not call it `cudaHostRegister`. State the conditions on the panel: default 4K host pages, process bound to the card's NUMA node. Pageable H2D there depends on page size (~6.4 GB/s on 4K pages vs ~23 GB/s on 2M THP, microbench only; e2e effect unmeasured; both numbers are unverified, because the microbench may have read an unwritten buffer, and the per-copy-pinning explanation comes from the driver team, not from our measurement), and a remote NUMA node costs pinned D2H ~26%. Saved-activation: runs on the NPU via the accelerator abstraction in DeepSpeedExamples; Qwen3-4B, batch 1, seq 2048, 1.14× (9.48→8.34 s, widened scope, credits 1, ZeRO-2 CPU AdamW). Qwen3.5-4B model-tensor rows are collected and required, off the Qwen3 chart.

**Dropped from this blog, even if a run lands:** NVIDIA consumer GPUs (GeForce, 2026-10-20) and GH200 / coherent CPU-GPU fabrics (2026-11-04). **Not in this blog’s evaluated list** unless a run lands: Radeon, MI300X, Gaudi, MI300A, datacenter Intel GPU, GB200.

---

## Prerequisites (merged)

| PR | Relevance to blog |
|----|-------------------|
| [#8207](https://github.com/deepspeedai/DeepSpeed/pull/8207) | Default `pin_memory: true`; async grad offload (not evaluated in this post) |
| [#8212](https://github.com/deepspeedai/DeepSpeed/pull/8212) / [#8236](https://github.com/deepspeedai/DeepSpeed/pull/8236) / [#8211](https://github.com/deepspeedai/DeepSpeed/pull/8211) | Shared manager + native backend (`DS_PIN_MEMORY_BACKEND`) |
| [#8283](https://github.com/deepspeedai/DeepSpeed/pull/8283) | Native `mlock` then **`cudaHostRegister`** on CUDA (default on); H200: registered native matches Torch (~53–55 GB/s), unregistered ~10–17 GB/s |
| [#8282](https://github.com/deepspeedai/DeepSpeed/pull/8282) | Activation checkpoint CPU offload (HF hooks + native `cpu_checkpointing`); `use_pin_memory` (default true). Replaced closed [#8181](https://github.com/deepspeedai/DeepSpeed/pull/8181). Published **async vs blocking on a pinned pool** — not pin vs pageable |

RTD: Host Memory Pinning in `docs/code-docs/source/memory.rst`. The stale “native is mlock-only / keep Torch because there is no `cudaHostRegister`” text was fixed on 2026-11-05 (uncommitted): native registers where a hook exists (CUDA, Ascend NPU, SUPA; not XPU). The same stale advice remains in `docs/_tutorials/ulysses-alst-sequence-parallelism.md` (activation offload paragraph); fix it before publishing.

Config orthogonality (document under Best practices, not as the intro):

- `zero_optimization.offload_*.pin_memory` → **whether** model-tensor CPU offload pins (default true).
- Activation `use_pin_memory` → **whether** the host pool pins (default true; HF `CheckpointHiddenStatesOffload` and native `cpu_checkpointing` share `_ActivationOffloadEngine`). Not the ZeRO config field.
- `DS_PIN_MEMORY_BACKEND=torch|native` → **how** to pin when code calls `get_accelerator().pin_memory()` (env only; default remains **torch**).
- `DS_PIN_MEMORY_REGISTER_DEVICE` → native only: on by default after `mlock` on CUDA; `0` is mlock-only ablation.

---

## Narrative arc

1. **Offload as strategy** — Cost-efficiency; capability only where a no-offload run does not fit. Not a ZeRO-only problem.
2. **Three peer applications** — Model tensors (params + optimizer). Activation checkpoint workload (checkpoint hidden states with remat, ZeRO-2 CPU AdamW, `--act-mem-opt`). Saved activations (whole-decoder-layer saved tensors, no remat). Name all three before DeepSpeed. Independent knobs.
3. **Pin vs non-pin as system efficiency** — Copies cross the host bus. Device-registered host memory makes DMA/overlap **eligible**; wall time is measured, not assumed. Applies to all three applications.
4. **What this post gives practitioners** — One-paragraph answer-first (when to pin, when to disable) plus the four deliverables.
5. **Evaluation: model-tensor offload** — Pin vs non-pin wall time on Arc Pro B60, H200, and **Ascend 910B4**, Qwen3 table then Qwen3.5 table, every collected sequence length. Same heading weight as (6) and (7). Ascend’s published models are Qwen3-4B and Qwen3.5-4B. Wall time is measured bound to the accelerator’s NUMA node, with written host buffers.
6. **Evaluation: activation checkpoint workload** — Same hardware and families, Qwen3 and Qwen3.5 tables, longest sequence only (H200 64K, Ascend 8192). Command is `activation_offload/bench.py --offload-optimizer --act-mem-opt` with async on. Pin on/off moves the activation pool and the optimizer host buffers together. Lead with capability (64K on one GPU), then why the pin gain is modest at that length. Cite #8282 async vs blocking (pinned) as a **separate overlap** inset. Do not reuse async-vs-blocking as pin vs pageable. GPU-AdamW-only and `--no-act-mem-opt` runs are not this workload.
7. **Evaluation: saved-activation offload** — No rematerialization. All three SKUs are required: H200 (Qwen3-8B), Ascend 910B4 (Qwen3-4B, measured), and B60 (Qwen3 0.6B/1.7B, batch 1, seq 2048). Qwen3 and Qwen3.5 tables, longest sequence only (H200 64K). Do not plot it as a variant of (6). Close with the same-job 64K comparison against (6): pinning makes skipping recomputation competitive, but on H200 recomputation is still faster and uses far less memory. Do not copy the H200 ratio onto other SKUs.
8. **Supporting DMA microbench** — After wall time, not before. CUDA: pageable vs pinned is the primary axis; Torch vs native-registered is a **parity inset**. B60: pageable vs Torch/XPU pin. Ascend 910B4: native-registered vs pageable, own panel.
9. **APIs and frameworks** — PyTorch, then HF hooks, then DeepSpeed configs. Saved-activation has no config surface. Then how-to-pin (Torch vs native), including explicit unpin as reclaiming locked host pages under memlock pressure — not a title claim.
10. **Best practices** — Independent on/off switches; default torch backend; disable pin under tight memlock/RAM; bind CPUs and memory to the accelerator’s NUMA node; do not port one machine’s numbers to another. “Default on” is the shipping default to **check**, not a result.
11. **What’s next** — Practitioner leftovers: XPU `register_host_memory` if claimed; remaining pin call sites only if they change what users can do. Not a Phase 2 routing tour. Saved-activation stays a benchmark unless it is later promoted. In the article this is the closing paragraph of the Conclusion, not its own heading.
12. **Appendix: reproducing the results** — Benchmark commands, one per workload plus H2D/D2H, under the NUMA binding; `--prefetch 2` passed explicitly; the saved-activation settings rationale.

All pin experiments live under DeepSpeedExamples `benchmarks/pin_memory` (`model_tensor_offload/`, `activation_offload/`, `saved_activation_offload/`, `h2d_d2h/`). Until they merge upstream, the dev fork is [`sfc-gh-truwase/DeepSpeedExamples` `tjruwase/pin-memory-experiments`](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory), and the README links there; switch the links to `deepspeedai/DeepSpeedExamples` master at merge. The fork's `PUBLISH_TRACKER.md` tracks the remaining rows.

**Dev fork synced with the blog (2026-11-05, `c535c6f`):** the `cuda/run.sh` launchers run the published H200 rows (activation checkpoint and saved-activation at seq 65536; saved-activation with `--offload-tensors -1 --credits 1 --prefetch 2 --act-mem-opt`, 10 steps / 1 warmup; `MODEL=` and `SEQ=` override). The saved-activation `npu/run.sh` uses the H200 flags for the bound Ascend rerun (the earlier unbound 1.14× row used K=3). The fork READMEs list the blog-row commands, and `PUBLISH_TRACKER.md` carries the remaining rows. The fork now holds only the three workloads plus `h2d_d2h/` (`7e76ba7` removed `grad_offload/`, `cpu_pin/`, `deepcompile_activation/`, and `dynamic_tensor_offload/`). Published step counts: 10 steps for every e2e row; warmup 1 for the activation workloads.

**Pin backend per row:** model-tensor and saved-activation pinned arms force `DS_PIN_MEMORY_BACKEND=native` with registration. Activation checkpoint defaults to native, but the Ascend `npu/run.sh` selects `torch` (faster on that host), so the Ascend activation rows are Torch-pinned. Do not claim native/Torch parity on Ascend; it is measured only on CUDA.

---

## Benchmark matrix

Three workloads, all required (peer applications):

- **Model-tensor offload (ZeRO-3 CPU param+optim):** use the Examples harness below. Scale `--batch` / `--seq` / model size to VRAM/host RAM. Document the exact command per SKU. Do not require identical parameter counts. In the **article**, call this model-tensor offload; ZeRO-3 is a method footnote.
- **Activation checkpoint workload:** `activation_offload/bench.py --model … --offload-optimizer --act-mem-opt` on a **real HF model**. Document `use_reentrant=False`, async on, and pin on/off. Pin on/off covers the activation pool and the ZeRO-2 optimizer host buffers. Qwen3 `--act-mem-opt` is Liger fused linear cross-entropy, TiledMLP, and fused cross-entropy. Qwen3.5 gets the same Liger fused linear cross-entropy plus TiledMLP recipe with Liger kernel 0.8.4 or later (`job-20261016T162911Z`); older Liger fell back to TiledMLP plus tiled fused logits loss. Shrink seq/model on smaller-VRAM SKUs. Publish up to seq 64K on H200, matching saved-activation's capacity ceiling so the two remat workloads share one sequence range; activation checkpoint itself fits to at least 128K (70.6 GB peak, 1.09× pin speedup, `job-20260928T140704Z`). Synthetic MLP is smoke and a method footnote, not a substitute for Qwen rows.
- **Saved-activation offload (no remat):** `saved_activation_offload/bench.py --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2` (accelerator-agnostic via DeepSpeed’s accelerator abstraction), whole-decoder-layer scope (the bench's default `--offload-modules`), ZeRO-2 CPU AdamW. H200 row: Qwen3-8B, batch 1, pin on/off, seq 8K/16K/32K/64K; seq ≥32K adds `--discover-calib-seqs 4096 8192`. Report step time, offloaded bytes, and peak HBM. Label `N` on each panel and do not copy the H200 ratio onto other SKUs. The script is accelerator-agnostic (DeepSpeed accelerator abstraction), so it runs on Ascend 910B4 (required: Qwen3-4B, batch 1, seq 2048, same flags) and on B60 (required: Qwen3 (same ID as the B60 model-tensor row: 0.6B or 1.7B), batch 1, seq 2048 (shrink seq if 24 GB VRAM requires it, and log it), same flags as H200, pin T/F; Qwen3.5-0.8B or 2B on its own chart). Qwen3.5-9B is required, own chart. Full spec: [`SAVED_ACTIVATION_OFFLOAD_PLAN.md`](SAVED_ACTIVATION_OFFLOAD_PLAN.md).

**Saved-activation settings, as measured (H200, Qwen3-8B, batch 1, ZeRO-2 CPU AdamW).** The published H200 settings differ from the original `N=2`, `M=2`, `--offload-tensors` 1/2/3 spec. Each change is backed by a measurement:

- **Scope: whole decoder layer, not MLP-only.** The original `\.mlp$` scope offloaded only 3–4 tensors per layer. It left attention activations resident and OOMed at seq 16K while activation checkpointing still ran. The scope is now `layers.N`, the same unit that gradient checkpointing wraps.
- **`--offload-tensors -1`, not a 1/2/3 sweep.** Under the whole-layer scope, a K cap *raises* peak HBM: unselected eligible tensors stay resident instead of passing through the bounded credit pool. At K=1, peak HBM went from 54.30 to 83.01 GB at seq 8K, and the run OOMed at 32K (job `20261011T105335Z`). The old MLP-only K=1/2/3 table (job `20261007T215537Z`, seq 4096, 2.00× / 2.09× / 2.20×) is superseded history, not a published row. `r` was a K-sweep metric and is not reported for the published rows.
- **`--credits 1` (`N=1`), not 2.** `N` bounds how many forward layer-groups are held resident. Offloaded bytes are identical at either setting. `N=1` cuts peak HBM at long sequence and costs no measurable step time:

  | seq | pin speedup (N=1), unbound | **pin speedup, bound** | pinned step, bound | peak HBM N=2 → N=1 |
  |---|---|---|---|---|
  | 8K | 3.12× (3.19× repeat) | **2.98×** (16.10 → 5.40 s) | 5.40 s | 54.30 → 54.30 GB (no change) |
  | 16K | 3.64× | **3.26×** (26.59 → 8.15 s) | 8.15 s | 59.44 → 59.44 GB (no change) |
  | 32K | 3.92× | **3.17×** (47.46 → 14.99 s) | 14.99 s | 82.04 → 75.54 GB (−8%) |
  | 64K | 3.85× | **2.74×** (91.22 → 33.25 s) | 33.25 s | 138.48 → 125.10 GB (−10%) |

  These speedups are from the 2026-11-07 rerun (job `20261107T043012Z`; the saved-activation Qwen3-8B 8K bound row and all Qwen3.5 rows are from `20261107T202125Z`). Bound = `numactl --cpunodebind=0 --preferred=0`; unbound has no NUMA policy and balancing on. Pinned step time at bound is within 3.5% of the earlier published bound times at every seq. Pin speedup is about 3× at 8K–32K and 2.7× at 64K, bound. **Print the bound column.** The peak HBM column is from the original runs; it was not re-extracted from the rerun. The pinned step time at `N=1` is within ±2.3% of `N=2` at every seq. Sources: `N=1` is job `20261011T171729Z` (dataset of record); `N=2` is job `20261011T005859Z`. At seq 8K/16K a different allocation sets the peak, so `N` does not move it.
- **`--prefetch 2` (`M=2`) kept for the published rows.** `M=1` cuts peak HBM further at 64K (125.10 → 97.88 GB, −22%), but pinned step time rises 12.3% (33.00 → 37.06 s, job `20261018T110757Z`). The pin speedup rises to 4.28× only because the pageable arm slows more. That makes `M=1` a memory-headroom ablation, not the published row. Qwen3.5-9B shows the same pattern: −29% HBM for +4–10% step time.
- **Ceiling: seq 64K.** At seq 98K the engine runs out of HBM: parameters, optimizer state, and held credit groups together exceed 139.8 GB. This is a real capacity limit, not a discovery artifact. Seq ≥32K uses linearly extrapolated discovery (`--discover-calib-seqs 4096 8192`), validated as an exact match against real discovery at 8K. Calibrated discovery threshold depends on the model: Qwen3-8B needs it at seq ≥32K, Qwen3.5-9B already at seq ≥16K (uncalibrated discovery at 16K exceeds 139.8 GiB before the measured arms run; `job-20261017T153604Z` OOMed there with no calibration flag, and the calibrated rerun `20261107T202125Z` passes at 16K–64K with finite losses).
- **Qwen3.5-9B** (own chart) uses the same settings, seq 2K–64K, all from job `20261107T202125Z`. Unbound: 2.42× / 2.68× / 3.58× / 3.91× / 4.61× / 5.41× at 2K/4K/8K/16K/32K/64K. **Bound:** 2.43× / 2.72× / 3.10× / 3.45× / 3.72× / 3.73×. The earlier Qwen3.5 rows from NaN-affected jobs are withdrawn: they had NaN losses, and their unbound values differ from the rerun by up to 2.5×.

### Real-world models (required diversity)

The post must show **published Hugging Face architectures**, not only a synthetic MLP. Instantiation may still be **config + random weights** (no pretrained load) so the pin comparison is not a quality eval.

**Qwen3** and **Qwen3.5** are both required families. Label them as different architectures — not a size ladder; Qwen3.5 is always its own chart, never a size-ladder twin of Qwen3.

| Family | Role in the post | Default IDs | Do not |
|--------|------------------|-------------|--------|
| **Qwen3** (dense full-attention `Qwen3ForCausalLM`) | Required. Compare pin ratios only inside one SKU | B60: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) if host RAM is under ~48 GB, else [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). H200: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) if host RAM is at least ~256 GB, else [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B). Ascend 910B4: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B). | Call a synthetic MLP “Qwen3-4B”; plot Ascend 4B against H200 8B as the same run |
| **Qwen3.5** (hybrid Gated DeltaNet + full attention; hub cards are often multimodal) | Required. Different architecture, not a size ladder with Qwen3 | B60: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) or [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B). H200: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | Plot Qwen3-8B next to Qwen3.5-9B as “the same model, bigger”; use MoE/FP8 |

**Host RAM (ZeRO-3 param+optim):** the published H200 rows used `zero.Init` (measured host-resident offload: Qwen3-8B 137.31 GB, Qwen3.5-9B 150.10 GB). Without `zero.Init`, plan **~27 bytes per parameter** (18 B/P states × 1.5 buffer), plus OS headroom — not “18× the checkpoint.” Qwen3-0.6B ~15 GiB; 1.7B ~43 GiB; 8B ~206 GiB. Raise `ulimit -l`. See [DeepSpeed memory](https://deepspeed.readthedocs.io/en/latest/memory.html).

**Software:** Qwen3.5 needs a Transformers that registers `qwen3_5` (newer than 4.57.6 in some installs). Prefer a **text-only** construction if `AutoModelForCausalLM.from_config` hits `Qwen3_5ForConditionalGeneration` + vision. Log transformers version.

**Fairness:** pin on/off only **within** one ID × SKU × batch × seq. One **Qwen3-0.6B** (or 1.7B if RAM allows) model-tensor command on B60 is the required PCIe anchor. Ascend Qwen3-4B is its own panel, not that anchor. Qwen3.5, including on B60, is required, own chart.

**Activation checkpoint workload:** `activation_offload/bench.py --model … --offload-optimizer --act-mem-opt` (checkpoint `use_reentrant=False`, mark hidden states, **bf16**, async on). ZeRO-2 CPU AdamW keeps parameters on GPU. Do not publish synthetic bars under a Qwen name. Do not publish a GPU-AdamW-only or `--no-act-mem-opt` arm as this workload.

**H2D/D2H:** still no HF model.

### Harnesses

**Model-tensor e2e (required):** [DeepSpeedExamples `benchmarks/pin_memory` (dev fork)](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory) — `model_tensor_offload/bench.py` (root `zero3_offload_bench.py` still delegates to ZeRO-3).

Default driver (two subprocesses, fresh rendezvous each):

| Arm | `offload_optimizer` / `offload_param` `pin_memory` | Backend |
|-----|-----------------------------------------------------|---------|
| unpinned | `false` | n/a |
| pinned | `true` | `DS_PIN_MEMORY_BACKEND=native`, `DS_PIN_MEMORY_REGISTER_DEVICE=1` |

`--ablate-register` adds `pinned-unregistered` (`REGISTER_DEVICE=0`). Metrics: avg/min step time, accelerator peak, tokens/s, JSON `DRIVERRESULT`.

```bash
# smoke only (not a published Qwen row)
python model_tensor_offload/bench.py --hidden 2048 --layers 12 --batch 4 --seq 128

# Qwen3 B60 (0.6B if host RAM < ~48 GB)
python model_tensor_offload/bench.py --model Qwen/Qwen3-0.6B --batch 2 --seq 256

# Qwen3 H200
python model_tensor_offload/bench.py --model Qwen/Qwen3-8B --batch 4 --seq 512

# Qwen3.5 (separate table; needs transformers with qwen3_5)
python model_tensor_offload/bench.py --model Qwen/Qwen3.5-0.8B --batch 2 --seq 256
python model_tensor_offload/bench.py --model Qwen/Qwen3.5-9B --batch 4 --seq 512

# CUDA register inset (optional extra arm, not the pin-vs-pageable story)
python model_tensor_offload/bench.py --ablate-register --hidden 2048 --layers 12 --batch 4 --seq 128

# optional ZeRO-1/2 optimizer-only
python model_tensor_offload/bench.py --zero-stage 2 --hidden 2048 --layers 12 --batch 4 --seq 128
```

**Activation checkpoint workload (required):** `activation_offload/bench.py --offload-optimizer --act-mem-opt` — pin on/off, async/`use_streams` on. Pin on/off covers the activation pool and the ZeRO-2 optimizer host buffers. **Published rows use `--model` Qwen3 and Qwen3.5 (own chart).** Synthetic `--hidden/--layers` is smoke only.

```bash
# B60 (bf16; shrink seq if host RAM or VRAM requires it)
python activation_offload/bench.py --model Qwen/Qwen3-0.6B --offload-optimizer --act-mem-opt --batch 1 --seq 8192

# H200 published shape
python activation_offload/bench.py --model Qwen/Qwen3-8B --offload-optimizer --act-mem-opt --batch 1 --seq 65536

# Qwen3.5 required, own chart (Liger >= 0.8.4: fused linear CE + TiledMLP; needs fla + causal-conv1d)
python activation_offload/bench.py --model Qwen/Qwen3.5-0.8B --offload-optimizer --act-mem-opt --batch 1 --seq 8192

# smoke only (not the published workload)
python activation_offload/bench.py --hidden 2048 --layers 12 --batch 1 --seq 8192
```

**Saved-activation (required on H200):** `saved_activation_offload/bench.py --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2` — ZeRO-2 CPU AdamW; pin on/off covers activation buffers and optimizer host buffers. Pass `--prefetch 2` explicitly: the bench default changed to 1 in DeepSpeedExamples `231034a` (2026-10-18), after the published rows were collected. Sweep seq 8K–64K on H200; add `--discover-calib-seqs 4096 8192` at seq ≥32K for Qwen3-8B and at seq ≥16K for Qwen3.5-9B. Checkpoint+remat (`--mode remat`) is an optional reference panel, not the pin-impact denominator. Momentum-free SGD is diagnostic only.

```bash
python saved_activation_offload/bench.py --model Qwen/Qwen3-8B --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2 --batch 1 --seq 8192
python saved_activation_offload/bench.py --model Qwen/Qwen3-8B --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2 --batch 1 --seq 65536 --discover-calib-seqs 4096 8192
python saved_activation_offload/bench.py --model Qwen/Qwen3.5-9B --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2 --batch 1 --seq 65536 --discover-calib-seqs 4096 8192
```

**H2D/D2H (supporting):** `h2d_d2h/bench.py` (pageable + torch + native-unregistered + native-registered). B60: `--skip-native-register`.

```bash
python h2d_d2h/bench.py
python h2d_d2h/bench.py --skip-native-register
```

Writer constraints for this script:

- The H2D/D2H bench must **write the host buffer before timing** (default since 2026-10-23; `--no-touch` reproduces the old behavior). An untouched pageable buffer reads the kernel zero page and inflates pageable H2D (H200: 20.8 vs 10.1 GB/s). Confirm every SKU's published pageable H2D was measured on a written buffer.
- Launch every e2e row bound to the accelerator’s NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`, as the `run.sh` scripts do from DSE `31db0b7`), and record `numa_balancing` and the policy per SKU. Unbound runs on `numa_balancing=1` hosts were not reproducible for the pageable arm on H200.
- Run the **same harness** on Arc Pro B60, H200, and Ascend 910B4. Change model/batch/seq per SKU; log the command. Published e2e rows are **Qwen3** and **Qwen3.5** (own chart), not synthetic. Ascend published e2e is Qwen3-4B.
- Default pinned arm is **native + register**, not Torch. That is the pin-vs-pageable wall-time pair for the blog (routing is on DeepSpeed `master`). Do not add a Torch-vs-native e2e axis unless the script grows a Torch arm.
- README targets CUDA and XPU. On B60, `register_host_memory` does not exist yet; still run unpinned vs pinned. Treat `--ablate-register` as CUDA-only. If native pin fails on XPU, fall back is a named limitation — do not invent B60 numbers.
- Native `mlock` needs `ulimit -l` (or root) for multi-GB models.
- **Does not** replace the activation harnesses: activation checkpoint rows use `activation_offload/` with HF `--model --offload-optimizer --act-mem-opt`; saved-activation rows use `saved_activation_offload/`.

### Required to publish

**Article order** (writer): (1) model-tensor pin on/off wall time, (2) activation checkpoint workload pin on/off with async on, (3) saved-activation pin on/off (no remat), (4) H2D/D2H as supporting DMA evidence. Collection order may differ (see Sequencing).

| Experiment | Hardware | Axes | Metric |
|------------|----------|------|--------|
| Model-tensor CPU offload (`model_tensor_offload/bench.py`) | Intel Arc Pro B60 | Qwen3-0.6B (or 1.7B if host RAM allows) | step time / tok/s |
| Model-tensor CPU offload (`model_tensor_offload/bench.py`) | One high-end CUDA PCIe (e.g. H200) | Qwen3-8B (or 4B); optional `--ablate-register` | step time / tok/s |
| Activation checkpoint workload (`activation_offload/bench.py --offload-optimizer --act-mem-opt`) | Intel Arc Pro B60 | Qwen3; pin T/F; async on | ms/step, tok/s, accelerator peak |
| Activation checkpoint workload (`activation_offload/bench.py --offload-optimizer --act-mem-opt`) | One high-end CUDA PCIe | Qwen3-8B; pin T/F; async on; batch 1 × seq 65536 | ms/step, tok/s, accelerator peak |
| Saved-activation offload (`saved_activation_offload/bench.py --offload-optimizer`) | One high-end CUDA PCIe (H200) | Qwen3-8B; batch 1; seq 8K/16K/32K/64K; ZeRO-2 CPU AdamW; pin T/F; `N=1`, `M=2`, `--offload-tensors -1` | step time, offloaded bytes, peak HBM |
| Saved-activation offload (`saved_activation_offload/bench.py`, same flags as H200) | Intel Arc Pro B60 | Qwen3 0.6B/1.7B; batch 1; seq 2048; ZeRO-2 CPU AdamW; pin T/F; required, not yet collected | step time |
| H2D/D2H microbench (`h2d_d2h/bench.py`) | One high-end CUDA (H200) | **pageable vs pinned** primary; Torch vs native-unregistered vs native-registered as CUDA parity inset | GB/s; `torch.is_pinned()` vs DeepSpeed-managed pin |
| H2D/D2H microbench (`h2d_d2h/bench.py`) | Intel Arc Pro B60 | pageable vs Torch/XPU pin (`--skip-native-register`) | GB/s |
| Model-tensor CPU offload (`model_tensor_offload/bench.py`) | Ascend 910B4 | Qwen3-4B; batch 1; seq 1024 and seq 2048; unpinned vs pinned | step time / tok/s |
| Activation checkpoint workload (`activation_offload/bench.py --offload-optimizer --act-mem-opt`) | Ascend 910B4 | Qwen3-4B; batch 1; seq 8192; pin T/F; async on | ms/step, tok/s, accelerator peak |
| Saved-activation offload (`saved_activation_offload/bench.py`, same flags as H200) | Ascend 910B4 | Qwen3-4B; batch 1; seq 2048; ZeRO-2 CPU AdamW; pin T/F (collected: 1.14×, 9.48→8.34 s) | step time |
| H2D/D2H microbench (`h2d_d2h/bench.py`) | Ascend 910B4 | native-registered vs pageable, 256 MiB | GB/s |

### Optional / add-on (do not block)

| Experiment | Hardware | Axes | Metric |
|------------|----------|------|--------|
| ZeRO-1/2 optimizer offload | High-end CUDA if VRAM allows | `model_tensor_offload/bench.py --zero-stage 1` or `2` | step time |

**Publish bar:** **Intel Arc Pro B60**, **H200**, and **Ascend 910B4**, on **Qwen3**. All three workloads and the H2D/D2H microbench are required on all three SKUs. B60 must have **Qwen3** model-tensor (and activation checkpoint unless the engine is a named miss). Ascend 910B4 must have Qwen3-4B model-tensor (batch 1, seq 1024 and seq 2048), the activation checkpoint workload (batch 1, seq 8192), and H2D/D2H native-registered vs pageable. Ascend saved-activation (Qwen3-4B, batch 1, seq 2048) is required and collected. Saved-activation **also blocks publish on H200** for Qwen3-8B (pin on/off, `N=1`, `M=2`, `--offload-tensors -1`, seq 8K–64K; collected), on Ascend 910B4 (Qwen3-4B, batch 1, seq 2048; collected), and on B60 (Qwen3 0.6B/1.7B, batch 1, seq 2048; required as of 2026-10-21, not yet collected). **Qwen3.5 is also a publish gate, on its own chart — never a size-ladder twin of Qwen3.** Do not ship synthetic-only e2e. Do not ship CUDA-only. Do not ship model-tensor-only. Do not treat #8282’s async-vs-blocking table as pin on/off. Do not treat saved-activation as a checkpoint-offload variant. Do not copy H200 numbers onto B60 or Ascend. ZeRO-1/2 optimizer-only offload is an add-on.

**B60 activation (no weasel words):** required, same as model-tensor. If the activation engine **does not run** on stock PyTorch XPU, the activation section **must open** with that named limitation (commit, error, or missing API). Do **not** imply Intel activation numbers exist. Do **not** drop B60 from model-tensor or H2D/D2H. That incomplete vendor-diversity for activations is explicit, not silent.

### Row notes

- **Model-tensor harness:** `model_tensor_offload/bench.py` with `--model Qwen/Qwen3-*` and `--model Qwen/Qwen3.5-*`. Synthetic is smoke. Use `--zero-stage 2` only if ZeRO-3 still OOMs after dropping to 0.6B / 0.8B.
- **Ascend 910B4:** required for model-tensor, activation checkpoint, and H2D/D2H on the shapes in the required table. Saved-activation is also required (Qwen3-4B, batch 1, seq 2048: 1.14×).
- **XPU:** B60 uses `model_tensor_offload/bench.py` and `h2d_d2h/bench.py --skip-native-register`. No `--ablate-register`.
- **Activation checkpoint workload:** HF `--model` Qwen3, `--offload-optimizer --act-mem-opt`, async **on**. Pin on/off moves the activation pool and the optimizer host buffers together. Cite #8282 async-vs-blocking as overlap inset only. Never label synthetic bars as Qwen.
- **Saved-activation:** H200 Qwen3-8B pin on/off at `N=1`, `M=2`, `--offload-tensors -1`, seq 8K–64K. The `N=1` justification is H200-measured; re-check it per SKU. B60 is required: Qwen3 (same ID as the B60 model-tensor row: 0.6B or 1.7B), batch 1, seq 2048 (shrink seq if 24 GB VRAM requires it, and log it), same flags as H200, pin T/F; Qwen3.5-0.8B or 2B on its own chart. If the engine fails on stock PyTorch XPU, open the section with a named limitation (commit + error); do not invent numbers. Ascend 910B4 Qwen3-4B seq 2048 is required (1.14×; no job id in the source table). Qwen3.5 is required, own chart.
- **PCIe flat results:** allowed. Soften “always pin” if host-optimizer-bound workloads show noise.
- **Activation checkpoint, 8×H200 capacity + placement (2026-11-10):** `activation_offload/bench_8gpu.py` (DSE `74daac2`), ZeRO-2 + GPU AdamW (fp32 state), packed 4096-token samples, `ds_async`. Restores the capacity claim the single-GPU section dropped: gc wall 233,472 vs offload 524,288 (Qwen3-8B, 2.25×); 208,896 vs ≥413,696 (Qwen3.5-9B, ≥1.98×). Pin rows at 256K per GPU, both placements (see Claims). Prints in the activation checkpoint section as its own block, not merged into the single-GPU tables.

### Figure guidance

- **H200 headline numbers to plot (bound to GPU0's node, jobs `20261107T043012Z` and `20261107T202125Z`):** model-tensor ZeRO-3 seq 4096 **2.82×** (10.53 → 3.74 s); activation checkpoint seq 64K **1.13×** (32.60 → 28.97 s); saved-activation seq 64K **2.74×** (91.22 → 33.25 s). The previously quoted 3.40× / 1.19× / 3.80× were unbound runs on other days. Do not print them as current; at most cite them in a NUMA footnote. State the launch policy in the hardware table.
- **H200 Qwen3.5-9B numbers to plot (unbound, mark provisional):** model-tensor seq 4096 **2.94×** (13.39 → 4.56 s, `job-20261107T043012Z`); activation checkpoint seq 64K **1.25×** (20.81 → 16.67 s, 55.7 GB, `job-20261016T193445Z`); saved-activation seq 64K **5.41×** (162.43 → 30.04 s, `job-20261107T202125Z`).
- **Activation workloads plot the longest sequence only**, per machine and family. Shorter rows (H200 activation checkpoint 8K–32K, saved-activation 2K–32K) stay in the handoff and in source comments, not in the article. The headline table and Conclusion use the same longest-sequence numbers (H200 Qwen3 range 1.13×–2.74×).
- Lead with **wall time**: model-tensor pin vs non-pin (unpinned=1.0), activation-checkpoint pin vs pageable with async on (pageable=1.0), and saved-activation pin vs pageable (pageable=1.0). Equal visual weight. Each workload has a Qwen3 figure and a separate Qwen3.5 figure; file names are listed in [`LAYOUT.md`](LAYOUT.md).
- H2D/D2H is a supporting figure. Do not plot 55 GB/s H200 next to B60, or next to Ascend (~26 GB/s). On CUDA, do not give four equal bars to Torch vs native; pin vs pageable is the story, register-parity is an inset. Ascend’s DMA panel is native-registered vs pageable.
- Optional third activation panel: #8282 async vs blocking (pinned) — different axis.
- Hardware table: SKU, host CPU, interconnect, Torch/CUDA or PyTorch XPU, DeepSpeed commit ≥ #8283, `ulimit -l`, `kernel.numa_balancing`, launch policy, models per SKU.
- Label H200 ~55 vs ~10–17 GB/s as one PCIe Gen5 datapoint. Use the bound touched-buffer remeasure (`job-20261024T171244Z`, `numactl --cpunodebind=0 --preferred=0`, median of 3 reps): 256 MiB H2D 10.23 -> 55.39 GB/s (5.41x), D2H 17.31 -> 54.94 (3.17x). Unbound on the same host: 5.42x / 3.41x, with one outlier rep (pageable 8.16 / 14.94 GB/s). The 2026-10-23 unbound figures (5.47x / 3.35x, `job-20261023T131838Z`) agree with these within noise. The older 20.76 GB/s pageable H2D (2.67x) was a zero-page artifact and must not be printed.

---

## Claims that need evidence

- Pin vs non-pin wall time for **model-tensor and activation checkpoint** on **Intel Arc Pro B60**, H200, and **Ascend 910B4** (Qwen3-4B), on **Qwen3**.
- Pin vs non-pin wall time for **saved-activation** on H200, Qwen3-8B, `N=1`, `--offload-tensors -1`, seq 8K/16K/32K/64K collected (bound: 2.98× / 3.26× / 3.17× / 2.74×; unbound: 3.12× / 3.64× / 3.92× / 3.85×); the article prints the 64K row only. B60: required, numbers pending. Ascend: Qwen3-4B, batch 1, seq 2048, 1.14× (9.48→8.34 s). Ascend Qwen3.5-4B saved-activation: not collected. Qwen3.5 is required, own table and chart.
- **Qwen3.5 open items:** H200 Qwen3.5-9B saved-activation 64K pageable step time is in the rerun (`job-20261107T202125Z`: 108.53 s bound, 162.43 s unbound). Ascend Qwen3.5-4B activation checkpoint at 8192 is ~6× slower per step than Qwen3-4B (92.58 vs 15.44 s pageable); confirm `fla` and `causal-conv1d` kernels were active on the NPU before publishing (the same missing-package fallback made H200 Qwen3.5 ~8× slower before `job-20261016T193445Z`).
- Ascend H2D/D2H: native-registered vs pageable at 256 MiB.
- Do not present synthetic MLP as a Qwen result.
- **Bound versus unbound (H200, rerun jobs `20261107T043012Z` and `20261107T202125Z`):** this is a main effect of the post, not a footnote. Unbound runs on a host with `kernel.numa_balancing=1` inflate the pageable arm. Bound pin speedups are lower than unbound ones in every row: model-tensor Qwen3-8B seq 4096 3.62× → 2.82×, activation checkpoint 64K 1.25× → 1.13×, saved-activation 64K 3.85× → 2.74×, Qwen3.5 saved-activation 64K 5.41× → 3.73×. Pinned times change by under 10% with placement; the pageable arm changes by 40% or more. Bound is the fairest pageable baseline: it is the best pageable placement we measured. Present bound numbers as the headline, unbound as the cost of default placement, and state the balancing setting per machine. B60 and Ascend have no bound runs yet; do not assume their gap matches H200.
- **Methodology (H200):** every published e2e row is measured bound to the accelerator's NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`, as the `run.sh` scripts do), with host buffers written before timing, 10 steps, and 1 warmup step. Unbound runs on `numa_balancing=1` hosts are not reproducible for the pageable arm: the earlier unbound Qwen3.5 64K figure (7.93×) differs from the rerun (5.41×) by 2.5×. Rerun, bound: model-tensor 2.82×, activation checkpoint 64K 1.13×, saved-activation 2.98× / 3.26× / 3.17× / 2.74× at 8K–64K. Rerun, unbound: model-tensor 3.62×, activation checkpoint 64K 1.25×, saved-activation 3.12× / 3.64× / 3.92× / 3.85×. **Weight initialization:** until 2026-11-06 the bench built models with uninitialized weights (`no_init_weights()` and no fill). Those runs have losses that are not meaningful, and some have NaN losses. The rows in this draft come from the 2026-11-07 rerun, which fills the weights with a fixed seed. Record `numa_balancing` and the launch policy for every SKU. B60 and Ascend rows have not been re-run bound. Qwen3.5 H200 rows have.
- **NUMA binding is a key practitioner message (user, 2026-11-10).** 8×H200 activation checkpoint at 256K per GPU, unbound `job-20261110T114439Z` vs bound `job-20261110T143411Z` (per-rank `numactl --cpunodebind=N --preferred=N`): Qwen3-8B pageable 54.44 → 47.61 s, pinned 43.24 → 43.28 s, pin speedup 1.26× → **1.10×**; Qwen3.5-9B pageable 50.42 → 45.35 s, pinned 41.39 → 41.41 s, 1.22× → **1.095×**. Messages: pinning alone recovers the whole gain regardless of placement; binding alone recovers 55–60%; bound, pinning still adds ~10%; torch and native backends tie. torchrun `--numa-binding=node` (`job-20261110T161516Z`, CPU affinity only, no memory policy) is **not enough**: Qwen3-8B pageable 54.09 s / pinned 43.28 s (1.25×), Qwen3.5-9B 49.75 / 41.36 s (1.20×), i.e. unbound-like; the memory policy (which also opts out of NUMA balancing) is what helps, but CPU-only vs memory-only was not split. **Open:** these rows used 5 timed steps after 2 warmup, not the 10-step method. Scaling of the NUMA effect with length is out of scope.
- Activation checkpoint workload: pin vs pageable **with async on**, `--offload-optimizer --act-mem-opt`. The pin arm moves the activation pool and the optimizer host buffers together. Do not claim pin from #8282 async-vs-blocking.
- On CUDA, native **with** register matches Torch H2D/D2H; native **without** register stays pageable-class (cite or remeasure #8283).
- Pin makes copies **eligible** for overlap; do **not** claim “usable vs stalled” or a relative win until measured (including PCIe).
- Torch pin needs accelerator-capable Torch; DS native works on CPU-only (mlock; register no-op).
- Torch pin freed by GC; DS native supports explicit unpin (unregister then `munlock`) — motivate with memlock / host RAM reclaim, not as the title.

---

## Outline

Canonical layout: [`LAYOUT.md`](LAYOUT.md).

Headings stay **accelerator-neutral**. No “ZeRO” and no PR numbers in H2/H3 titles. Vendor APIs and PR ids belong in body text.

```
# Pinned Host Memory for CPU Offload: A Practitioner Guide
## Introduction
### CPU offload for cost-efficiency
### Three applications: model tensors, checkpoint activations, and saved activations
### Pin vs pageable as system efficiency
### What this post covers (evaluations, APIs, frameworks, best practices)
## Background: why pin can change offload efficiency
## Real-world evaluation
### Setup (Qwen3; host RAM)
### Method: written buffers, bound placement, step counts
### Model-tensor offload (Arc Pro B60, H200, Ascend 910B4)
### Activation checkpoint workload (same hardware and families)
### Saved-activation offload, no rematerialization (Arc Pro B60, H200, Ascend 910B4)
### Supporting H2D/D2H microbench
## APIs and popular frameworks
### PyTorch
### Hugging Face checkpoint hooks
### DeepSpeed (model-tensor CPU offload and cpu_checkpointing)
### How to pin: Torch or native (parity once registered; when unpin matters)
## Best practices
### Independent knobs: offload pin_memory vs use_pin_memory
### Backend and device register
### When to disable pin (memlock, host RAM)
### Host memory placement: NUMA node, automatic NUMA balancing, page size
## Conclusion
## Contributors
## Appendix: reproducing the results
```

**Writer notes (not headings):**

- **Answer first:** the “What this post covers” section carries the headline table (Qwen3, longest sequence for the activation workloads), three or four one-sentence findings, and the decision hint (pin on unless host RAM/memlock is tight; measure on a new host link).
- **Qwen3.5 placement:** no standalone Qwen3.5 section. Describe the architecture once under Setup; each workload section then has a Qwen3 table and a Qwen3.5 table, each with its own chart.
- **Article vs collection:** benches may be gathered ZeRO-first; the README still leads with all three applications’ wall time.
- **Frameworks:** three sibling H3s. Do not collapse into `get_accelerator().pin_memory()`. DataLoader pin is orientation only.
- **Registered host memory:** page-lock for the OS; register with the driver for the copy engine. `mlock` without register stays pageable-class for GPU copies.
- **Why pageable is slower (Background), per driver:** CUDA (H200): pageable copies stage through a driver-managed buffer; a probe found no per-copy `mlock`/`ioctl`/`mmap` difference against pinned, 2M pages do not help, the issuing thread blocks for the whole copy (27.9 ms for 256 MiB), and the staging work runs on CPU threads, so it is placement-sensitive. Ascend 910B4: the driver team reports that pageable memory is pinned and SMMU page tables are built on every copy, and that async copies become synchronous; that explains the page-size sensitivity but is not measured by us. B60: not characterized. Mark which statements are measured and which are reported.
- **Observability:** point readers to `get_accelerator().is_pinned()`, which covers both backends. Do not print that `torch.is_pinned()` is false on DeepSpeed-native buffers: once registered with `cudaHostRegister`, PyTorch may report them as pinned. Verify on CUDA before making either claim.
- **Numbers:** H200 / #8283 under supporting microbench; H200 / #8282 async vs blocking under the activation checkpoint workload as an overlap inset. New pin on/off numbers for that workload are the activation eval. Ascend 910B4 numbers stay on the Ascend panel. Qwen3 vs Qwen3.5 never share an unlabeled chart.
- **hipHostRegister / IPEX** are not DeepSpeed #8283 APIs.
- **Page size, H200 bound, all workloads (`job-20261027T132751Z`):** between 4K and 2M, the end-to-end pin speedup moves ≤0.08× for model-tensor, activation checkpoint, and saved-activation 8K–64K. The H2D/D2H copy speedup moves up to 0.17× (5.43× vs 5.26× H2D at 256 MiB; 5.87× vs 5.70× at 1 GiB), which is ≤3% relative (table in EXPERIMENTS_HANDOFF §6). Use this for the page-size claim: huge pages are not a pinning substitute and not a pinning enhancer on H200. Staging probe: no per-copy pinning syscalls on CUDA. Ascend's page-size result is separate and unverified.
- **Host page size (huge pages), under “Host memory placement”:** do not present 2M pages as an alternative to, or an enhancer of, pinning. H200, bound (`job-20261027T132751Z`): 2M changes any end-to-end step time by ≤4% and the end-to-end pin speedup by ≤0.08× on every workload (the microbench D2H pageable arm moves 5%), and the staging probe found no per-copy pinning syscalls for pageable CUDA copies. The earlier unbound 2M gains (average −7% to −16%) were NUMA-balancing outliers and placement, not page size (`job-20261024T022512Z`, `032725Z`). Ascend 910B4: 2M pages bring pageable H2D near pinned in the microbench (~6.4 → ~23 GB/s), but that is unverified (see the Ascend note) and e2e is unmeasured. Say that, and that the mechanism differs by driver.
- **NUMA, under Best practices:** bind the process to the accelerator's NUMA node. Ascend: a remote node costs pinned D2H ~26%. H200: ≤6%. Measure per platform; do not port the number.
- **Host memory placement (practitioner tip, H200 measured):**
  - With `kernel.numa_balancing=1` the kernel samples and migrates the pages of every large host buffer each step. Pinning offload buffers does not prevent this, because DeepSpeed's CPU Adam state is ordinary pageable memory. On H200 (Qwen3-8B, 4K pages) it cost 1.6–6.8M minor faults and 10–70 CPU-s of kernel time per step, in pinned and pageable runs alike.
  - Any explicit task policy (bind or preferred) turns it off. `numactl --membind=all` does that but leaves CPU and memory placement to the scheduler, which swung the same command between 7.4 and 8.8 s/step.
  - Launch single-accelerator CPU-offload training under `numactl --cpunodebind=<accelerator node> --preferred=<accelerator node>`. H200 activation checkpoint, pageable: 7.2 s/step local vs 10.5 s with CPUs on the far node. Pinned is less sensitive (4.2 vs 5.5 s), which is itself an argument for pinning: staged pageable copies run on CPU threads and need them near the device. Model-tensor is placement-insensitive (3.76 vs 3.77 s pinned). For multi-rank runs, bind each rank to its own device's node.
  - Chart: faults/step and step time, unbound vs bound, pinned and pageable arms.

---

## Artifacts

- Layout: [`LAYOUT.md`](LAYOUT.md). Draft: `blogs/pinned-offload/08-2026/README.md` + `media/` figures (model-tensor, activation checkpoint, **and** saved-activation wall time, equal weight, one Qwen3 and one Qwen3.5 figure each; microbench supporting).
- Repro notes: the README appendix, with the `model_tensor_offload/bench.py`, `activation_offload/bench.py`, `saved_activation_offload/bench.py`, and `h2d_d2h/bench.py` commands and the NUMA launch policy. Per-SKU commands (B60, Ascend 910B4) are added when those rows land.
- Hardware/software table for **measured** SKUs only.
- RTD lock-vs-register text now matches #8283 (fixed 2026-11-05, uncommitted); the README links it. Fix the Ulysses/ALST tutorial's stale paragraph too.

---

## Sequencing vs Phase 2

Collection order (hardware access) is not article order.

1. Pin experiments: [DeepSpeedExamples `benchmarks/pin_memory` (dev fork)](https://github.com/sfc-gh-truwase/DeepSpeedExamples/tree/tjruwase/pin-memory-experiments/benchmarks/pin_memory) on DeepSpeed **`master`**. Land `activation_offload --model` before published HF rows.
2. Collect **model-tensor** Qwen3 + **H2D/D2H** on B60, then H200. Ascend 910B4 Qwen3-4B rows are already measured.
3. Collect the **activation checkpoint workload** (`--offload-optimizer --act-mem-opt`, async on, pin on/off) on B60 and H200. Ascend 910B4 Qwen3-4B seq 8192 is already measured. Named limitation if the B60 engine fails — do not invent numbers.
4. Collect **saved-activation** pin on/off on H200 (Qwen3-8B, `N=1`, `--offload-tensors -1`, seq 8K–64K: collected, jobs `20261011T171729Z` unbound, `20261026T133233Z` and `20261024T171244Z` bound; Qwen3.5-9B collected in rerun job `20261107T202125Z`; the earlier unbound Qwen3.5 run is withdrawn). Ascend 910B4 Qwen3-4B seq 2048 is measured. B60 saved-activation is required and not yet collected. Qwen3.5 is required, own chart.
5. Re-run the non-H200 rows and Qwen3.5 bound, recording `numa_balancing` and the launch policy.
6. README drafted 2026-11-05 ahead of the B60 rows, with placeholders for B60 and for the bound Ascend reruns. Fill those in as they land; publish only when every required row is present. Synthetic stays in an appendix or method note.
