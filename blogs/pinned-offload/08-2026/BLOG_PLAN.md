# Blog Plan — Pinned Host Memory for CPU Offload

**Audience:** agent or writer drafting the blog and collecting benchmarks.  
**Venue:** [`blogs/pinned-offload/08-2026/README.md`](README.md) (create when drafting).  
**Layout peer only:** [`blogs/deepspeed-offloadpp/README.md`](../../deepspeed-offloadpp/README.md) (README/media shape). **Do not** copy its ZeRO-product lede, hero speedup, or “we release feature X” voice.  
**Related:** experiment/DSE work is [`EXPERIMENTS_HANDOFF.md`](EXPERIMENTS_HANDOFF.md) (out of scope for the blog writer). Phase 2 routing is already on DeepSpeed `master` ([`PHASE2_PR_HANDOFF.md`](PHASE2_PR_HANDOFF.md) is historical).

**Out of scope:** NVMe, DeepNVMe, AIO, GDS, ZeRO-Inference disk offload, DataLoader pin as an experiment. Do not mention NVMe as a reason to pick native pin. DataLoader `pin_memory` may appear once as the PyTorch API practitioners already know — not as a benchmark.

---

## Working title

**Pinned Host Memory for CPU Offload: A Practitioner Guide**

Do **not** use “DeepSpeed Offload,” “Faster Transfers,” or “Explicit Lifetime Control” in the title. Lifetime/unpin belongs under APIs / best practices. Speedups belong in Evaluation after they are measured.

**Pitch:** CPU offload is a **cost-efficiency** and **capability** tool for deep learning: run on smaller or cheaper GPUs, and train workloads that would not fit otherwise. It has **more than one application**, presented as peers:

1. **Model-tensor offload** — parameters and optimizer states (DeepSpeed ZeRO CPU offload is the mechanism, not the headline). Gradients are the same idea; this post does not measure them.
2. **Activation checkpoint workload** — checkpoint inputs moved to host during forward, then rematerialized, so long sequences fit. The published workload is that offload together with ZeRO-2 CPU AdamW (`--offload-optimizer`; parameters stay on GPU) and `--act-mem-opt` (Qwen3: Liger fused linear cross-entropy, TiledMLP, and fused cross-entropy). Async copies stay on. Pin on/off moves the activation pool and the optimizer host buffers together. Do **not** call this generic “input offload.”
3. **Saved-activation offload** — every saved-for-backward tensor in each decoder layer saved to host and reloaded, **without** rematerialization. Credit-bounded: published H200 rows use `N=1` (`--credits 1`), `M=2` (`--prefetch 2`), no per-layer tensor cap (`--offload-tensors -1`). See “Saved-activation settings, as measured” under Benchmark matrix. Published rows use ZeRO-2 CPU AdamW (`--offload-optimizer`); pin toggles activation buffers and optimizer host buffers. A different algorithm from (2), not a config of `CheckpointHiddenStatesOffload`. No `ds_config` field; the blog reports the benchmark, not a shipping API.

Do **not** tell this as a ZeRO story with activations as a sequel. Name all three applications before any DeepSpeed tour.

**Pin vs non-pin** is the **system-efficiency** question on top of that: whether host buffers are device-visible so copies are **eligible** for DMA and overlap. Pinning does **not** guarantee end-to-end overlap or wall-time wins; that depends on workload and interconnect. Pageable copies often bounce; do not write that they always “stall the bus.”

This blog is a **practitioner guide**. It should deliver:

- **Real-world evaluations** — pin on/off wall time for **all three** applications on **Qwen3** (dense). Qwen3.5 is required alongside Qwen3, on its own chart. All three applications, (1) model-tensor, (2) activation checkpoint, and (3) saved-activation, run on every evaluated accelerator: Intel Arc Pro B60, high-end PCIe (H200), and **Ascend 910B4** (Qwen3-4B), through DeepSpeed’s accelerator abstraction. Synthetic MLP is smoke, not a published substitute. H2D/D2H is supporting DMA evidence, not the lede figure. Keep each machine on its own panel. Wall time is measured bound to the accelerator’s NUMA node (see Best practices).
- **Available APIs** — PyTorch `Tensor.pin_memory()` (and, in passing, `DataLoader(pin_memory=True)` as the familiar hook); DeepSpeed `get_accelerator().pin_memory()` / unpin / `is_pinned`; ZeRO `offload_*.pin_memory`; activation `use_pin_memory`; `DS_PIN_MEMORY_BACKEND` and `DS_PIN_MEMORY_REGISTER_DEVICE`.
- **Popular frameworks** — three sibling sections: PyTorch, Hugging Face non-reentrant checkpoint hooks, DeepSpeed (model-tensor CPU offload and `cpu_checkpointing`). Do not invent APIs other stacks do not have.
- **Best practices** — shipping defaults are pin-on; confirm with the e2e numbers. When to turn pin off (memlock / host RAM). Torch vs native (how to pin). Host memory placement (bind to the accelerator’s NUMA node; page size is not a lever on H200). Do not copy one machine’s numbers onto another. A **flat** pin-on/off result on PCIe is publishable.

Cost/capability is **framing**. Pin on/off numbers do not prove “ran on a cheaper GPU.” For activation capability, you may cite #8282’s no-offload vs offload peak/seq arms; do not relabel them as pin vs pageable.

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

**Voice:** (1) offload as cost + capability, three peer applications; (2) pin vs non-pin as system efficiency (eligibility, then measured wall time); (3) evaluations, then APIs / frameworks, then best practices. **Article order is not collection order.** Do not open as “ZeRO offload, plus a note on activations.”

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

RTD: Host Memory Pinning in `docs/code-docs/source/memory.rst`. **Fix stale RTD** that still says native is mlock-only / keep Torch because there is no `cudaHostRegister` before linking it from the blog.

Config orthogonality (document under Best practices, not as the intro):

- `zero_optimization.offload_*.pin_memory` → **whether** model-tensor CPU offload pins (default true).
- Activation `use_pin_memory` → **whether** the host pool pins (default true; HF `CheckpointHiddenStatesOffload` and native `cpu_checkpointing` share `_ActivationOffloadEngine`). Not the ZeRO config field.
- `DS_PIN_MEMORY_BACKEND=torch|native` → **how** to pin when code calls `get_accelerator().pin_memory()` (env only; default remains **torch**).
- `DS_PIN_MEMORY_REGISTER_DEVICE` → native only: on by default after `mlock` on CUDA; `0` is mlock-only ablation.

---

## Narrative arc

1. **Offload as strategy** — Cost-efficiency and capability. Not a ZeRO-only problem.
2. **Three peer applications** — Model tensors (params + optimizer). Activation checkpoint workload (checkpoint hidden states with remat, ZeRO-2 CPU AdamW, `--act-mem-opt`). Saved activations (whole-decoder-layer saved tensors, no remat). Name all three before DeepSpeed. Independent knobs.
3. **Pin vs non-pin as system efficiency** — Copies cross the host bus. Device-registered host memory makes DMA/overlap **eligible**; wall time is measured, not assumed. Applies to all three applications.
4. **What this post gives practitioners** — One-paragraph answer-first (when to pin, when to disable) plus the four deliverables.
5. **Evaluation: model-tensor offload** — Pin vs non-pin wall time on Arc Pro B60, H200, and **Ascend 910B4** on **Qwen3**. Same heading weight as (6) and (7). Qwen3.5 is required, own chart. Ascend’s published model is Qwen3-4B. Wall time is measured bound to the accelerator’s NUMA node, with written host buffers.
6. **Evaluation: activation checkpoint workload** — Same hardware and families. Command is `activation_offload/bench.py --offload-optimizer --act-mem-opt` with async on. Pin on/off moves the activation pool and the optimizer host buffers together. Cite #8282 async vs blocking (pinned) as a **separate overlap** inset. Do not reuse async-vs-blocking as pin vs pageable. GPU-AdamW-only and `--no-act-mem-opt` runs are not this workload.
7. **Evaluation: saved-activation offload** — No rematerialization. All three SKUs are required: H200 (Qwen3-8B), Ascend 910B4 (Qwen3-4B, measured), and B60 (Qwen3 0.6B/1.7B, batch 1, seq 2048). Qwen3.5-9B is required, own chart. Do not plot it as a variant of (6). Do not copy the H200 ratio onto other SKUs.
8. **Supporting DMA microbench** — After wall time, not before. CUDA: pageable vs pinned is the primary axis; Torch vs native-registered is a **parity inset**. B60: pageable vs Torch/XPU pin. Ascend 910B4: native-registered vs pageable, own panel.
9. **APIs and frameworks** — PyTorch, then HF hooks, then DeepSpeed configs. Saved-activation has no config surface. Then how-to-pin (Torch vs native), including explicit unpin as reclaiming locked host pages under memlock pressure — not a title claim.
10. **Best practices** — Independent on/off switches; default torch backend; disable pin under tight memlock/RAM; bind CPUs and memory to the accelerator’s NUMA node; do not port one machine’s numbers to another. “Default on” is the shipping default to **check**, not a result.
11. **What’s next** — Practitioner leftovers: XPU `register_host_memory` if claimed; remaining pin call sites only if they change what users can do. Not a Phase 2 routing tour. Saved-activation stays a benchmark unless it is later promoted.

All pin experiments live under [DeepSpeedExamples `benchmarks/pin_memory`](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/pin_memory) (`model_tensor_offload/`, `activation_offload/`, `saved_activation_offload/`, `h2d_d2h/`).

---

## Benchmark matrix

Three workloads, all required (peer applications):

- **Model-tensor offload (ZeRO-3 CPU param+optim):** use the Examples harness below. Scale `--batch` / `--seq` / model size to VRAM/host RAM. Document the exact command per SKU. Do not require identical parameter counts. In the **article**, call this model-tensor offload; ZeRO-3 is a method footnote.
- **Activation checkpoint workload:** `activation_offload/bench.py --model … --offload-optimizer --act-mem-opt` on a **real HF model**. Document `use_reentrant=False`, async on, and pin on/off. Pin on/off covers the activation pool and the ZeRO-2 optimizer host buffers. Qwen3 `--act-mem-opt` is Liger fused linear cross-entropy, TiledMLP, and fused cross-entropy. Qwen3.5 has no Liger; that arm is TiledMLP plus tiled fused logits loss. Shrink seq/model on smaller-VRAM SKUs. Publish up to seq 64K on H200, matching saved-activation's capacity ceiling so the two remat workloads share one sequence range; activation checkpoint itself fits to at least 128K (70.6 GB peak, 1.09× pin speedup, `job-20260928T140704Z`). Synthetic MLP is smoke and a method footnote, not a substitute for Qwen rows.
- **Saved-activation offload (no remat):** `saved_activation_offload/bench.py --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2` (accelerator-agnostic via DeepSpeed’s accelerator abstraction), whole-decoder-layer scope (the bench's default `--offload-modules`), ZeRO-2 CPU AdamW. H200 row: Qwen3-8B, batch 1, pin on/off, seq 8K/16K/32K/64K; seq ≥32K adds `--discover-calib-seqs 4096 8192`. Report step time, offloaded bytes, and peak HBM. Label `N` on each panel and do not copy the H200 ratio onto other SKUs. The script is accelerator-agnostic (DeepSpeed accelerator abstraction), so it runs on Ascend 910B4 (required: Qwen3-4B, batch 1, seq 2048, same flags) and on B60 (required: Qwen3 (same ID as the B60 model-tensor row: 0.6B or 1.7B), batch 1, seq 2048 (shrink seq if 24 GB VRAM requires it, and log it), same flags as H200, pin T/F; Qwen3.5-0.8B or 2B on its own chart). Qwen3.5-9B is required, own chart. Full spec: [`SAVED_ACTIVATION_OFFLOAD_PLAN.md`](SAVED_ACTIVATION_OFFLOAD_PLAN.md).

**Saved-activation settings, as measured (H200, Qwen3-8B, batch 1, ZeRO-2 CPU AdamW).** The published H200 settings differ from the original `N=2`, `M=2`, `--offload-tensors` 1/2/3 spec. Each change is backed by a measurement:

- **Scope: whole decoder layer, not MLP-only.** The original `\.mlp$` scope offloaded only 3–4 tensors per layer. It left attention activations resident and OOMed at seq 16K while activation checkpointing still ran. The scope is now `layers.N`, the same unit that gradient checkpointing wraps.
- **`--offload-tensors -1`, not a 1/2/3 sweep.** Under the whole-layer scope, a K cap *raises* peak HBM: unselected eligible tensors stay resident instead of passing through the bounded credit pool. At K=1, peak HBM went from 54.30 to 83.01 GB at seq 8K, and the run OOMed at 32K (job `20261011T105335Z`). The old MLP-only K=1/2/3 table (job `20261007T215537Z`, seq 4096, 2.00× / 2.09× / 2.20×) is superseded history, not a published row. `r` was a K-sweep metric and is not reported for the published rows.
- **`--credits 1` (`N=1`), not 2.** `N` bounds how many forward layer-groups are held resident. Offloaded bytes are identical at either setting. `N=1` cuts peak HBM at long sequence and costs no measurable step time:

  | seq | pin speedup (N=1), unbound as published | **pin speedup, bound** | pinned step, bound | peak HBM N=2 → N=1 |
  |---|---|---|---|---|
  | 8K | 3.08× | **3.03×** (16.16 → 5.32 s) | 5.32 s | 54.30 → 54.30 GB (no change) |
  | 16K | 4.24× | **3.32×** (26.47 → 7.98 s) | 7.98 s | 59.44 → 59.44 GB (no change) |
  | 32K | 4.36× | **3.25×** (47.26 → 14.53 s) | 14.53 s | 82.04 → 75.54 GB (−8%) |
  | 64K | 3.80× | **2.83×** (91.5 → 32.3 s) | 32.3 s | 138.48 → 125.10 GB (−10%) |

  These speedups are **unbound** runs (no NUMA policy, balancing on). Bound = `numactl --cpunodebind=0 --preferred=0`, medians. 8K–32K are from `job-20261026T133233Z`; 64K is from `job-20261024T171244Z`, on a different host, so it is comparable within its own row only. Same-host unbound re-runs gave 3.60× / 3.73× / 4.98× / 4.46×, with run-to-run spread up to 5.4 s at 8K. Bound spread was ≤0.37 s. The pinned arm moves 4–8%; the pageable arm moves 16–38%. **Print the bound column.** Its message: pinning gives about 3× at 8K–32K and 2.8× at 64K. The pinned step time at `N=1` is within ±2.3% of `N=2` at every seq. Sources: `N=1` is job `20261011T171729Z` (dataset of record); `N=2` is job `20261011T005859Z`. At seq 8K/16K a different allocation sets the peak, so `N` does not move it.
- **`--prefetch 2` (`M=2`) kept for the published rows.** `M=1` cuts peak HBM further at 64K (125.10 → 97.88 GB, −22%), but pinned step time rises 12.3% (33.00 → 37.06 s, job `20261018T110757Z`). The pin speedup rises to 4.28× only because the pageable arm slows more. That makes `M=1` a memory-headroom ablation, not the published row. Qwen3.5-9B shows the same pattern: −29% HBM for +4–10% step time.
- **Ceiling: seq 64K.** At seq 98K the engine runs out of HBM: parameters, optimizer state, and held credit groups together exceed 139.8 GB. This is a real capacity limit, not a discovery artifact. Seq ≥32K uses linearly extrapolated discovery (`--discover-calib-seqs 4096 8192`), validated as an exact match against real discovery at 8K. Calibrated discovery threshold depends on the model: Qwen3-8B needs it at seq ≥32K, Qwen3.5-9B already at seq ≥16K (uncalibrated discovery at 16K exceeds 139.8 GiB before the measured arms run; `job-20261016T141158Z` and `20261105T120256Z` both OOMed there, the calibrated reruns `20261017T235002Z` and `20261105T162755Z` pass).
- **Qwen3.5-9B** (own chart) uses the same settings, seq 2K–64K. Unbound (`job-20261017T235002Z`): 3.68× / 2.73× / 3.19× / 4.30× / 5.14× / 7.93× at 2K/4K/8K/16K/32K/64K. **Bound** (`20261105T120256Z`, `20261105T162755Z`, repeats `20261105T165925Z`): 2.45× / 2.77× / 3.13× / 3.44× / 3.70× / 3.67× at 2K / 4K / 8K / 16K / 32K / 64K. 8K, 16K and 64K are medians of two repeats; 2K, 4K and 32K are from the first run. The first run's 8K pinned time (4.98 s) did not reproduce (6.14 s in both repeats), so the 8K row uses the repeats.

### Real-world models (required diversity)

The post must show **published Hugging Face architectures**, not only a synthetic MLP. Instantiation may still be **config + random weights** (no pretrained load) so the pin comparison is not a quality eval.

**Qwen3** and **Qwen3.5** are both required families. Label them as different architectures — not a size ladder; Qwen3.5 is always its own chart, never a size-ladder twin of Qwen3.

| Family | Role in the post | Default IDs | Do not |
|--------|------------------|-------------|--------|
| **Qwen3** (dense full-attention `Qwen3ForCausalLM`) | Required. Compare pin ratios only inside one SKU | B60: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) if host RAM is under ~48 GB, else [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). H200: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) if host RAM is at least ~256 GB, else [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B). Ascend 910B4: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B). | Call a synthetic MLP “Qwen3-4B”; plot Ascend 4B against H200 8B as the same run |
| **Qwen3.5** (hybrid Gated DeltaNet + full attention; hub cards are often multimodal) | Required. Different architecture, not a size ladder with Qwen3 | B60: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) or [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B). H200: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | Plot Qwen3-8B next to Qwen3.5-9B as “the same model, bigger”; use MoE/FP8 |

**Host RAM (ZeRO-3 param+optim, current harness without `zero.Init`):** plan **~27 bytes per parameter** (18 B/P states × 1.5 buffer), plus OS headroom — not “18× the checkpoint.” Qwen3-0.6B ~15 GiB; 1.7B ~43 GiB; 8B ~206 GiB. Raise `ulimit -l`. See [DeepSpeed memory](https://deepspeed.readthedocs.io/en/latest/memory.html).

**Software:** Qwen3.5 needs a Transformers that registers `qwen3_5` (newer than 4.57.6 in some installs). Prefer a **text-only** construction if `AutoModelForCausalLM.from_config` hits `Qwen3_5ForConditionalGeneration` + vision. Log transformers version.

**Fairness:** pin on/off only **within** one ID × SKU × batch × seq. One **Qwen3-0.6B** (or 1.7B if RAM allows) model-tensor command on B60 is the required PCIe anchor. Ascend Qwen3-4B is its own panel, not that anchor. Qwen3.5, including on B60, is required, own chart.

**Activation checkpoint workload:** `activation_offload/bench.py --model … --offload-optimizer --act-mem-opt` (checkpoint `use_reentrant=False`, mark hidden states, **bf16**, async on). ZeRO-2 CPU AdamW keeps parameters on GPU. Do not publish synthetic bars under a Qwen name. Do not publish a GPU-AdamW-only or `--no-act-mem-opt` arm as this workload.

**H2D/D2H:** still no HF model.

### Harnesses

**Model-tensor e2e (required):** [`deepspeedai/DeepSpeedExamples` `benchmarks/pin_memory`](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/pin_memory) — `model_tensor_offload/bench.py` (root `zero3_offload_bench.py` still delegates to ZeRO-3).

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

# Qwen3.5 required, own chart (no Liger; TiledMLP + tiled fused logits loss)
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

### Figure guidance

- **H200 headline numbers to plot (bound to GPU0's node, `job-20261024T171244Z`):** model-tensor ZeRO-3 seq 4096 **2.82×** (10.46 → 3.71 s); activation checkpoint seq 64K **1.15×** (30.68 → 26.78 s); saved-activation seq 64K **2.83×** (91.5 → 32.3 s). The previously quoted 3.40× / 1.19× / 3.80× were unbound runs on other days. Do not print them as current; at most cite them in a NUMA footnote. State the launch policy in the hardware table.
- Lead with **wall time**: model-tensor pin vs non-pin (unpinned=1.0), activation-checkpoint pin vs pageable with async on (pageable=1.0), and saved-activation pin vs pageable (pageable=1.0). Equal visual weight. Qwen3.5 figures are required; keep them off the Qwen3 chart.
- H2D/D2H is a supporting figure. Do not plot 55 GB/s H200 next to B60, or next to Ascend (~26 GB/s). On CUDA, do not give four equal bars to Torch vs native; pin vs pageable is the story, register-parity is an inset. Ascend’s DMA panel is native-registered vs pageable.
- Optional third activation panel: #8282 async vs blocking (pinned) — different axis.
- Hardware table: SKU, host CPU, interconnect, Torch/CUDA or PyTorch XPU, DeepSpeed commit ≥ #8283, `ulimit -l`.
- Label H200 ~55 vs ~10–17 GB/s as one PCIe Gen5 datapoint. Use the bound touched-buffer remeasure (`job-20261024T171244Z`, `numactl --cpunodebind=0 --preferred=0`, median of 3 reps): 256 MiB H2D 10.23 -> 55.39 GB/s (5.41x), D2H 17.31 -> 54.94 (3.17x). Unbound on the same host: 5.42x / 3.41x, with one outlier rep (pageable 8.16 / 14.94 GB/s). The 2026-10-23 unbound figures (5.47x / 3.35x, `job-20261023T131838Z`) agree with these within noise. The older 20.76 GB/s pageable H2D (2.67x) was a zero-page artifact and must not be printed.

---

## Claims that need evidence

- Pin vs non-pin wall time for **model-tensor and activation checkpoint** on **Intel Arc Pro B60**, H200, and **Ascend 910B4** (Qwen3-4B), on **Qwen3**.
- Pin vs non-pin wall time for **saved-activation** on H200, Qwen3-8B, `N=1`, `--offload-tensors -1`, seq 8K/16K/32K/64K (bound: 3.03× / 3.32× / 3.25× / 2.83×; unbound as published: 3.08× / 4.24× / 4.36× / 3.80×). B60: required, numbers pending. Ascend: Qwen3-4B, batch 1, seq 2048, 1.14× (9.48→8.34 s). Qwen3.5 is required, own chart.
- Ascend H2D/D2H: native-registered vs pageable at 256 MiB.
- Do not present synthetic MLP as a Qwen result.
- **Methodology (H200, jobs `20261024T022512Z`, `114211Z`, `171244Z`, `20261026T133233Z`, `20261027T132751Z`):** every published e2e row is measured bound to the accelerator's NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`, as the `run.sh` scripts do), with host buffers written before timing and at least 10 steps. Unbound runs on `numa_balancing=1` hosts are not reproducible for the pageable arm: same-host unbound re-runs differed from the published unbound figures by up to 0.7× (saved-activation 8K 3.60× vs 3.08×). Bound H200: model-tensor 2.82×, activation checkpoint 64K 1.15×, saved-activation 3.03× / 3.32× / 3.25× / 2.83× at 8K–64K; the published unbound figures were 3.40× / 1.19× / 3.08–4.36× / 3.80×. Pinned times barely change with placement; the pageable arm does. Record `numa_balancing` and the launch policy for every SKU. B60, Ascend, and Qwen3.5 rows have not been re-run bound.
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
### CPU offload for cost-efficiency and capability
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
### Qwen3.5 hybrid architecture (own chart)
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
```

**Writer notes (not headings):**

- **Answer first:** after the intro, two or three sentences plus a small decision hint (pin on unless host RAM/memlock is tight; measure on a new host link).
- **Article vs collection:** benches may be gathered ZeRO-first; the README still leads with all three applications’ wall time.
- **Frameworks:** three sibling H3s. Do not collapse into `get_accelerator().pin_memory()`. DataLoader pin is orientation only.
- **Registered host memory:** page-lock for the OS; register with the driver for the copy engine. `mlock` without register stays pageable-class for GPU copies.
- **Why pageable is slower (Background), per driver:** CUDA (H200): pageable copies stage through a driver-managed buffer; a probe found no per-copy `mlock`/`ioctl`/`mmap` difference against pinned, 2M pages do not help, the issuing thread blocks for the whole copy (27.9 ms for 256 MiB), and the staging work runs on CPU threads, so it is placement-sensitive. Ascend 910B4: the driver team reports that pageable memory is pinned and SMMU page tables are built on every copy, and that async copies become synchronous; that explains the page-size sensitivity but is not measured by us. B60: not characterized. Mark which statements are measured and which are reported.
- **Observability:** DeepSpeed-managed pin is not `torch.is_pinned()`. Report both on CUDA.
- **Numbers:** H200 / #8283 under supporting microbench; H200 / #8282 async vs blocking under the activation checkpoint workload as an overlap inset. New pin on/off numbers for that workload are the activation eval. Ascend 910B4 numbers stay on the Ascend panel. Qwen3 vs Qwen3.5 never share an unlabeled chart.
- **hipHostRegister / IPEX** are not DeepSpeed #8283 APIs.
- **Page size, H200 bound, all workloads (`job-20261027T132751Z`):** pin speedup moves ≤0.08x between 4K and 2M for model-tensor, activation checkpoint, saved-activation 8K–64K, H2D and D2H (table in EXPERIMENTS_HANDOFF §6). Use this for the page-size claim: huge pages are not a pinning substitute and not a pinning enhancer on H200. Staging probe: no per-copy pinning syscalls on CUDA. Ascend's page-size result is separate and unverified.
- **Host page size (huge pages), under “Host memory placement”:** do not present 2M pages as an alternative to, or an enhancer of, pinning. H200, bound (`job-20261027T132751Z`): 2M changes any arm by ≤4% and the pin speedup by ≤0.08× on every workload, and the staging probe found no per-copy pinning syscalls for pageable CUDA copies. The earlier unbound 2M gains (average −7% to −16%) were NUMA-balancing outliers and placement, not page size (`job-20261024T022512Z`, `032725Z`). Ascend 910B4: 2M pages bring pageable H2D near pinned in the microbench (~6.4 → ~23 GB/s), but that is unverified (see the Ascend note) and e2e is unmeasured. Say that, and that the mechanism differs by driver.
- **NUMA, under Best practices:** bind the process to the accelerator's NUMA node. Ascend: a remote node costs pinned D2H ~26%. H200: ≤6%. Measure per platform; do not port the number.
- **Host memory placement (practitioner tip, H200 measured):**
  - With `kernel.numa_balancing=1` the kernel samples and migrates the pages of every large host buffer each step. Pinning offload buffers does not prevent this, because DeepSpeed's CPU Adam state is ordinary pageable memory. On H200 (Qwen3-8B, 4K pages) it cost 1.6–6.8M minor faults and 10–70 CPU-s of kernel time per step, in pinned and pageable runs alike.
  - Any explicit task policy (bind or preferred) turns it off. `numactl --membind=all` does that but leaves CPU and memory placement to the scheduler, which swung the same command between 7.4 and 8.8 s/step.
  - Launch single-accelerator CPU-offload training under `numactl --cpunodebind=<accelerator node> --preferred=<accelerator node>`. H200 activation checkpoint, pageable: 7.2 s/step local vs 10.5 s with CPUs on the far node. Pinned is less sensitive (4.2 vs 5.5 s), which is itself an argument for pinning: staged pageable copies run on CPU threads and need them near the device. Model-tensor is placement-insensitive (3.76 vs 3.77 s pinned). For multi-rank runs, bind each rank to its own device's node.
  - Chart: faults/step and step time, unbound vs bound, pinned and pageable arms.

---

## Artifacts

- Layout: [`LAYOUT.md`](LAYOUT.md). Draft: `blogs/pinned-offload/08-2026/README.md` + `media/` figures (model-tensor, activation checkpoint, **and** saved-activation wall time, equal weight; microbench supporting).
- Repro notes: exact `model_tensor_offload/bench.py`, `activation_offload/bench.py`, `saved_activation_offload/bench.py`, and `h2d_d2h/bench.py` commands per SKU, including Ascend 910B4, with the NUMA launch policy.
- Hardware/software table for **measured** SKUs only.
- Link RTD only after lock-vs-register docs match #8283.

---

## Sequencing vs Phase 2

Collection order (hardware access) is not article order.

1. Pin experiments: [DeepSpeedExamples `benchmarks/pin_memory`](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/pin_memory) on DeepSpeed **`master`**. Land `activation_offload --model` before published HF rows.
2. Collect **model-tensor** Qwen3 + **H2D/D2H** on B60, then H200. Ascend 910B4 Qwen3-4B rows are already measured.
3. Collect the **activation checkpoint workload** (`--offload-optimizer --act-mem-opt`, async on, pin on/off) on B60 and H200. Ascend 910B4 Qwen3-4B seq 8192 is already measured. Named limitation if the B60 engine fails — do not invent numbers.
4. Collect **saved-activation** pin on/off on H200 (Qwen3-8B, `N=1`, `--offload-tensors -1`, seq 8K–64K: collected, jobs `20261011T171729Z` unbound, `20261026T133233Z` and `20261024T171244Z` bound; Qwen3.5-9B collected unbound, job `20261017T235002Z`). Ascend 910B4 Qwen3-4B seq 2048 is measured. B60 saved-activation is required and not yet collected. Qwen3.5 is required, own chart.
5. Re-run the non-H200 rows and Qwen3.5 bound, recording `numa_balancing` and the launch policy.
6. Draft README when the Qwen rows for B60 (including saved-activation), H200, and Ascend 910B4 are in place. Synthetic stays in an appendix or method note.
