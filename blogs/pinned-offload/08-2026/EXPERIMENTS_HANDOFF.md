# Handoff — Pin-memory DSE harnesses and blog experiments

**Audience:** agent owning DeepSpeedExamples benches and collecting numbers.  
**Do not** draft or rewrite [`BLOG_PLAN.md`](BLOG_PLAN.md) or `blogs/pinned-offload/08-2026/README.md`. Return JSON/tables/repro commands to the blog owner.

The activation checkpoint workload is `activation_offload/bench.py --offload-optimizer --act-mem-opt` with async on. Pin on/off moves the activation pool and the ZeRO-2 optimizer host buffers together. GPU-AdamW-only and `--no-act-mem-opt` arms are not this workload.

**GitHub account:** `sfc-gh-truwase` (fork + PRs).  
**DeepSpeed:** `deepspeedai/DeepSpeed` `master` (pin routing already merged: #8256, #8257, #8258, #8332, #8283).  
**Examples:** `deepspeedai/DeepSpeedExamples`.

---

## 0. Where the code is

| Item | Location |
|------|----------|
| Local clone | `/code/users/truwase/projects/deepspeed/sfc_ds_pin_memory/DeepSpeedExamples` (inside the DeepSpeed workspace; gitignored there) |
| Branch | `tjruwase/pin-memory-experiments` |
| Remotes | `origin` = `deepspeedai/DeepSpeedExamples`; `sfc-gh-truwase` = fork |
| PR status | #1015 (https://github.com/deepspeedai/DeepSpeedExamples/pull/1015) **merged 2026-09-30** — folder-split refactor only, not an open PR. Current branch `tjruwase/pin-memory-experiments` has no PR open yet; open a fresh one against `deepspeedai/DeepSpeedExamples` when ready to cite repro commands. |
| Harness root | `benchmarks/pin_memory/` |

HF `--model`, `--offload-optimizer`, `--act-mem-opt`, and offload-size reporting live in this nested clone. Push those to `sfc-gh-truwase` and open a **new** PR against `deepspeedai/DeepSpeedExamples` before citing repro commands in the blog — #1015 is already merged, do not try to reuse or reopen it.

---

## 1. Harnesses

Subfolders + `common.py` (subprocess arms, `ARMRESULT` / `DRIVERRESULT`):

| Path | Role |
|------|------|
| `model_tensor_offload/bench.py` | Required e2e: ZeRO CPU offload pin on/off. `--zero-stage 3` default; `1`/`2` optimizer-only. Also now supports `--act-mem-opt`/`--tiled-mlp`/`--gradient-checkpointing` (ported from `activation_offload/bench.py`'s wiring into `common.py`, default off) — see "H200 model-tensor offload + act-mem-opt" below. |
| `activation_offload/bench.py` | Activation checkpoint workload: `CheckpointHiddenStatesOffload`, `use_pin_memory` on/off, **`use_streams=True`**, HF `--model`, `--offload-optimizer --act-mem-opt`. |
| `h2d_d2h/bench.py` | Supporting GB/s: pageable, torch, native-unregistered, native-registered. `--skip-native-register` on XPU. Requires DSE `48de051`+ (writes host buffers before timing; reports THP coverage). Follow §2a for every SKU. |
| `grad_offload/bench.py` | Optional; execs model-tensor ZeRO-3 |
| `cpu_pin/bench.py` | Optional CPU-only native vs Torch |
| `deepcompile_activation/bench.py` | Optional `compile.offload_activation_pin_memory` |
| `zero3_offload_bench.py` | Compat shim → model-tensor `--zero-stage 3` |
| `saved_activation_offload/bench.py` | **Peer publish application (3)** ([`SAVED_ACTIVATION_OFFLOAD_PLAN.md`](SAVED_ACTIVATION_OFFLOAD_PLAN.md)): credit-bounded saved-tensor offload over the whole decoder layer, **no rematerialization**. Published command: `--offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2` (ZeRO-2 CPU AdamW; parameters stay on GPU; optimizer `pin_memory` follows the pin arm). Pass `--prefetch 2` explicitly: the bench default became 1 in `231034a` (2026-10-18), after the published rows were collected. H200 blocks publish: pin T/F, seq 8K–64K. Settings are justified in `BLOG_PLAN.md` (“Saved-activation settings, as measured”). Accelerator-agnostic via DeepSpeed’s accelerator abstraction (runs on CUDA, XPU, and Ascend NPU). Momentum-free SGD is diagnostic only. B70 is required. |

Pinned model-tensor arm: `DS_PIN_MEMORY_BACKEND=native`, `DS_PIN_MEMORY_REGISTER_DEVICE=1`. `--ablate-register` is CUDA-only inset, not the pin-vs-pageable story.

Native `mlock` needs `ulimit -l`.

Activation checkpoint pin/pageable toggles **both** the activation pool and ZeRO-2 optimizer host buffers. Label it a combined CPU-offload training workload, not an activation-copy-only ablation.

---

## 2. Experiments to collect

**Hardware (required):** **Intel Arc Pro B70** (stock PyTorch `xpu`, not IPEX), one high-end CUDA PCIe (H200), **Ascend 910B4**. GH200 and GB200 are optional.

**Models (required for published e2e):** Hugging Face **Qwen3** (dense `Qwen3ForCausalLM`). **Qwen3.5 is also required** (per `BLOG_PLAN.md`, 2026-10-18), always on its own chart. Instantiation may be config + random weights. Do **not** ship synthetic-only e2e. If a Qwen3.5 figure is included, do **not** plot it against Qwen3-8B as a size twin.

| SKU | Qwen3 (required) | Qwen3.5 (required, own chart) |
|-----|-------|---------|
| Intel Arc Pro B70 | `Qwen/Qwen3-0.6B` if host RAM is under ~48 GB, else `Qwen/Qwen3-1.7B` | `Qwen/Qwen3.5-0.8B` or `Qwen/Qwen3.5-2B` |
| High-end CUDA | `Qwen/Qwen3-8B` | `Qwen/Qwen3.5-9B` |
| Ascend 910B4 (29.5 GiB HBM usable) | `Qwen/Qwen3-4B`, batch 1, seq 1024 and seq 2048 | Qwen3.5-4B collected |
| GH200 (`galleon` GPU0) | Qwen3-8B, batch 1, seq 8192/16384/32768 for activation checkpoint and saved-activation. Model-tensor not yet run | optional |

H200 model-tensor used `deepspeed.zero.Init` (not the ~27 bytes/param no-Init planning figure). Measured CPU-resident offload: 8B 137.31 GB, 9B 150.10 GB. For B70 without `zero.Init`, plan ~27 bytes/param or use `zero.Init`. Raise `ulimit -l`.

**Qwen3.5 software:** Transformers that registers `qwen3_5` (newer than 4.57.6 on some boxes). Prefer **text-only** `from_config`. Log transformers version. No MoE/FP8. **Update (2026-10-16, `job-20261016T162911Z`):** liger-kernel `0.8.4` (torch214 env rebuild) now does attach to `qwen3_5_text` -- `apply_liger_kernel_to_qwen3_5` accepts `fused_linear_cross_entropy` (confirmed via `inspect.signature` in the job's VERIFY step), so the activation checkpoint workload gets the same `liger_qwen3_5` + `liger_fused_linear_ce` recipe as Qwen3-8B's `liger_qwen3` + `liger_fused_linear_ce`, not the TiledMLP + `TiledFusedLogitsLoss` fallback described below (stale as of liger-kernel 0.6.1/0.8.3).

**Still open:** XPU native pin may fail on B70 (named limitation). ZeRO-3 unpinned param-buffer validity unverified.

**Required rows**

1. Model-tensor: `python model_tensor_offload/bench.py --model Qwen/...` unpinned vs pinned on B70, H200, and Ascend 910B4 (GH200 optional). Synthetic `--hidden/--layers` is **smoke only**. B70: no `--ablate-register`. Ascend published model is Qwen3-4B.
2. Activation checkpoint workload: `python activation_offload/bench.py --model Qwen/Qwen3-... --offload-optimizer --act-mem-opt` pageable vs pinned, async on. Pitch: **longest demonstrated sequence length**. If B70 cannot run the engine, **named limitation** (commit + error) — do not invent numbers; still run Qwen3 model-tensor + H2D/D2H on B70. Do not label synthetic bars as Qwen.
3. H2D/D2H: `python h2d_d2h/bench.py` on H200, Ascend 910B4, and GH200; B70 with `--skip-native-register`. No HF model. Pageable vs pinned is the story; Torch vs native-registered is a CUDA parity inset. Ascend’s measured arm is native-registered vs pageable. Keep GH200 and Ascend off the H200 GB/s chart. **Follow §2a**: written host buffers (never `--no-touch`), NUMA-bound plus unbound runs, recorded THP mode and page backing; optional 4K-vs-2M page-size arms.
4. Saved-activation offload (peer publish application; H200 is required, GH200 optional): `python saved_activation_offload/bench.py --model Qwen/Qwen3-8B --offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2` on H200, pin T/F, seq 8K/16K/32K/64K (add `--discover-calib-seqs 4096 8192` at seq ≥32K). GH200 was measured at `N=2`; keep that label and do not copy the H200 ratio. ZeRO-2 CPU AdamW; pin toggles activation buffers and optimizer host buffers. Qwen3.5-9B is required (own chart). The script is accelerator-agnostic (DeepSpeed accelerator abstraction). B70 is required: Qwen3 (same ID as B70 model-tensor), batch 1, seq 2048, same flags, pin T/F; Qwen3.5-0.8B/2B own chart; named limitation (commit + error) only if it actually fails on XPU. Ascend 910B4 is required and collected (Qwen3-4B, batch 1, seq 2048, 1.14×). Full details in `SAVED_ACTIVATION_OFFLOAD_PLAN.md` §10.
5. Ascend 910B4 (required, already measured): model-tensor Qwen3-4B batch 1 seq 1024 and seq 2048; activation checkpoint workload Qwen3-4B batch 1 seq 8192; H2D/D2H native-registered vs pageable at 256 MiB. Qwen3.5-4B model-tensor is collected.
6. GH200 (`ds-pin-Xinyu-experiment.pdf`, `galleon` GPU0, node 0, Examples `5b16fe4`, October 2026): H2D/D2H, activation checkpoint, and saved-activation are measured. Model-tensor is not in the PDF; GH200 is optional so it does not block. Do not copy PCIe or H200 numbers. The PDF’s DeepNVMe figure is out of scope.

**Do not** reuse DeepSpeed #8282 async-vs-blocking (pinned pool) as pin vs pageable. **Do not** reuse #8283 H200 GB/s as the only H2D/D2H figure if you can remeasure with the pageable arm.

**Optional:** GB200, model-tensor `--zero-stage 1`/`2` (optimizer-only; distinct from the activation checkpoint workload), `grad_offload/`, `cpu_pin/`, `deepcompile_activation/`, model-tensor `--ablate-register`.

**Out of scope:** NVMe, DataLoader pin, blog prose.

---

## 2a. Host-memory measurement guidance for a new SKU (first write, huge pages, NUMA)

This section stands alone. Follow it on any accelerator (CUDA, XPU, Ascend NPU, C2C) before publishing a pin-vs-pageable number. It comes from the H200 study of 2026-10-23/24 (`job-20261023T131838Z`, `job-20261023T145721Z`, `job-20261023T221332Z`, `job-20261024T022512Z`; results in §6, “H200 host page size (transparent huge pages) and NUMA” and “H200 automatic NUMA balancing”). Use the H200 numbers only to compare against. Do not copy them onto another SKU's panel.

### Why this matters

1. **First write.** A host buffer that is allocated but never written is backed by the kernel's shared zero page. A pageable H2D copy from it reads that one page over and over and looks far faster than real data. On H200, 256 MiB pageable H2D measured 20.8 GB/s untouched vs 10.1 GB/s written, so the published pin speedup was 2.7x when the true value is 5.5x. D2H writes the host buffer, so it is not affected.
2. **Host page size.** Pageable copy cost can depend on whether host pages are 4 KiB or 2 MiB (transparent huge pages, THP). On Ascend 910B4, 4K pages give ~6.4 GB/s and 2M pages ~23 GB/s, close to pinned (microbench). On H200 there is no bandwidth effect, because CUDA stages pageable copies through a driver buffer. In e2e training on H200, 2M pages mostly removed slow outlier steps. Those outliers turned out to be automatic NUMA balancing (item 4), which 2M pages make ~500x cheaper. Separately, the activation checkpoint pageable arm keeps a ~1.3-1.5 s backward-phase gain at 2M that is not fault-related (open, §6). Every SKU must be measured; the effect does not transfer between SKUs. Every SKU must be measured; the effect does not transfer between SKUs.
3. **NUMA placement.** A host buffer on a NUMA node remote from the device can cost a lot: Ascend lost 26% on pinned D2H, H200 at most 6%. Record which node you used.
4. **Automatic NUMA balancing (`kernel.numa_balancing=1`).** On multi-socket hosts the kernel periodically unmaps the pages of long-lived host buffers to sample access, then migrates them. Each re-touch is a minor "hinting" fault. On H200 at 4K pages this was **all** of the steady-state minor faults: 1.6-6.8M per step, in both pinned and pageable arms, in backward and in the CPU Adam step, with 10-70 CPU-s of sys time per step. It makes step times noisy and can shift results by several percent. The DeepSpeed CPU Adam state (`exp_avg`/`exp_avg_sq`, `torch.zeros_like` on CPU) is pageable in **both** arms, so it is hit even when offload buffers are pinned. Opt out per process with `numactl --membind=all` (Rule 2).

### Required setup

- **Harness version.** `deepspeedai/DeepSpeedExamples`-layout clone at `sfc-gh-truwase/DeepSpeedExamples`, branch `tjruwase/pin-memory-experiments`, commit **`48de051` or later**. Check with `python benchmarks/pin_memory/h2d_d2h/bench.py --help | grep -- --no-touch`. If the flag is missing, the bench predates the first-write fix, and its pageable H2D numbers are not publishable.
- **Record the host environment** alongside every result:
  ```bash
  cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
  touch /sys/kernel/mm/transparent_hugepage/enabled 2>&1   # read-only in most pods
  uname -r; ldd --version | head -1; ulimit -l; numactl -H | head -4
  cat /proc/sys/kernel/numa_balancing                       # 1 = automatic NUMA balancing on
  python -c "import torch; print(torch.__version__)"
  ```
  Also record the driver/runtime version, the device-to-NUMA mapping, and the host link (PCIe gen/width or C2C). For the NUMA mapping, read `cat /sys/bus/pci/devices/<device-BDF>/numa_node` on any accelerator; on CUDA, `nvidia-smi topo -m` also works.

### Rule 1 — write host buffers before timing (always)

- `h2d_d2h/bench.py` does this by default from `48de051`. Every result row has `"touched": true`.
- **Never publish a run made with `--no-touch`.** That flag only exists to reproduce the old artifact for comparison.
- In any custom script, `fill_()` (or otherwise write) every host buffer before the first timed copy.
- **Re-check old numbers.** Pageable H2D on this SKU measured with a bench older than `48de051` must be re-measured or confirmed to have used a written buffer. A pageable H2D that comes out higher than pageable D2H at large sizes is a warning sign; on H200 the artifact showed exactly that pattern.

### Rule 2 — bind to the device's NUMA node, and report it

- Microbench: run with `numactl --cpunodebind=<N> --membind=<N>`, where `<N>` is the device's node. Also run once unbound, so the number is comparable to unbound history.
- E2E: match how the published rows were run. If the published rows were unbound, run unbound and record the node the allocations landed on. Do not mix bound and unbound numbers in one chart.
- E2E: run under `numactl --cpunodebind=<dev node> --preferred=<dev node>`. The cuda `run.sh` scripts do this from DSE `31db0b7`; npu needs `NUMA_NODE=<n>`; `NUMA_BIND=0` opts out. Any explicit task policy (bind or preferred) also opts the process out of NUMA balancing. On H200, CPU threads or memory on the node remote from the GPU cost 1.1-3.3 s/step on activation checkpoint offload. `numactl --membind=all` disables balancing but leaves placement to the scheduler, which swung the same command between 7.38 and 8.82 s/step; use it only when the device node is unknown. Arm subprocesses inherit it; check that `/proc/<arm pid>/numa_maps` shows `bind:`. Report which mode you used. If you also want the balancing-on number, run both, back to back.

### Rule 3 — page-size conditions (if you measure huge pages)

Measure two conditions in a 2x2 design (page size x pin), back to back in the same job:

| condition | how to request it |
|---|---|
| `4k` (default) | nothing; confirm THP is `madvise` or `never`, so ordinary allocations land on 4K pages |
| `2m` | `THP_MEM_ALLOC_ENABLE=1 GLIBC_TUNABLES=glibc.malloc.hugetlb=1` in the environment of every process, both driver and arms |

Each host allocation path has its own switch:

| allocation path | used by | made 2M by |
|---|---|---|
| PyTorch CPU tensors (`torch.empty`/`zeros`, c10 `alloc_cpu`) | pageable arms; ZeRO and offload host buffers | `THP_MEM_ALLOC_ENABLE=1` (check that your torch supports it: `grep -c THP_MEM_ALLOC_ENABLE $(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib/libc10.so` must be ≥1) |
| DeepSpeed native pin (DeepNVMe `posix_memalign`) | native pinned arms | `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (glibc ≥ 2.35) |
| Accelerator-runtime pinned alloc (`cudaHostAlloc`, `torch.Tensor.pin_memory()`) | `torch` arm of `h2d_d2h` | nothing changes it; it is the control (expect 0%) |

- **Always set both variables for `2m`.** If only the PyTorch path gets 2M pages, the pageable arm also gains CPU-side speed (fewer TLB misses and page faults), and that gain would be misread as a transfer effect.
- If `/sys/kernel/mm/transparent_hugepage/enabled` is writable and you have root, `always` is an alternative. Record which method you used.
- Do not use hugetlbfs (`MAP_HUGETLB`, `HugePages_Total`) for this; the Ascend result and this method use THP.

### Rule 4 — verify the page backing; a null result is meaningless otherwise

- **Microbench:** each `h2d_d2h` result row reports `host_thp_fraction_before`, `host_thp_fraction_after`, `thp_mem_alloc_enable`, and `glibc_tunables`. Accept the run only if the pageable and native arms show ≥0.90 at `2m` and ≈0 at `4k`. Sizes below 2 MiB cannot be fully THP-backed; report them, but do not gate on them.
- **E2E:** sample every arm process during the run. Read `Anonymous` and `AnonHugePages` from `/proc/<pid>/smaps_rollup`, and `minflt`/`majflt` from fields 10/12 of `/proc/<pid>/stat`, every 5 s. Accept `2m` only if `AnonHugePages / Anonymous` ≥ 0.90 for the arm processes.
- The H200 study's helper scripts do exactly this. They live in `autorun/hugepage/` of the `sfc_ds_pin_memory` workspace, not in the DSE repo; copy them or reimplement them:
  - `thp_probe.py`: per-allocation-path coverage gate.
  - `thp_sampler.py`: per-process coverage and fault sampler for e2e.
  - `overlap_probe.py`: copy concurrent with a GEMM. It uses `torch.cuda` directly; port it to `get_accelerator()` streams and events on non-CUDA devices.

### Rule 5 — e2e timing hygiene

- Use `--steps 10` or more (the activation and model-tensor benches default to 4, which is too noisy for a few-percent effect).
- Repeat each condition at least twice, interleaving `4k`/`2m`.
- **Report both average and minimum step time per arm.** On H200 the 2M effect was almost entirely in the gap between average and minimum, i.e. it removed slow outlier steps; reporting only the average overstated it.
- Report pin speedup within each page condition. Never compare pageable at 2M against pinned at 4K.
- To attribute a step-time difference, set `PIN_BENCH_STEP_PROBE=<dir>` (DSE `152e17c`). Per step and per phase (fwd/bwd/opt/sync) it logs wall time, minor faults, user/sys CPU, faults by thread name, and host `numa_hint_faults`/`numa_pages_migrated`. Summarize with `autorun/hugepage/analyze_steps.py`. If per-step faults ≈ `numa_hint_faults + numa_pages_migrated`, the noise is NUMA balancing; rerun bound.

### Commands (adapt model, sizes, and NUMA node to the SKU)

```bash
B=DeepSpeedExamples/benchmarks/pin_memory
N=<device NUMA node>
# microbench, 3 reps, interleaved; add --skip-native-register on XPU
for rep in 1 2 3; do for c in 4k 2m; do
  E="THP_CONDITION=$c"; [ $c = 2m ] && E="$E THP_MEM_ALLOC_ENABLE=1 GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
  env $E numactl --cpunodebind=$N --membind=$N python $B/h2d_d2h/bench.py --sizes-mib 4 64 256 1024
  env $E python $B/h2d_d2h/bench.py --sizes-mib 4 64 256 1024          # unbound
done; done
# e2e example (published SKU shape, e.g. Ascend Qwen3-4B), each condition twice, interleaved
env $E numactl --cpunodebind=$N --preferred=$N python $B/model_tensor_offload/bench.py --model Qwen/Qwen3-4B --batch 1 --seq 2048 --zero-stage 3 --steps 10
```

### What to report per SKU

| field | example |
|---|---|
| SKU, host CPU, link | Ascend 910B4, Kunpeng-920, PCIe Gen4 x16 |
| THP mode, `/sys` writable, kernel, glibc, torch, runtime | `madvise`, no, 6.x, 2.39, 2.14.0, CANN x.y |
| harness commit, `touched` | `48de051`, true |
| NUMA node used, bound/unbound | node 0, bound |
| `numa_balancing`, e2e launch policy | 1, `--cpunodebind=0 --preferred=0` |
| per arm x size x condition: H2D/D2H GB/s (median and min-max of reps), `host_thp_fraction` | pageable 256 MiB 4k: 6.4 / …, 0.0 |
| e2e per arm x condition: avg and min step, pin speedup, THP coverage, minflt | … |

### How to read the result

- **2M pageable bandwidth within ~10% of 4K pageable:** page size does not matter for this SKU's DMA path (the H200 outcome).
- **2M pageable reaches ≥70% of pinned:** the SKU's pageable path is page-count bound (the Ascend outcome). Run e2e before claiming anything; the microbench alone does not show a training benefit.
- **Pinned arms move between 4K and 2M:** the effect is CPU-side, not transfer. Say so; do not credit it to DMA.
- **Huge pages are never presented as a replacement for pinning** unless e2e pin speedup at 2M is within noise of 1.0x on that SKU.

---

## 3. Suggested agent sequence

1. Push nested-clone harness (HF `--model`, activation checkpoint workload flags, offload size) to `sfc-gh-truwase` and open a new PR against `deepspeedai/DeepSpeedExamples` — #1015 is already merged, not a target to push onto.
2. Collect B70: model-tensor + H2D/D2H (`--skip-native-register`; no `--ablate-register`) + the activation checkpoint workload + saved-activation (smaller IDs from §2; saved-activation batch 1 × seq 2048). Log DeepSpeed commit ≥ #8283, transformers, SKU, `ulimit -l`, Torch version. Named limitation if activation engine or Qwen3.5 fails.
3a. Ascend 910B4 rows below are already measured. GH200 H2D/D2H, activation checkpoint, and saved-activation are in `ds-pin-Xinyu-experiment.pdf`. GH200 model-tensor is optional.
4. Drop `DRIVERRESULT=` JSON (and H2D/D2H CSV) in a results dir **outside** the blog README unless the blog owner asks to copy figures.
5. Hand numbers back. Do not start `blogs/pinned-offload/08-2026/README.md`.

---

## 4. DeepSpeed leftovers (only if they block a harness)

Routing for ZeRO/FPDT/writer is on `master`. Not a mega-PR. Ignore `tjruwase/pin-memory-route-bypass` WIP.

Only touch DeepSpeed if a bench cannot run (e.g. activation engine missing on XPU — document, do not silently skip Intel model-tensor).

Stale RTD (`docs/code-docs/source/memory.rst` still describing native as mlock-only) is a **docs** fix for a later DeepSpeed PR, not this DSE PR.

---

## 5. Rules

- Commits: `--signoff`. Push DSE to **`sfc-gh-truwase`**. **No PRs or pushes to `deepspeedai/*` until the blog is completed** (owner instruction, 2026-10-23).
- Host-memory measurements follow §2a: written buffers, NUMA recorded, THP backing verified.
- Do not update `git config`.
- Do not commit the nested clone into the DeepSpeed repo (`/DeepSpeedExamples/` is gitignored).

---

## 6. Blog results checklist (2026-09-26)

**H200 box:** 8×H200 node `tunji-h200-n1g8-ds-pin-0`, **one GPU** (`WORLD_SIZE=1`). Torch `2.14.0+cu130`, transformers `5.16.0.dev0`, flash-attn `2.8.3`, `ulimit -l unlimited`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, DeepSpeed on `PYTHONPATH`. Logs: `autorun/logs/job-<id>.log`. Liger `0.8.3` applied only to the activation checkpoint workload on Qwen3-8B (not model-tensor, H2D/D2H, or Qwen3.5-9B text) -- **superseded 2026-10-16**: the torch214 env's liger-kernel `0.8.4` now also attaches to the activation checkpoint workload on Qwen3.5-9B text (`job-20261016T162911Z`); model-tensor and H2D/D2H remain unaffected.

**Publish bar:** Arc Pro B70 + H200 + **Ascend 910B4** (GH200 optional as of 2026-10-20) for model-tensor and activation checkpoint on **Qwen3**; saved-activation blocks on H200 Qwen3-8B (GH200 optional), on B70, and on Ascend 910B4 (collected). **Qwen3.5 is a publish gate, own chart** (H200 Qwen3.5-9B rows all collected). GeForce was dropped from the blog (2026-10-20). **Not met** — H200 Qwen3 rows, Ascend 910B4 Qwen3-4B rows, and GH200 H2D/D2H, activation checkpoint, and saved-activation rows are collected. B70 model-tensor is missing (GH200 model-tensor is optional). **Note (2026-10-11):** the H200 saved-activation row below was collected under the old narrow (MLP-only) `--offload-modules` scope and the old `--offload-tensors` 1/2/3 cap; the scope was since widened to the whole decoder layer (see `SAVED_ACTIVATION_OFFLOAD_PLAN.md` §6) and K-capping was found to be a negative result. The widened-scope `--credits`=1 data (`job-20261011T171729Z`) is now the dataset of record for this row; **confirmed 2026-10-20** as the published H200 figure, and `BLOG_PLAN.md` now specifies these settings (`N=1`, `M=2`, `--offload-tensors -1`), so this row is met.

### H200 workloads to be published (summary table)

| Workload | Description | Model & seqlen | Pinned speedup |
|---|---|---|---|
| **(1) Param-optimizer offload** | ZeRO-3 parameter + optimizer CPU offload (`model_tensor_offload/bench.py`) | Qwen3-8B, batch 1 x seq 4096 | **3.40x** (13.583->3.998s, 302->1025 tok/s)^1 |
| **(2) Activation checkpoint workload** | ZeRO-2 CPU AdamW optimizer offload + activation checkpointing - Liger fused-linear-CE + TiledMLP + fused CE | Qwen3-8B, batch 1 x seq 64K (65536) | **1.19x** (29.990->25.303s, 44.66 GB) |
| **(3) Saved-activation offload** | Credit-bounded saved-tensor-hook offload, no rematerialization, widened whole-decoder-layer scope, `--credits 1`, ZeRO-2 CPU AdamW | Qwen3-8B, batch 1 x seq 64K (65536) | **3.80x** (125.345->33.002s) |
| **(4) H2D/D2H** | Host<->device transfer microbenchmark, native-registered pin vs. pageable (256 MiB transfers) | n/a (no model) | H2D **5.47x** (10.13->55.40 GB/s), D2H **3.35x** (16.38->54.93 GB/s)^2 |

^2 Remeasured 2026-10-23 (`job-20261023T131838Z`, median of 3 reps, unbound like the original). The earlier 2.67x (20.76 GB/s pageable H2D, `job-20260925T211614Z`) was a measurement artifact: the bench never wrote the pageable buffer, so H2D read the kernel's shared zero page. Re-running the old untouched behavior reproduces it (20.39-20.86 GB/s); with the buffer written, pageable H2D is 10.13 GB/s. D2H writes the host buffer, so it was not affected (15.35 then, 16.38 now).

^1 The published 3.40x is a historical snapshot (`job-20260928T140704Z`); 2026-10-12 re-runs on the same node/config land ~2.8-2.96x due to day-to-day node drift, not a regression - see the regression-check section below.

### Ascend 910B4 workloads collected (summary table)

| Workload | Description | Model & seqlens | Pinned speedup |
|---|---|---|---|
| Model-tensor | ZeRO-3 CPU param+optim offload | Qwen3-4B (4.0B), bs1 x seq1024 | **1.85x** (21.26->11.48 s, 48->89 tok/s) |
| Model-tensor | ZeRO-3 CPU param+optim offload | Qwen3-4B (4.0B), bs1 x seq2048 | **1.47x** (19.32->13.16 s, 106->156 tok/s) |
| Model-tensor | ZeRO-3 CPU param+optim offload | Qwen3.5-4B (4.2B), bs1 x seq512 | **1.32x** (14.45->10.95 s, 35->47 tok/s) |
| Model-tensor | ZeRO-3 CPU param+optim offload | Qwen3.5-4B (4.2B), bs1 x seq1024 | **1.29x** (17.95->13.95 s, 57->73 tok/s) |
| Activation checkpoint | ZeRO-2 CPU AdamW optimizer offload + activation checkpointing - Liger fused-linear-CE + TiledMLP + fused CE (recipe C) | Qwen3-4B (4.0B), bs1 x seq8192 | **1.23x** (15.44->12.60 s) |
| Activation checkpoint | ZeRO-2 CPU AdamW optimizer offload + activation checkpointing - Liger fused-linear-CE + TiledMLP + fused CE (recipe C) | Qwen3.5-4B (4.2B), bs1 x seq8192 | **1.08x** (92.58->85.85 s) |
| Saved-activation offload | Credit-bounded saved-tensor-hook offload, no rematerialization, widened whole-decoder-layer scope, `--credits 1`, ZeRO-2 CPU AdamW | Qwen3-4B, batch 1 x seq 2048 | **1.14x** (9.48->8.34 s) |
| H2D (host -> device) | native-registered vs pageable | 256 MiB | **4.01x** (6.48->25.97 GB/s) |
| D2H (device -> host) | native-registered vs pageable | 256 MiB | **3.95x** (6.71->26.49 GB/s) |

Environment: single Ascend 910B4, **29.5 GiB HBM** (32 GiB physical, ~2.5 GiB firmware-reserved); Kunpeng-920 host, 42-core container CPU quota.

**Resolved (2026-10-21):** saved-activation offload is accelerator-agnostic in DeepSpeedExamples (DeepSpeed accelerator abstraction), so it runs on CUDA, XPU, and Ascend NPU. Ascend 910B4 (1.14×, Qwen3-4B batch 1 × seq 2048) is a required, collected row, and B70 saved-activation is now required. `BLOG_PLAN.md` is updated; `LAYOUT.md` may still carry the old CUDA-only language.

### H200 headline results (quick reference)

| Workload | Key config | Result |
|---|---|---|
| **Model-tensor offload** (Qwen3-8B) | ZeRO-3 param+optim CPU offload, batch 1 x seq 4096 | **3.40x** (13.583->3.998 s), 302->1025 tok/s (2026-10-12 re-runs land ~2.8-2.96x — see note below) |
| **Activation checkpoint workload** (Qwen3-8B) | ZeRO-2 CPU AdamW (`--offload-optimizer`, optimizer-only, params on GPU), Liger + TiledMLP + fused linear CE, batch 1, seq 64K | **1.19x** (29.990->25.303 s), 44.66 GB |
| **H2D/D2H microbench** | 256 MiB transfers, native-registered pin vs pageable | H2D **5.47x** (10.13->55.40 GB/s), D2H **3.35x** (16.38->54.93 GB/s) (`job-20261023T131838Z`, host buffer written before timing; supersedes the 2.67x zero-page artifact) |
| **Saved-activation offload** (Qwen3-8B) | Widened scope, `--offload-tensors -1`, `--credits 1`, ZeRO-2 CPU AdamW, batch 1, seq 64K | **3.80x** (125.345->33.002 s) |

Full evidence, job IDs, and per-seqlen breakdowns for each row are in the tables below.

### Required (blocks publish)

| Row | Status | Evidence |
|-----|--------|----------|
| Model-tensor Qwen3, B70 | missing | — |
| Activation checkpoint workload, Qwen3, B70 (or engine miss: commit + error) | missing | — |
| H2D/D2H B70 `--skip-native-register` | missing | — |
| Saved-activation, Qwen3, B70 (or engine miss: commit + error) | missing | Same ID as B70 model-tensor, batch 1 × seq 2048, `--offload-optimizer --act-mem-opt --offload-tensors -1 --credits 1 --prefetch 2`. Required as of 2026-10-21. |
| Model-tensor Qwen3-8B, H200 | **have (headline, per BLOG_PLAN.md's required-SKU table)** | `job-20260928T140704Z` (steps=10, cleaner sample; supersedes steps=4 `job-20260928T131344Z`). ZeRO-3 param+optim CPU offload, batch 1 × seq 4096 (realistic context, supersedes the earlier seq-512 shape). 13.583→3.998 s (**3.40×**, 302→1025 tok/s), CPU 137.31 GB, GPU 38.88 GB. Speedup is stable/mildly increasing across seqlen — same job also ran seq 2048 (2.90×, steps=10); earlier steps=4 runs (seq 4096 3.26×, seq 2048 2.82×, and the original `job-20260925T223201Z` seq 512 2.89×) land in the same ~2.8–3.4× band. Pinned step time stays ~flat (~4.0 s) across shapes while unpinned scales with seqlen, consistent with the gap being driven by per-layer weight-gather transfer, not compute. Re-run on 2026-10-12 (no new flags) landed at ~2.8-2.96x instead of 3.40x — isolated to environmental drift on the shared node, not a code regression (see new section below), so 3.40x is kept here as a valid historical snapshot. |
| Model-tensor Qwen3-4B/8B + act-mem-opt, H200 | **have (secondary capability aside, not headline)** | New capability: `--act-mem-opt`/`--tiled-mlp`/`--gradient-checkpointing` added to `model_tensor_offload/bench.py` (default off; existing invocations unaffected). seq=4096 and seq=65536 sweep across Qwen3-4B/8B in "H200 model-tensor offload + act-mem-opt" below. Qwen3-8B stays the required/headline model-tensor figure per BLOG_PLAN.md's required-SKU table; Qwen3-32B was measured (see memory for the raw numbers) but dropped from blog messaging. |
| Activation checkpoint workload, Qwen3-8B, H200 | **have** | `job-20260926T165140Z` exit 0. Pitch: **64K capability**. Batch 1. Liger + TiledMLP + fused linear CE. At 64K: 29.990 → 25.303 s (**1.19×**, 44.66 GB, 17.5 GB act DMA). |
| H2D/D2H H200 | **have (remeasured 2026-10-23)** | `job-20261023T131838Z`, 4/64/256/1024 MiB, 3 reps, host buffer written before timing (`h2d_d2h/bench.py` now does this by default; `--no-touch` reproduces the old behavior). 256 MiB, unbound, H2D / D2H GB/s: pageable **10.13 / 16.38**; torch 55.40 / 54.92; native-unregistered 10.15 / 16.07; native-registered **55.40 / 54.93**. Pageable H2D depends on size: ~16 GB/s at 4-64 MiB, ~10 GB/s at 256 MiB and above. Native-unregistered (mlock only) is pageable-class at every size. **Superseded:** `job-20260925T211614Z` reported pageable H2D 20.76, read from the zero page because the buffer was never written. That also produced the false "native-unregistered H2D 9.72 is worse than pageable" note. |
| Saved-activation, Qwen3-8B, H200, widened-scope credits=1 | **have (dataset of record, confirmed 2026-10-20)** | `job-20261011T171729Z`. **Widened whole-decoder-layer scope** (`--offload-modules` default; supersedes narrow MLP-only scope), `--offload-tensors -1` (unlimited; K=1/2/3 capping found to be a negative result — raises peak HBM via always-resident fallback), ZeRO-2 CPU AdamW (`--offload-optimizer`), seq 8192/16384/32768/65536. `--credits 1` ablation: 8-10% HBM savings at seq≥32768 vs `--credits 2`, negligible step-time cost (±2.3%). seq≥32768 uses seq-extrapolated calibrated discovery (`--discover-calib-seqs 4096 8192`). Explicit decision: stop the seqlen sweep at seq64k. |
| Model-tensor, Qwen3.5-9B, H200 | **have (required, own chart)** | `job-20261016T193445Z` (repeated in `20261017T235002Z`), batch 1 × seq 4096, `fla` installed. 13.718 → 4.507 s (**3.04×**), offload 150.10 GB, GPU peak 53.68 GB. The pre-`fla` `20261016T141158Z` number (3.00×, 77.48 GB) is superseded. |
| Activation checkpoint workload, Qwen3.5-9B, H200 | **have (required, own chart)** | `job-20261016T193445Z` (`fla` rerun). 8K **1.85×** (8.661→4.675 s, 33.33 GB), 16K **1.80×** (10.661→5.926 s, 35.48 GB), 32K **1.52×** (13.340→8.763 s, 39.77 GB), 64K **1.25×** (20.807→16.671 s, 55.71 GB). Do not use `20261016T162911Z` (pre-`fla`, about 9× slower). |
| Saved-activation, Qwen3.5-9B, H200 | **have (required, own chart)** | `job-20261017T235002Z`, same settings as the Qwen3-8B row (widened scope, `--offload-tensors -1`, `--credits 1`, `--prefetch` 2 = the default at the time). Pin speedup / peak HBM reserved / pinned median: 2K **3.68×** / ~55.7 GB / 3.00 s; 4K **2.73×** / ~57.6 GB / 5.12 s; 8K **3.19×** / ~61.4 GB / 6.54 s; 16K **4.30×** / ~69.0 GB / 9.58 s; 32K **5.14×** / ~85.1 GB / 15.97 s; 64K **7.93×** / 138.92 GB / 30.10 s. Supersedes `20261016T141158Z` and `20261017T153604Z` (both OOMed at 16K before the discovery-calibration threshold dropped to seq ≥16K). |
| Saved-activation `--prefetch 1` ablation, H200, seq 64K | have (not the headline) | Qwen3-8B `job-20261018T110757Z`: 3.80× → 4.28×, pinned 33.00 → 37.06 s (+12.3%), HBM 125.10 → 97.88 GB (−22%). Qwen3.5-9B `job-20261018T062030Z` (duplicate `080244Z`): 7.93× → 5.92×, pinned 30.10 → 31.41 / 33.15 s (+4–10%), HBM 138.92 → 98.43 GB (−29%). M=1 is a memory-headroom lever with a latency cost, so the published rows keep M=2. |
| Dynamic `offload_states()`/`reload_states()`, Qwen3.5-9B, 8×H200 | have (extra, not in `BLOG_PLAN.md`) | `job-20261019T134229Z`, `dynamic_tensor_offload/bench.py`, real `torchrun` world_size=8, ZeRO-3, batch 4 × seq 512, offloads hp_params + optim_states only. 12.51 GB/GPU offloaded, GPU peak 46.30 GB. Transfer (reload+offload) 10.842 → 5.029 s (**2.16×**). Unlike every other row here it truly shards; others run `WORLD_SIZE=1`. |
| Saved-activation, Qwen3-8B, H200, K=1/2/3 (narrow scope, superseded) | historical | `job-20261007T215537Z`. Old narrow (MLP-only) `--offload-modules` scope, old K=1/2/3 cap. ZeRO-2 CPU AdamW (`--offload-optimizer`), seq 4096. K=1/2/3: 2.00× (7.222→3.616 s) / 2.09× (7.722→3.697 s) / 2.20× (8.268→3.763 s). Pin also pins the optimizer host buffers. Qwen3.5-9B from the same job is optional, not a gate. Kept for history only; not the figure to publish. |
| Model-tensor Qwen3-4B, Ascend 910B4 | **have** | Single Ascend 910B4, 29.5 GiB HBM usable (32 GiB physical). ZeRO-3 CPU param+optim. Batch 1 × seq 1024: **1.85×** (21.26→11.48 s, 48→89 tok/s). Batch 1 × seq 2048: **1.47×** (19.32→13.16 s, 106→156 tok/s). Source table has no job id. Seq 2048 unpinned step time is lower than seq 1024; report the table as measured. |
| Activation checkpoint workload, Qwen3-4B, Ascend 910B4 | **have** | ZeRO-2 CPU AdamW + Liger/TiledMLP. Batch 1 × seq 8192: **1.23×** (15.44→12.60 s). |
| Activation checkpoint workload, Qwen3.5-4B, Ascend 910B4 | have (required, own chart) | ZeRO-2 CPU AdamW + Liger fused-linear-CE + TiledMLP + fused CE (recipe C). Batch 1 × seq 8192: **1.08×** (92.58→85.85 s). Different architecture from Qwen3-4B (table labels 4.2B vs 4.0B). Keep off the Qwen3 chart. |
| H2D/D2H, Ascend 910B4 | **have** | 256 MiB, native-registered vs pageable. H2D **4.01×** (6.48→25.97 GB/s). D2H **3.95×** (6.71→26.49 GB/s). Not `cudaHostRegister` and not the B70 Torch/XPU arm. Do not plot next to H200. |
| Saved-activation, Qwen3-4B, Ascend 910B4 | **have (2026-10-16, supersedes prior named-limitation status)** | Credit-bounded saved-tensor-hook offload, no rematerialization, widened whole-decoder-layer scope, `--credits 1`, ZeRO-2 CPU AdamW. Batch 1 x seq 2048: **1.14x** (9.48->8.34 s). Source table has no job id. |
| Model-tensor, GH200 | missing | Not in `ds-pin-Xinyu-experiment.pdf`. Optional as of 2026-10-20; does not block publish. |
| H2D/D2H, GH200 | **have** | `galleon` GPU0, node 0, 256 MiB chart (`ds-pin-Xinyu-experiment.pdf` p.18). H2D GB/s pageable / native-unregistered / torch / native-registered: 4 MiB 196 / 189 / 262 / 294; 64 MiB 360 / 337 / 370 / 355; 256 MiB 366 / 358 / 383 / 357. D2H: 4 MiB 180 / 181 / 263 / 263; 64 MiB 284 / 285 / 295 / 295; 256 MiB 293 / 294 / 297 / 297. Large copies differ by under 4%. At 4 MiB, pinned is 1.3–1.5×. mlock-only tracks pageable. Separate nsys point (p.19, 256 MiB): H2D pageable/pinned 351/350, D2H 296/297; single-thread CPU copy 14.6 GB/s; pageable sometimes stalls 4–20 ms. Do not plot next to H200. |
| Activation checkpoint, Qwen3-8B, GH200 | **have** | Slide label Recipe C + ZeRO-2 CPU AdamW. Batch 1. One arm each. The slide does not state Liger or `--act-mem-opt`. 8K: 3.56→3.40 s (**1.05×**), HBM 27.3 GB both. 16K: 5.29→5.29 s (**1.00×**), 29.8 GB. 32K: 10.43→10.29 s (**1.01×**), 34.6 GB. |
| Saved-activation, Qwen3-8B, GH200 | **have** | K=-1, N=2, ZeRO-2 CPU AdamW, batch 1. Median of two arms each. 8K: 3.52→3.08 s (**1.14×**), HBM 30.9→32.6 GB. 16K: 4.93→4.38 s (**1.13×**), 38.1→41.5 GB. 32K: 9.38→8.05 s (**1.17×**), 57.4→64.2 GB. No-offload at 8K: 3.13 s, 88.6 GB. Pinned HBM is not lower than pageable at 16K/32K. Host note: SAO at 32K pins 238 GB with 29 GB left on node 0. |
| Saved-activation ablations, GH200 | have (not the headline) | Pinned. K=1: 8K 3.03 s / 82.0 GB, 16K 4.28 s / 139.1 GB, 32K GPU OOM. credits=1 (N=1): 8K 3.33 s / 30.9 GB, 16K 4.59 s / 38.1 GB, 32K 8.28 s / 57.4 GB (1.7 GB less HBM at 8K and 6.8 GB less at 32K than N=2). K=1 is the same negative result as on H200: uncapped tensors stay resident. |
| GH200 overlap microbench | have (mechanism, not a publish row) | 2 GiB D2H, ~10 ms spin, median of 7, `pgoverlap.cu`. Compute alone 10.2 ms. Kernel then copy: pageable 10.4 ms, pinned 10.2 ms. Copy then kernel: pageable 39.9 ms serialized, pinned 10.2 ms. Copy from another thread: pageable 25.0 ms (issuing thread blocked 24.9 ms), pinned 10.9 ms. |
| Model-tensor Qwen3.5-4B, Ascend 910B4 | have (required, own chart) | Batch 1 × seq 512: **1.32×** (14.45→10.95 s, 35→47 tok/s). Batch 1 × seq 1024: **1.29×** (17.95→13.95 s, 57→73 tok/s). Different architecture from Qwen3-4B (table labels 4.2B vs 4.0B). Keep off the Qwen3 chart. |

### Diagnostic only (not the activation checkpoint workload)

GPU-AdamW activation-pin-only (`130057Z`, `121302Z`) and ZeRO-2 CPU AdamW without `--act-mem-opt` (`145043Z`) are not the activation checkpoint workload. Do not plot them.

| Row | Job | What it isolates |
|-----|-----|------------------|
| Saved-activation, momentum-free SGD | `024322Z` | Activation-pin-only. Superseded as the published figure by ZeRO-2 CPU AdamW `215537Z`. 8B seq 4096: 1.87× / 2.88× / 3.52×. 9B seq 2048: 1.43× / 1.87× / 2.30×. |

### H200 activation checkpoint workload (`165140Z`)

Lead with **64K fits**. Pin/pageable wall time is secondary. Say “longest demonstrated sequence length,” not “maximum possible.”

| model | 8K | 16K | 32K | 64K | 96K | 128K |
|---|---|---|---|---|---|---|
| Qwen3-8B | 1.71× (7.456 → 4.357 s), 32.76 GB | 1.56× (9.050 → 5.816 s), 32.76 GB | 1.43× (14.946 → 10.427 s), 34.58 GB | **1.19× (29.990 → 25.303 s), 44.66 GB** | 1.17× (57.189 → 48.735 s), 57.64 GB | 1.09× (85.903 → 78.895 s), 70.63 GB |

Do not attribute 64K (or 8K 1.71×) solely to activation DMA. The activation checkpoint workload is checkpoint offload plus ZeRO-2 CPU AdamW plus tiled/fused kernels.

96K/128K (`job-20260928T140704Z`, steps=10; supersedes the noisier steps=4 probe `job-20260928T132833Z`, which had shown a non-monotonic 96K bump to 1.35× — that was a small-sample artifact) both ran successfully well under H200's ~140 GB capacity, extending the demonstrated range past 64K. The pin/pageable ratio now decreases monotonically with seqlen (1.19×→1.17×→1.09×): as sequence length grows, compute increasingly dominates the step and the activation-DMA saving becomes a shrinking fraction of it. Still lead the blog claim with 64K as the "capability" headline; 96K/128K are evidence the mechanism keeps working further out, not a new headline number.

Qwen3.5-9B from this job was pre-`fla` and is superseded — see the Qwen3.5-9B section below.

**Canonical Qwen3-8B 64K number: 1.19× (`165140Z`).** Two other 64K measurements exist with the same recipe. `job-20260930T173721Z` gave 1.23× (27.17 s pinned, 50.56 GB) and is the baseline for the sao-vs-ckpt comparison in memory. The `job-20261016T193445Z` regression re-run gave 1.34×. That spread is node drift; cite 1.19× in the blog and do not mix the three.

### H200 model-tensor offload + act-mem-opt (new capability, 2026-10-12)

`model_tensor_offload/bench.py` gained `--act-mem-opt`/`--tiled-mlp`/`--gradient-checkpointing` (ported
from `activation_offload/bench.py`'s proven wiring, deduped into `common.py`; all three default `False`,
so every existing invocation — including the published 8B 3.40x figure above — is unaffected unless passed
explicitly). One behavior change applies even with the new flags off: the `--model` branch's loss moved
from `logits.sum()` to a real `labels=ids` cross-entropy, since `logits.sum()` always forces full-logits
materialization and would defeat the fused-CE path.

**Regression check on the existing 8B/seq4096/no-new-flags config — resolved, not a code regression.**
Re-running the published headline config 5 times gave 2.96x/2.87x/2.89x/2.82x/2.89x — a tight cluster, not
noise bouncing toward 3.40x. The gap is entirely in the unpinned arm (pinned stays ~4.0-4.13s across every
run). A back-to-back A/B (old `logits.sum()` loss vs new `labels=ids` loss, same job, same node) gave
11.658s/2.82x vs 11.779s/2.89x — statistically indistinguishable, ruling out the loss-function change as
the cause. The 3.40x figure is a valid snapshot from a different day's node conditions (cluster
load/thermal/driver state on the shared H200), not an error; today's node reproducibly gives ~2.8-2.96x
for that exact config.

**Capacity sizing.** Single-arm pinned probes (seq 2048/8192/16384, act-mem-opt+tiled-mlp+gc on) showed
peak scales linearly with seq (4B ~0.284 GB/1K tok, 8B ~0.320), extrapolating to ~25.7/29.0 GB at
seq=65536 for 4B/8B — comfortably under the ~140 GB H200 budget. The real sweep below mostly confirmed the
extrapolation, though 8B overshot at seq=65536 by ~20% (34.75 GB actual vs ~29.0 GB predicted) —
bigger than "slightly above" suggests, and comparable to the dropped Qwen3-32B datapoint's ~16%
overshoot (raw numbers in memory, not reproduced here). Treat this slope as a rough sizing check,
not a tight bound, past seq=16384.

**seq=4096 sweep** (act-mem-opt + tiled-mlp + gradient-checkpointing on, ZeRO-3 param+optim CPU offload):

| model | step_avg unpinned | step_avg pinned | speedup | gpu_peak_gb |
|---|---|---|---|---|
| Qwen3-4B  | 6.292s  | 2.370s  | 2.65x | 8.24 |
| Qwen3-8B  | 13.662s | 4.177s  | 3.27x | 9.37 |

**seq=65536 sweep** (same flags):

| model | step_avg unpinned | step_avg pinned | speedup | gpu_peak_gb |
|---|---|---|---|---|
| Qwen3-4B  | 26.720s  | 22.192s | 1.20x | 25.63 |
| Qwen3-8B  | 34.993s  | 26.404s | 1.33x | 34.75 |

**Finding: pinning speedup is seqlen-dependent once act-mem-opt is on.** At seq=4096 speedup is ~2.65-3.27x
(close to the no-flags 2.8-2.96x baseline above); at seq=65536 (same flags, same models) it collapses to
~1.2-1.33x. Confirmed to be a pure seqlen/compute-dilution effect, not an act-mem-opt-on/off artifact —
with tiled MLP/CE compute now scaling with seq, per-step compute increasingly dominates over the fixed
per-step param/optimizer H2D transfer that pinning speeds up, so pinning's *relative* benefit shrinks even
though it still helps in absolute terms (e.g. 8B: ~8.6s/step saved either way). Legitimate, expected
tradeoff for the writeup, not a bug.

**Decision (2026-10-12), after reading `BLOG_PLAN.md` in full:** keep **Qwen3-8B** as the published
model-tensor headline (quick-reference table above). `BLOG_PLAN.md`'s "Required to publish" table
explicitly pins the high-end-CUDA model-tensor row to Qwen3-8B (or 4B), and the B70 rows can
only ever use small IDs (0.6B/1.7B) — so a 32B figure has no cross-SKU partner and would orphan the
required three-SKU pin-vs-pageable comparison the blog is built around. **Qwen3-32B dropped from blog
messaging entirely (2026-10-12)** — the act-mem-opt capability story above now stays scoped to
Qwen3-4B/8B only. The raw Qwen3-32B seq=4096/seq=65536 numbers were measured and remain in local
project memory if needed later, but are not part of this doc or the blog.

### Qwen3.5-9B (required, own chart)

Required per `BLOG_PLAN.md` (2026-10-18). Do not cite these beside Qwen3-8B as a size comparison. **Only cite numbers from runs with `fla` + `causal-conv1d` installed** (`job-20261016T193445Z` onward). Earlier Qwen3.5-9B numbers measure a missing-package fallback.

**Model-tensor, Qwen3.5-9B:** `job-20261016T141158Z` (MODEL_TENSOR_QWEN35_9B_SEQ4096_EXIT=0; **superseded 2026-10-16, see update below**). Batch 1 × seq 4096 — now matches the Qwen3-8B headline methodology exactly (`job-20260928T140704Z`), superseding the earlier batch 4 × seq 512 number above (which was not a fair comparison against the 8B headline). 14.605 → 4.875 s (3.00×, 280 → 840 tok/s), offload 150.10 GB (params 16.68 GB + optimizer 133.42 GB), GPU peak 77.48 GB (same both arms). Flag (root-caused and fixed below): at matched batch/seq, 9B's GPU peak was a clean 1.99× 8B's (38.88 GB) despite only ~1.12× more params (8.954B vs 8B nominal).

**Update (2026-10-16, `job-20261016T193445Z`) — root cause of the 1.99× GPU-peak gap found and fixed.** Qwen3.5-9B's hybrid layers (`qwen3_next` architecture) dispatch `chunk_gated_delta_rule` and `causal_conv1d_fn`/`causal_conv1d_update` through `transformers`' `use_kernel_func_from_hub_with_fallback`, which needs the `fla` (flash-linear-attention) and `causal-conv1d` packages for a fused kernel and otherwise silently falls back to a pure-PyTorch chunked implementation. Neither package was installed in the `torch214` env — confirmed via `is_new_implementation=False` on `torch_chunk_gated_delta_rule`'s closure before the fix. Installed both (`pip install flash-linear-attention causal-conv1d`; confirmed `fla 0.5.2`, `causal_conv1d 1.7.0`, `is_new_implementation=True` after) and re-ran: 13.718 → 4.507 s (**3.04×**, same ballpark as the no-fla 3.00×), offload unchanged (150.10 GB, by construction), **GPU peak dropped 77.48 GB → 53.68 GB (-30.7%)**. Against 8B's 38.88 GB, that's now **1.38×** (down from 1.99×) for 1.12× more params — still somewhat disproportionate, but no longer wildly so. The dominant driver of the original 1.99× was the missing `fla`/`causal-conv1d` dependency, not an inherent architectural cost.

**Activation checkpoint workload, Qwen3.5-9B (8K/16K/32K/64K), re-measured with real Liger FLCE:** `job-20261016T162911Z` (`ACT_MEMOPT_Qwen3.5-9B_s*_EXIT=0`, all 4 exit 0). Same recipe as 8B this time -- `liger_qwen3_5, liger_fused_linear_ce, tiled_mlp:Qwen3_5MLP, flash_attn_2_if_available` -- superseding the prior confounded numbers (which silently fell back to tiled fused CE because the old liger-kernel's `apply_liger_kernel_to_qwen3_5` didn't accept `fused_linear_cross_entropy`). New: 1.42× (7.852→5.545 s, **33.33 GB**) / 1.20× (17.390→14.507 s, **35.48 GB**) / **1.21×** (54.342→44.827 s, 46.29 GB) / 1.07× (162.436→151.599 s, 71.57 GB). Compare to the old (confounded) numbers: 1.42×/35.82 GB, 1.22×/35.82 GB, 1.05×/46.29 GB, 1.08×/71.57 GB. Net effect of real Liger FLCE: 8K and 16K GPU peak drop ~2.5 GB and ~0.3 GB; 32K speedup jumps from 1.05× to 1.21× (the one seqlen where it clearly mattered); 64K is within noise of the old number. So the Liger-availability confound was real but modest in practice for most seqlens here -- it is **not** the dominant explanation for 9B's steeper step-time/memory scaling vs 8B (e.g. 64K pinned step is still ~5.6× 8B's 27.102 s, memory jump 32K→64K is still ~2.3× 8B's); that scaling gap persists with matched kernels on both sides and is a real architectural difference (9B's hybrid gated-delta-rule layers), not a kernel-availability artifact. Qwen3-8B was re-run in the same job as a regression check and landed at 1.96×/1.74×/1.48×/1.34× (8K/16K/32K/64K) -- same recipe, within the node's day-to-day drift band seen elsewhere in this doc, not a code regression.

**Update (2026-10-16, `job-20261016T193445Z`): the "real architectural difference, not a kernel-availability artifact" conclusion directly above was wrong.** It was a second, larger kernel-availability artifact -- same root cause as the model-tensor GPU-peak gap in the Qwen3.5-9B model-tensor section above: `fla` (flash-linear-attention) and `causal-conv1d` were never installed in the `torch214` env, so Qwen3.5-9B's hybrid gated-delta-rule/causal-conv1d layers ran an unoptimized pure-PyTorch fallback for every Qwen3.5 number in this doc up to this point. Installed `fla 0.5.2` + `causal_conv1d 1.7.0` (confirmed `fla_kernel_active: True`) and re-ran all 4 seqlens: 1.85× (8.661→4.675 s, 33.33 GB) / 1.80× (10.661→5.926 s, 35.48 GB) / 1.52× (13.340→8.763 s, **39.77 GB**, down from 46.29 GB) / 1.25× (20.807→16.671 s, **55.71 GB**, down from 71.57 GB). The pageable-arm step time at 64K dropped from 162.436 s to 20.807 s -- a **7.8× reduction from the fla/causal-conv1d install alone**, independent of pin vs pageable. Qwen3.5-9B's 64K pageable step (20.807 s) is now *faster* than Qwen3-8B's (36.433 s), and GPU peak (55.71 GB vs 8B's 50.56 GB) is within 1.10× despite 1.12× more params -- essentially matched. **Do not cite the pre-fla numbers in this paragraph, or the pre-fix 1.99× GPU-peak ratio in the model-tensor section, as evidence of an inherent Qwen3.5-9B cost** -- they measure a missing-package bug, not the architecture.

An earlier ZeRO-2 CPU AdamW run without `--act-mem-opt` OOMed Qwen3.5-9B at 64K in cross-entropy (`145043Z`). That run is not the activation checkpoint workload. It is consistent with CE being the expensive part for 9B.

**Saved-activation offload (widened whole-decoder-layer scope, `--credits 1`), Qwen3.5-9B:** `job-20261016T141158Z` (Job B). Matches the Qwen3-8B dataset-of-record methodology (`job-20261011T171729Z`): `--offload-tensors -1` (unlimited), ZeRO-2 CPU AdamW (`--offload-optimizer`), batch 1. seq 2048: **3.71×** (8.562→2.306 s). seq 4096: **3.37×** (12.792→3.796 s) — this seqlen OOMed under the old narrow (MLP-only) `--offload-modules` scope at every K (see the K-sweep investigation below); the widened scope now clears it. seq 8192: **4.60×** (34.052→7.397 s; flagged `pin_separated=false` in the raw result — the two pageable-arm samples disagreed by ~17s, so treat this ratio as noisy rather than a clean measurement). seq 16384: **OOM** during the discovery/calibration pass itself (before either timed arm ran) — `torch.OutOfMemoryError` inside `torch_chunk_gated_delta_rule`, ~139.7 GiB resident on a 139.8 GiB device — so the ladder stopped here per its own abort-on-failure design; seq 32768/65536 were not attempted. **Superseded by `job-20261017T235002Z`** (full 2K–64K ladder, all pass, `fla` installed): see the Qwen3.5-9B saved-activation row in the §6 required table.

### H200 host page size (transparent huge pages) and NUMA — 2026-10-23

Prompted by Ascend 910B4: pageable H2D is ~6.4 GB/s on 4K pages and ~23 GB/s on 2M pages, close to pinned (microbench only). Plan and scripts: `autorun/hugepage/` (`PLAN_hugepage_h200.md`). The pod's THP mode is `madvise` and `/sys` is read-only, so the 2M condition is requested per process: `THP_MEM_ALLOC_ENABLE=1` (torch CPU allocations) plus `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (DeepNVMe `posix_memalign`, i.e. native pinned buffers). Both pageable and pinned buffers get 2M pages (2x2 page x pin design). Coverage was verified from `/proc/<pid>/smaps`: 99%+ in every 2M arm, 0% at 4K. Torch `cudaHostAlloc` pins stay 0% (control).

**Microbench (`job-20261023T131838Z`): no effect.** 256 MiB, NUMA-bound, 4K -> 2M: pageable H2D 10.30 -> 10.39, D2H 17.31 -> 17.33 GB/s; native-registered 55.40 -> 55.54 / 54.97 -> 55.30. All sizes (4-1024 MiB), bound and unbound, are within run-to-run noise. CUDA pageable copies go through the driver's pinned staging buffer, so page count does not set the bandwidth. The Ascend effect does not transfer to H200.

**E2E, Qwen3-8B (same job), average step time, 4K -> 2M:**

| workload | pageable | pinned | pin speedup |
|---|---|---|---|
| model-tensor, bs1 x seq 4096, steps 10 | 11.99 -> 10.75 s (-10.4%) | 4.04 -> 4.00 s (-1.1%) | 2.97x -> 2.69x |
| activation checkpoint, bs1 x seq 8192, steps 4 | 8.50 -> 7.78 s (-8.5%; -2.5% by min step) | 4.52 -> 4.48 s (-1.0%) | 1.88x -> 1.74x |
| saved-activation, bs1 x seq 8192 | 18.81 -> 16.64 s (-11.5%) | 5.71 -> 5.53 s (-3.3%) | 3.29x -> 3.01x |

Microbench bandwidth did not move, so the pageable e2e gain is CPU-side, not DMA.

**Follow-up (`job-20261023T145721Z`, steps 10, activation checkpoint repeated twice, interleaved 4K/2M, per-arm `minflt` sampled from `/proc/<pid>/stat`):** the gain is mostly fewer slow outlier steps, not a faster typical step.

| run | pageable avg 4K -> 2M | pageable **min** 4K -> 2M | pinned min 4K -> 2M | pin speedup by min |
|---|---|---|---|---|
| activation checkpoint s8192, rep 1 | 8.97 -> 7.56 s (-15.7%) | 7.50 -> 7.49 s (-0.1%) | 4.46 -> 4.31 s (-3.3%) | 1.68x -> 1.74x |
| activation checkpoint s8192, rep 2 | 8.68 -> 7.66 s (-11.8%) | 7.64 -> 7.57 s (-0.9%) | 4.47 -> 4.34 s (-2.8%) | 1.71x -> 1.74x |
| model-tensor s4096 | 11.62 -> 10.84 s (-6.7%) | 10.79 -> 10.67 s (-1.1%) | 3.97 -> 3.88 s (-2.4%) | 2.72x -> 2.75x |

The avg-minus-min gap on the pageable arm is about 1-1.5 s at 4K and about 0.1 s at 2M. Page faults drop about 170x: 88-103M minor faults per arm run at 4K (~580k/s steady) vs 0.5-0.8M at 2M (0.5-1.4k/s). There are no major faults. At 4K, **both arms fault at the same rate** (activation checkpoint pageable 584k/s, pinned 583k/s), so fault count alone does not explain why only the pageable arm benefits. *(Superseded 2026-10-24: the faults are automatic NUMA balancing, in both arms. See “H200 automatic NUMA balancing” below.)* The first job's model-tensor min step did drop 5.5% (11.14 -> 10.53 s), so put the steady-state pageable gain at 0-5%, with most of the average gain coming from removed outlier steps. **Bottom line:** huge pages make pageable step times steadier and trim 0-5% off the typical step. They do not substitute for pinning: pin speedup by min step is unchanged or slightly higher at 2M (2.75x model-tensor, 1.74x activation checkpoint).

**Overlap probe (256 MiB H2D with a ~10.9 ms GEMM on another stream, NUMA-bound):** a pageable `copy_(non_blocking=True)` blocks the issuing thread for the whole copy (27.9 ms issue = 27.9 ms wall, ~9.6 GB/s). Its bandwidth does not drop under concurrent compute, unlike B60. Native-registered issues in 0.017 ms and fully overlaps (11.25 ms total, about the same as the GEMM alone). 2M pages change neither arm.

**NUMA:** H200 GPU0 is on node 0. Binding to it gains 2-6% on pageable D2H and at most 2% on pinned D2H at 1 GiB, far smaller than Ascend's 26% pinned-D2H loss when remote.

**Cross-SKU caveat:** any H2D/D2H number from a bench version that did not write the pageable buffer before timing is suspect (zero-page reads). Ask the Ascend, B60, and GH200 owners whether their pageable H2D arms wrote the buffer first. Ascend's 6.48 GB/s and the GH200 PDF numbers are unconfirmed on this point.

### H200 automatic NUMA balancing — root cause of the 4K fault storm and step jitter (2026-10-24)

**Setup.** The H200 nodes run `kernel.numa_balancing=1` on 2 NUMA nodes (Xeon 8488C, 192 CPUs, ~1 TB per node; GPU0-3 on node 0, GPU4-7 on node 1). Probe: DSE `152e17c` `StepProbe`. Jobs: `job-20261023T221332Z` (per-phase attribution) and `job-20261024T022512Z` (confirmation, on a new host after a node recycle). Script: `autorun/hugepage/hugepage_numa_job.sh`. Workloads: Qwen3-8B, activation checkpoint (ACT) bs1 x seq 8192 and model-tensor (MT) ZeRO-3 bs1 x seq 4096, steps 10.

**Attribution (`221332Z`).** At 4K, faults land in two places:
- The **optimizer step, in both arms**: 0.3-7M faults/step and 10-70 CPU-s of sys time across the CPU Adam threads. This is all of the pinned arm's 4K penalty. MT pinned steps with no optimizer-step faults ran 3.91-3.98 s, the same as at 2M.
- The **backward phase, in the pageable arm only**: 1-1.7M faults/step, partly on `pt_autograd_0`.

The fault pattern drifts over a run, so 4K->2M min-step deltas vary from run to run.

**Confirmation (`022512Z`).** Steady-state minor faults per step equal the host-wide `numa_hint_faults + numa_pages_migrated` to within 0.01M:

| arm (4K) | faults/step | hint + migrated |
|---|---|---|
| ACT pageable | 1.99M | 1.72 + 0.26 |
| ACT pinned | 1.90M | 1.60 + 0.30 |
| MT unpinned | 6.81M | 6.12 + 0.69 |
| MT pinned | 1.63M | 1.39 + 0.23 |

Under `numactl --membind=all`, every arm has 0 faults/step and ~0 sys time. The policy was verified as `bind:0-1` in the arm subprocess. So the "pageable-only backward faults" above are hinting faults too, not freshly allocated copy destinations. DeepSpeed is not re-allocating; the kernel samples and migrates long-lived buffers (including the ~61 GiB of pageable CPU Adam state).

**Step time, avg / min (s), same host and job:**

| | 4K | 4K + `--membind=all` | 2M |
|---|---|---|---|
| MT unpinned | 12.828 / 11.610 | **11.870 / 11.166** | 12.343 / 11.628 |
| MT pinned | 4.062 / 3.851 | **3.781 / 3.699** | 3.999 / 3.896 |
| ACT pageable | 8.881 / 8.462 | 8.813 / 8.779 | **7.636 / 7.417** |
| ACT pinned | 4.661 / 4.394 | 4.673 / 4.626 | **4.513 / 4.460** |

- MT: binding is the best condition for both arms; pinned is 7% faster than 4K and 5% faster than 2M.
- ACT: binding leaves the average unchanged but removes the variance. Pinned max goes 5.12 -> 4.70 s; pageable spread goes from 8.46-9.61 s to 8.78-8.84 s.
- Pin speedup is stable across conditions: ACT ~1.9x, MT ~3.1x. Under binding, average ≈ min.
- One rep per condition, on a different host from the 2026-10-23 jobs (MT unpinned is slower here in every condition). Do not compare absolute times across these jobs.

**Resolved (`job-20261024T032725Z`, `autorun/hugepage/hugepage_act2m_job.sh`): the ACT "2M gain" was NUMA placement, not page size.** ACT Qwen3-8B s8192, page size x placement, balancing off in all conditions. Placement was verified per arm from `/proc/<pid>/numa_maps` (sampler `THP_SAMPLER_NUMA_EVERY=6`). Avg step (s):

| placement | pageable 4K -> 2M | pinned 4K -> 2M |
|---|---|---|
| local: `--cpunodebind=0 --membind=0` (GPU0's node) | 7.226 -> 7.109 (-1.6%) | 4.214 -> 4.192 (-0.5%) |
| remote: `--cpunodebind=0 --membind=1` | 8.790 -> 8.775 | 5.619 (outlier steps; min 5.33) -> 5.310 |
| all: `--membind=all`, CPUs free (~50/50 placement) | 7.383 -> 7.232 | 4.330 -> 4.630 (this arm landed 78% on node 1) |

- **Page size:** with placement fixed, 2M is worth ≤1.6% pageable and ≤0.5% pinned. Microbench at 1 GiB, local: pageable H2D 9.49 -> 9.83 GB/s, D2H 17.3 -> 18.4 GB/s; pinned unchanged.
- **Remote memory:** costs +1.5 s/step pageable and +1.1-1.4 s pinned, almost all in the **optimizer step**: pageable 3.61 -> 5.31 s, pinned 2.19 -> 3.30 s. That is CPU Adam on node-0 CPUs streaming its state from node 1. Backward stays at 3.0-3.2 s.
- **Copies:** pinned DMA is placement-insensitive (55 GB/s local and remote). Pageable H2D at 1 GiB drops 27% when remote (9.49 -> 6.94 GB/s); D2H drops ~2%.
- **The 4.64 s pageable backward is CPU-thread placement (`job-20261024T114211Z`, `autorun/hugepage/numa_place_job.sh`, new host).** CPU on node 1 with memory on node 0 (`--cpunodebind=1 --membind=0`) reproduces `022512Z`'s per-phase signature exactly. Pageable: fwd 0.54 s, bwd 4.63 s (local: 0.42 / 3.15), 10.46 s/step overall because the optimizer step is also remote (5.28 s). Pinned is barely affected in backward (1.75 -> 1.93 s). The staged pageable copies run on the CPU threads, so they slow down when those threads sit on the socket far from the GPU. Pinned DMA does not. The same job's `--membind=all` repeat landed in the slow state again (8.82 s/step, bwd 4.66 s), vs 7.38 s in `032725Z`. So `--membind=all` is a scheduler coin flip for the pageable arm.
- **`--preferred` = `--membind` for local placement:** `--cpunodebind=0 --preferred=0` gives 7.20 / 4.24 s (ACT pageable / pinned), vs 7.16 / 4.22 s with `--membind=0`. It shows 0 faults/step, so a preferred task policy also opts out of NUMA balancing. Unlike bind, it spills to the other node instead of failing when one node is full. MT is placement-insensitive: unpinned 10.49 vs 10.39 s, pinned 3.77 vs 3.78 s for local-preferred vs `--membind=all`. Binding to one node halves the CPU Adam threads (96 of 192), which costs MT's optimizer step what local placement saves in backward.
- **Best condition:** CPU and memory bound to the GPU's node, at plain 4K. Pageable 7.23 s, pinned 4.21 s, ~0.05 s step spread, pin speedup 1.71x. `--membind=all` only disables balancing; placement stays random and swings runs by 0.3-1.5 s.

**Methodology consequence.** Single-device e2e benches: launch under `numactl --cpunodebind=<dev node> --preferred=<dev node>`. The cuda `run.sh` scripts do this from DSE `31db0b7`; npu takes `NUMA_NODE=<n>`. Use `numactl --membind=all` only when the node is unknown (§2a Rule 2). Multi-rank runs must bind each rank to its own device's node, never all ranks to one node. The cuda/npu `run.sh` scripts do this from DSE `895f338`. Published H200 rows measured unbound on `numa_balancing=1` hosts carry this noise; decide per chart whether to re-run them bound (preferred) or footnote them.

**Possible DeepSpeed follow-up (not started):** set an explicit bind mempolicy (`mbind`) on DeepSpeed-allocated offload buffers and the CPU Adam state, so users get this without `numactl`.

### Writer constraints

- Qwen3 and Qwen3.5 are both published. Qwen3.5 always goes on its own chart, never next to Qwen3.
- Model-tensor, activation checkpoint, and saved-activation get equal visual weight, different axes.
- H200 GB/s is supporting only; do not plot next to B70, Ascend, or GH200.
- #8282 is async vs blocking (both pinned). #8283 GB/s was remeasured with a pageable arm here.
- Next: B70 activation checkpoint workload + model-tensor + H2D/D2H + saved-activation, (GH200 model-tensor optional). All H200 rows (Qwen3 and Qwen3.5-9B) are collected. Ascend 910B4 Qwen3-4B rows are collected. GH200 H2D/D2H, activation checkpoint, and saved-activation are collected. Do not start `README.md` until B70 model-tensor exists, and B70 saved-activation exist (or B70 activation/saved-activation are named limitations).

### Application (3): saved-activation offload (credit-bounded, no remat) — 2026-09-27/28

Separate algorithm from `activation_offload/` above (checkpoint hidden states with remat). Full spec in `SAVED_ACTIVATION_OFFLOAD_PLAN.md`. **Peer publish application.** Published H200 rows are ZeRO-2 CPU AdamW, widened scope, `--credits 1` (`job-20261011T171729Z` for Qwen3-8B, `job-20261017T235002Z` for Qwen3.5-9B). The narrow-scope `job-20261007T215537Z` K-sweep below is history. Momentum-free SGD is diagnostic only. Ascend 910B4 and B70 are required (accelerator-agnostic script).

**Phase 2/3 correctness + n-sweep validation (H200, `--n-sweep` N={1,2,4}, K=3, batch 1 × seq 2048) — all PASS:**

| Model | pin_speedup | pin_separated | Credit backpressure at N=1/2/4 |
|---|---|---|---|
| Qwen3-1.7B | ~3.76× | true | None (`fwd_max_held=1` flat, `credit_wait_fwd_ms=0`) |
| Qwen3-8B | ~3.58–3.62× | true | None (same pattern) |
| Qwen3.5-9B (dense, `qwen3_5_text`) | ~2.18–2.22× | true | None (same pattern; lower speedup, not backpressure-related) |

Manifests byte-identical across pin arms (V1), `is_pinned_verified=true` on pin arms (V0), no rematerialization (V3: scope-module forward calls per step == L − skip_last).

**Phase 4 K-sweep attempt at batch 1 × seq 8192 (`job-20260928T015918Z`) — mostly OOM, only one arm survived:**

| Model | K | Result |
|---|---|---|
| Qwen3-8B | 1 | OOM (`torch.OutOfMemoryError`, ~135.6 GB resident, tried to alloc 4.64 GiB) |
| Qwen3-8B | 2 | OOM (~136.3 GB resident, tried to alloc 4.64 GiB) |
| Qwen3-8B | 3 | **PASS** |
| Qwen3.5-9B | 1 | OOM (~137.2 GB resident, tried to alloc 7.58 GiB) |
| Qwen3.5-9B | 2 | OOM (same signature) |
| Qwen3.5-9B | 3 | OOM (same signature) |

Qwen3-8B K=3 (the only surviving arm): pin_speedup **3.56×** (pageable 4.369 s → pinned 1.226 s median step), `pin_separated=true`, `offloaded_bytes` 21,139,292,160 (≈19.7 GiB, 105 tensors), `r_d2h_over_mlp_span` **0.93 (pageable arm) → 4.06 (pinned arm)** — pinned arm is bandwidth-bound relative to MLP span (r > 1), pageable arm is not (r < 1).

**No-remat HBM ceiling confirmed and now known to bind Qwen3.5-9B at every K, not just Qwen3-8B at K<3** (see `SAVED_ACTIVATION_OFFLOAD_PLAN.md`'s no-remat-ceiling note): lower K keeps more HBM resident since less gets offloaded, so batch 1 × seq 8192 is infeasible for Qwen3-8B below K=3, and infeasible for Qwen3.5-9B at **any** K.

**Seqlen-ceiling sweep below 8192 (`job-20261007T024322Z`, pod `tunji-h200-n1g8-ds-pin-0-s4l9r`, re-recreated env: deepspeed `0.19.8+581ac41bd` editable, transformers `5.16.0.dev0` (git `69a7fb1`), liger-kernel `0.8.4`, flash-attn `2.8.3`) — found a feasible operating point for both models, full K={1,2,3} table at each:**

| Model | seq | K=1 OOM? | K=2 OOM? | K=3 OOM? |
|---|---|---|---|---|
| Qwen3-8B | 4096 | PASS | PASS | PASS |
| Qwen3-8B | 2048 | PASS | PASS | PASS |
| Qwen3-8B | 1024 | PASS | PASS | PASS |
| Qwen3.5-9B | 4096 | OOM | OOM | OOM |
| Qwen3.5-9B | 2048 | PASS | PASS | PASS |
| Qwen3.5-9B | 1024 | PASS | PASS | PASS |

**Feasible-ceiling picks for the real K-sweep table — the largest seqlen where all three K values pass (so the comparison isn't missing arms, unlike the seq-8192 attempt above):** Qwen3-8B at **seq=4096**, Qwen3.5-9B at **seq=2048**. (Qwen3.5-9B's no-remat HBM ceiling binds somewhere between 2048 and 4096 regardless of K, consistent with the "ceiling, not a K effect" diagnosis above.)

The SGD sweep above only located the feasible ceilings. **Published K-sweep** is the same grid with `--offload-optimizer` (`job-20261007T215537Z`, finished 2026-10-08 01:34 UTC, torch `2.9.1+cu130`). ZeRO-2, parameters on GPU, CPU AdamW `lr=1e-4`, optimizer `pin_memory` follows the pin arm. Qwen3.5-9B seq 4096 OOMed again at K=1/2/3 (CUDA OOM, ~138 GiB resident).

| Model | seq | K | pin_speedup | step (pageable → pinned) | offloaded_bytes | r (pageable→pinned) |
|---|---|---|---|---|---|---|
| Qwen3-8B | 4096 | 1 | 1.997× | 7.222 → 3.616 s | 3.28 GiB | 0.808 → 1.262 |
| Qwen3-8B | 4096 | 2 | 2.088× | 7.722 → 3.697 s | 6.56 GiB | 0.910 → 2.526 |
| Qwen3-8B | 4096 | 3 | 2.197× | 8.268 → 3.763 s | 9.84 GiB | 0.932 → 3.810 |
| Qwen3.5-9B | 2048 | 1 | 1.828× | 7.595 → 4.155 s | 1.45 GiB | 0.793 → 1.158 |
| Qwen3.5-9B | 2048 | 2 | 2.033× | 7.975 → 3.923 s | 2.91 GiB | 0.875 → 2.376 |
| Qwen3.5-9B | 2048 | 3 | 2.030× | 8.480 → 4.178 s | 4.36 GiB | 0.905 → 3.356 |

All 15 passing Adam arms `pin_separated=true`. Speedup stays near **2×** as K grows: the CPU Adam step is in both arms, so extra activation bytes move `r` but add little wall-time ratio. Do not quote the SGD 3.52× / 2.30× figures as this result. Do not plot 8B-at-4096 next to 9B-at-2048 as one comparison. seq=1024 Adam rows are in `job-20261007T215537Z.log`.

**SKU split:** H200, Ascend 910B4, and B70 are publish gates (above); GH200 is optional.

#### Widened scope, seq-extrapolated discovery, k/credits ablations (H200, `job-20261011T171729Z`) — 2026-10-11

Supersedes the narrow-scope `job-20261007T215537Z` K=1/2/3 table above as the dataset of record for this
application; that job is kept only as historical/superseded (checklist table in §6). This is sao's own
pin-on/off story at the widened scope, not a sao-vs-ckpt-activation comparison (out of scope per
`SAVED_ACTIVATION_OFFLOAD_PLAN.md`).

**Widened `--offload-modules` scope fix.** The narrow default (`r"\.mlp$"`, MLP submodule only) undercounted
eligible saved tensors — attention projections and residuals outside the MLP were never offloaded. Widened to
`r"(?:^|\.)layers\.\d+$"` (whole decoder layer); every published row below uses this scope.

**Discovery-residue / subprocess-isolation fix.** Discovery's own full-residency forward (every candidate
saved-for-backward tensor held resident at once, by design, to catalog it) was leaving resident bytes behind
that leaked into the first measured arm. Discovery now always runs in an isolated subprocess
(`_discover_via_subprocess`) so its one-time resident-memory cost never contaminates a measured arm.

**Seq-extrapolated calibration (`--discover-calib-seqs`).** Discovery's full-residency forward still doesn't
fit in 139.8 GB at seq>=32768 even in isolation. `--discover-calib-seqs LO HI` runs Discovery at two cheap
small seqlens instead and linearly extrapolates `shape`/`stride`/`nbytes` per record to the real target
seq (validated: exact match against a real `--seq 8192 --discover-only` manifest when calibrating from
2048/4096). seq 32768 and 65536 below use `--discover-calib-seqs 4096 8192`.

**K=1-under-widened-scope: negative result.** Capping `--offload-tensors` to 1 under the widened scope OOMs
and gives *worse* HBM than `-1` (unlimited) — the ~20 non-selected eligible tensors/layer per decoder layer
fall back to always-resident instead of being offloaded, raising peak HBM rather than lowering it. This is a
distinct, separately-validated negative result from the old narrow-scope K=1/2/3 table above ("Published
K-sweep"), which should no longer be read as a publish-worthy K sweep — K-capping does not help in either
scope, and under the widened scope it actively hurts.

**`--credits` ablation: the real tuning lever.** With `--offload-tensors -1` (unlimited coverage) held fixed,
halving the forward credit-pool depth (`--credits 1`, down from the default 2; `--prefetch` left at default
2) bounds the resident held-group staging buffer (`fwd_max_held_bytes`) directly, independent of K:

| seq | HBM savings (credits=1 vs credits=2) | Step-time cost |
|---|---|---|
| 8192 | negligible (below the >=32768 regime) | negligible |
| 16384 | negligible (below the >=32768 regime) | negligible |
| 32768 | **8-10%** | +/-2.3% |
| 65536 | **8-10%** | +/-2.3% |

This is the tuning axis to publish for application (3), replacing the old `--offload-tensors` 1/2/3 ask.

**Per-seqlen breakdown (credits=1, dataset of record, `job-20261011T171729Z`)** — backs the quick-reference headline's 3.80x/125.345→33.002s figure with the full sweep, not just the seq=65536 endpoint:

| seq | pin_speedup | pinned step_s | host D2H GiB/step | peak HBM reserved (GB) |
|---|---|---|---|---|
| 8192 | 3.08x | 5.76 | 55.4 | 54.30-54.49 |
| 16384 | 4.24x | 8.45 | 110.9 | 59.40-59.44 |
| 32768 | 4.36x | 14.94 | 221.8 | 75.48-75.54 |
| 65536 | 3.80x | 33.00 | 443.5 | 124.95-125.10 |

Pageable-arm step times aren't restated here (available in the job log); pin_speedup × pinned step_s reproduces them (e.g. 65536: 33.00 × 3.80 ≈ 125.3s, matching the headline's 125.345s).

**Seq8192/16384/32768/65536 capability result — stop at seq64k (explicit decision).** The widened-scope,
`--offload-tensors -1` sweep passes cleanly at seq 8192, 16384, 32768, and 65536 (seq>=32768 via calibrated
discovery above). Decision: do not push the capability sweep past seq 65536 for this publish — seq64k is
sufficient to demonstrate the capability story (credit-bounded saved-activation offload scaling to long
context) without spending further GPU time chasing a higher ceiling that isn't needed for the publish bar.

Repro commands, environment setup (H200 torch 2.14 + flash-attn 2.8.3 C++20 rebuild), and the matched
ckpt-activation comparison baseline are in `saved_activation_offload/README.md`.
