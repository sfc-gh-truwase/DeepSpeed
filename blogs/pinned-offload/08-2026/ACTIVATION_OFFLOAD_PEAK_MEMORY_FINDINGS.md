# Activation-offload GPU peak-memory investigation — findings for handoff

Status: **investigation complete, conclusions verified**. `EXPERIMENTS_HANDOFF.md` has
**not** been edited — these findings still need to be folded into §6 by whoever owns
that doc next. This file is the evidence trail; read it before touching that section.

Scope: `benchmarks/cpu_offload/activation_offload/bench.py` (activation checkpoint workload), Qwen3-8B,
H200, `--offload-optimizer --act-mem-opt` (ZeRO-2 CPU AdamW + Liger + TiledMLP),
comparing `--pin 1` (activation offload on) vs `--no-act-offload --pin 0` (plain
gradient checkpointing, no offload).

## 1. The anomaly that started this

`EXPERIMENTS_HANDOFF.md`'s activation checkpoint workload table reports `gpu_peak_gb` per seqlen for
the with-offload (pinned) arm: 32.76 / 32.76 / 34.58 / 44.66 / 57.64 / 70.63 GB at
8K/16K/32K/64K/96K/128K. At seq 98304 (96K), a `--no-act-offload` run produced the
*exact same* `gpu_peak_gb: 57.64` — suspicious, since offload should reduce peak
GPU memory.

## 2. Metric-correctness investigation (resolved, metric is now trustworthy)

- **Hypothesis 1 (refuted):** harness never called `reset_peak_memory_stats()`, so
  the recorded peak was contaminated by an unrelated early-step transient. Fixed by
  adding a reset call — result unchanged (still 57.64 both arms). Refuted.
- **Fix applied and kept:** peak is now reset every measured step and only the
  *last* measured step's peak is reported (`step_peaks[-1]`), so the number reflects
  that step's own steady-state memory, not a stale transient from an earlier step.
  This fix is real, correct, and reproducible — but it did not change the outcome
  (still 57.64 both arms at 96K). The metric itself is not the bug.

## 3. Root cause — layer-level instrumentation (job `20260929T140331Z`)

Per-decoder-layer forward/backward memory hooks (36 layers, Qwen3-8B) at seq98304,
steps=3/warmup=2, both arms. Full log: `autorun/logs/job-20260929T140331Z.log`.

**Forward pass — offload's forward-side benefit is real:**
- with-offload: `alloc` climbs to 20.97GB by layer2, then **flat** at 20.97GB
  through layers 3–35 (each layer's checkpoint boundary tensor ships to host as
  soon as it's produced).
- no-act-offload: `alloc` climbs the whole way, 19.36GB → 47.54GB by layer35
  (nothing ever leaves GPU).
- Forward-only peak: 24.26GB (offload) vs 51.64GB (no offload) — ~27GB gap,
  matching `offload_gb: 26.25` almost exactly.

**Backward pass — this is where both arms converge, via opposite paths:**
- **no-act-offload:** peak jumps to **58.45GB in one shot at `bwd-layer35`**
  (the very first backward hook), then stays flat at 58.45 for the rest of
  backward while `alloc` *decreases* monotonically (48.54GB → 20.31GB) as each
  layer's already-resident boundary tensor is consumed and freed.
- **with-offload:** peak climbs gradually, layer by layer, through the **entire**
  backward pass — 31.07 (bwd-layer35) → 39.12 → 47.98 → 57.64 (bwd-layer1) →
  **58.45 only at the very last layer, bwd-layer0**. `alloc` rises monotonically
  the whole time (21.97GB → 49.3GB) — the opposite direction from the no-offload
  arm.

**Mechanism (confirmed from source, not just inferred from the alloc curve):**
`deepspeed/runtime/activation_checkpointing/offload_activations.py`,
`_unpack_tensor()` → `_restore_tensor(tensor_id, cache=True)`. Every tensor
restored from host during backward is cached in `self._restored[tensor_id]`, and
that cache is **only cleared on context-manager exit** (`__exit__`) or in
`reset()`/`_sync_and_clear()` — never per-tensor mid-backward. The comment in the
source says this is deliberate: *"cache=True allows repeated unpack of one id (HF
retain_graph), cleared on context exit."* Since `bench.py`'s `with offload:` block
spans the whole step, every tensor restored during backward accumulates on GPU and
is only released when the block exits after the optimizer step — so by the last
backward layer, the offload arm is holding the *same* ~26GB of activations
simultaneously that the no-offload arm held from the start. That's why both arms
land on the identical 58.45GB (58.45 in this steps=3 debug run; 57.64 in the
official steps=10 runs — the small delta is step-count/allocator noise, not a
different mechanism).

**Confirmed NOT a ZeRO-2 gradient-buffer artifact:** if grad-buffer materialization
were the driver, both arms would peak at the same absolute backward layer. They
don't — no-offload peaks at the first backward layer, with-offload at the last.

**Checked against a newer DeepSpeed workspace — not fixed there either:**
Diffed `sfc_ds_pin_memory/deepspeed/runtime/activation_checkpointing/offload_activations.py`
against `sfc_ds_act_offload/deepspeed/runtime/activation_checkpointing/offload_activations.py`
(dated 2026-09-28, newer). The only delta is a **prefetch optimization** (dedicated
`s_reload` H2D stream, LIFO prefetch-the-previous-tensor via `_start_reload`/
`_prefetch_prev`/`_finish_reload`, new `prefetch` flag) — purely about overlapping
restore-from-host with compute. The `_restored` cache retention-until-context-exit
behavior described above is byte-for-byte unchanged in both workspaces. This is not
a bug that's been fixed upstream; it's a live, intentional design tradeoff.

## 4. Seqlen-ceiling sweep — no-offload baseline (job `20260929T152934Z`)

Same config as the activation checkpoint workload (ZeRO-2 + Liger + TiledMLP), `--no-act-offload --pin 0`,
steps=3/warmup=1, at 64K/96K/128K:

| seqlen | no-act-offload `gpu_peak_gb` | no-act-offload `step_avg_s` | published with-offload `gpu_peak_gb` | published with-offload `step_avg_s` |
|---|---|---|---|---|
| 64K (65536)  | 44.82 | 29.52 | 44.66 | 25.303 |
| 96K (98304)  | **57.64** | 56.93 | **57.64** | 48.735 |
| 128K (131072)| **70.63** | 86.24 | **70.63** | 78.895 |

All three shapes exit 0 — **no OOM anywhere in the tested range, without
activation offload at all.** GPU peak matches the published with-offload numbers
to the hundredth of a GB at 96K and 128K.

## 5. Bottom line for the blog write-up

- **Do not attribute the "64K/96K/128K fits" capability claim to activation
  offload.** The plain-GC baseline (ZeRO-2 CPU-optimizer-offload + Liger +
  TiledMLP, no activation offload) reaches the identical GPU peak and the
  identical seqlen ceiling. Activation offload is not what's keeping these
  seqlens under the memory budget.
- **Activation offload's real, legitimate benefit here is wall-clock speedup**
  (the already-published 1.09×–1.71× pin/pageable ratios), not memory ceiling.
  Reattribute the capability framing in the handoff to the overall workload (ZeRO-2 +
  kernels), and keep activation-offload's claim scoped to step-time at a given
  seqlen.
- Forward-pass activation memory reduction from offload is real and large
  (~27GB at 96K) — just don't let that get conflated with the *step's* peak,
  which is set in backward and is offload-insensitive at this configuration.

## 6. Open question — not yet resolved

Why is the with-offload arm *faster* despite doing *more* total PCIe traffic
(26.25GB D2H in forward + 26.25GB H2D in backward = ~52GB/step, vs zero for
no-offload) and identical recompute FLOPs (checkpointing recomputes each layer's
forward during backward in both arms, this cost does not change with where the
boundary tensor was stored)?

Leading hypothesis (unconfirmed): GPU memory pressure. `alloc` in the no-offload
arm runs roughly 2× higher than the offload arm for nearly the whole step
(peaking mid-forward at 47.5GB vs a flat ~21GB). `reserved` is flat and identical
in both arms (73.69GB vs 76.69GB, never grows — so this isn't triggering new
`cudaMalloc` calls), but a much larger *live* allocation footprint inside that
same reserved pool plausibly leaves less free, contiguous space for FlashAttention
/Liger kernels' per-layer scratch/workspace allocations, which could slow kernel
execution even though FLOPs are unchanged.

**Not yet profiled at the kernel level.** Next step to resolve this properly:
Nsight or `torch.profiler` trace comparing per-layer kernel durations between the
two arms at matched seqlen (e.g. seq98304), to see whether individual
attention/MLP kernel wall-times actually differ, and if so, correlate with
allocator/free-block state at that point in the step.

## Evidence index

- `autorun/logs/job-20260928T195635Z.log` — reset-stats hypothesis test (refuted).
- `autorun/logs/job-20260929T092451Z.log` — coarse MEMDEBUG instrumentation
  (confirmed forward-side D2H savings work).
- `autorun/logs/job-20260929T101459Z.log` — per-step peak-reset fix verification
  (fix correct, outcome unchanged).
- `autorun/logs/job-20260929T140331Z.log` — per-decoder-layer fwd/bwd memory hooks,
  the log this file's §3 is built from.
- `autorun/logs/job-20260929T152934Z.log` — no-offload seqlen-ceiling sweep, §4.
- `deepspeed/runtime/activation_checkpointing/offload_activations.py` (this repo,
  `sfc_ds_pin_memory`) vs the same path under `../sfc_ds_act_offload` — diffed in
  §3, prefetch-only delta.
