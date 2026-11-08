# Implementation Plan — Credit-Bounded Saved-Intermediate Activation Offload (blog application 3)

**Audience:** agent implementing the benchmark and collecting numbers.
**Purpose:** experiment collection for the pinned-offload blog — **pin on/off impact on this algorithm**. PCIe primary, C2C optional.
**Repo for the work:** `deepspeedai/DeepSpeedExamples`, branch `tjruwase/pin-memory-experiments`.
**Blog plan:** [`BLOG_PLAN.md`](BLOG_PLAN.md) · **Sibling harness handoff:** [`EXPERIMENTS_HANDOFF.md`](EXPERIMENTS_HANDOFF.md) · **Reproduce the H200 numbers:** [`SAVED_ACTIVATION_REPRO.md`](SAVED_ACTIVATION_REPRO.md)
**Not a DeepSpeed feature.** No `ds_config` surface, no production API. See §12 for the promotion path.

This is **application (3)** in the blog, and a **peer in the publish set** ([`BLOG_PLAN.md`](BLOG_PLAN.md)): saved-intermediate (MLP) activation offload **without** rematerialization. It is a different algorithm from application (2) (`CheckpointHiddenStatesOffload`, checkpoint hidden states **with** remat). Checkpoint+remat appears only as a contextual reference panel, never as the pin-impact denominator.

---

## 0. Settled design (requirements, not open questions)

| # | Requirement |
|---|-------------|
| D1 | Credit unit = **one layer's offload group**, not one tensor |
| D2 | Acquire credit **before** the layer allocates its targeted activations (`forward_pre_hook`), not in the pack hook |
| D3 | Issue D2H **only after** acquiring the credit; never enqueue copies beyond the credit limit |
| D4 | Forward release order: D2H event complete → drop GPU **source** refs → return **forward** credit |
| D5 | Forward acquisition blocks with **host-side polling of `event.query()`**, *not* device-side `compute_stream.wait_event()`. Device-side waits leave HBM unbounded while timings still look correct. Do not use unbounded `event.synchronize()` for the timeout path |
| D6 | Separate forward (N) and backward (M) credit pools; acquisition timeout **raises**, never hangs |
| D7 | Alias guard against **scope-input** storage (`untyped_storage()` identity) and `model.parameters()` storage. Do not rely on pack-time output aliases — outputs are not known until the post-hook |
| D8 | Cache repeat **unpack of the same slot** during one backward. Reject a **second backward** (`retain_graph=True` then another `backward()`), not a second unpack |
| D9 | Credits bound **HBM only**. Host RAM holds *all* offloaded layers until their H2D completes; pin locks that whole footprint (`ulimit -l`) |
| D10 | Published config **N=2, M=2**. N sweep `{1,2,4}` validates **credit caps and backpressure**, not a rigid timing equation |
| D11 | Pin arms differ **only** in host buffer pinning: identical N, M, frozen selector, bytes, streams, policy |
| D12 | Backward credit lifetime: dest alloc → H2D → last backward consumer → compute-stream completion event → drop GPU dest refs → return **backward** credit. Arming on `h2d_event` only bounds copies, not restored HBM |
| D13 | H2D waits on that group's D2H event (`s_copy.wait_event(g.d2h_event)`) before reading host buffers |
| D14 | Capture `compute = current_stream()` **before** entering the copy-stream context, then `s_copy.wait_stream(compute)`, then copy. Waiting inside `with stream(s_copy)` waits on the copy stream itself |
| D15 | Frozen selector from a **discovery pass**. Pack-time "K largest" is unknowable until every save in the layer has occurred |
| D16 | `saved_tensors_hooks` wraps **forward and backward**. Drain both pools after the step. `arm` / cancel in `finally` so exceptions do not leak credits |

**Steady-state intuition (not a correctness oracle).** If a layer were a single two-stage machine (all tensors ready together, then one D2H), Little's law would give

```
T = max( t_f , t_d , (t_f + t_d) / N )
```

so `T = t_f + t_d` at N=1 and `T = max(t_f, t_d)` for **N ≥ 2**. Real MLPs save tensors mid-layer and other layer work sits between MLPs, so N=2 need not match `max(t_f, t_d)` within 10%. Occupancy saturates near **2** for N≥2 unless `t_d ≳ (N-1) t_f`. Extra HBM at N=4 ≈ N=2 on a compute-bound Qwen layer is **expected**, not a fail.

**Expected pageable behaviour (platform-dependent, measure it).** `copy_(non_blocking=True)` into pageable host memory is often host-synchronous, so credits may never bind: `T_pageable ≈ t_f + t_d` at any N. State in the method if that is what you observe. The claim is *"pinning is what makes credit-paced overlap possible when pageable copies block the host,"* not *"pinning always makes copies faster."* Do not hard-require `blocking_acquisitions ≈ 0` on pageable.

---

## 1. Where the code lives, and why

**All of it in DeepSpeedExamples.** New subfolder `benchmarks/pin_memory/saved_activation_offload/`, alongside the existing `activation_offload/`, `model_tensor_offload/`, `h2d_d2h/`.

Rationale:

1. **No DeepSpeed change is required.** The credit engine only calls already-public accelerator APIs on `master`: `get_accelerator().pin_memory(t, make_copy=False)`, `.is_pinned()`, `.unpin_memory()`, `.Stream`, `.Event`, `.current_stream()`, `.on_accelerator()`, `.is_synchronized_device()`. Verified against `accelerator/abstract_accelerator.py` and `accelerator/cuda_accelerator.py`.
2. It is an experiment, not a feature. Landing credit scheduling on DeepSpeed `master` would force a config surface decision that the blog explicitly does not want to make.
3. `benchmarks/pin_memory/common.py` already provides the subprocess-arm plumbing every sibling bench uses (`run_arm_subprocess`, `print_arm_result`, `print_driver_result`, `parse_arm_result`, `dist_env`, `build_hf_causal_lm`, `param_numel`, `patch_cpp_extension_drop_cxx17`).

**XPU / Arc Pro B70 is out of scope for this experiment plan.** Do not treat the native-vs-torch pin-backend issue on XPU as a prerequisite or publish gate for application (3). Applications (1) and (2) may still fix that harness separately.

**Ascend 910B4 is not a collection target.** This bench is CUDA-only and does not run on that NPU. The blog states that limitation and does not invent a number. **GH200 is measured** (`ds-pin-Xinyu-experiment.pdf`, `galleon` GPU0): Qwen3-8B, batch 1, K=-1, N=2, ZeRO-2 CPU AdamW, seq 8K/16K/32K. Pin vs pageable step time 1.14× / 1.13× / 1.17×. Report it on its own panel. Do not copy the H200 ratio. GB200 stays optional.

---

## 2. Out of scope — XPU pin-backend

Ignored here. The existing DSE benches force `DS_PIN_MEMORY_BACKEND=native` on pinned arms; that is CUDA-correct and insufficient on XPU. This plan does not fix those files and does not collect B70 rows for application (3).

Pinned arm for this bench: CUDA `native` + `REGISTER_DEVICE=1`, plus `is_pinned` assertion. No XPU backend branch.

---

## 3. File layout

Keep the bench small. Policy and JSON assembly live in `bench.py` until they become independently large.

```
benchmarks/pin_memory/saved_activation_offload/
├── README.md            # repro commands, arms, known limitations
├── bench.py             # CLI, arms, driver, discovery/manifest, metrics JSON
└── credit_offload.py    # CreditPool, HostBufferPool, CreditBoundedOffloadEngine,
                         # SavedIntermediateOffload (saved_tensors_hooks ctx)
```

Keep `credit_offload.py` free of argparse and HF imports — it is the piece that could be promoted (§12). Do not add `policy.py` / `metrics.py` unless a file exceeds ~400 lines.

---

## 4. Reuse vs. new code

**Decision: compose, do not subclass or fork `_ActivationOffloadEngine`.**

`deepspeed/runtime/activation_checkpointing/offload_activations.py` is **not** credit-bounded. `_reap_forward_stash` (L132) drops GPU references *without blocking*, relying on `record_stream` to defer allocator reuse; `_start_offload` (L183) enqueues a copy unconditionally; `_restore_tensor` (L236) has no prefetch. The credit protocol inverts ownership of exactly those three paths. Subclassing would mean overriding almost every method except the buffer pool while staying coupled to a private class on DeepSpeed `master`. That is negative reuse.

| Reused | How |
|--------|-----|
| Pin call | `get_accelerator().pin_memory(host, make_copy=False)` — the identical call `_empty_cpu_like` makes at `offload_activations.py:180`, so pin accounting, `DS_PIN_MEMORY_BACKEND`, and the tracker behave exactly as in #8282 |
| Skip predicate | Port `_should_skip` (L123): synchronized device, not on accelerator, `< min_offload_bytes`, `nn.Parameter`, `nn.Buffer`, stride-0 / internally overlapping views. Copy with a comment crediting the source |
| Host buffer pool *design* | Key `(shape, dtype, layout, pinned)`; pending-reap list gated on a stream event. Reimplemented because the credit scheduler must own release ordering |
| Restore metadata | Same as #8282: save `(device, size, stride, dtype)` at pack; restore with `empty_strided`. Host staging buffers are dense, so `empty_strided_like(host)` is wrong |
| Remat reference panel | Import `CheckpointHiddenStatesOffload` **as-is** for `--mode remat`. Also call `gradient_checkpointing_enable(use_reentrant=False)` or there is no remat |

| Newly written | Why |
|---------------|-----|
| `CreditPool` | No equivalent exists |
| Layer-scoped acquire/arm hooks | #8282 marks at `GradientCheckpointingLayer.__call__`; we scope to MLP submodules with checkpointing **off** |
| Discovery pass + frozen selector (D15) | Pack-time K-largest is impossible |
| Input/parameter storage-identity alias guard (D7) | `isinstance(nn.Parameter)` misses transposed views of weights |
| Bounded H2D prefetch with dest-HBM credits (D12–D13) | `_restore_tensor` is just-in-time and has no D2H barrier |
| Generation / second-backward guard (D8) | #8282 caches repeat unpacks; we cache slot unpacks too, and reject a second graph traversal |

v1 is single-GPU, no ZeRO-3 param gather. Still assert zero selected storages shared with `model.parameters()`.

---

## 5. Credit scheduler API and protocol

v1 uses **one copy stream** (`s_copy`). D2H then H2D on the same stream still requires recording `d2h_event` and waiting on it before H2D of that group (later groups' D2H may already be queued). A second stream is optional after D13 is proven.

### 5.1 API

```python
class CreditTimeout(RuntimeError): ...
class CreditDeadlock(RuntimeError): ...

@dataclass
class SlotMeta:
    shape: tuple
    stride: tuple
    dtype: torch.dtype
    device: torch.device
    storage_id: int
    nbytes: int
    pack_index: int

@dataclass
class Credit:
    tag: str                      # "fwd:L12" / "bwd:L12"
    layer_idx: int
    id: int                       # stable group id
    nbytes: int = 0
    gpu_refs: list = field(default_factory=list)   # sources (fwd) or dests (bwd)
    host_bufs: list = field(default_factory=list)
    slots: list = field(default_factory=list)      # SlotMeta
    d2h_event: "torch.Event | None" = None
    h2d_event: "torch.Event | None" = None
    consume_event: "torch.Event | None" = None     # bwd: last consumer done
    release_event: "torch.Event | None" = None     # pool waits on this

class CreditPool:
    def __init__(self, capacity: int, timeout_s: float = 60.0): ...
    def acquire(self, tag: str, layer_idx: int) -> Credit: ...  # host-blocking; raises
    def arm(self, credit: Credit, event) -> None: ...
    def cancel(self, credit: Credit) -> None: ...  # unarmed acquire; return credit
    def drain(self) -> None: ...                   # host-sync outstanding, drop refs, free=capacity
    # instrumentation: held = capacity - free, includes unarmed in-compute credits
    wait_s: float
    max_held: int
    max_held_bytes: int
    max_outstanding: int          # armed only; do not use as the HBM bound
    acquisitions: int
    blocking_acquisitions: int
```

`drain()`: host-sync all outstanding → drop `gpu_refs` / dest bufs → `free = capacity`. Call after every step and in `except`.

### 5.2 Acquire / release (D2–D6)

```python
def _reap(self):
    # D4 / D12: wait event complete -> drop GPU refs -> return credit.
    while self._outstanding and self._outstanding[0].release_event.query():
        credit = self._outstanding.popleft()
        credit.gpu_refs.clear()
        self._free += 1

def _host_wait(self, event, deadline, tag):
    # D5 / D6: poll. event.synchronize() cannot be interrupted by the deadline.
    t0 = time.perf_counter()
    while not event.query():
        if time.perf_counter() >= deadline:
            raise CreditTimeout(f"{tag}: no credit after {self.timeout_s}s")
        time.sleep(self._poll_s)          # ~1ms
    self.wait_s += time.perf_counter() - t0
    self.blocking_acquisitions += 1

def acquire(self, tag, layer_idx):
    self._reap()
    deadline = time.perf_counter() + self.timeout_s
    while self._free == 0:
        if not self._outstanding:
            raise CreditDeadlock(f"{tag}: no free credit and none outstanding")
        self._host_wait(self._outstanding[0].release_event, deadline, tag)
        self._reap()
    self._free -= 1
    self.acquisitions += 1
    self.max_held = max(self.max_held, self.capacity - self._free)
    return Credit(tag=tag, layer_idx=layer_idx, id=self._next_id())

def arm(self, credit, event):
    credit.release_event = event
    self._outstanding.append(credit)
    self.max_outstanding = max(self.max_outstanding, len(self._outstanding))
    self.max_held_bytes = max(self.max_held_bytes,
                              sum(c.nbytes for c in self._held_credits()))
```

Host-side wait on **forward** acquire stops the host allocating future activations. A device-side `compute_stream.wait_event()` here produces similar step times and **unbounded** HBM. Instrument `max_held` (includes the unarmed in-compute credit). `max_outstanding` under-counts by 1 if used as the bound.

Timeout test (V5): arm a credit with a dummy Event that is **never recorded**; `--credit-timeout-s 1` must raise `CreditTimeout` and not hang.

### 5.3 Discovery, then forward path (D15, D7, D14)

**Discovery (once per recipe, before either pin arm).** Run one untimed forward with `saved_tensors_hooks` that **records** but does not copy: `(layer_idx, pack_index, storage_id, shape, stride, dtype, nbytes, aliases_param, aliases_input)`. Freeze a selector keyed by `(layer_idx, pack_index, signature)`: eligible, not param/input alias, not overlapping view, `nbytes >= min_offload_bytes`, then keep K unique storages by `(-nbytes, pack_index)`. Dedup repeated saves of the same storage to the same slot. Persist the manifest; both pin arms load it and **abort on signature drift**.

Do **not** rank K-largest online in the pack hook.

```python
def scope_pre_hook(module, args):            # D2
    if tls.in_scope:
        raise RuntimeError("nested target-module execution is not supported")
    idx = module._ds_layer_idx
    tls.credit = fwd_pool.acquire(f"fwd:L{idx}", idx)
    tls.in_scope = True
    tls.scope_input_storages = storages_of(args)   # untyped_storage() ids

def pack_hook(t):
    if not tls.in_scope or should_skip(t):
        return t
    if aliases_storage(t, tls.scope_input_storages) or aliases_params(t, model):
        return t                             # D7
    key = (tls.credit.layer_idx, tls.pack_index, signature(t))
    tls.pack_index += 1
    if key not in frozen_selector:
        return t
    slot = frozen_selector[key]
    if slot.storage_id in tls.credit.seen_storages:
        return _Ref(group_id=tls.credit.id, slot=tls.credit.storage_to_slot[slot.storage_id])
    compute = accel.current_stream()         # D14: capture BEFORE copy-stream ctx
    host = host_pool.get_dense(t, pinned=self.pin)
    self.s_copy.wait_stream(compute)
    with accel.stream(self.s_copy):
        host.copy_(t.detach(), non_blocking=True)
    t.record_stream(self.s_copy)
    tls.credit.gpu_refs.append(t)
    tls.credit.host_bufs.append(host)
    tls.credit.slots.append(SlotMeta(...))
    tls.credit.nbytes += slot.nbytes
    return _Ref(group_id=tls.credit.id, slot=len(tls.credit.slots) - 1)

def scope_post_hook(module, args, output):   # register with always_call=True
    try:
        if tls.credit is None:
            return
        if not tls.credit.slots:
            fwd_pool.cancel(tls.credit)
            return
        tls.credit.d2h_event = self.s_copy.record_event()
        fwd_pool.arm(tls.credit, tls.credit.d2h_event)
        groups[tls.credit.layer_idx] = tls.credit
    finally:
        tls.in_scope, tls.credit, tls.pack_index = False, None, 0
```

`record_stream` is allocator safety. HBM bounding is the explicit `gpu_refs` list released at reap.

Install **matching no-op backward hooks** in `--mode none` when checking V2, so view/storage side effects of `full_backward_pre_hook` are present in the baseline.

Reject unsupported views (internal overlap, nonzero storage offset if restore assumes offset 0).

### 5.4 Backward path (M dest-HBM credits, published M=2)

```python
def _issue_h2d(g):                           # always under a bwd credit
    if g.h2d_started:
        return
    credit = bwd_pool.acquire(f"bwd:L{g.layer_idx}", g.layer_idx)
    compute = accel.current_stream()
    self.s_copy.wait_stream(compute)
    self.s_copy.wait_event(g.d2h_event)      # D13: host buf must be complete
    g.gpu_bufs = [
        torch.empty_strided(s.shape, s.stride, dtype=s.dtype, device=s.device)
        for s in g.slots
    ]
    with accel.stream(self.s_copy):
        for dst, src in zip(g.gpu_bufs, g.host_bufs):
            dst.copy_(src, non_blocking=True)
        g.h2d_event = self.s_copy.record_event()
    g.h2d_started = True
    credit.gpu_refs = g.gpu_bufs             # dest HBM held until consume
    g.bwd_credit = credit
    # do NOT arm on h2d_event — that would release dest HBM too early (D12)

def last_layer_backward_pre_hook(module, grad_output):
    # skip-last layer never offloaded; use it to seed prefetch of L-2 .. 
    seed_prefetch_window(start=module._ds_layer_idx - skip_last)

def scope_full_backward_pre_hook(module, grad_output):
    # Issue the next group entering the M-layer window only after a slot is free.
    issue_window_ending_at(module._ds_layer_idx - 1)

def scope_full_backward_post_hook(module, grad_input, grad_output):
    g = groups.get(module._ds_layer_idx)
    if g is None or g.bwd_credit is None:
        return
    g.consume_event = accel.current_stream().record_event()
    bwd_pool.arm(g.bwd_credit, g.consume_event)   # D12
    # after consume_event: host bufs may already have been pooled at h2d_event

def unpack_hook(ref):
    if not isinstance(ref, _Ref):
        return ref
    g = groups[ref.group_id]                 # group_id == layer_idx
    if not g.h2d_started:
        _issue_h2d(g)                        # still acquires a bwd credit
    if g.restored[ref.slot] is None:
        accel.current_stream().wait_event(g.h2d_event)
        g.restored[ref.slot] = g.gpu_bufs[ref.slot]
    g.unpack_count[ref.slot] += 1
    return g.restored[ref.slot]              # D8: cache; do not raise on >1
```

**Prefetch window (no deadlock).** At most M dest groups live. Seed when backward hits the skipped last layer (attach a prefetch-only `full_backward_pre_hook` there). At each offloaded module's **post-hook**, retire that layer's dest credit, then issue the next layer entering the window. Do **not** at layer `ℓ` acquire credits for `ℓ-1 … ℓ-M` while still holding dests for `ℓ` and `ℓ-1`.

**Host vs dest events.** Recycle host staging after `h2d_event`. Recycle GPU dest after `consume_event`. Separate events.

**Hooks context.** `with SavedIntermediateOffload(...):` must cover `loss.backward()`. Exiting before backward drops packed refs.

**Hook order (Phase 1).** On a 2-layer toy, assert layer-`ℓ` backward pre-hook runs before any `_Ref` unpack for layer `ℓ`. Qwen3 `MLP` returns one tensor; v1 supports that contract. If the order test fails, default the prefetch trigger to a sentinel `torch.autograd.Function` on the scope output (DeepSpeed already warns about `full_backward_pre_hook` in `OutputBackwardHookManager`).

**Second backward (D8 / V5).** Per-step generation counter. First `backward(retain_graph=True)` may unpack slots twice — allowed. A second `backward()` on the same graph raises.

**Device-side wait on unpack** is correct: dests are already allocated; M bounds issuing dest HBM. Still wait D2H→H2D on the **copy** stream (D13). That is not the same as a device-side wait on forward acquire.

**M=2.** M=1 issues ℓ−1 at ℓ's backward entry; short backward of ℓ stalls every layer. Sweep M only as a diagnostic if dest H2D tails dominate.

---

## 6. Module scoping and tensor selection

Lives in `bench.py` (discovery + frozen manifest).

- `--offload-modules REGEX`, default `r"(?:^|\.)layers\.\d+$"` — widened from the original `r"\.mlp$"` (HF Qwen3 `model.layers.<i>.mlp`) to match the **whole decoder layer** `model.layers.<i>`, not just its MLP submodule. Narrow MLP-only scope undercounted eligible saved tensors (attention projections/residuals outside the MLP were never offloaded); the widened scope is the one used for all published credits/k ablations below.
- Resolve matches, attach `_ds_layer_idx`, **assert exactly one scope module per decoder layer**. More than one breaks D1; raise.
- `--skip-last-layers` (default `1`): last MLP stays resident (no D2H). Count those bytes as `+K` outside N, same on both pin arms. This is **not** `#8282` keep-last (one tensor).
- Eligibility: ported `_should_skip`, plus D7 storage-identity vs scope **inputs** and `model.parameters()`.
- `--offload-tensors K` (int, default `-1`): K unique eligible storages from the **frozen** discovery ranking `(-nbytes, pack_index)`. `-1` is a sentinel meaning **unlimited** — offload every eligible storage in scope, no top-K cap; this replaces the original CLI restriction to `choices=(1,2,3)` and is the default for all published runs. Capping to a small K (validated at K=1 under the widened scope) is a **negative result**: the ~20 non-selected eligible tensors/layer fall back to always-resident instead of being offloaded, which *raises* peak HBM relative to `-1` — do not treat the old K=1/2/3 sweep as a recommended tuning axis. Unfused SwiGLU may save four similar `[B,S,d_ffn]` tensors; do **not** caption the set as gate/up/product unless the manifest names them.
- `--min-offload-bytes` (default 1 MiB).
- `--manifest-out PATH`: per-layer `(pack_index, shape, stride, dtype, nbytes, storage_id, selected)`. V1 diffs this across pin arms.

---

## 7. Bench CLI and arms

`bench.py` mirrors the driver/arm subprocess shape of `activation_offload/bench.py`.

```
--model / --hidden --layers        HF id (published) or synthetic (smoke only)
--batch --seq --steps --warmup     default --steps 20 --warmup 5
--credits N                        forward credits, default 2 (validated HBM-tuning
                                    lever: credits=1 ablation gives 8-10% HBM savings
                                    at seq>=32768, negligible step-time cost, +/-2.3%)
--prefetch M                       backward dest credits, default 2
--offload-tensors K                int, default -1 (unlimited; was 1|2|3, now a
                                    sentinel -- K-capping is a validated negative
                                    result, see Sec. 6)
--offload-modules REGEX            default (?:^|\.)layers\.\d+$ (whole decoder
                                    layer; was \.mlp$)
--min-offload-bytes                default 1048576
--skip-last-layers                 default 1
--credit-timeout-s                 default 60
--mode {offload,none,remat}        none = no-offload; remat = AC reference
--manifest-out PATH
--n-sweep                          driver: N in {1,2,4} x pin (validation)
--pin 0|1                          internal: single arm
```

Default driver: two subprocesses, fresh rendezvous, **A/B/B/A** (or ≥3 fresh processes per arm, alternating).

| Arm | Host pool | Backend |
|-----|-----------|---------|
| `pageable` | `torch.empty(..., device="cpu")` | n/a |
| `pinned` | `get_accelerator().pin_memory(..., make_copy=False)` | CUDA: `native` + `REGISTER_DEVICE=1` |

Everything else identical (D11), including the frozen manifest from discovery.

`--mode remat`: `model.gradient_checkpointing_enable(use_reentrant=False)` **then** `CheckpointHiddenStatesOffload`. Without enabling checkpointing, V3 never sees remat.

`--mode none`: may use a **smaller V2 recipe** (short seq / fewer layers) if full saved activations OOM. Pin T/F on the published recipe must still match each other.

Published defaults: `N=2, M=2, K=-1, skip-last-layers=1, steps=10, warmup=1` (widened scope; the earlier `K=3, steps=20, warmup=5` defaults were from the narrow-scope/MLP-only phase and are superseded). `--credits 1` is the validated tuning ablation off these defaults, not a new default.

---

## 8. Validation — correctness gates before performance gates

Run in order. A failed correctness gate blocks all performance collection.

### Correctness

| Gate | Check | Pass |
|------|-------|------|
| **V0** Environment | pinned arm `native` + `REGISTER_DEVICE=1`; sample buffer `is_pinned`; `ulimit -l` ≥ projected pinned bytes | CUDA SKUs, logged |
| **V1** Manifest | frozen selector loaded by both arms; byte-identical; `offloaded_bytes` equal; unique non-param non-input-alias storages; groups `== L - skip_last` | exact |
| **V2** Numerical parity | (a) pin T vs F, same recipe; (b) vs `--mode none` on a recipe that **fits** | (a) tight (same graph); (b) bf16 rel 1e-2 loss / 2e-2 grad norm when (b) is run. Matching no-op backward hooks on `none` |
| **V3** No remat | scope-module forward calls per step | `== L` for offload and none. Remat count is **descriptive** (non-reentrant checkpoint may not be exactly `2L`) |
| **V4** Credit cap (replaces linear-peak-HBM) | `max_held_fwd ≤ N`, `max_held_bwd ≤ M` every step, both arms; no layer starts targeted alloc without a fwd credit; live D2H source groups ≤ N | hard. Transfer-bound **synthetic**: blocks at N, never exceeds N. **Do not** fail HF if peak allocated is flat for N=2 vs N=4 |
| **V5** Failure modes | dummy never-recorded Event + `--credit-timeout-s 1`; `backward(); backward()` after `retain_graph=True` on the first | `CreditTimeout`; second backward raises; neither hangs. First `retain_graph=True` backward must **succeed** |
| **V6** Host accounting | host bytes vs **sum of manifest bytes** for offloaded groups | within 5%. Confirms D9 |

Optional synthetic for D5 (host-wait vs device-wait): enlarge tensors / shrink compute until `r > 3`, then extra HBM may rise with N. Not required on Qwen.

### Performance

| Gate | Check | Pass |
|------|-------|------|
| **P1** N-sweep backpressure | transfer-bound synthetic | N=1: no overlap beyond noise. Credit cap never violated. N=2 vs N=4 **reported**, not forced equal. Continued improvement past N=2 is investigated, not auto-classified as a release bug |
| **P2** Bandwidth | in-situ D2H GB/s vs `h2d_d2h` **or** vs the same tensor-size manifest | diagnostic; flag if pageable in-situ is far below microbench (inflated pin win). Not a 20% hard fail |
| **P3** Statistics | ≥20 measured steps after ≥5 warmup, A/B/B/A | median and p10–p90; pin "win" only if median gap > 2× larger within-arm spread. The driver emits `pin_gap_s`, `arm_spread_s`, `pin_separated`; treat `pin_separated: false` as noise |
| **P4** Pageable vs pin occupancy | `blocking_acquisitions`, host enqueue latency, `max_held` | **measured**, no required pageable-zero. Document if pageable never binds credits |
| **P5** `r` sweep | ~~`--offload-tensors` 1/2/3 at N=2~~ superseded: K-capping is a validated negative result (raises peak HBM via always-resident fallback), so the meaningful ablation is `--credits` 1 vs 2 at `--offload-tensors -1` | **report** `r`. Bracketing 1 is a sizing **goal** (use synthetic B/S if HF stays `r<1`). Not a collection accept gate |

Clock state locked or explicitly reported on consumer SKUs.

---

## 9. Metrics / JSON schema

`print_arm_result` payload (one per arm subprocess):

```json
{
  "experiment": "saved_activation_offload",
  "mode": "offload",
  "use_pin_memory": true,
  "pin_backend": "native", "pin_registered": true, "is_pinned_verified": true,
  "credits_n": 2, "prefetch_m": 2, "offload_tensors_k": -1,
  "k_effective_min": 35, "k_effective_max": 35,
  "offload_modules": "(?:^|\\.)layers\\.\\d+$", "skip_last_layers": 1,

  "device": "cuda", "sku": "H200",
  "model": "Qwen/Qwen3-8B", "hf_model_type": "qwen3", "transformers": "4.5x.y",
  "params_b": 8.19, "layers": 36, "batch": 1, "seq": 8192, "tokens_per_step": 8192,

  "step_median_s": 0.0, "step_p10_s": 0.0, "step_p90_s": 0.0, "steps": 20,
  "inter_mlp_entry_mean_ms": 0.0, "inter_mlp_entry_samples": 0,
  "mlp_span_mean_ms": 0.0,
  "t_d_layer_ms": 0.0, "r_d2h_over_mlp_span": 0.0, "t_h2d_layer_ms": 0.0,

  "credit_wait_fwd_ms": 0.0, "credit_wait_bwd_ms": 0.0,
  "fwd_blocking_acquisitions": 0, "bwd_blocking_acquisitions": 0,
  "fwd_max_held": 2, "fwd_max_held_bytes": 0,
  "bwd_max_held": 2, "bwd_max_held_bytes": 0,
  "fwd_max_outstanding": 2, "bwd_max_outstanding": 2,

  "d2h_copy_ms_total": 0.0, "d2h_group_span_ms_total": 0.0,
  "h2d_event_ms_total": 0.0,
  "d2h_effective_gbps": 0.0, "h2d_effective_gbps": 0.0,
  "host_enqueue_latency_ms": 0.0,

  "hbm_peak_allocated_gb": 0.0, "hbm_peak_reserved_gb": 0.0,
  "hbm_peak_discovery_gb": 0.0,
  "host_offload_bytes": 0, "host_pinned_bytes": 0, "ulimit_l_kb": 0,

  "offloaded_tensors": 0, "offloaded_bytes": 0, "groups": 35,
  "manifest_sha256": "...",
  "scope_forward_calls_per_step": 35,
  "loss_step0": 0.0, "grad_norm_step0": 0.0,
  "loss_first_step": 0.0, "grad_norm_first_step": 0.0,

  "deepspeed_commit": "...", "torch": "...", "arm_order": "A"
}
```

`pcie_gen` / `lanes` are optional logs, not required schema. `manifest_sha256` is the V1 check. `host_pinned_bytes` from `deepspeed/utils/pin_memory_tracker.py`. Do not report a derived `overlap_fraction` unless defined as `(compute + copy - step) / min(compute, copy)`.

**There is no `t_f` field, deliberately.** An entry-to-entry interval measured inside the offload run absorbs the credit stall, so a ratio built on it *falls* exactly when transfer is the binding constraint — backwards from the truth, and biased against the pinned arm, which is the arm where credits bind. Take the reference forward period from `inter_mlp_entry_mean_ms` in a `--mode none` arm with the same model, batch, sequence, and hook set; `inter_mlp_entry_mean_ms` under offload is the observed layer period `T` from §0's model, which makes it the P1 oracle rather than a denominator. `mlp_span_mean_ms` excludes the host credit wait but can still include compute queued ahead of the module.

`t_d_layer_ms` and `d2h_effective_gbps` come from individually bracketed copies. `d2h_group_span_ms_total` is the wider first-save-to-scope-exit span, which also covers the compute the pack hook waits on between saves; use it for credit-hold analysis, never for bandwidth (P2).

`hbm_peak_*` covers measured steps only: peak stats are reset after discovery and after warmup. Discovery is a full forward with every activation resident, reported separately as `hbm_peak_discovery_gb`, and it is the step most likely to run out of memory at the configurations where transfer is most constrained. `is_pinned_verified` is `null` in the pageable arm, since there is nothing there to verify.

`k_effective_min` / `k_effective_max` record what the frozen manifest actually selected. `K` is a request: a fused SwiGLU or a different transformers graph legitimately exposes fewer offloadable storages per layer, and only a layer with *nothing* offloadable is an error. With the published `K=-1` (unlimited) default, `k_effective_min`/`k_effective_max` instead report how many eligible storages the widened whole-decoder-layer scope actually found per layer (the example above uses 35, from the published Qwen3-8B widened-scope runs) rather than echoing a requested cap.

---

## 10. Hardware / run matrix for the blog

Application (3) **inverts** the SKU priority used for (1) and (2): bandwidth-critical application, bandwidth-relevant SKU leads.

| SKU | Role | Model | Rows |
|-----|------|-------|------|
| **H200 (PCIe Gen5)** | **Primary.** SKU Poolside analyses as infeasible on PCIe | `Qwen/Qwen3-8B` (or 4B if host RAM < ~256 GB). `Qwen/Qwen3.5-9B` is optional, not a publish gate | `--offload-optimizer` (ZeRO-2 CPU AdamW); pin T/F at N=2; `--offload-tensors -1` (published config, widened whole-decoder-layer scope) with `--credits` 1/2 as the sweep axis (report HBM savings, not `r`); `--mode remat` optional panel |
| One consumer CUDA (4090/5090) | Secondary | `Qwen/Qwen3-1.7B` (or 0.6B) | pin T/F at N=2 |
| **GH200** | **Measured.** `galleon` GPU0. Own panel. Do not copy the H200 ratio | `Qwen/Qwen3-8B`, batch 1, seq 8192/16384/32768 | pin T/F, K=-1, N=2, ZeRO-2 CPU AdamW. 1.14× / 1.13× / 1.17× |
| GB200 | Optional | same | no section unless a node was actually run |
| Ascend 910B4 | Named limitation, not a row | — | CUDA-only script, not runnable on the NPU. Do not invent a number |

Validation (`--n-sweep`, synthetic) on whichever CUDA SKU is available; **not** blog rows.

**Qwen3.5 dense is optional for application (3), not a publish gate.** `Qwen/Qwen3.5-9B` resolves to a
text-only config (`model_type=qwen3_5_text`, `num_experts=None`, 32 layers, hidden 4096) via
`get_text_config()` — confirmed dense, not MoE. It already passed the V0/V1/V2(a)/V3/V6 correctness
gates and `--n-sweep` performance gates used for Qwen3-1.7B and Qwen3-8B in Phase 2/3. Those gate results are withdrawn (untraceable to a source job); re-run before citing them. The old Phase 4 K-sweep (`--offload-tensors` 1/2/3 at N=2) is superseded by the widened-scope negative result in Sec. 6 — if Qwen3.5-9B is run, use the `--credits` 1/2 sweep at `--offload-tensors -1` alongside Qwen3-8B on H200 instead. Multimodal or MoE Qwen3.5 checkpoints remain excluded (§13).

**Host RAM (D9):** sum of manifest bytes, not `N` groups. Qwen3-8B, B·S=8192, K=3, ~35 groups: plan ~21 GB locked. V0 fails if `ulimit -l` is too small.

---

## 11. Intel XPU / B70

Out of scope. Do not block implementation or collection on XPU pin-backend, overlap smoke, or B70 rows.

---

## 12. If this is later promoted to a DeepSpeed feature

Notes only — do not structure the bench around an unapproved API.

| Area | Change |
|------|--------|
| Location | `credit_offload.py` → `deepspeed/runtime/activation_checkpointing/credit_offload.py`. Apache-2.0 + DeepSpeed Team header |
| Shared pool | One host-buffer helper shared with `_ActivationOffloadEngine` |
| Config | Undesigned here |
| D8 | Support `retain_graph` with per-slot refcount |
| Distributed | Storage-identity vs ZeRO-3 gathered weights; credit stall vs comm watchdog |
| Tests | `tests/unit/v1/pin_memory/` plus on-device `SimpleModel` per `AGENTS.md` |

---

## 13. Non-goals

- No `ds_config` field, no DeepSpeed commit for the benchmark itself.
- No MoE, no FP8, no multimodal or MoE Qwen3.5 checkpoints for this application. Dense text-only
  Qwen3.5 (e.g. `Qwen/Qwen3.5-9B`, resolved via `get_text_config()`) may be run, but it is not an H200
  publish gate (§10).
- No fused SwiGLU, no `torch.compile`, no FSDP / TP / PP / EP / multi-GPU / ZeRO-3 in v1.
- No `retain_graph` **support** (second backward is rejected; first `retain_graph=True` unpacks are cached).
- No required `r=1` bracket, no required linear HBM-vs-N slope, no M sweep unless dest H2D tails dominate.
- No NVMe, no DataLoader pin.
- No XPU / B70 rows as a prerequisite.
- No C2C section unless a node was run. Poolside 6–13% cited **once**, attributed to GB200, never on PCIe charts.
- No single "offload vs recompute" speedup number.

---

## 14. Phased task list

### Phase 1 — Credit engine, synthetic

`credit_offload.py` + synthetic MLP stack.
**Accept:** V5 (timeout via dummy Event; second backward raises; first `retain_graph` unpacks succeed). V4: `max_held ≤ N/M`. 2-layer toy: backward pre-hook of layer `ℓ` before `_Ref` unpack of `ℓ`. D14 wait-stream-before-copy-context. Activations dominate peak (size the stack so V4 is not noise). P1 synthetic: N=1 serializes; cap never exceeded.

### Phase 2 — HF model, frozen selector, correctness

Discovery pass, `--model`, alias/param storage guard, manifest.
**Accept:** V0, V1, V2(a), V3 (`== L`), V6 on one CUDA SKU with `Qwen/Qwen3-1.7B` (or a seq that fits). Manifests byte-identical across pin arms. No remat.

### Phase 3 — Performance validation

`--n-sweep`, event probes, enqueue latency.
**Accept:** P1 (synthetic intuition + cap), P3, P4 **measured**. Do not block on P2 20% or P5 bracketing.

### Phase 4 — Blog collection

§10 at published defaults `N=2, M=2`.
**Accept:** H200 pin T/F JSON; consumer CUDA or a named gap; `DRIVERRESULT` outside the blog README per `EXPERIMENTS_HANDOFF.md` §3. Do not wait on XPU/B70 or `r>1`.

### Phase 5 — Reference panel (optional, do not block)

`--mode remat` with checkpointing **enabled**. Separate captioned panel. Expect (3) to **lose** to recompute on PCIe.

---

## 15. Rules

- Commits `--signoff`; push DSE to `sfc-gh-truwase`; PR against `deepspeedai/DeepSpeedExamples`.
- Land or close [DSE #1015](https://github.com/deepspeedai/DeepSpeedExamples/pull/1015) before opening this one; keep the subfolder layout.
- Do not edit `BLOG_PLAN.md` or `blogs/pinned-offload/08-2026/README.md`. Hand numbers back to the blog owner.
- Do not update `git config`. Do not commit the nested `DeepSpeedExamples/` clone into the DeepSpeed repo.
