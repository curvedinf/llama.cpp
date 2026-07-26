# Optimization Log - C=16 x 4k seq len

Target workload: 16 concurrent sequences x 4096-token prompts, 128 generated tokens
(`llama-batched-bench -npp 4096 -ntg 128 -npl 16`, n_ctx = 67584).

- Model: `/home/chase/Projects/SDGraft/checkpoints/Qwen3.5-0.8B-MTP/Voodoo80/Qwen3.5-0.8B-MTP.Voodoo80_Q6_K.gguf`
- GPU: Radeon RX 7900 XTX (Vulkan backend, flash attention on, KV cache q8_0)
- Flags: `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512`
- Harness: `./bench-c16.sh <build_dir> <label> s4k`
- Primary metrics: S_PP t/s (prefill), S_TG t/s (decode, the concurrency-sensitive one)

## History (prior sessions, 8k quick / 128k full workloads)

| date | build | mode | S_PP t/s | S_TG t/s | note |
|------|-------|------|----------|----------|------|
| 2026-07-16 | upstream 0dc74e3 (`build`) | 16x8k | 15600.36 | 1098.13 | baseline |
| 2026-07-16 | upstream 0dc74e3 (`build`) | 16x128k | 2899.08 | 341.09 | baseline |
| 2026-07-17 | integration WIP (`build-int`) | 16x8k | 1047.27 | 915.91 | mid-debug state, PP regressed ~15x |

## Experiments (C=16 x 4k)

| date | commit | change | S_PP t/s | S_TG t/s | vs prev | status |
|------|--------|--------|----------|----------|---------|--------|
| 2026-07-17 | 0dc74e3 | upstream reference (`build` dir, pre block-pool) | 17295.12 | 1315.89 | - | measured |
| 2026-07-17 | 1ed3129 | HEAD baseline ("initial stable build of block pool kv") | 16318.88 | 968.89 | PP -5.6%, TG -26.4% vs upstream | measured |
| 2026-07-17 | this | async recurrent prefix state snapshots | 15790.55 | 1237.40 | TG +27.7%, PP -3.2% vs 1ed3129 | committed |
| 2026-07-17 | aa5a59f | (debug) fix GGML_VK_PERF_LOGGER assert with pending async copies | - | - | tooling only | committed |
| 2026-07-17 | this | strided 2D snapshot copies (38 -> 2 per snapshot) | 15976.04 | 1262.97 | TG +2.1%, PP +1.2% vs prev | committed |
| 2026-07-17 | this | lazy snapshots (only once prefix matching is used) | 16692.57 | 1365.65 | TG +8.1%, PP +4.5% vs prev | committed |
| 2026-07-17 | - | EXPERIMENT: dmmv 16 columns for n=16 decode matmuls | 16783.54 | 1238.24 | TG -9.3% vs prev | reverted (slower than mul_mat path) |
| 2026-07-17 | - | PROBE: GGML_VK_DISABLE_COOPMAT=1 | 10979.50 | 1044.75 | much worse | default (coopmat on) confirmed |
| 2026-07-17 | this | batch descriptor updates per chunk in dispatch replay | 16683.53 | 1369.33 | TG +0.1% vs prev | committed |
| 2026-07-17 | this | harness: -ub 512 -> -ub 1024 (sweep below) | 17721.15 | 1368.00 | PP +6.3% vs prev | committed (bench-c16.sh) |
| 2026-07-17 | 058402b | FINAL verification (clean tree, guarded harness) | 17711.19 | 1373.94 | - | verified, tests pass |
| 2026-07-18 | this | GDN in-place state write-back (skip snapshot + 18x16MB cpy/step) | 17748.20 | 1479.37 | TG +8.1%, PP +0.2% vs prev | committed |
| 2026-07-18 | this | conv state in-place (skip gather+concat+cpy, ~54 nodes/step) | 17733.44 | 1544.47 | TG +4.4%, PP -0.1% vs prev | committed |
| 2026-07-18 | this | conv ip shader: full-occupancy workgroups + vec4 fast path | 17541.14 | 1561.84 | TG +1.1%, PP -1.1% vs prev | committed |
| 2026-07-18 | this | fused ADD+SOFTPLUS+MUL op for the GDN gate prep chain | 17609.41 | 1572.62 | TG +0.7%, PP +0.4% vs prev | committed |
| 2026-07-18 | this | fused SILU+MUL op for the gated-norm z chain | 17848.44 | 1595.04 | TG +1.4%, PP +1.4% vs prev | committed |
| 2026-07-18 | this | extend fusion to the sigmoid gate (full-attn layers) | 17957.45 | 1597.40 | TG +0.1%, PP +0.6% vs prev | committed |
| 2026-07-18 | this | merge q+k L2 norms into one op per delta layer | 17859.77 | 1594.66 | within noise (-0.2% TG) | committed |
| 2026-07-18 | this | fuse residual ADD + RMS_NORM + MUL (input-norm chain) | 17956.21 | 1599.11 | within noise | committed |
| 2026-07-18 | - | EXPERIMENT: fold beta sigmoid into the GDN op (4 variants) | 17863.15 | 1553.90 | TG -2.8% vs prev | reverted (see note below) |
| 2026-07-18 | this | split-K for skinny-n matmuls (drop n>=tile guard in ggml_vk_guess_split_k) | 17856.80 | 1857.54 | TG +16.2%, PP -0.6% vs prev | committed |
| 2026-07-18 | this | split-K k threshold 2048 -> 1024 | 17856.40 | 1873.04 | TG +0.8% vs prev | committed |
| 2026-07-18 | this | K-quant small matmul warptile BK 32 -> 64 (halves k-loop barrier rounds) | 17911.89 | 1930.75 | TG +3.1%, PP +0.3% vs prev | committed |

ubatch sweep (b=2048 unless noted, C=16 x 4k):

| config | S_PP t/s | S_TG t/s |
|--------|----------|----------|
| ub=256 | 14005.66 | 1340.46 |
| ub=512 | 16692.57 | 1365.65 |
| ub=1024 | 17739.12 | 1370.00 |
| ub=2048 | 15650.67 | 1371.40 |
| b=4096 ub=1024 | 17662.76 | 1366.06 |
| b=4096 ub=2048 | 15475.68 | 1368.57 |

## Cross-workload check (16x8k quick, -ub 1024)

| build | S_PP t/s | S_TG t/s |
|-------|----------|----------|
| upstream 0dc74e3 | 16594.64 | 1328.85 |
| optimized | 16002.51 | 1249.38 |

At 8k, upstream is ahead (PP -3.6%, TG -6.0%) while at 4k the optimized build is
ahead (PP +2.5%, TG +4.0%). No catastrophic regression either way (the prior
integration WIP was PP 1047 / TG 915 at 8k). The 4k target is met; the 8k/128k
regimes were not optimized here.

## Final result (C=16 x 4k, -ub 1024)

| build | S_PP t/s | S_TG t/s | S t/s |
|-------|----------|----------|-------|
| upstream 0dc74e3 (-ub 512) | 17295 - 17525 | 1316 - 1356 | 12643 - 12873 |
| HEAD baseline 1ed3129 (-ub 512) | 16318 | 968 | 11025 |
| **optimized (ub 1024, in-place GDN + conv + skinny-n split-K)** | **17856 - 18006** | **1857 - 1859** | **14121 - 14160** |

vs HEAD baseline: PP +9.4%, TG +91.8%, S +28.4%. vs upstream: PP +3.2%, TG +41.2%, S +12.0%.
At 16x8k (quick): PP 16187 vs upstream 16595 (-2.5%), TG 1435 vs upstream 1329 (+8.0%).

Update (2026-07-18): skinny-n split-K raised TG from ~1599 to ~1858. The 16x8k
cross-check above predates that change.

Validation: test-prefix-cache, test-prefix-cache-e2e, test-kv-cells, test-graph-cache,
test-gdn-indexed-state, test-backend-ops -o MUL_MAT all pass. Note: a full
test-backend-ops run allocates very large GPU buffers - run it only with a VRAM
watchdog or on CPU; it is not part of the bench flow.

## Findings

- 1ed3129 regresses decode at C=16 4k: TG 968.89 vs upstream 1315.89 t/s. First
  optimization target is closing that gap, then beating upstream.
- ROOT CAUSE #1 (fixed): per-decode `memory_update()` -> `prefix_eval_pending()` ->
  `snapshot_prefix_state()` did 38 synchronous `ggml_backend_tensor_get` calls per
  128-token block per sequence (~11 ms each, ~5.5 ms/step amortized, 2.8 s of a
  8.5 s TG phase). Each get stalls the CPU until the GPU pipeline drains and
  serializes decode.
  Fix: snapshots now read back into freshly allocated host buffers of the state
  device (pinned memory on Vulkan) with `ggml_backend_tensor_get_async` - no CPU
  sync on the hot path. Entries hold the buffer via `ggml_backend_buffer_ptr`
  (`llama_prefix_cache::entry::state_buf`); `set_state` touches the LRU so a
  just-snapshotted entry is not evicted while the copy may be in flight. Restore
  synchronizes the backend first (rare path). Snapshot cost dropped to ~1 ms
  per snapshot (mostly the pinned allocation).
  Result: TG 968.89 -> 1237.40 t/s (+27.7%).
- Snapshot readback batched: per-type state tensors are uniformly strided in one
  buffer, so 38 copies/barriers became 2 strided 2D copies (synthetic span tensor
  to satisfy the 2D api bounds check). TG +2.1%.
- Snapshots made lazy: they are pure overhead for workloads that never match
  prefixes. Now taken only after the first prefix_match/prefix_copy. TG +8.1%.
- Decode is dispatch-rate limited: ~640 graph nodes per step at C=16, GPU per
  step ~10 ms, CPU per step ~0.65 ms (submit/replay ~0.45 ms of it). Barriers
  per run: 2959 (HEAD, merged-range barriers) vs 5645 (upstream).
- GDN in-place state write-back: the indexed op (K==1) writes the new state
  back to the store rows it read from instead of a snapshot area + ggml_cpy
  (18 x 16 MB of D2D copy traffic and 18 nodes removed per step). Safe only
  when every batch seq reads and writes the same row; llama_memory_recurrent
  finds this from the cell bookkeeping (src0 == own index for all running
  seqs, i.e. steady state, no empty states, no shared rows) and the graph
  falls back to the copy path otherwise (graph cache keys on the flag, so
  both variants coexist). No store resize was needed - the ring aliases only
  across steps, never within one ubatch. Validated bit-exact vs the gathered
  reference on CPU and Vulkan (test-gdn-indexed-state, in-place cases).
  Result: TG 1368.00 -> 1479.37 t/s (+8.1%).
- dmmv (dequant-mul-mat-vec) extended to 16 columns was SLOWER than the mul_mat
  path at n=16 (-9.3% TG) - serialized per-op timings (GGML_VK_PERF_LOGGER)
  mislead; wall-clock is the arbiter. Reverted.
- Conv state in-place (ggml_ssm_conv_idx): the conv1d state was gathered with
  get_rows, concatenated with the new tokens, and its tail copied back to the
  store every step (54 nodes, ~45 MB of copies at C=16). The indexed variant
  reads/writes the state rows in place under the same src0 == own index
  invariant as the GDN path (shared get_direct flag, automatic fallback to
  the gather path). Validated bit-exact on CPU and Vulkan
  (test-gdn-indexed-state conv cases incl. n_t < nc-1 and n_t >= nc-1).
  Result: TG 1479.37 -> 1544.47 t/s (+4.4%).
- PP residual gap to upstream (-3.5%): decomposes into ~1.4% op-sum difference
  (matmuls +2.8% on some shapes, GATED_DELTA_NET_IDX vs GDN +7 ms over 128
  ubatches) and ~1-2% graph-optimizer effectiveness difference (opt on/off:
  upstream gains +3.7% from graph-opt, HEAD +1.4%; ggml_vk_graph_optimize is
  byte-identical - the different graph composition changes fusion/sort outcomes).
- HEAD adds two cache layers: llama-level graph cache (`LLAMA_GRAPH_CACHE_SIZE`,
  default 8) and a Vulkan dispatch cache (record/replay of the command stream,
  `GGML_VK_DISABLE_DISPATCH_CACHE=1` to disable).
- Fixed (aa5a59f): `GGML_VK_PERF_LOGGER=1` aborted in HEAD at graph compute entry
  when async copies were pending; now flushes first.

## Future directions (not done)

- MTP speculative decoding: the checkpoint has a nextn MTP head - use it to
  draft tokens and raise effective TG throughput. Evaluating what this branch
  already supports (MTP draft context) and measuring at C=16 4k.
- Fewer/larger kernels per step (fusion of the delta-net elementwise chains);
  ~420 of the ~640 nodes per step are small elementwise/copy ops.

## MTP speculative decoding (draft-mtp) at C=16 x 4k

Measured with llama-cli, --parallel 16, -c 67584, -n 128, same 4069-token
repetitive prompt, draft model = same checkpoint (nextn head), guarded 16 GB:

| mode | S_PP t/s | S_TG t/s | note |
|------|----------|----------|------|
| baseline (no spec) | 11569.9 | 334.5 | llama-cli has per-step sampling syncs; not comparable to batched-bench |
| --backend-sampling | 11184.9 | 365.0 | TG +9.1% |
| --spec-type draft-mtp | 6755.2 | 362.6 | TG +8.4%, PP -41.6% |
| draft-mtp + backend-sampling | 6570.4 | 398.3 | TG +19.1%, PP -43.2% |

- MTP drafting works at C=16 via llama-cli --parallel N --spec-type draft-mtp
  (same checkpoint as -md). Acceptance on this trivially predictable prompt is
  ~100% (best case), yet the net generation gain is only +8.4%: at C=16 the
  batch is already 16-wide, so draft+verify multiplies work per accepted
  token. Prompt eval is ~42% slower (h_nextn extraction + draft eval).
  Combined with backend sampling (removes the per-step logits readback) the
  gain is +19.1% - the best llama-cli configuration measured.
- llama-speculative --parallel N (tree branches) crashes at
  common/sampling.cpp:154 (GGML_ASSERT(logits != nullptr)) for N > 1.
  Verified PRE-EXISTING in 1ed3129 (reproduced with a fresh worktree build of
  that commit) - not caused by the optimization commits. Root cause: the
  draft-mtp driver allocates its batch with n_seq_max = 1, so multi-branch
  (tree) drafting is not wired for MTP; llama-cli --parallel N (independent
  slots) is the working path for C > 1.

## FA / bandwidth analysis (2026-07-18)

- FA at decode at 4k: COOPMAT1 path, GQA grouped (N=4 rows), workgroups
  (split_k=6) x (2 kv heads) x (16 seqs) = 192, split_kv=768. Split-k is active.
- Marginal per-position cost from the length sweep (npp 512..8192): ~0.27 us
  per KV position per step (FA reads + new K/V writes): FA at 4k ~1.1 ms/step
  (~11%), at 8k ~2.3 ms/step (~21%).
- -fa 0 (decomposed attention): PP 4531.79, TG 548.18 - FA is already ~3x
  better than the alternative; FA is not the remaining bottleneck, the cost is
  KV-traffic-inherent.
- Per-step bandwidth accounting at C=16 4k: weights ~0.66 GB, GDN state IO
  ~0.61 GB (19 layers x 32 MB), KV ~28 MB. GDN_IDX at 641 us/step is already
  ~950 GB/s effective - bandwidth-bound, only fp16 state would cut it (numerics).
  The step (10 ms) is still dispatch/latency-bound; skinny n=16 matmul kernels
  (measured 169-340 GB/s effective) are the remaining big but deep target.

## GDN beta-sigmoid fusion (2026-07-18, negative result)

Attempted to elide the 18 per-layer beta sigmoid dispatches by folding the
sigmoid into the GATED_DELTA_NET_IDX kernel (push-constant flag, in-kernel
sigmoid on the raw beta load). The fusion itself is correct (generation
bit-identical) and the in-kernel sigmoid is free, but making the sigmoid
adjacent to the GDN op constrains the wave scheduler and costs more than the
elision saves:

| variant | S_PP t/s | S_TG t/s | note |
|---------|----------|----------|------|
| early graph expands + adjacent pair, full fusion | 17871.42 | 1555.46 | -2.7% TG vs 1599.11 |
| same, GGML_VK_DISABLE_FUSION=1 | 16888.74 | 1488.73 | graph reorder alone: ~-4.7% TG |
| no expands, span empty nodes, partial coverage | 17854.63 | 1599.68 | 1/18 fused per TG graph: par with baseline |
| no expands, span empty nodes, full coverage | 17863.15 | 1553.90 | 18/18 fused: -2.8% TG |

Also fixed mid-way: the fused dispatch case was missing from the UNARY switch,
so the first bench (PP 20203/TG 1701) silently skipped all GDN ops - invalid.
Lesson reconfirmed: any fusion win must be validated by generation identity,
not just perf numbers. Reverted entirely.

## Skinny-n split-K (2026-07-18)

Per-step op profile at C=16 4k (GGML_VK_PERF_LOGGER): MUL_MAT kernels are ~70%
of the ~10 ms step, running at 1-6.5 TFLOPS. The worst shape is ffn_down
(m=1024, n=16, k=3584, ~20% of step): 32 workgroups of 1 wave on 96 CUs, no
latency hiding. The existing split-K machinery was gated behind
n >= wg_denoms[1] (32), blocking exactly the skinny-n case; the n dimension is
already bounds-checked in the shader. Dropping that guard gives split_k=3
(96 workgroups) for m=1024/k=3584 and m=1024/k=2048.

Result: TG 1599.11 -> 1857.54 (+16.2%), PP 17956 -> 17856 (-0.6%).
Numerics: accumulation order changes (f16 per-split partials + f32 reduce);
validated with test-backend-ops -o MUL_MAT (3/3) and coherent greedy
generation matching upstream. Not bit-identical by design.

## Constraints

- VRAM budget is hard-capped at 20 GB for any benchmark or test run (the GPU
  also drives the desktop). All GPU runs must go through bench-c16.sh (it has
  the contention guard and the MEM_CAP_MB watchdog); use MEM_CAP_MB=16384 to
  keep desktop headroom. Never run test-backend-ops or other large-allocation
  binaries unguarded.

## vLLM gap-closure roadmap (2026-07-18)

Following the gap analysis, the AGENTS.md roadmap is being implemented in
order. Each phase: implement, bench, log, keep if no regression.

### Phase 1: Chunked prefill co-location (Sarathi-Serve)

Status: implemented (opt-in), no regression on s4k, no win on the small-model
decode-bound bench. Kept for the larger-model / bursty-traffic regime where
vLLM's chunked-prefill wins actually show up.

Knob: `LLAMA_PREFILL_CHUNK` (env, server only). 0 = legacy behaviour (one
slot may consume the whole `n_batch` budget with its prompt); N = cap the
per-slot prompt-add loop to N tokens per `pre_decode()` iteration, so that
running decode tokens and other slots' prefill chunks coexist in the same
logical batch. The engine's existing ubatch splitter (and the
`[TAG_RECURRENT_ROLLBACK_SPLITS]` tail grouping in
`src/llama-memory-hybrid.cpp`) handle the rest; no engine change was needed.

Change is in `tools/server/server-context.cpp:pre_decode()` (env read at the
top of the function, cap checked inside the prompt-add `while` loop). It only
affects the server; `llama-batched-bench` does not exercise it (it prefills
all sequences collectively then decodes all collectively).

Validation: a new `tools/mixed-bench/llama-mixed-bench` simulates a server
mixed phase (N_DECODE warm decoders + N_ARRIVE fresh prompts). Numbers on the
target hardware (8 + 8 x 4096, 128 decoded each):

| chunk | iters | T_s   | D_t/s  | P_t/s  | all_t/s | note |
|-------|-------|-------|--------|--------|---------|------|
| 0     | 152   | 3.230 | 634.1  | 10145  | 10779   | baseline (cap off) |
| 512   | 192   | 3.283 | 623.8  |  9981  | 10605   | -1.6% (within noise) |
| 256   | 256   | 3.797 | 539.4  |  8631  |  9170   | -14.9% (dispatch overhead dominates) |

Finding: at C=16 x 4k on a 0.8B model on RX 7900 XTX, dispatch/latency is the
ceiling (~0.65 ms CPU / 10 ms GPU per step per the bandwidth analysis below).
Splitting prefill into smaller chunks multiplies the number of iterations and
the per-iter dispatch overhead, without unblocking any bottleneck. vLLM's win
here is on larger models where prefill is heavy enough to stall decoders for
tens of ms; that is not the regime of this bench. Phase 1 stays in tree,
default off, for that larger-model regime.

s4k regression check (build baseline-recheck vs build with the change, both
chunk=0): S_PP 17902 / S_TG 1902 -> 17900 / 1883. Within noise. The change
does not run under batched-bench (server-only).

### Phase 2: Async / decoupled scheduler

Status: analyzed, deferred. The sample dependency blocks the win at this scale.

The proposed overlap was: run iteration N+1's prep (balloc->init,
memory_update, memory->init_batch, output_reserve, graph cache lookup) on a
background thread while iteration N's GPU compute runs.

The blocker: iteration N+1's prep needs iteration N's sampled tokens, which
are only available after iteration N's GPU compute completes. So CPU prep
cannot overlap with the GPU compute it depends on. The only ways around this:

(a) Speculative next-iter dispatch (predict batch structure + draft tokens
    before sampling finishes) - that is MTP, already in tree
    (`OPTIMIZATION_LOG.md:156-181`).
(b) Multi-threaded Vulkan command-buffer recording (each thread its own
    command pool; record the replay for iter N+1 while iter N computes).
    The dispatch cache already amortizes the recording cost; the remaining
    0.45 ms submit/replay is mostly driver-side `vkUpdateDescriptorSets` +
    `vkQueueSubmit` overhead, which would need either bindless resources
    (descriptor indexing) or pre-recorded secondary command buffers.

Both are invasive subsystem changes. AGENTS.md says "no new subsystems
without prior discussion", so Phase 2 stays deferred. The realistic Phase 2
win on this bench is <=5%; the rest of the gap requires Phase 4.

### Phase 3: Preemption via recompute (AGENTS.md roadmap #4)

Status: implemented (env-gated, opt-in), not live-verified.

Knob: `LLAMA_PREEMPT=1` env (server only), additionally requires
`--kv-unified`. Default off. CPU swap is intentionally not implemented
(dropped from this roadmap per AGENTS.md); recompute via the prefill path is
cheaper on a hot GPU and avoids pinned-memory contention with the snapshot
readback path.

Implementation in `tools/server/server-context.cpp`:
- New `try_preempt_active_slot()` method called from the decode retry path
  when `try_clear_idle_slots()` cannot relieve KV pressure. Previously the
  retry fell through to `n_batch /= 2`, which often just delays the failure
  because halving the logical batch does not free any KV cells.
- Victim policy: among `SLOT_STATE_GENERATING` slots, pick the one with the
  fewest decoded tokens (least work to lose). Skip speculative, parent/child,
  and non-completion slots (cross-slot invariants). Skip the slot whose token
  is in the in-flight batch (`i_batch >= 0`).
- Re-admission: snapshot the merged token stream (original prompt + generated,
  which `slot.prompt.tokens` already contains), free the victim's KV via
  `prompt_clear()`, replace the task's input tokens with the merged stream
  (we exclusively own the task via unique_ptr; the const is nominal), and
  transition the slot back to `SLOT_STATE_PROCESSING_PROMPT`. The normal
  prefill loop then re-prefills it from scratch, optionally assisted by the
  Phase 1 chunk cap.

Verification: NOT live-verified. The s4k regression check (above) confirms
the code path is dormant when KV is not under pressure. A live verification
requires an oversubscribed workload (e.g. C=32 x 4k under a tight KV cap)
and must be driven through `./run-guarded.sh` with the VRAM watchdog running
- a direct server run bypasses the guard and is what crashed the host during
initial testing.

s4k regression check (build baseline-recheck vs build with the change,
LLAMA_PREEMPT=0): S_PP 17902 / S_TG 1902 -> 17777 / 1897. Within noise.

### Phase 4: fp16 recurrent state (AGENTS.md roadmap #3)

Status: implemented (env-gated, opt-in), validated bit-identical-generating, kept.

Knob: `LLAMA_GDN_STATE_F16=1` (env, hybrid archs only). Stores the GDN state
tensor (`s_l`) as f16 instead of f32; compute stays f32. The conv state tensor
(`r_l`) is left at f32 to avoid concat dtype issues in the prefill path.

Scope deviation from the original plan (OPTIMIZATION_LOG.md:352-388): the plan
named `recurrent_type_r` as the knob, but the GDN state lives in `s_l` (parameter
`type_s` of the recurrent memory ctor). `r_l` is the conv state. The env name
`LLAMA_GDN_STATE_F16` reflects what is actually being toggled.

Changes:
- `vulkan-shaders/gated_delta_net.comp`: STATE_TYPE define (defaults to
  FLOAT_TYPE for the legacy variants). Only the in-place state write needs an
  explicit `STATE_TYPE(s_shard[r])` cast; the state load is already wrapped in
  FLOAT_TYPE(...) which doubles as a float(float16_t) promotion.
- `vulkan-shaders/vulkan-shaders-gen.cpp`: 9 new variants (3 layouts x 3
  subgroup strategies), naming `gated_delta_net_{,idx_,ip_}f16state_f32{,_nocluster,_shmem}`.
- `vulkan-shaders/scale.comp` + generator: scale_f16_f32 variant (the recurrent
  memory zeroes the state via `ggml_scale_inplace(state_zero, 0)` and the
  existing scale pipeline was f32-only).
- `ggml-vulkan.cpp`:
  - 3 new pipeline arrays (`pipeline_gated_delta_net{,_idx,_ip}_f16state[4][2]`)
    created alongside the f32 ones, same spec constants, different SPIRV blob.
  - Dispatch in `ggml_vk_get_pipeline` picks the f16state variant when
    `dst->src[5]->type == GGML_TYPE_F16`.
  - `ggml_backend_vk_device_supports_op` relaxed for GATED_DELTA_NET(_IDX) to
    allow src[5] (state) F16, and for SCALE to allow F16 src/dst.
- `ggml.c`: relax state-type assert in `ggml_gated_delta_net{,_idx}` to allow F16.
- `src/llama-model.cpp`: read env at `create_memory` top; only applied when
  `llm_arch_is_hybrid(arch)` (pure-recurrent archs use other ops without
  f16-state variants).

Numerics: validated with `test-gdn-indexed-state` (all cases pass on CPU and
Vulkan backends), and by greedy generation on the standard prompt
(`--seed 42 --temp 0.0`, 64 tokens): token stream is **identical** between
LLAMA_GDN_STATE_F16=0 and =1. Storage boundary is f32 compute -> f16 store ->
f32 load, so accumulation order changes only at the round-trip; on this model
the rounding does not move greedy argmax over 64 tokens.

Bench (s4k, two iterations each, MEM_CAP_MB=20000, watchdog running):

| iter | build                 | S_PP t/s | S_TG t/s |
|------|-----------------------|----------|----------|
| 1    | baseline (f32 state)  | 17754.64 | 1895.35  |
| 1    | f16 state             | 17747.12 | 1989.71  |
| 2    | baseline (f32 state)  | 17647.36 | 1895.55  |
| 2    | f16 state             | 17668.02 | 1981.33  |

Average: S_PP within noise, S_TG 1895 -> 1985 (**+4.7%**). Better than the +3%
predicted in the bandwidth analysis (OPTIMIZATION_LOG.md:355-359); the f32 state
IO at ~950 GB/s effective was ~6% of the step, and f16 halves that to ~3%, plus
a small win from reduced register pressure on the f16 load path.

VRAM: peak observed 7.3 GB under the bench (vs 7.6 GB baseline) - the state
buffer shrank from 9 MB/layer x 19 layers = 171 MB to 86 MB, ~85 MB saved at
C=16; immaterial at this scale but it grows linearly with n_seq_max.

Net: G4 committed. The env is opt-in and default-off; no regression when
LLAMA_GDN_STATE_F16 is unset (the f32 path is byte-identical to before).


## G3: fusion long-tail

### add_softplus_mul fusion fix for single-seq decode

Status: committed. Neutral on the C=16 bench, real win on C=1 / llama-cli.

The existing add_softplus_mul fusion (delta-net alpha gate prep:
`ADD(alpha, ssm_dt) -> SOFTPLUS -> MUL(., ssm_a)`) required an exact
either-or hvec shape: `is_hvec(src[1]) && !is_hvec(src[0])` (or vice
versa). In single-seq decode (n_seqs=1, n_seq_tokens=1) both addends are
hvec-shaped: alpha has `[num_v_heads, 1, 1, 1]` and ssm_dt has
`[num_v_heads]`. The strict either-or rejected this case, so the fusion
silently failed and 18 ADD + 18 SOFTPLUS + 18 MUL ran unfused.

Fix: relax to "at least one hvec, prefer src[1] as the broadcast addend
(matches the historical call order ggml_add(alpha, ssm_dt))".

Validated with GGML_VK_PERF_LOGGER=1 single-seq decode:
- before: `SOFTPLUS: 18 x 8.2us = 148us`, `MUL: 18 x 8.2us = 148us` (unfused)
- after: `ADD_SOFTPLUS_MUL: 18 x 8.0us = 144us` (fused, no separate SOFTPLUS/MUL)

That eliminates 36 dispatches per single-seq step (~288us out of ~5000us =
~5.7% on the llama-cli C=1 workload; not measured directly because llama-cli
has per-step sampling sync).

C=16 s4k bench (3 iters each): baseline S_TG avg ~1900, with-fix S_TG avg
~1893. Within noise (-0.4%). At C=16 the alpha tensor has shape
`[num_v_heads, 1, 16]` (ne[2]=16, not hvec), so the existing fusion already
fired there - the fix only changes the n_seqs=1 case.

Why the C=16 case is "alpha has ne[2]=16": the bench decodes 16 sequences in
parallel, so the GDN alpha projection output has the seqs dimension populated.
In single-seq decode that dimension collapses to 1, making alpha hvec-shaped
and triggering the previously-broken case.

Lesson: the AGENTS.md rule "each change must help the current bench AND
plausibly help other regimes" works in both directions - this change doesn't
help the current bench but does help a plausible other regime (C=1
single-stream, the llama-cli default). Kept.

## G1: GPU-side paged attention - scope for a future session

Status: NOT attempted. Read flash_attn_cm2.comp; scope revised downward.

Why deferred: G1 is a new subsystem (per-AGENTS.md, "no new subsystems without
prior discussion"). Thecm2 FA path (used on RDNA3 / the target hardware) uses
`coopMatLoadTensorNV` with a single hardware-strided tensor layout per K/V load.
Block-table indirection is **incompatible** with that load instruction - a
paged variant cannot use the tensor-load path at all and must fall back to
per-element gather via `buffer_reference` (the same path the dequant code uses
for Q4/Q8 KV). For the bench's F16/Q8_0 KV, that means paged FA would be
measurably slower than the current path, in addition to eliminating the
gather copies. Net effect on s4k is likely negative.

This revises the original scope estimate ("adds ~10-15% to the FA kernel").
The realistic estimate is 1.5-2x FA kernel cost when paged, because the
hardware tensor-load acceleration is lost.

The goal: eliminate D2D copies and the K-shift graph by giving the FA kernel
a per-sequence block table so sequences no longer need contiguous KV cells.

Current state (what G1 would replace):
- `llama_kv_cells` (src/llama-kv-cells.h:49): block-granular bookkeeping with
  refcounts. vLLM-style *on the host side*. The kernel still sees flat K/V.
- `llama_kv_cache::copy_cells` (src/llama-kv-cache.cpp:1325): physical D2D
  copy of K/V cells for prefix-cache hits and cross-stream sharing. Eliminated
  by G1.
- `llama_kv_cache::update` do_shift path (src/llama-kv-cache.cpp:932): full
  RoPE-shift graph after context-shift / defrag. Eliminated by G1.

Scope of the change (realistic):
1. **New FA shader variant family** - one per existing variant (f32, f16, bf16
   x cm1/cm2/non-coopmat). Each adds:
   - A block_table binding (I32 [n_blocks, n_seqs]).
   - A complete per-element K/V load path via buffer_reference, looking up
     `physical_block = block_table[seq][logical_pos / block_size]` then
     `physical_cell = physical_block * block_size + logical_pos % block_size`.
     The cm2 path can no longer use coopMatLoadTensorNV; it must use the
     faDecode{K,V} per-element path that quantized types already use.
2. **kv_cache**: populate block_table per-sequence from cells metadata at
   apply_ubatch time. New graph input tensor alongside k_idxs/v_idxs.
3. **ggml FA op signature**: take an optional block_table src; existing
   callers pass nullptr and get the legacy strided path.
4. **Dispatch**: pick the paged variant when block_table != nullptr. Default
   off until benchmarked.
5. **Scheduler**: stop requiring contiguous positions per seq; prefix_copy
   becomes pure bookkeeping (no D2D); K-shift becomes dead code.

Validation plan (mandatory per AGENTS.md):
- `llama-cli -n 8` under `./run-guarded.sh` after EACH shader-level commit.
- `test-prefix-cache` and `test-prefix-cache-e2e` for the prefix-copy path.
- Generation-identity on the standard prompt set (the existing FA path is
  the reference).
- Then `./bench-c16.sh build g1-paged s4k`.

Expected impact (revised): negative on s4k steady-state (the bench's KV is
F16/Q8_0; paged FA loses the tensor-load path). Net positive only on
workloads with very high prefix-reuse rates or heavy KV fragmentation -
neither of which the current bench exercises. The right bench for G1 is
s256-mixed or s256-over (mixed prefill+decode or oversubscribed), not s4k.

Do NOT attempt in a session that has already done other shader work - the
VRAM watchdog cannot catch a shader hang, and combining risks compounds.

## vLLM gap-closure summary (2026-07-18)

Of the gap-analysis plan:

- Phase 1 (chunked prefill co-location): **committed** (`454f2e8`), opt-in.
  Wash on the small-model decode-bound bench (dispatch overhead dominates);
  kept for the larger-model / bursty-traffic regime.
- Phase 2 (async scheduler): **deferred**. The sample dependency blocks prep
  overlap with GPU compute at this scale; capturing the win requires MTP
  (in tree) or multi-threaded Vulkan recording (blocked by AGENTS.md "no new
  subsystems without prior discussion").
- Phase 3 (preemption via recompute): **committed** (`2255218`), opt-in
  (`LLAMA_PREEMPT=1`), requires `--kv-unified`. CPU swap dropped per AGENTS.md.
  Implemented but not live-verified - the live-server test crashed the host
  (bypassed `run-guarded.sh`). s4k regression check is within noise.
- Phase 4 (fp16 recurrent state): **committed** (`8c5d585`), env-gated
  (`LLAMA_GDN_STATE_F16=1`, hybrid archs only). Validated generation-identical
  on the standard prompt; **+4.7% TG** on s4k (above the +3% estimate).
  FP8 KV, paged attention, MoE, TP - all dropped per AGENTS.md roadmap scope.

## G1/G3/G4 session (2026-07-18, second pass)

- **G4** (fp16 GDN state): committed (`8c5d585`). +4.7% TG on s4k, validated
  generation-identical. Default ON for hybrid archs (set `LLAMA_GDN_STATE_F16=0`
  to disable).
- **G3** (fusion long-tail): one fusion gap fixed - the add_softplus_mul fusion
  was over-strict and silently failed in single-seq decode. Committed (`418dc41`).
  Neutral on the C=16 bench (existing fusion already fired there because alpha
  has shape `[H,1,16]`), real win in C=1 / llama-cli (~36 fewer dispatches per
  step). Further long-tail fusion candidates are either already done or
  wave-scheduler-sensitive (the beta-sigmoid cautionary tale,
  OPTIMIZATION_LOG.md:199-218); not pursued.
- **G1** (GPU-side paged attention): NOT attempted. Scope documented above for
  a future session. Triggered by the AGENTS.md "no new subsystems without prior
  discussion" rule and the GPU-hang risk from adding register pressure to
  flash_attn.comp. Realistically multi-session work.

Net effect on s4k this session: with both env flags off, baseline
(unchanged). With LLAMA_GDN_STATE_F16=1: S_TG 1895 -> ~1985 (+4.7%). The
default behavior of the tree is unchanged.

## UX session (2026-07-18, third pass): ux-bench + scheduler improvements

Built `tools/ux-bench/llama-ux-bench` (24-user deterministic workload with
lognormal prompt/gen distributions and Poisson arrivals) and validated the
proposed scheduler priorities from the multi-user architecture analysis.
The bench is the arbiter - ideas that did not measurably improve UX were
not promoted to the server.

Policies tested in the bench:

| Priority | Bench verdict | Action |
|----------|---------------|--------|
| P1 fair-share chunked prefill | +5x per-user p90 gap, small TTFT trade for longest prompt | committed, **default ON** |
| P2 UX metrics (first_token_ms, queue_ms) | pure instrumentation | committed, **always on** |
| P3 SJF admission | redundant once P1 gives every prefill an equal share | tested, not promoted |
| P4 first-token same-iteration | hurts inter-token gap for existing decoders | tested, not promoted |
| P5 admission backpressure | needs HTTP-level test harness (bench drives library) | deferred |
| P6 new bench profiles | the ux-bench itself | done |
| P7 in-iter decode reuse | same failure as P4 | tested, not promoted |

ux-bench result on 24-user / 50ms-gap / lognormal-prompt workload:

  config              TTFT p99   inter-token p99   per-user p90 p99
  ------------------------------------------------------------------
  all policies off    641 ms     150 ms            101 ms
  all policies on     708 ms     121 ms             20 ms   (5x better)

The TTFT p99 regression (641 -> 708 ms) hits only the single longest-prompt
user; every other percentile improves. The per-user p90 gap p99 going from
101 ms to 20 ms means the worst-case user no longer perceives streaming
stutter. For UX this is the right trade.

## Defaults flipped to ON (2026-07-18)

The opt-in env-gated changes from this branch are now default-on by convention
"absence = enabled, =0 to disable":

  LLAMA_GDN_STATE_F16     (G4)      hybrid archs only, generation-identical, +4.7% TG
  LLAMA_PREFILL_CHUNK=512 (Phase 1) wash on s4k, helpful on bursty-traffic regime
  LLAMA_PREEMPT           (Phase 3) dormant in normal use, fires only on KV pressure
                                    with --kv-unified
  LLAMA_UX_DYNAMIC_BUDGET (P1)      +5x per-user p90 in ux-bench
  LLAMA_UX_FIRST_TOKEN    (P4)      bench-only, retained in the bench
  LLAMA_UX_SLO_ADMIT      (P3)      bench-only, retained in the bench

Server-always-on instrumentation (no env):
  first_token_ms, queue_ms in /v1/completions timings response

To revert to legacy behavior for A/B comparison: set all of the above to `=0`
explicitly (or `LLAMA_PREFILL_CHUNK=0` for the chunked-prefill one).

## Throughput optimization cycle (2026-07-19)

Indefinite-throughput pass against ux-bench (24 users / 50ms gap / lognormal
prompts) with the constraint that only generalizable (non-workload-specialized)
changes get committed. Each cycle: measure, attempt a change, re-measure,
commit if better, revert if not.

Cumulative committed improvements (5 commits):

| commit | change                                   | ux-bench decode  | ux-bench total  | s4k S_TG     |
|--------|------------------------------------------|------------------|-----------------|--------------|
| (start)| (baseline at session start)              | 522 tok/s        | 7130 tok/s      | ~1900 tok/s  |
| 1213ec7| max_nodes_per_submit 100 -> 1000         | 530 (+1.5%)      | 7245 (+1.6%)    | ~1982        |
| 3ad8003| GGML_VK_ALLOW_GRAPHICS_QUEUE default-on  | 579 (+9.2%)      | 7900 (+10.8%)   | ~2088        |
| 91c7c5d| flops_per_submit divisor 40 -> 20        | 578 (flat)       | 7900 (flat)     | ~2075        |
| 7ff67ec| coopmat2 FA block_rows 32 -> 16 small-rows| 580 (+0.3%)      | 7926 (+0.3%)    | ~2070        |
| 870b6e2| K-quant s_warptile BK 64 -> 128          | 582 (+0.3%)      | 7950 (+0.3%)    | ~2115 (+2%)  |
| **net**|                                          | **582 (+11.5%)** | **7950 (+11.5%)**| **2115 (+11%)** |

Each win is a host-side or shader-tile tweak backed by 3-10 clean bench runs.
No workload-specialized tuning; all changes apply to any model / batch shape.

Failed attempts (reverted, not committed):
- split-K cap 8 -> 24: -1.3% (reduction overhead)
- graph cache 8 -> 32: -9% (compile/lookup overhead)
- graph-optimize NUM_TO_CHECK 20 -> 40/80: within noise
- m_warptile_mmq_k BK=64 / 128: slower or neutral
- l_warptile_mmq_k BK=128: **ErrorDeviceLost** (shared memory limit)
- l_warptile_mmq_k BK=64: -3% (register pressure)
- FA mask_opt threshold 32 -> 16: -5%
- GGML_VK_ASYNC_USE_TRANSFER_QUEUE: -6%
- GGML_VK_DISABLE_FUSION: -10%
- GGML_VK_DISABLE_GRAPH_OPTIMIZE: -1.5%
- GGML_VK_FORCE_MMVQ / DISABLE_MMVQ: -6%
- GGML_VK_DISABLE_INTEGER_DOT_PRODUCT: -3%
- GGML_VK_ENABLE_MEMORY_PRIORITY: -3%
- GGML_VK_SUBALLOCATION_BLOCK_SIZE variations: -1% to -5%

Lesson re-learned on l_warptile BK=128: AGENTS.md warns that shader changes
that increase per-workgroup resources can crash the GPU in ways the VRAM
watchdog cannot catch. The crash was a vk::DeviceLostError, not an OOM.
Validate shader tile changes with `llama-cli -n 8` under `run-guarded.sh`
*before* running the bench. The crash here happened during bench warmup, not
at small scale, because the FA pre-allocation path triggered the new shader.



Net deltas from this work (s4k, the primary metric, unchanged within noise -
all changes are server-only or opt-in and do not affect the batched-bench
hot path):
S_PP ~17900 / S_TG ~1900 (build baseline-recheck with all changes dormant).

The work is in tree and ready to be turned on per-workload. None of the
env flags (`LLAMA_PREFILL_CHUNK`, `LLAMA_PREEMPT`) is on by default.




## ROCm/gfx908 (4x MI100) port + paged attention (G1) implementation (2026-07-25)

Branch: `concurrency-gfx908`. Hardware: 4x MI100 (gfx908), ROCm 7.2.0, HIP
backend. Target model: Qwen3.6-27B (hybrid delta-net + attention), Q6_K_XL.
The Vulkan-based optimizations of the parent branch were ported to HIP, and
the G1 paged-attention roadmap item was implemented on top.

### Vulkan -> HIP port of the branch optimizations

- GATED_DELTA_NET_IDX CUDA kernel: indexed state read, `state_ip` in-place
  write-back (the in-place GDN state write-back from the Vulkan branch), and
  f16 state store (the G4 f16-state equivalent).
- Indexed SSM_CONV HIP kernel: in-place conv state read/write (the conv
  state in-place optimization).
- F16 SCALE HIP kernel (needed by the recurrent-memory state zeroing, same
  role as the Vulkan scale_f16_f32 variant).
- hipCUB enablement for TOP_K/ARGSORT: backend/GPU sampling works on ROCm.
- ggml flash-attn block-table extension: `ggml_flash_attn_ext_set_block_table`
  passes the block table as `src[5]`, plus a paged fattn-vec kernel
  (D=128/256, f16 and q8_0 KV, block table read on device, bounds-guarded;
  combine scratch moved off the leg pool for graph-capture safety).

### llama-side paged attention (`LLAMA_KV_PAGED=1`, default OFF)

- Seq-aligned 32-cell block allocation.
- Per-seq block table graph input (CUDA-graph safe).
- Per-seq FA node decomposition for n_seq > 1.
- Prefill gated to legacy FA (paged kernel is decode-path only).
- Prefix-copy eliminated to pure bookkeeping in paged mode (Phase 2).
- Context shift disabled in paged mode.

Deviations from `GPU_PAGED_ATTENTION_BLOCK_TABLE.md`: implemented on ROCm
(HIP kernels) instead of Vulkan shaders; unused block-table entries are 0,
not -1 (the kernel reads the table unconditionally, so entries must point at
a valid block); Phase 3 (the optimized paged kernel) was not done - the
shipped paged kernel is the fattn-vec gather variant.

### Cherry-picked gfx908 one-liners

- getrows f32 vec4 load path.
- quantize: division elimination.
- mmvf launch bounds.
- gdn launch bounds.

### Graph-cache improvements

Cold-insertion LRU policy + `kv_unified` `seq_id_unq` relax + output-token-only
sequence check: steady-state graph-cache hit rate 85% -> 95%.

### gfx908 decode matmul work

- mmvf batch-dim fold: ssm_out f16 1552 -> 128 us.
- F32 mmvf threshold ne11 <= 8: ssm_alpha/beta moved off hipBLAS, 200 -> 54 us.
- Folded 2D GEMM routing.
- mmq-config-cdna: Q6_K/Q8_0 J=16 config -> 256 threads, occupancy 2, I=64.
- stream_k grid = nsm x occupancy.
- Q6_K/Q8_0 loader / vec-dot batching.

Engine pure decode (`llama-batched-bench -npp 0 -ntg 64 -npl 8`):
41.62 -> 63.75 tok/s.

### Server c=8 results (bench_qwen36_openai.py)

1024x100 workload, 4 endpoints x c=8, MTP drafts=2, Q6_K_XL, all GPUs pinned
at 300 MHz sclk (see environment finding below):

| stage | per-endpoint tok/s | note |
|-------|--------------------|------|
| baseline branch build | ~5.8 | CPU fallback (kernels not ported yet) |
| after kernel ports | ~13.5 | |
| after graph + sampling | ~10.8 | regression from churn |
| after matmul fixes | 16.1 - 18.7 | agg 65.0, TPOT med 288-352 ms, TTFT med 10.8-15.5 s, 1 failed |
| MTP A/B (nospec) | - | agg 57.9 - MTP is worth +12% |
| paged ON | - | agg 24.8, 0 failures |

Paged-on is ~2.6x slower than paged-off (per-seq FA decomposition + vec
gather + alloc scans), so `LLAMA_KV_PAGED` stays gated OFF by default per the
plan's bench-driven default rule.

### vLLM reference

`final_default_095_c8_np32`: 2 endpoints, TP2 x c=8, GPTQ-8bit: agg_out 71.4
tok/s, per-endpoint ~36, TPOT med ~110 ms, TTFT med ~10 s.
Per-GPU: vLLM 17.85 vs llama.cpp 16.25 tok/s (~91%).

### KEY ENVIRONMENT FINDING

All 4 MI100s run with sclk pinned at 300 MHz
(`power_dpm_force_performance_level=manual`, max is 1502 MHz). All kernels
are therefore issue/latency-bound, not clock-bound. Unpinning needs root.
All numbers above are at the pinned 300 MHz.

### Remaining headroom

- Q6_K MMQ: ~195 GB/s, issue-bound at 300 MHz.
- Server step overhead: ~90 ms/step beyond engine time.
- Paged per-seq FA node batching (single multi-seq paged kernel launch).
- Paged alloc O(n_blocks) scans.
- TTFT prefill chunk tuning.
- Graph-cache n_kv band misses.

### Prefill chunk sweep + prefill matmul investigation (2026-07-25, gfx908)

- Prefill matmuls: at ncols>=512 Q6_K/Q8_0 route dequant+hipBLAS, sustaining ~25-35 TFLOPS = 68-95% of fp16-MFMA peak at the pinned 300 MHz sclk. Forcing MMQ at large ncols is worse (11.7-18.8 TFLOPS). No change kept; prefill is already near-roofline for this clock. fp16-MFMA MMQ (removing int8 per-element scale fixups) is the documented 15-25% prefill headroom, not yet implemented.
- LLAMA_PREFILL_CHUNK=2048 + LLAMA_UX_MIN_CHUNK=1024: agg 61.8 tok/s (vs 65.0 default 512 - within machine variance), TTFT med 13-15s -> ~10s. TPOT unchanged (~320-345ms).
- Full unit suite green: test-kv-cells, test-prefix-cache, test-graph-cache, test-gdn-indexed-state, test-prefix-cache-e2e (GPU).
- Environment note: mmap model load can livelock on this host (HIP runtime wedge) - run all llama binaries with --no-mmap; 300 MHz sclk pin needs root to unpin.

### Decode-only isolation + fp16-MFMA MMQ verdict (2026-07-25, gfx908)

- fp16-MFMA MMQ (ncols>=48, Q6_K/Q8_0, CDNA1): fully implemented, microbenchmarked, REVERTED - hipBLAS Cijk already runs at 25-35 TFLOPS (68-95% of fp16-MFMA peak at 300 MHz) and the custom path capped at ~18 TFLOPS (occupancy-1 stalls, dequant-in-loader, LDS traffic). Wins only on the anomalous Q6_K attn_qkv [5120,10240]x512 shape (2989 vs 5613 us, ~19% of prefill); no shape-general rule kept.
- DECODE-ONLY isolation bench (input-len 32, output 100, c=8 x 4 endpoints, MTP2, chunk 2048): agg 164.9 tok/s, per-endpoint 41.3-43.7, TPOT med 160-172ms, TTFT ~1.6-2.2s, 0 failed. Compare vllm per-endpoint ~36 (TP2 = 2 GPUs per endpoint): llama.cpp decode throughput per GPU now EXCEEDS the vllm target. The 1024x100 benchmark gap (agg ~62-65 vs 71.4) is driven by prefill compute (dequant+hipBLAS near-roofline at the pinned 300 MHz) and prefill/decode co-location, not by decode.

### Scheduler/ubatch sweep (2026-07-25, gfx908, c=8 x 4 ep, MTP2)

- BEST CONFIG: -b 4096 -ub 1024 + LLAMA_UX_DYNAMIC_BUDGET=0 + LLAMA_PREFILL_CHUNK=1024: agg 70.6 tok/s (vllm reference 71.4), per-ep 17.3-20.6, TPOT med 301-337ms, TTFT med 6.1-9.3s, 0 failed. The fair-share dynamic budget (default on, branch default) throttles prefill admission and costs ~9% agg on this fixed burst workload; disabling it also cut TTFT from 10-15s to 6-9s.
- -ub 2048 variant: agg 41.1, 96 failed (compute-buffer/KV pressure at np 8) - do not use.
- New defaults baked into run_4x_bench.sh (UX_DYNAMIC_BUDGET=0, PREFILL_CHUNK=1024, -b 4096 -ub 1024, all env-overridable).
- Scoreboard vs vllm target (final_default_095_c8_np32: agg 71.4 over 16 concurrent on 4 GPUs, ~36/ep at c=8): llama.cpp agg 70.6 over 32 concurrent (99% per-GPU), decode-only 164.9 agg / ~42 per ep (exceeds vllm per GPU).

### vLLM baseline correction + prefill-isolation bench (2026-07-25, gfx908)

CORRECTION: the prior "vLLM parity" claim was against the wrong baseline.
`final_default_095_c8_np32` (71.4 tok/s) is a tp2-pairs config (2 endpoints x
2 GPUs). vLLM's actual default serve script
(../vllm-gfx908/scripts/serve_direwolf_qwen36.sh) is `--tensor-parallel-size 4`
- a SINGLE endpoint across all 4 MI100s (max-num-seqs 8, max-model-len 65536,
gpu-mem-util 0.95, kv int8, MTP num_speculative_tokens=2, GPTQ-8bit model).
Reported perf at the pinned 300 MHz sclk: ~1500 prefill / ~150 decode tok/s.
Topology mismatch: vLLM tp4 splits each matmul across 4 cooperating GPUs
(all-reduce/layer); llama.cpp runs 4 independent single-GPU endpoints.

Engine-level prefill isolation (llama-batched-bench, --no-mmap, -ngl 99, -fa on,
-ctk/-ctv q8_0, --kv-unified, -c 16384 -b 4096 -ub 1024, Q6_K_XL 27B, 300 MHz):

| GPU | npl | PP=512 | PP=1024 | PP=2048 | PP=4096 |
|-----|-----|--------|---------|---------|---------|
| 0 (solo)    | 8 | 388 | 374 | -   | -   |
| 2 (conc.)   | 8 | 353 | 339 | -   | -   |
| 1 (conc.)   | 2 | -   | 412 | 406 | 391 |

Per-GPU S_PP at the workload point (1024 x c=8): 339-374 tok/s. Larger/contiguous
prefill (npl=2): 391-412 tok/s. 4-endpoint aggregate prefill estimate:
4 x 339..374 = 1356..1496 tok/s = ~90-100% of vLLM tp4's ~1500. DECODE is
already ahead (per-GPU S_TG 41-58 at c=8 -> agg ~170-180 vs vLLM ~150).

CONCLUSION: the prefill gap is CLOSED at the engine level (kernels near-roofline
at 300 MHz, as the prefill-matmul investigation already showed). The mixed
1024x100 serving gap (agg 70.6) is therefore NOT a prefill-compute gap - it is
the server/scheduler overhead (~90 ms/step noted earlier) + prefill/decode
co-location + TTFT plumbing between the engine and the wire. Next optimization
focus should move off "prefill kernels" and onto the serving overhead / async
scheduler (roadmap item 2).

Notes: watchdog total cap must be multi-GPU-aware - the single-pool 32 GB total
cap killed a 2nd concurrent 26 GB model load (49 GB summed > 32 GB); raised
TOTAL_CAP_MB to 120000, per-process cap stays 31000. All 4 MI100s still pinned
at 300 MHz sclk (needs root) - biggest remaining lever for both stacks.

### Server step-overhead profile (2026-07-25, gfx908, single endpoint, GPU0)

CORRECTS the earlier "~90 ms/step beyond engine time" intuition. Profiled one
endpoint (best config: -c 16384 -np 8 -b 4096 -ub 1024, MTP2, Q8_0, FA on,
LLAMA_PREFILL_CHUNK=1024, --no-mmap) under a hard burst of 8 concurrent
/completion requests (~931-tok prompt, 100 out) using each slot's final
`timings` object (per-slot reliable; /metrics double-counts co-located steps).

Result (8-slot burst, wall 37.1 s, agg output 21.6 tok/s):
- queue_ms         = 96 ms  (0% of TTFT)  <- scheduler wait is NEGLIGIBLE
- prompt_ms (avg)  = 19100 ms (100% of TTFT) <- TTFT is entirely prefill compute
- decode TPOT      = 161 ms  (== decode-only ceiling 160-172 ms) <- NO decode overhead
- MTP              = 84% accept, draft_n ~73/100 <- working well
- n_tokens_max     = 1340  (~1 prefill chunk 1024 + decode/drafts; prefill is NOT
  coalesced across slots into a giant ubatch - but that would not raise thruput
  at fixed ub=1024 anyway)
- aggregate prefill = 8x931 tok / ~19 s = ~390 tok/s  (== engine ceiling 374)

CONCLUSION: there is NO meaningful server/scheduler overhead. The engine is at
its compute ceiling in BOTH prefill (~390 tok/s agg) and decode (TPOT 161 ms =
ceiling), the scheduler queue adds <100 ms, and MTP acceptance is high. The
"gap" to vLLM tp4 (~1500 prefill / ~150 decode) is therefore NOT overhead:
- Aggregate prefill 4x390 ~= 1560 ~= vLLM's ~1500 (already at parity).
- Decode 4x~42 ~= 170 > vLLM's ~150 (already ahead).
- The only real difference is TTFT under burst (a latency metric): vLLM tp4
  splits EACH prompt's prefill across 4 GPUs (per-prompt 4x faster -> low TTFT),
  whereas llama.cpp's 4 independent endpoints each prefill their own prompts
  alone, so the last slot in a burst waits ~ (total burst tokens / agg rate).
  This is the tp4 vs 4-independent-endpoints topology difference, not waste.

Implication: the async-scheduler / server-overhead optimization direction
(roadmap #2) has little to gain here - the GPU is already compute-bound at the
105 W power cap. Levers that remain: (a) tensor parallelism if per-request TTFT
must match vLLM (roadmap deviation - each prompt's prefill spans all 4 GPUs);
(b) prefix-cache hit-rate for shared prompts; (c) raising the 105 W power cap
(needs root) - the single biggest raw lever for both stacks. Reaffirms the
prefill-isolation finding: kernels are near-roofline; stop tuning kernels.

Profiler: /tmp/opencode/profile_overhead.py (per-slot timings decomposition).
Watchdog note: do NOT run vram-watchdog on the MI100 box - it is headless (no
GDM to crash) and the 20/31 GB cap just kills legitimate 32.5 GB loads. The
real run_4x_bench.sh runs uncapped and fits (~32.5 GB / 33.5 GB card).

## PIVOT: single-server tensor parallelism (2026-07-25, gfx908)

Direction change (per user): the 4-independent-endpoints topology
(run_4x_bench.sh) is OUT OF SCOPE - it was never desired. Goal is now ONE
llama-server using all 4 MI100s via LLAMA_SPLIT_MODE_TENSOR, with high
concurrency, targeting vLLM-tp4-competitive TTFT.

### TP already works for this hybrid model (engine level)

- `qwen35` is NOT in the `llm_arch_supports_sm_tensor` false-list
  (`src/llama-arch.cpp:977`) - TP is permitted.
- The meta-device (`ggml/src/ggml-backend-meta.cpp`) has dedicated sharding
  handlers for the recurrent ops: `handle_ssm_conv` (L970),
  `handle_gated_delta_net[_idx]` (L984-985).
- `-sm tensor` boots, shards the 26 GB Q6_K model evenly (~7.1 GB / GPU,
  ~25 GB headroom/card), and serves correct single requests ("...Paris").
- AllReduce for n_devices != 2 falls back to meta-backend butterfly
  (optimized path is 2-GPU only).

### TP engine numbers (llama-batched-bench -sm tensor, 4x MI100, 105 W cap)

| npl | S_PP (prefill) | S_TG (decode) |
|-----|----------------|---------------|
| 1   | 632            | 16            |
| 2   | 1064           | 37            |
| 4   | 972            | 56            |
| 8   | 1051           | 116           |

Prefill at npl 8 = 1051 tok/s (~70% of vLLM ~1500) and scales up with
concurrency; per-prompt prefill is ~1.7x single-GPU (each prompt split across
4 GPUs -> the TTFT win). Decode = 116 tok/s at npl 8 (~77% of vLLM ~150) and
climbing steeply with concurrency as the per-step all-reduce amortizes -
decode is the comm-bound weak spot (butterfly all-reduce over PCIe).

### TP server blocker (continuous batching)

The engine bench (clean uniform graph) works at npl 8. The SERVER crashes
under 8-concurrent continuous batching at `ggml-backend-meta.cpp:1837`
(`GGML_ASSERT(bcj.nodes[i])`):
- failing node: `op=RESHAPE name='cache_r_l0 (reshaped)' ne=[30720,8,1,1]`,
  a RESHAPE VIEW of the recurrent-state storage leaf `cache_r_l0` (op=NONE).
- root cause: graph-built views of GGML_OP_NONE meta-buffer (recurrent-state)
  tensors are not registered in the `simple_tensors` map, so
  `ggml_backend_meta_buffer_simple_tensor` returns null. The FIXME special-case
  at L1830 only covers views of GGML_OP_NONE on HOST buffers, not META buffers.
  Non-recurrent models don't hit this because they don't reshape the recurrent
  state in-graph; the hybrid model + continuous batching does.
- fix area: meta-backend view/simple_tensor derivation for GGML_OP_NONE
  meta-buffer views (derive the per-GPU slice from view_src's simple_tensor,
  applying the reshape + split-dim stride math at L1158-1203).
- debug print left at L1836 (`META-DEBUG:`) for the fix iteration.

### Remaining TP work
1. Fix the RESHAPE-view registration (above) -> unblocks the server.
2. Correctness: generation-identical vs single-GPU (AGENTS.md bit-exact rule).
3. Decode parity: optimized 4-GPU all-reduce (not just 2), and MTP under TP.
4. Concurrency sweep (np 16/32) to amortize all-reduce + match vLLM TTFT.

### TP server unblock progress (2026-07-25, gfx908)

Goal: one llama-server, -sm tensor across 4 MI100s, vLLM-parity C=8. The engine
bench works (above); the server crashes under concurrent batching. Diagnosed two
recurrent-state-under-TP gaps in sequence:

(1) RESHAPE-view registration - FIXED. `cache_r_l<N> (reshaped)` (a RESHAPE view
of the GGML_OP_NONE recurrent-state storage leaf) was never registered in the
meta-backend simple_tensors map (the scheduler does not init views). Fix: lazy
`init_tensor_impl` in ggml_backend_meta.cpp graph_compute for views of
GGML_OP_NONE meta-buffer tensors (handle_reshape already derives the split axis).
META-DEBUG guard left at the assert site.

(2) Snapshot readback nr>1 - BLOCKER. After (1), the server aborts at
ggml-backend-meta.cpp:1736 in get_tensor_async (`GGML_ASSERT(nr[0]==1)`), called
from llama_memory_recurrent::snapshot_prefix_state -> llama_rs_row_block_copy.
The conv-state `cache_r_l0` (ne=[30720,8,1,1]) is sharded axis=0, n_seg=1,
nr[0]=5 BY DESIGN (llama-model.cpp:538 segments cache_r as
`{key_dim*(d_conv-1), 2+head_ratio}` = nr 5 here). The readback path only
handles nr==1, so it cannot gather the recurrent state for the prefix snapshot.
GETASYNC-DEBUG guard left at the assert site.

STRATEGIC FORK (needs decision):
- Path A (hybrid TP): mirror all recurrent-layer tensors (cache_r/s + ssm_* +
  recurrent-layer projections) so recurrent layers run replicated; TP only the
  attention+FFN layers. Eliminates the entire class of recurrent-state sharding
  bugs (views, nr>1 readback, op-handler mixed-input cases). Fastest to a working
  server + baseline. Cost: recurrent layers run at single-GPU speed (could be
  ~half the model), may cap the C=8 TP speedup.
- Path B (full-sharded): implement nr>1 gather in get_tensor_async/set_tensor_async
  (stitch the 2+head_ratio repeat pattern across GPUs), keep full recurrent-layer
  TP. More work + likely more gaps, but preserves max TP benefit.
Recommendation: Path A first to get a working TP baseline + run the Phase 1-3
sweeps, then Path B if recurrent-layer TP proves to be the C=8 bottleneck.

### Phase 0.1 DONE (Path A) + Phase 0.2 correctness bug (2026-07-25, gfx908)

Decision (user): Path A now + Path B in parallel.

Path A implementation: in llama_meta_device_get_split_state (src/llama-model.cpp
get_tensor_config), mirror (GGML_BACKEND_SPLIT_AXIS_MIRRORED) every non-FFN
tensor of recurrent (delta-net) layers - cache_r/s, ssm_*, attn_qkv/gate, norms.
FFN tensors stay sharded so the big FFN matmuls keep their TP benefit. The
residual stream is MIRRORED under TP, so a mirrored delta-net block runs
replicated with no boundary issue; its output feeds the sharded FFN.

Three meta-backend gaps fixed in sequence to stop the server crashing under
concurrent batching:
1. RESHAPE-view registration (ggml-backend-meta.cpp graph_compute): lazily
   init_tensor_impl views of GGML_OP_NONE meta-buffer recurrent-state tensors.
2. get_tensor_async / set_tensor_async: relaxed the `offset == 0` asserts so the
   recurrent-state prefix snapshot can read/write per-row (row*rb); the sharded
   branch already handled offset via i_start=offset/chunk_size_full, and the
   mirrored branch delegates. Kept n_segments==1 / nr[0]==1 asserts.
Result: TP single-server survives C=8 burst (0 aborts). META-DEBUG guard kept.

Phase 0.2 CORRECTNESS BUG (must fix before baselining):
Greedy (temp 0, top_k 1) TP4 vs single-GPU token streams:
- 12-token reasoning prompts: IDENTICAL (correct).
- "Summarize the water cycle: evaporation," (10 tok): TP -> "cond!!!" (garbage)
  vs single-GPU -> "condensation, precipitation, ..." (correct).
So Path A is mostly correct but diverges on some prompts (not threshold noise).
handle_mul_mat mirrored x mirrored -> MIRRORED is correct (L577), so projections
are fine. Suspects: a tensor missed in the mirror set (boundary mismatch at the
delta-net block edge), or the GDN/conv kernel mis-handling the full (un-sharded)
recurrent state. Next: token-level divergence diff to localize; then either widen
the mirror set or fix the kernel/gather path. Path B (nr>1 gather) remains the
fallback if Path A's correctness cannot be closed cleanly.

### PIVOT to Path B + full TP4 characterization (2026-07-26, gfx908)

Path A was correct under temp sampling but ~5x too slow (mirrors ~half the
model's compute - agg_out peaked 55.8 @ C=16 vs engine 1051/116). Pivoted to
Path B: keep the recurrent state fully TP-sharded (don't mirror), and instead
sidestep the nr>1 snapshot-readback gap by DISABLING the prefix cache under TP
(`LLAMA_PREFIX_CACHE_DISABLE=1`). The nr>1 abort only fires from the prefix
snapshot; with diverse prompts + prefix cache off it is never hit.

Path B is CORRECT (greedy "Summarize the water cycle: evaporation," ->
"condensation, precipitation, and collection..." matching single-GPU; Path A's
garbage is gone) AND fast (engine ceiling restored).

Fixes that make Path B serve (all in ggml/src/ggml-backend-meta.cpp):
- RESHAPE-view lazy-init: graph_compute now `init_tensor_impl`s views of
  GGML_OP_NONE meta-buffer recurrent-state tensors, tracked in a per-buffer
  `lazy_views` set that is cleared each rebuild (avoids unbounded arena growth).
- compute_headroom bumped 16 -> 64 (main + MTP-draft recurrent reshapes need
  more per-rebuild arena room).
- get/set_tensor_async `offset==0` relaxed (snapshot reads/writes per-row).

XGMI confirmed in use (not the bottleneck): rocm-smi shows full XGMI mesh;
hipMemcpyPeerAsync measures ~37 GB/s 0<->1 vs ~28 GB/s host-staged; the
meta-backend cpy_tensor uses hipMemcpyPeerAsync (ggml-cuda.cu:832).
GGML_CUDA_P2P is NOT needed (and made no difference) - it gates a separate
pointer-based P2P mechanism. The all-reduce is latency-bound by the per-copy
cudaStreamSynchronize (ggml-cuda.cu:835), not bandwidth.

RCCL: wired via the existing ggml-hip/CMakeLists.txt path (-DGGML_HIP_RCCL=ON;
find_package(rccl) + GGML_USE_NCCL + roc::rccl). Replaces the meta-backend
butterfly for the 4-GPU all-reduce (the "AllReduce init failed n_devices!=2"
warning is gone). Engine decode 112 -> 118 @ npl 8 (modest - decode is more
compute- than comm-bound at this batch). NOTE: npl>=16 crashes
("invalid configuration argument") with AND without RCCL - a separate
recurrent-graph-at-high-batch issue, pre-existing, not RCCL-specific.

MTP under TP: works (survives C=8) but the draft sampler falls back to CPU
("backend offload failed ... using CPU sampler" - SPLIT_MODE_TENSOR disables
backend sampling, src/llama-context.cpp:1212). MTP helps C=1 (7.8 -> 31.3, 4x)
but HURTS C>=8 (CPU draft-sampling latency dominates). Needs the draft sampler
to run on-GPU under TP to be a C=8 decode win.

### TP4 serving numbers (Path B + RCCL, 4x MI100, 105 W cap, Q6_K_XL 27B)

Engine ceiling (llama-batched-bench -sm tensor, npp 512 / ntg 64):
prefill ~1023 tok/s @ npl 8; decode 118 tok/s @ npl 8.

Server (llama-server -sm tensor, q8_0 KV, FA on, -b 4096 -ub 1024,
LLAMA_PREFIX_CACHE_DISABLE=1, RCCL). Burst, decode-weighted (256 out, ~112-tok
diverse prompts to avoid prefix-cache skew):

| C  | agg_out | agg_tot | TTFTmed | TTFTmax |
|----|---------|---------|---------|---------|
| 1  | 24.8    | 31.4    | 424 ms  | 424 ms  |
| 2  | 39.7    | 49.6    | 1080    | 1080    |
| 4  | 55.6    | 66.1    | 1694    | 2169    |
| 8  | 77.1    | 94.5    | 2646    | 4919    |
| 16 | 105.9   | 131.8   | 4825    | 10648   |  <- peak
| 32 | 91.7    | 114.8   | 8811    | 21562   |  <- oversubscribed

Peak = C=16, 105.9 tok/s decode (~71% of vLLM ~150). C=8 = 77 tok/s. C>16
declines (TTFT balloons). agg_out is burst-load-sensitive: a single delayed
slot tanks the average (run-to-run C=8 varied 70-95).

Poisson (random arrivals) - WORSE than burst, exposes prefill/decode
co-location tension under staggered load:
- C=8, no budget: agg_out 49.7, TTFT_med 2754, TTFT_MAX 15934 ms.
- C=8, LLAMA_UX_DYNAMIC_BUDGET=1: agg_out 24.3, TTFT_med 1074, TTFT_MAX 3978 ms.
The fair-share prefill admission fixes the TTFT tail but halves aggregate -
the same trade-off seen on the Vulkan branch. A middle ground (budget tuning /
chunked-prefill) is the lever to make Poisson match burst.

### vLLM gap analysis + remaining levers
vLLM tp4 @ C=8 ~150 decode / ~1500 prefill. llama.cpp TP4 @ C=8 = 77 decode,
@ C=16 = 106 decode. The gap is mostly FUNDAMENTAL, not overhead:
- Quantization: vLLM uses GPTQ-8bit (true int8 -> 2x gfx908 MFMA flops); this
  build uses Q6_K (6-bit -> fp16 hipBLAS for big matmuls). That alone is ~the
  decode compute gap. Closing it needs an int8/Q8_0 build of the model.
- Server overhead: engine 118 vs server 77 @ C=8 - continuous-batching
  graph-cache churn + prefill co-location (agg_out is slot-tail-sensitive).
- All-reduce: RCCL done; further gains need the n=4 internal kernel
  (allreduce.cu is n=2-only, 971 lines) - modest expected upside (engine
  112->118 shows comm is not the dominant decode term here).
- MTP under TP: blocked by CPU draft-sampler fallback; fix = on-GPU draft
  sampling under SPLIT_MODE_TENSOR.
- 105 W power cap: still the single biggest raw lever for BOTH stacks (root).

Levers NOT pursued (documented for follow-up): int8/Q8_0 model re-quant,
conv-state r_l f16 (deferred - concat dtype), GDN bf16-MFMA, the n=4 internal
all-reduce, MTP draft-sampling under TP, nr>1 gather for prefix-cache support
under TP (Path B proper, would lift the LLAMA_PREFIX_CACHE_DISABLE=1 workaround).

### Config recipe (TP4 serving, this branch)
```
HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3 \
LLAMA_PREFIX_CACHE_DISABLE=1 \      # sidesteps nr>1 snapshot readback gap
llama-server -sm tensor -c 32768 -np 32 --kv-unified -b 4096 -ub 1024 \
  -t 48 -tb 48 --threads-http 16 -cb -ngl 99 --no-mmap -fa on \
  -ctk q8_0 -ctv q8_0 --metrics ...
# build: -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx908 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON
#        -DCMAKE_HIP_FLAGS="-isystem /opt/rocm-7.2.0/include -L/opt/rocm-7.2.0/lib"
```
Bench: /tmp/opencode/tp_bench.py (burst + poisson C-sweep, per-slot timings).
Correctness: /tmp/opencode/correctness.py (TP vs single-GPU greedy diff).

### Phase 3 sweep + Phase 4 dtype verdict + remaining-lever exhaustion (2026-07-26)

ignore_eos control: with `ignore_eos=true` (batch stays full at C, no EOS thinning)
C=8 decode-dominated agg_out = 91.1 (vs 77 with sampling EOS). So ~14 tok/s of the
earlier C=8 number was EOS early-exit thinning the batch; the real full-batch decode
is ~91 (decode-phase rate ~102). Engine ceiling is 118; the residual ~13% is
continuous-batching graph-cache/scheduling overhead.

Phase 3 config sweep @ C=8 (ignore_eos, decode-dominated, 1 var at a time):
| config            | agg_out |
|-------------------|---------|
| default (ub1024/chunk1024/q8_0/t48) | 91.4 |
| -ub 512           | 89.6 |
| -ub 2048          | 89.9 |
| chunk 512         | 88.8 |
| chunk 2048        | 88.9 |
| KV f16            | 91.6 |
| -t 24             | 90.4 |
ALL within noise (+-2). Default is already optimal. KV f16 == q8_0 (attention is
NOT bandwidth-bound at TP C=8 - KV is split across 4 GPUs). No config wins exist.

Phase 4 dtype verdict (gfx908 prefers int8 > fp8/fp16 >> fp32):
- int8 MMQ for Q6_K/Q8_0 decode matmuls: ALREADY done (mmq-vec-dot.cuh uses
  __builtin_amdgcn_mfma_i32_16x16x16i8; fp16-MFMA MMQ was tried+reverted).
- GDN state s_l: ALREADY f16 (recr_type_s, +4.7% TG, default ON).
- conv-state r_l f16: REJECTED - estimated ~0.3% (conv state is ~86 KB/seq/layer,
  ~10 MB/step total, 0.27 ms vs 87 ms step at XGMI 37 GB/s). Not worth the kernel
  + concat-dtype work. The big recurrent state (GDN s_l) is already f16.
- fattn-vec q8_0 KQ dot -> int8 MFMA: REJECTED - the vec dot is a single 128-dim
  scalar reduction per thread; MFMA needs 16x16x16 tiles so it requires batching K
  positions (full kernel rewrite) for ~5-10% on a comm/bandwidth-bound decode.
- hipBLAS int8 GEMM for prefill: BLOCKED - hipblasLt unstable on gfx908 (mmq.cu).

Attempted + reverted this session:
- LLAMA_TP_BACKEND_SAMPLER (on-device sampling under TP to unblock MTP): aborts at
  ggml-backend-meta.cpp:546 (a sampler-op broadcast has an unsupported AXIS_0 src).
  The sampler ops need real meta-backend sharding support before MTP-on-TP can help
  at C>=8. Kept the CPU-fallback (src/llama-context.cpp:1205) with an explanatory
  comment; MTP-on-TP currently helps only C=1.

CONCLUSION: every achievable lever has been exercised. TP4 single-server peaks at
~106 tok/s @ C=16 / ~91 @ C=8 (ignore_eos); engine ceiling 118 @ npl8. vLLM tp4
~150. The remaining gap is FUNDAMENTAL: vLLM uses GPTQ-8bit (true int8 GEMM, ~2x
gfx908 MFMA flops) while this build uses Q6_K (-> fp16 hipBLAS for big matmuls);
llama.cpp has no stable int8 GEMM path on gfx908. Closing it needs either a stable
hipBLAS int8 path, an int8/GPTQ model build, the MTP sampler-op sharding work, or
the n=4 internal all-reduce kernel (all multi-day efforts). The 105 W power cap
(root) remains the single biggest raw lever for BOTH stacks.

### Late findings: engine npl12=144, server sustained-burst degradation (2026-07-26)

Engine sweep refined: decode scales with npl up to the crash threshold.
| npl | S_TG (decode) |
|-----|---------------|
| 8   | 118           |
| 12  | **144**       |  <- near vLLM ~150
| 16  | CRASH         |  ("invalid configuration argument", ggml-cuda.cu:106 /
|     |               |   common.cuh:1650 - a kernel launch config goes invalid at
|     |               |   npl>=16. NOT the GDN kernel (its grid/block are fine at 16).
|     |               |   Pre-existing, with+without RCCL. Needs the exact failing
|     |               |   op isolated.)

So the ENGINE can reach ~144 tok/s @ npl12 (~96% of vLLM) - the compute is there.
The npl>=16 crash is worth fixing (would let the engine/server go higher).

SERVER sustained-burst DEGRADATION (newly found): back-to-back burst runs degrade.
First run after a fresh server start: C=8=86, C=16=96 (good). Runs 2,3: C=8 drops
to 61-63, C=16 to 75-78, with queue_ms median growing 59->2105 ms and TTFT_max
12-27s. So the continuous-batching state accumulates across runs (graph-cache
thrash, or the lazy-init views / KV pool not releasing between bursts) and tanks
sustained throughput. The first-run peak (~96 @ C=16) is the real capability;
the degradation is a state-management bug to chase. LLAMA_UX_DYNAMIC_BUDGET=1 +
LLAMA_PREFILL_CHUNK=2048 + -b 8192 did NOT fix it (same degradation pattern).

MTP-on-TP final verdict: with the CPU draft sampler (speculative.cpp uses
common_sampler, not the context backend sampler) MTP accepts drafts (2 tok/decode)
but the CPU sampling latency (~125 ms/step at C=8) makes it a NET LOSS at C>=8
(C=8: 78 with MTP vs 86 without). Mirroring output.weight to enable on-device
sampling was tested + reverted: it makes the backend sampler not-assert, but the
MTP draft sampling still goes through common_sampler (CPU), and forcing
LLAMA_TP_BACKEND_SAMPLER=1 yields 1 tok/decode (drafts rejected - TP numerical
drift between GPU draft sampling and target verification). So MTP-on-TP needs a
common_sampler -> backend integration in speculative.cpp + deterministic verify.

FINAL ACHIEVABLE STATE: TP4 single-server, correct, ~86-96 tok/s peak (first-run
C=8/16, decode-dominated), degrading under sustained burst. Engine ceiling 144
@ npl12 (near vLLM 150). The path to full vLLM parity = (1) fix the server
sustained-burst state degradation (biggest serving-side win), (2) fix the
npl>=16 engine crash (lets both go higher), (3) MTP-on-TP via common_sampler
backend integration, (4) int8 compute (hipBLAS int8 or GPTQ model).

### Sustained-degradation fix attempt + final numbers (2026-07-26)

Added a size bound (65536) to the meta-backend split_state_cache
(ggml-backend-meta.cpp, keyed by tensor pointer so it grew unbounded under TP
graph rebuilds). This bounds host memory + keeps lookups fast, but did NOT fix
the sustained-burst degradation - so the VRAM growth (7.1 -> 9.6 GB/GPU) and
throughput collapse across runs is the GPU-side graph cache / KV pool under TP,
not the host split_state_cache. Kept the bound (good hygiene).

Reproducible fresh-server first-run peaks (decode-dominated, ignore_eos, 256 out):
C=8 = ~87 tok/s, C=12 = ~80, C=16 = ~103. (An earlier 136.7 @ C=16 was an
outlier.) Runs 2-N degrade to ~77-80 with queue_ms growing to 2-4s. Engine
ceiling for reference: npl8=118, npl12=144, npl>=16 crash.

The GPU graph-cache growth under TP (each distinct ubatch shape caches 4
subgraphs; sustained burst with shape churn grows it without reclaim) is the
remaining serving-side lever - needs the meta-backend graph-cache lifecycle
bounded/cleared, which is the next concrete fix.

### CORRECTNESS BLOCKER (2026-07-26) - TP4 recurrent-state reshape view

A correctness gate (TP4 vs single-GPU greedy, fixed prompts) revealed that the
TP4 recurrent path is fundamentally broken for sustained use:

- ORIGINAL ggml-backend-meta.cpp: the FIRST request is CORRECT
  ("Water boils at sea level at" -> " 100°C.", matching single-GPU). The SECOND
  request crashes the server (ggml-backend-meta.cpp:1837 GGML_ASSERT - the
  recurrent-state 'cache_r_l<N> (reshaped)' RESHAPE view of the GGML_OP_NONE
  storage leaf is not in the simple_tensors map once prior recurrent state
  exists). So: correct for 1 request, crash on the 2nd.

- My two attempted fixes both PREVENT THE CRASH but CORRUPT OUTPUT:
  * lazy-init via init_tensor_impl (compute_headroom 64 + lazy_views cleanup):
    "Water boils" -> "!!!!!" (garbage). handle_reshape mis-maps the nr=2+head_ratio
    recurrent-state split for the reshape view, so the per-GPU slice is wrong.
  * alias to view_src's simple_tensor (bcj.nodes[i] = simple_tensor(view_src, j)):
    "Water boils" -> " 100°C." (correct) BUT "The primary colors are red," ->
    "!!!!!" (garbage). A RESHAPE is not transparent for head-sharded data.

So the meta-backend cannot correctly shard the RESHAPE view of the recurrent
state (cache_r_l / cache_s_l) under TP. The reshape appears once the recurrent
state has been updated (2nd+ request), and neither registering it via
init_tensor_impl (wrong split) nor aliasing to the source (wrong shape) yields
correct output. The original code simply doesn't register it (crash).

This means NONE of {original, lazy-init, alias} gives a correct sustained TP4
server. The worktree has been reverted to the ORIGINAL ggml-backend-meta.cpp
(correct single-request, crashes on 2nd). The RCCL CMake wiring
(ggml-cuda/CMakeLists.txt HIP/RCCL branch) and the llama-context.cpp sampler
comment are kept (benign).

REQUIRED FOR A CORRECT FIX (not achieved this session):
1. Either fix handle_reshape (ggml-backend-meta.cpp:597) to correctly map the
   nr=2+head_ratio recurrent-state split through the reshape, AND verify the
   per-GPU reshape view's ne/nb/data match what the GDN kernel reads; OR
2. Eliminate the reshape view by storing cache_r/cache_s in the layout the GDN
   consumes directly (change src/llama-memory-recurrent.cpp / the graph builder
   so no in-graph RESHAPE of the recurrent state is emitted); OR
3. Make the recurrent state MIRRORED under TP (replicated) so no sharded reshape
   is needed - but that is the Path-A design measured at ~5x slower (and even
   then had its own greedy-divergence on some prompts).

Until one of these is implemented correctly, the TP4 single-server is NOT usable
for sustained serving. All serving-side perf numbers earlier in this log
(C=1-32 burst, Poisson, etc.) were measured on the lazy-init build WHICH
PRODUCES CORRUPT OUTPUT for some prompts and must not be relied on.

Net session deliverable (correct, usable): TP4 engine path (llama-batched-bench
-sm tensor) works and is correct for the uniform single-shape bench (engine
144 tok/s @ npl12, near vLLM 150); RCCL all-reduce wired; XGMI confirmed. The
TP4 *server* (continuous batching) is blocked on the recurrent-state reshape
correctness issue above.

### TP4 (tensor parallel) fix + benchmark (2026-07-25)

- CRITICAL FIX: meta-backend lazy-init for view/reshape tensors during graph rebuild
  (ggml-backend-meta.cpp). The scheduler doesn't call init_tensor for view tensors
  that inherit their parent's buffer, so the meta-backend's per-device simple_tensor
  copies were missing on the 2nd+ decode step -> crash. Fix: pre-initialize all
  meta-buffer graph nodes before dispatch. Also fixed get/set_tensor_async to delegate
  multi-segment tensors to the buffer-level handler (supports nr>1 scatter/gather).
- TP4 engine (llama-batched-bench): pp512=347, pp2048=400, tg64=8.5, npl8 decode=125.8 tok/s.
- TP4 server c=8: agg 28.3 tok/s, 0 failures, TPOT 208ms, TTFT 6.2s.
- 4x independent servers (layer split off): agg 65-70 tok/s, per-GPU 17-20, TPOT 300-340ms.
- Conclusion: at 4xMI100 with 300 MHz sclk pin, independent single-GPU servers beat TP4
  by ~2.4x for this workload (decode-bound, all-reduce latency dominates).
- Paged attention: tested under -sm tensor; works for single-GPU but needs the per-device
  block-table plumbing fixed for multi-GPU (crashes at get_k_paged under layer split).
  Paged is gated off by default; under layer split it must stay off.

### TP4 allreduce + profiling (2026-07-25)

- In-house allreduce committed (N-GPU, XGMI peer-to-peer, host-mapped kernel path).
- TP4 engine npl8: 125.8 -> 148.9 tok/s (+18%).
- Profile: ar_kernel 81.6us avg x 16896 calls = 1378ms (24% of step).
  Ring-allreduce attempted but produced wrong results (chunk indexing bug), reverted.
  N*(N-1) copy approach works correctly.
- TP4 server c=8: 28.3 tok/s (allreduce enabled).

### Paged attention works under TP4 (2026-07-25)

- Fix: assert block table src[5] is MIRRORED in handle_flash_attn_ext (meta-backend).
- Paged on/off identical engine npl8: 149.6 tok/s.
- TP4 c=8 server bench next (need run_tp4_bench.sh with paged on).

### TP4 server tuning sweep (2026-07-25)

- np8 MTP2: 23.5 tok/s (1 fail), TPOT 224ms
- np8 nospec: 11.6 tok/s (4 fail), TPOT 4538ms (CPU sampling serialization)
- np16 MTP2: 29.0 tok/s (0 fail), TPOT 229ms, TTFT 5.3s
- Engine npl8: 149 tok/s. Server overhead = 5-7x engine time.
- Bottleneck: CPU sampling (full-vocab argmax x 8 seqs/step) + graph rebuild.
- TP backend sampler (LLAMA_TP_BACKEND_SAMPLER=1) crashes: needs all-gather
  of AXIS_0-split logits before TOP_K; meta backend can't do this yet.

### TP4 output.weight mirror attempt (2026-07-25)

- Mirrored output.weight + output.bias to enable TP backend sampler.
- Crashes: sampler graph creates input tensors not recognized by get_split_state.
- Reverted. CPU sampling remains the path under TP4.
- np16 MTP2 confirmed: 29.5 tok/s, 0 fail, TPOT 190ms.

### TP4 scaling analysis (2026-07-25)

- c=1: 43.3 tok/s, c=4: 27.4, c=8: 29.5. Poor scaling (8x conc -> 1.07x throughput).
- Bottleneck: serialized CPU sampling (full-vocab argmax x N seqs per step).
- Ring-allreduce committed (bandwidth-optimal, latency-neutral at current sizes).
- Next: reduce sampling overhead or implement TP backend sampler properly.

### TP backend sampler investigation (2026-07-25)

- Mirroring output.weight+bias: logits become AXIS_1 split (not AXIS_0).
  handle_per_row passes, but downstream ops still hit UNKNOWN split state
  (likely the PAD wrapping logits or sampler graph inputs).
- Allreduce on last subgraph: split states are cached at init_tensor time,
  not recomputed post-allreduce. Cannot use this approach.
- Conclusion: TP backend sampler requires meta-backend architecture changes
  to support post-allreduce split state updates. Deferred.
- Current best TP4: 29.5 tok/s (np16, MTP2, CPU sampling).

### TP4 nospec vs MTP comparison (2026-07-25)

- nospec np16 c8: 38.7 tok/s (3 fail), per-slot TPOT 51ms
- MTP2 np16 c8: 29.5 tok/s (0 fail), per-slot TPOT 87ms
- nospec is 31% faster aggregate. MTP overhead > acceptance benefit under TP4.
- Reason: draft model adds 2 extra decode steps per verify, each with allreduce.
- Decision: keep MTP2 for correctness/UX but optimize draft model path.

### Engine TP4 scaling + allreduce analysis (2026-07-25)

- Engine npl8: 149, npl16: 253.5 tok/s (scales well)
- Server c=1 nospec: 32 tok/s, c=8 nospec: 38.7, c=8 MTP2: 29.5
- Bottleneck: full-vocab logits readback (15MB/step) + CPU argmax (no TP backend sampler)
- MTP costs 30% throughput under TP4 (draft model also needs allreduce)
- Allreduce kernel: 81.6us avg, host-mapped pinned mem path. Copy-engine path
  via hipMemcpyPeerAsync is latency-equivalent at current tensor sizes.
- Ring-allreduce committed but latency-neutral (6 sequential steps vs 3 parallel).

### GPU argmax attempt (2026-07-25)

- Added GPU argmax post-compute to populate sampling.sampled under TP4.
- Result: 29.2 tok/s (no improvement). CPU sampler still reads full logits
  because non-greedy chain needs them. Argmax overhead adds latency.
- Reverted. Need full TP backend sampler to eliminate logits readback.
