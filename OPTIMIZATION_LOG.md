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


