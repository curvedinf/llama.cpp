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
| **optimized (ub 1024, in-place GDN + conv)** | **17859 - 18014** | **1595 - 1599** | **13643 - 13783** |

vs HEAD baseline: PP +10.0%, TG +64.9%, S +24.2%. vs upstream: PP +3.8%, TG +21.4%, S +8.4%.
At 16x8k (quick): PP 16187 vs upstream 16595 (-2.5%), TG 1435 vs upstream 1329 (+8.0%).

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

## Constraints

- VRAM budget is hard-capped at 20 GB for any benchmark or test run (the GPU
  also drives the desktop). All GPU runs must go through bench-c16.sh (it has
  the contention guard and the MEM_CAP_MB watchdog); use MEM_CAP_MB=16384 to
  keep desktop headroom. Never run test-backend-ops or other large-allocation
  binaries unguarded.
