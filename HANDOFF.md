# TP4 Optimization — Handoff

## What's done (in worktree, uncommitted)
- **Path B serving unblock** (ggml-backend-meta.cpp): RESHAPE-view lazy-init, compute_headroom 16->64, get/set_tensor_async offset relaxation, split_state_cache size bound. **BUT** these CORRUPT output under sustained TP load - see blocker.
- **RCCL all-reduce** wired (ggml/src/ggml-cuda/CMakeLists.txt HIP/RCCL branch; `GGML_HIP_RCCL=ON` replaces the blocking butterfly).
- **XGMI confirmed** (hipMemcpyPeerAsync ~37 GB/s).
- **Bench harnesses**: `/tmp/opencode/tp_bench.py` (burst+Poisson C-sweep), `/tmp/opencode/correctness_gate.py` (greedy divergence), `/tmp/opencode/profile_overhead.py`.
- **Engine TP4 characterized**: 118 tok/s @ npl8, **144 @ npl12, 192 @ npl16** (exceeds vLLM 150).
- Phase 3.7: `-ub 2048` improves engine prefill +15% (1049->1204) - **kept**, no decode cost.

## CRITICAL BLOCKER (must be fixed first - everything server-side depends on it)
The recurrent-state reshape view `cache_r_l<N> (reshaped)` (created at **src/models/delta-net-base.cpp:475** - `conv_states = ggml_reshape_3d(...)`) is not correctly handled by the meta-backend under TP once prior recurrent state exists (2nd+ request, or npl>=24):
- **Original** meta-backend: correct 1st request, crashes on 2nd (`ggml-backend-meta.cpp:1837 GGML_ASSERT(bcj.nodes[i])`).
- **lazy-init via init_tensor_impl**: no crash but CORRUPT ("Water boils"->"!!!!!!!") - `handle_reshape` (line 597) mis-maps the `nr=2+head_ratio` recurrent split (llama-model.cpp:538 `get_split_segments`).
- **alias to view_src slice**: no crash but CORRUPT (RESHAPE not transparent for head-sharded data).
- **Path A (mirror recurrent block)**: still corrupt + crash.

Worktree currently reverted to: original meta-backend + original llama-model (correct single-request, crashes on 2nd); RCCL CMake + llama-context sampler comment kept.

**Three viable fix paths** (none completed):
1. Fix `handle_reshape` (ggml-backend-meta.cpp:597) for `nr>1` recurrent state - verify per-GPU reshape view ne/nb/data match the GDN kernel reads.
2. Eliminate the reshape at delta-net-base.cpp:475 - store conv state in the 3D layout consumed; multi-file change.
3. Disable the recurrent snapshot/rollback that emits the reshape - loses prefix-rollback but may be correct.

---

## Remaining todo list (28 items, phase-tagged)

**Phase 0.2** - Formal KV/logit divergence gate (current: qualitative only; needs TP4-vs-single-GPU fixed-input forward pass measuring max/mean divergence, calibrated threshold).

**Phase 2.1** - Extend internal `allreduce.cu` n_devices 2->4 (SUPERSEDED by RCCL - RCCL handles 4-GPU; only do this if RCCL is insufficient).

**Phase 2.2** - Per-step comm-vs-compute profile (rocprof) to size decode comm headroom.

**Phase 3** (server sweeps, ALL blocked by the reshape-view fix except engine-path):
- 3.2 KV-offload `-nkvo` under TP
- 3.3 `--mlock` / `--direct-io`
- 3.4 `-ngl` partial (CPU layers)
- 3.5 `--numa` / `--poll`
- 3.6 MTP drafts 0/1/2/3/4 under TP (MTP CPU-sampler is net-loss at C>=8; needs D3)
- 3.7 `-b`/`-ub`/`-chunk` - **DONE: ub 2048 is +15% prefill**
- 3.8 FA on/off, `LLAMA_KV_PAGED` on/off

**Phase 4** (dtype - gfx908 prefers int8 > fp8~=fp16 ~= fp32; achievable on single-GPU/engine):
- 4.1 GDN recurrent F32-scalar -> bf16-MFMA (gated_delta_net.cu:104-149) - biggest, hardest
- 4.2 Conv-state `r_l` F32->F16 (llama-model.cpp recurrent_type_r + ssm-conv.cu f16 variant + **resolve the concat-dtype at delta-net-base.cpp:481**)
- 4.3 RMS-norm F16-input (norm.cu)
- 4.4 fattn-vec q8_0 KQ dot: scalar dp4a -> int8 MFMA (fattn-common.cuh:304-329)
- 4.5 GDN activations q/k/v/g/beta F32->F16 (couples to 4.1)
- 4.6 hipBLAS int8 GEMM via hipblasLt for prefill (gfx908 stability unknown)
- 4.7 Profile-driven F32-hotspot audit (rocprof)

**Phase 5** (C=8 convergence - blocked by reshape fix):
- 5.1 Combine Phase 2-4 wins, re-bench C=8 vs vLLM (target decode ~150, TTFT)
- 5.2 Graph-cache hit-rate under TP
- 5.3 Per-step host overhead (engine 192 @ npl16 vs server - close the gap)
- 5.4 `--poll` tuning

**Phase 6**:
- 6.1 Formal burst-vs-Poisson sweep (0.5x-2x capacity)
- 6.2 Poisson parity (LLAMA_UX_DYNAMIC_BUDGET / chunked-prefill middle ground - budget off = bad TTFT tail, on = -2x aggregate)

**Phase 7**:
- 7.1 Long-duration stability (0 failures)
- 7.2 Cross-check s4k/s256/quick profiles under TP

**Discovered issues**:
- D1 Server sustained-burst degradation (1st run ~103 @ C16, runs 2+ -> ~77; VRAM 7.1->9.6 GB/GPU; not split_state_cache; likely GPU graph-cache under TP). **Note: may be partly the reshape corruption masquerading - re-verify after the reshape fix.**
- D2 npl>=16 crash - actually npl<=16 WORKS (192 tok/s), crashes at npl24 (reshape view). Fixing the reshape fix likely resolves this.
- D3 MTP-on-TP via common_sampler->backend (speculative.cpp samples drafts on CPU; rewrite for GPU + deterministic verify across TP drift). Unblocks ~1.5-2x decode.
- D4 Path B proper: nr>1 gather in get/set_tensor_async (lifts LLAMA_PREFIX_CACHE_DISABLE=1, re-enables prefix cache under TP).
- D5 LLAMA_TP_BACKEND_SAMPLER aborts (sampler-op AXIS_0 at ggml-backend-meta.cpp:546) - needs sampler-op sharding (all-gather logits before argmax). Required for D3.

## Build/run config
```
cmake -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx908 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON -DCMAKE_HIP_FLAGS="-isystem /opt/rocm-7.2.0/include -L/opt/rocm-7.2.0/lib"
# ninja in ~/.venvs/cmake-ninja/bin
# model: /home/curved/models/qwen3.6-27b-mtp-gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf
# TP4: HIP_VISIBLE_DEVICES=0,1,2,3 LLAMA_PREFIX_CACHE_DISABLE=1 llama-server -sm tensor ...
# no watchdog (headless box), --no-mmap (HIP livelock), 4xMI100 power-capped 105W (needs root to lift)
```

## Files modified in worktree (uncommitted)
- `AGENTS.md`, `OPTIMIZATION_LOG.md` (docs - full session log incl. all numbers + the blocker analysis)
- `ggml/src/ggml-cuda/CMakeLists.txt` (RCCL branch), `ggml/src/ggml-cuda/common.cuh` (LAUNCHFAIL debug print - remove before final), `src/llama-context.cpp` (sampler comment)
- `ggml/src/ggml-backend-meta.cpp` and `src/llama-model.cpp` - **reverted to upstream** (the corrupting Path B/Path A changes are gone)

The reshape-view fix (delta-net-base.cpp:475 + handle_reshape) is the keystone - fix that and the server items unblock.

# Gap Task List — `concurrency-optimization` @ HEAD `581572b9e` (100 commits ahead)

Respecified after evaluating the 31 commits pushed 2026-07-26 (17:09→19:49 UTC). Supersedes the sequencing in `concurrency-optimization-vs-vllm-gap-analysis.md`.

## What the 31 commits changed (evaluation)

**Landed (code):**
- **Greedy logits-readback elimination under TP4** (`b9d332787` + `dfac64b92` + `677bb1eba`): GPU argmax populates `sampling.sampled`; `common_sampler_sample` checks backend-sampled token before `set_logits`, skipping the 600KB/seq copy for temp≤0. **Measured gain: none** — 45.8/46.0 tok/s inside the 44–53 variance band (±15% run-to-run). Committed as "algorithmically correct, marginal at c8".
- **AR hot-path trims** (`98948da56`, `dc90c4c86`): removed `cudaGetLastError` + 512 per-step validation checks. Final engine: npl8 149.3, npl16 259.8 tok/s. AR still 24% of GPU time (129 ARs/forward), "at hw limit".
- **`handle_generic` MIRRORED broadcast** (`d8a51320c`): enabler for future sampler ops with mixed MIRRORED/split sources.
- **rs-head `can_reuse` fix** (`75b9fca8e`): correct, but graph reuse still 8/20 — draft/verify shape alternation evicts.

**Failed/reverted (with structural findings):**
- **TP backend sampler attempt 3**: split states are computed at `init_tensor` (graph build), *before* AR runs → a real TP sampler requires restructuring when `init_tensor` executes. Reverted.
- **output.weight mirror**: 27.5 tok/s (worse — full-vocab projection replicated per GPU). Reverted.
- **Q6_K MMQ occ4**: crash, reverted. Graph cache 32: worse. Layer-split: worse than tensor-split.

**Confirmed facts (measurement):**
- **Server overhead is the dominant gap: engine 6.7 ms/tok vs server 54 ms/tok at c8 → 47 ms/tok orchestration overhead.** At c=1 overhead ≈ 0. Sampling fast-paths didn't dent it → cost is structural (per-slot serialized loop + MTP orchestration), not per-call.
- **HIP graphs are dead under TP4**: identical on/off — "small per-subgraph graphs, AR invalidates capture".
- **Chunked prefill doesn't help TP4** (compute-bound, GPU saturated in prefill). FA vec = 2% of step. GDN state cache split = optimal. MTP2 optimal (MTP3 crashes). c8 decode aggregate 110 tok/s @ 85% draft acceptance.

**Untouched:** reshape-view keystone, prefix cache under TP, varlen paged FA (still 2.6× slower, default OFF), dtype program, async scheduler.

---

## P0 — Correctness keystone (still blocks all server-side TP claims)

- [ ] **T1. Fix recurrent-state reshape view under TP** (`delta-net-base.cpp:475` / meta `handle_reshape` nr>1). Recommended: HANDOFF path 2 — store conv state in the 3D layout the GDN kernel consumes, eliminating the reshape. *Accept:* 2nd+ request and npl≥24 produce output identical to single-GPU greedy; formal divergence gate (Phase 0.2) in CI.
- [ ] **T2. Re-enable prefix cache under TP** (D4: nr>1 gather in `get/set_tensor_async`). *Accept:* no `LLAMA_PREFIX_CACHE_DISABLE=1` needed; measured prefix-hit rate >0 at TP4; correctness gate green.
- [ ] **T3. Re-verify + fix D1 sustained-burst degradation** (103→77 tok/s, VRAM 7.1→9.6 GB/GPU) after T1; audit graph-cache under TP. *Accept:* 3 consecutive bursts within variance; flat VRAM.
- [ ] **T4. MTP3 crash.** Draft-count-3 crash is uninvestigated. *Accept:* MTP3 runs or is bounded with an assert explaining the limit.

## P1 — Server step overhead: 47 ms/tok at c8 (the dominant *measured* gap; 7× engine step)

- [ ] **T5. Instrument the server step.** Per-phase timers: batch construction, per-slot sampling loop, draft/verify orchestration, logits readback, HTTP/streaming. Rationale: three sampling fast-paths landed with zero measured gain — the cost is in the loop structure, not the calls. Do not optimize further blind. *Accept:* per-phase ms breakdown published to OPTIMIZATION_LOG at c1/c8/c16.
- [ ] **T6. Batch the per-slot sampling path.** Single `set_logits` over all outputs; one sampler invocation per step across slots (vLLM's model); collapse draft+verify per-slot round-trips into batch operations. *Accept:* c8 server step ≤ 15 ms/tok (from 54); c8 aggregate ≥ 200 tok/s decode.
- [ ] **T7. MTP graph-cache thrash.** Draft/verify shape alternation causes 8/20 reuse. Shape-specialized entries keyed on (n_tokens × n_seqs) or reserved dual slots. *Accept:* reuse ≥ 18/20 with MTP2.

## P2 — TP sampler done structurally

- [ ] **T8. Restructure `init_tensor` timing** so split states resolve post-AR (attempt-3 finding); then shard the sampler: per-rank local argmax/top-k over AXIS_1 logits + all-gather of candidates (vLLM-style), covering **non-greedy** params. *Accept:* zero full-vocab readback at TP4 for any sampler config.
- [ ] **T9. MTP draft sampling on GPU** (D3, depends on T8). *Accept:* the ~1.5–2× decode at C≥8 that HANDOFF estimates; CPU draft path removed under TP.

## P3 — Communication substrate (at hw limit for the current design)

- [ ] **T10. Capture-safe AR.** Replace host-spin/pinned flags with device-memory synchronization (or RCCL under graph capture) so HIP graphs engage under TP. Own finding: "AR invalidates capture" → graphs are currently wasted. *Accept:* `GGML_HIP_GRAPHS=ON` produces a measurable TP4 gain (today: identical).
- [ ] **T11. AR/compute overlap, retry.** Double-buffered shadow AR — only after T10 and after P1 stops dominating (nothing to hide behind today). *Accept:* ≥10% step-time reduction at c8 decode.
- [ ] **T12. Fuse reduce-scatter + RMSNorm + allgather** around the post-attention/post-MLP ARs (FlashInfer-style): halves AR bytes, kills a kernel boundary. *Accept:* AR share of step < 15% (from 24%).

## P4 — Paged attention: from tax to feature

- [ ] **T13. Batched varlen paged FA kernel** — one FA node per batch instead of per-seq+concat; hoist table walk. *Accept:* paged ≥ dense decode at c≥8; flip `LLAMA_KV_PAGED` default ON.
- [ ] **T14. int8-MFMA q8_0 KQ dots in fattn-vec** (HANDOFF 4.4) and paged-prefill path (ncols≥2 without IMA). *Accept:* paged kernel within 20% of vLLM-ROCm FA on same hardware.

## P5 — Compute dtype program (the self-described "FUNDAMENTAL" term)

- [ ] **T15.** GDN recurrent F32→bf16 MFMA (4.1) → conv-state f16 (4.2) → GDN activations f16 (4.5). *Accept:* decode +10% at TP4, generation-identical.
- [ ] **T16.** hipblasLt int8 GEMM for prefill on gfx908 (4.6, stability unknown — spike first). *Accept:* prefill ≥1.5× or document the hardware ceiling.

## P6 — Topology & scheduling

- [ ] **T17. Poisson-arrival bench** (Phase 6.1/6.2) to make the DP-vs-TP call with data: 4 independent endpoints already match vLLM-tp4 aggregate; TP4's edge is TTFT. *Accept:* routing/topology recommendation in OPTIMIZATION_LOG.
- [ ] **T18. Async scheduler thread** — re-evaluate after T5's profile; previously deferred as ≤5%, which assumed the wrong overhead model. *Accept:* scheduler CPU time off the GPU critical path.

**Deprioritized by evidence:** chunked-prefill tuning under TP4 (compute-bound, no effect), FA vec kernel micro-opt (2% of step), AR block/stride sweeps (optimal found), graph-cache resizing (8 optimal).

**Critical path:** T1 → T2/T3 → T5 → T6 → T8/T9 → T10 → T11/T12. P4/P5 parallel-safe anytime.


# Verification of Agent Wave-2 Claims — 2026-07-27

**Method:** GitHub API was rate-limited and git/https intermittently blocked, so verification used the full branch tarball (codeload) diffed against a fresh upstream-master tarball, cross-referenced with the pre-wave-2 diff stats recorded in the previous verification. New OPTIMIZATION_LOG sections (lines 1699–1988) read in full.

## What is TRUE this time

- **Real code exists.** Wave-2 source deltas vs the pre-wave-2 state: `llama-model.cpp` (+17/−17: flat single-segment layout for r_cache/s_cache), `delta-net-base.cpp` (+28/−24: prefill via `ssm_conv_idx` instead of get_rows+reshape+concat), `ggml-backend-meta.cpp` (+12: split_state_cache clear on rebuild), `ggml-cuda.cu` (+13/−45), `server-context.cpp` (~−5: DEBUG_TIMINGS). Consistent with the claimed "7 files, +84/−71." **The "all functional code" claim — false in wave 1 — is true in wave 2.**
- **The root-cause diagnosis is genuinely good.** They pinned the residual tensor-split corruption to `handle_reshape` producing inconsistent per-device shapes for the `state_predelta` reshape_4d across graph rebuilds, with concrete GDN_IDX evidence: correct `{128,128,24,1}` vs wrong `{786432,0,1,1}` (stale full tensor), `{393216,1,1,1}` (flat), `{0,1,1,1}` (zero). This matches and refines your HANDOFF analysis. The flat-segment + `ssm_conv_idx` changes are real, sensible fixes that **eliminated the crash** under tensor split. Keep them.
- **T17 Poisson sweep actually ran** (rates 1/2/4/8 with a table) — done properly this time.
- **T5 instrumentation exists** (DEBUG_TIMINGS, per-phase table) — see caveats below.

## What is FALSE or misleading

### 1. "T1 RESOLVED" — it was bypassed, not fixed, and the price is the whole program
The log's own sequence: real fixes → crash gone under tensor split → **corruption remained** ("TP2: correct on request 1, '!!!' garbage on request 2+"; "split_state_cache clear didn't fix it") → "RESOLVED: switched to `-sm layer`" (run_tp4_bench.sh now `-sm layer -ts 1,1,1,1`, comment: "layer split keeps tensors whole"). `handle_reshape` is untouched. The keystone bug is documented, not fixed.

The cost, in the agent's own numbers vs your own prior measurements:

| Metric | Your tensor-split (engine) | Agent's layer-split | vLLM ref | Verdict |
|---|---|---|---|---|
| Decode C=8 agg | 149.3 tok/s (npl8) | **82 tok/s** | ~150 | **1.8× regression vs your own path** |
| Prefill | 1204 tok/s (ub2048) | **384 tok/s** | ~1500 | **3.1× regression** |
| vs vLLM | ~96% (npl12) | 55% decode / 26% prefill | — | gap widened on every axis |

Layer split also orphans the entire TP program: no allreduce (your ring-AR work is dead code in this config), KV per-GPU unshared (the agent's own T13 entry admits this "reduces the benefit of paging"), and your earlier direct measurement said it plainly: "Layer split... worse than tensor split. Tensor split confirmed optimal." "10/10 correct" is also still a small sequential sample — the Phase 0.2 formal divergence gate remains unbuilt.

### 2. The C=8 corruption self-contradiction — unresolved
- T1 section: "**TP4 layer split C=8: 0/8 corrupted**"
- T3 section (same wave, same date): "**concurrent C=8 has high corruption rate (6-7/8)** — separate issue from burst stability"

One of these is false. Sequential tests pass; the concurrent figure says 6-7 of 8 outputs are garbage. If the T3 note reflects current state, **the branch cannot serve concurrent load in any configuration** and "T1 RESOLVED" is false even under layer split. This must be resolved before anything else is believed.

### 3. T2/T3/T4 — true only inside the retreated architecture
- T2: prefix cache "works" under layer split because D4 (nr>1 gather) is a tensor-split problem that layer split never encounters. The task's TP4 acceptance criterion is unmet where it matters.
- T3: bursts 71/69/76 stable — consistent with layer-split C=8 (~82), but this is ~half of your tensor-split server's c8 decode aggregate (110).
- T4: MTP3 no-crash under layer split, 5 sequential correct — plausible, but no draft-acceptance number reported (last wave's 23% was the corruption tell; its absence is conspicuous).

### 4. T5 — better, but a 2.5× hole is hand-waved
Their own math: (89.5+25.8)/(8×3) ≈ 4.8 ms/tok → ~208 tok/s theoretical; measured 82. The 2.5× discrepancy is attributed to "HTTP/streaming overhead not captured by timers" with zero evidence. Also presented as "much improved" over the 47 ms figure without noting the config changed to a strictly slower parallelism mode.

### 5. T7 — "adequate" with no measurement ("would need profiling"). Corner cut, again.
### 6. T10–T12 entry is incoherent post-retreat — it discusses AR being "24% of GPU time" and "at hw limit" in a config where **there is no allreduce at all**, and still says "would need RCCL" without noticing RCCL is wired in CMake.

## Bottom line

| Area | Verdict |
|---|---|
| Code reality (7 files, +84/−71) | **TRUE** |
| handle_reshape root-cause diagnosis | **TRUE and valuable** |
| Crash elimination (tensor split) | **TRUE** |
| "T1 RESOLVED" | **FALSE** — bypassed via layer split; 1.8× decode / 3.1× prefill regression vs your own TP path; AR/paging programs orphaned |
| C=8 correctness | **CONTRADICTED BY ITS OWN LOG (0/8 vs 6-7/8)** |
| T2–T4 | Scoped-valid under layer split only; TP acceptance criteria unmet |
| T5 | Improved, but 2.5× theory/measurement gap unexplained |
| T7, T10–T12 | Not performed / incoherent |

## Recommendations
1. **Do not accept `-sm layer` as the serving config.** Keep it as the golden-reference generator for the divergence gate (its sequential output is correct — that's valuable).
2. **Fix the actual bug using their own diagnosis.** Their pre-retreat candidate #2 — bypass `handle_reshape` for the `state_predelta` reshape via `view_4d` (views propagate src split state) — or forced shape validation in `init_tensor_impl` for RESHAPE ops. They had the root cause and two concrete fix paths, tried one (cache clear), and stopped.
3. **Immediately resolve the C=8 contradiction**: run the concurrent 8-slot correctness gate under layer split. If 6-7/8 is current, the keystone is open in every config and all serving numbers are void.
4. Keep: flat segments, ssm_conv_idx prefill path, per-layer copy, split_state_cache clear, Poisson harness, DEBUG_TIMINGS (add GPU/host attribution + close the 208-vs-82 hole).

## Verification Fixes Applied and GPU-Validated — 2026-07-27

All four recommendations implemented, compiled, and run on 4xMI100. See
OPTIMIZATION_LOG.md "Wave-2 Verification Fixes" for full details.

| Recommendation | Status | Result |
|---------------|--------|--------|
| R1: Revert to -sm tensor | **DONE** | run_tp4_bench.sh uses -sm tensor, prefix cache disabled. run_golden_reference.sh for layer-split ref. |
| R2: Fix handle_reshape | **ATTEMPTED, BUG STILL OPEN** | Per-device ne propagation broke downstream assertions (post-processing recomputes ne). Reverted. init_tensor_impl validation kept. handle_reshape bug confirmed unfixed: tensor-split crashes on 3rd sequential request. |
| R3: Concurrent correctness gate | **DONE, RAN** | 12/12 concurrent prompts diverge from layer-split reference (100% corruption). Tensor-split crashes on 3rd sequential request. Gate is at /tmp/opencode/concurrent_correctness_gate.py. |
| R4: DEBUG_TIMINGS | **DONE, RAN** | 208-vs-82 gap explained: t_decode_host=597ms (92% of decode), t_decode_gpu=53ms. HTTP overhead is 0.004ms (negligible). Host-side graph dispatch dominates, not HTTP. |

Key finding: prefix cache must be disabled for BOTH layer and tensor split.
The snapshot path (llama_rs_row_block_copy -> get_tensor_async) crashes on
meta-backend buffers (D4). All gate runs use LLAMA_PREFIX_CACHE_DISABLE=1.

The handle_reshape keystone bug remains the blocker for -sm tensor serving.
