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
