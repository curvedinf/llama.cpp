# llama.cpp - divergent branch (concurrency-optimization)

This is a divergent fork of llama.cpp. It is **not** intended for upstream PR.
All upstream-specific workflow (PR conventions, contributor guidelines, reviewer
etiquette, commit footer rules, `gh` usage) is intentionally absent - develop
directly on the branch, commit freely, log experiments in OPTIMIZATION_LOG.md.

Target: close the concurrency gap with vLLM on single-GPU small-model workloads
using only generalizable techniques (each change must help the current bench
**and** plausibly help other regimes - larger C, longer context, larger models).

## Code style

- ASCII only. No emdash (`-` not `-`), no unicode arrows (`->` not `->`), no
  unicode punctuation (`x` not `x`, `...` not `...`).
- Comments only when a non-obvious invariant would otherwise be lost. Never
  restate what the code says.
- Reuse existing infrastructure. Read the surrounding code first - changes must
  blend in. No new subsystems without prior discussion.
- For any change that touches the decode hot path or the GPU pipeline, bench it
  before and after on the standard workload.

## GPU safety rules (mandatory, learned from 3 machine crashes)

- **20 GB VRAM hard cap** for any process. The GPU also drives the desktop.
- Before ANY GPU work, start the watchdog and keep it running:
  `./vram-watchdog.sh 20480 &`
  It SIGKILLs any compute process over 20 GB (and the largest one if total
  compute VRAM exceeds 22 GB). Kills are logged to
  `bench-results/vram-watchdog.log`.
- All GPU runs must go through `./bench-c16.sh` or `./run-guarded.sh 16384 --`.
  Never run `test-backend-ops` or other large-allocation binaries unguarded;
  only targeted subsets (`-o MUL_MAT` etc.) under `run-guarded.sh`.
- Shader experiments that increase per-workgroup resources (LDS/registers,
  e.g. larger BK tiles) can HANG the GPU - a VRAM watchdog cannot stop that.
  Validate such changes with the smallest possible run first
  (`llama-cli -n 8` under `run-guarded.sh`) before any benchmark.

## Bench workflow

Target hardware: Radeon RX 7900 XTX, Vulkan backend, FA on, KV q8_0.
Target model: `Qwen3.5-0.8B-MTP.Voodoo80_Q6_K.gguf` (hybrid delta-net).

Profiles (all under `./bench-c16.sh <build_dir> <label> <profile>`):

| profile | definition | use |
|---------|------------|-----|
| `s4k`   | `-npp 4096 -ntg 128 -npl 16` | prior-baseline continuity |
| `s256`  | `-npp 4096 -ntg 256 -npl 16`, MTP on | **primary metric** |
| `s256-mixed` | 8 sequences decoding + 8 fresh 4k prompts arriving staggered | exercises chunked prefill |
| `s256-over`  | 32 sequences x 4k under 16 GB cap | exercises preemption |
| `quick` | `-npp 8192 -ntg 128 -npl 16` | fast sanity (~1 min) |
| `full`  | `-npp 130816 -ntg 128 -npl 16` | long-prefill regression watch |

- Log every experiment (tok/s, kept or reverted) in `OPTIMIZATION_LOG.md`.
- Commit each successful optimization; revert and log the failures.
- Cross-check at 16x8k stays in `OPTIMIZATION_LOG.md` for regression-watching.

## Hybrid model invariants (must hold across all changes)

The Qwen3.5-MTP target is a hybrid attention + delta-net model. Every change
must keep these invariants valid:

- `[TAG_RECURRENT_ROLLBACK_SPLITS]`: the trailing `1 + n_rs_seq` tokens of each
  recurrent sequence must land in the same ubatch (see `src/llama-memory-hybrid.cpp`).
- In-place GDN/conv state write-back: the `inp->direct` flag (`src/llama-memory-recurrent.cpp`)
  is only safe when every batch seq reads and writes the same row; the graph
  cache must keep both the in-place and copy-path variants co-resident.
- Prefix-end-anchored snapshot protocol: recurrent-state snapshots are taken
  only at aligned 32-token block ends (`src/llama-memory-hybrid.cpp:323-368`).
- Any new op or shader variant must be validated bit-exact (or generation-
  identical, if accumulation order changes by design) against the reference
  path. The GDN beta-sigmoid fusion experiment
  (`OPTIMIZATION_LOG.md:199-218`) is the cautionary tale.

## Roadmap (in execution order)

1. **Chunked prefill co-location** - mix 512-tok prefill chunks with running
   decode tokens in the same `llama_batch`. Fixed chunk size, env-tunable.
2. **Async scheduler** - run next-iteration CPU prep on a second thread while
   the GPU computes. Thread-safety via cell snapshots (not mutexes).
3. **fp16 recurrent state** - halve the delta-net state IO bandwidth. New
   GDN/conv shader variants. (FP8 KV is dropped from this roadmap - bench
   stays Q8_0.)
4. **Preemption via recompute** - on KV pressure, drop victim sequence and
   re-admit via chunked prefill. CPU swap is dropped.

Each phase: bench cycle, log to `OPTIMIZATION_LOG.md`, revert on regression.

## ROCm/gfx908 port (branch concurrency-gfx908)

A ROCm/HIP port of this branch's optimizations exists on branch
`concurrency-gfx908` (4x MI100, ROCm 7.2.0). See `OPTIMIZATION_LOG.md` for
the full session log.

- Paged attention (G1) is implemented for the HIP backend, gated by
  `LLAMA_KV_PAGED=1` (default OFF - measured ~2.6x slower than paged-off on
  the server c=8 bench).
- Build (gfx908):
  `cmake -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx908 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON -DCMAKE_HIP_FLAGS="-isystem /opt/rocm-7.2.0/include -L/opt/rocm-7.2.0/lib"`
  Gotchas: the system `/usr/include/hip` is ROCm 5.7 and breaks builds - the
  `-isystem /opt/rocm-7.2.0/include` flag above is required. Use the cmake
  and ninja in `~/.venvs/cmake-ninja/bin`. `GGML_HIP_RCCL=ON` wires the RCCL
  all-reduce (consumed in `ggml/src/ggml-hip/CMakeLists.txt`); the
  `ggml-cuda/CMakeLists.txt` NCCL block is skipped on HIP builds.
- Single-server tensor parallelism (`-sm tensor`, 4 MI100s, TP4) is the target
  serving topology (NOT the 4-independent-endpoints `run_4x_bench.sh`). It needs:
  - `LLAMA_PREFIX_CACHE_DISABLE=1` - the recurrent state is sharded with
    `nr = 2+head_ratio` (`src/llama-model.cpp:538`), and the meta-backend
    snapshot readback (`get_tensor_async`) only handles `nr==1`. Disabling the
    prefix cache avoids the snapshot path entirely. Correct + fast (engine
    1023 prefill / 118 decode @ npl 8). Server peak ~106 tok/s @ C=16 (~71% of
    vLLM). See OPTIMIZATION_LOG.md "PIVOT to Path B" for the full fix set and
    characterization. The meta-backend fixes (RESHAPE-view lazy-init,
    compute_headroom=64, get/set_tensor_async offset relaxation) are required.
  - MTP under TP works but the draft sampler falls back to CPU
    (SPLIT_MODE_TENSOR disables backend sampling) - helps C=1, hurts C>=8.
  - npl>=16 crashes the engine bench ("invalid configuration argument") with
    and without RCCL - a pre-existing high-batch recurrent-graph issue.
- XGMI: confirmed in use for inter-GPU copies (`hipMemcpyPeerAsync` ~37 GB/s,
  `ggml-cuda.cu:832`). `GGML_CUDA_P2P` is NOT needed (gates a separate pointer
  P2P mechanism; made no perf difference).
- Environment caveat: all 4 MI100s run power-capped at 105 W (stock 290 W);
  clocks are variable (idle ~300 MHz, ramp under load, max 1502 MHz) but
  power-bound, so all perf numbers are issue/latency-bound. Raising the cap
  needs root and is the single biggest raw lever for both stacks.
- Do NOT run `vram-watchdog.sh` on this box: it is headless (no GDM to crash)
  and the 20/31 GB caps just kill legitimate 32.5 GB model loads. The MI100 is
  33.5 GB; `run_4x_bench.sh` runs uncapped and fits (~32.5 GB/card). The
  watchdog + the GPU-safety rules above are for the 7900 XTX workstation only.
- vLLM reference target: its default serve config
  (`../vllm-gfx908/scripts/serve_direwolf_qwen36.sh`) is `--tensor-parallel-size
  4` (a single endpoint across all 4 MI100s), ~1500 prefill / ~150 decode tok/s
  at this power cap. llama.cpp's 4 independent single-GPU endpoints are at
  aggregate parity (4x~390 prefill, 4x~42 decode) - the difference is per-request
  TTFT (tp4 parallelizes each prompt's prefill across 4 GPUs).

## Useful references (load on demand)

- `OPTIMIZATION_LOG.md` - experiment history, current numbers, bandwidth analysis.
- `docs/build.md` - build flags.
- `tools/server/README.md` and `tools/server/README-dev.md` - server scope.
- `docs/development/HOWTO-add-model.md` - model integration.
- `common/jinja/README.md` - chat template engine.
- `docs/development/parsing.md` and `docs/autoparser.md` - parsers.
