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

## Useful references (load on demand)

- `OPTIMIZATION_LOG.md` - experiment history, current numbers, bandwidth analysis.
- `docs/build.md` - build flags.
- `tools/server/README.md` and `tools/server/README-dev.md` - server scope.
- `docs/development/HOWTO-add-model.md` - model integration.
- `common/jinja/README.md` - chat template engine.
- `docs/development/parsing.md` and `docs/autoparser.md` - parsers.
