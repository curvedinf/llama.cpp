#!/usr/bin/env bash
# C=16 x 128k benchmark harness for the concurrency-optimization branch.
# Usage: ./bench-c16.sh [build_dir] [label] [quick|full]
#   quick: 16 x 8k prompt  (fast sanity, ~1 min)
#   full:  16 x 128k prompt (the target workload; long prefill)
#
# Safety: refuses to start on a contended GPU, and a watchdog kills the bench
# if its VRAM usage exceeds MEM_CAP_MB (default 20000 MB) so the rest of the
# system always keeps breathing room.
set -euo pipefail

MODEL=/home/chase/Projects/SDGraft/checkpoints/Qwen3.5-0.8B-MTP/Voodoo80/Qwen3.5-0.8B-MTP.Voodoo80_Q6_K.gguf
BUILD=${1:-build}
LABEL=${2:-run}
MODE=${3:-quick}
OUT=bench-results
mkdir -p "$OUT"

MEM_CAP_MB=${MEM_CAP_MB:-20000}
MEM_CAP_BYTES=$(( MEM_CAP_MB * 1024 * 1024 ))

# refuse to run on a contended GPU unless FORCE=1
# note: a display-connected GPU idles at a few % use - check compute processes and VRAM instead
KFD_PIDS=$(rocm-smi --showpids 2>/dev/null | grep -cE "^[0-9]+" || true)
GPU_USE=$(rocm-smi --showuse 2>/dev/null | grep -oP 'GPU use \(%\): \K\d+' | head -1 || echo 0)
VRAM_FREE_MB=$(amd-smi metric --mem-usage 2>/dev/null | grep -oP 'FREE_VRAM:\s*\K[0-9]+' | head -1 || echo 0)
VRAM_FREE=$(( ${VRAM_FREE_MB:-0} / 1024 ))
# full mode needs ~16 GiB (KV + weights + state + compute); require 18 GiB free headroom
NEED_FREE=18
[ "$MODE" = quick ] && NEED_FREE=8
if [ "${FORCE:-0}" != 1 ] && { [ "${KFD_PIDS:-0}" -gt 0 ] || [ "${GPU_USE:-0}" -gt 20 ] || [ "${VRAM_FREE:-0}" -lt "$NEED_FREE" ]; }; then
    echo "error: GPU not clear (${KFD_PIDS} compute processes, ${GPU_USE}% used, ${VRAM_FREE} GB VRAM free; need ${NEED_FREE} GB) - results would be invalid/dangerous."
    echo "       wait for it to free up, or run with FORCE=1 to proceed anyway."
    exit 1
fi

if [ "$MODE" = full ]; then
    NPP=130816
else
    NPP=8192
fi
NTG=128
NPL=16
NCTX=$((NPL * (NPP + NTG)))

LOG="$OUT/${LABEL}-${MODE}-$(date +%Y%m%d-%H%M%S).log"
{
    echo "# date:   $(date -Is)"
    echo "# branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    echo "# build:  $BUILD  mode: $MODE  npp=$NPP ntg=$NTG npl=$NPL nctx=$NCTX  mem_cap=${MEM_CAP_MB}MB"
    echo "# model:  $MODEL"
} > "$LOG"

"$BUILD/bin/llama-batched-bench" \
    -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 \
    -c "$NCTX" -npp "$NPP" -ntg "$NTG" -npl "$NPL" >> "$LOG" 2>&1 &
BENCH_PID=$!

# watchdog: kill the bench if its VRAM exceeds the cap
while kill -0 "$BENCH_PID" 2>/dev/null; do
    USED=$(rocm-smi --showpids 2>/dev/null | awk -v pid="$BENCH_PID" '$1 == pid {print $4; exit}')
    if [ -n "$USED" ] && [ "$USED" -gt "$MEM_CAP_BYTES" ]; then
        echo "error: bench process ${BENCH_PID} exceeded ${MEM_CAP_MB} MB VRAM cap (${USED} bytes) - killed" | tee -a "$LOG"
        kill -TERM "$BENCH_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$BENCH_PID" 2>/dev/null || true
        wait "$BENCH_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
wait "$BENCH_PID"
RC=$?

cat "$LOG"
echo "# written: $LOG"
exit $RC
