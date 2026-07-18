#!/usr/bin/env bash
# guarded wrapper: run a GPU binary only if the GPU is clear, with a VRAM watchdog.
# usage: ./run-guarded.sh [mem_cap_mb] -- <cmd...>
set -euo pipefail

MEM_CAP_MB=${1:-16384}; shift
[ "${1:-}" = "--" ] && shift

MEM_CAP_BYTES=$(( MEM_CAP_MB * 1024 * 1024 ))

KFD_PIDS=$(rocm-smi --showpids 2>/dev/null | grep -cE "^[0-9]+" || true)
GPU_USE=$(rocm-smi --showuse 2>/dev/null | grep -oP 'GPU use \(%\): \K\d+' | head -1 || echo 0)
VRAM_FREE_MB=$(amd-smi metric --mem-usage 2>/dev/null | grep -oP 'FREE_VRAM:\s*\K[0-9]+' | head -1 || echo 0)
VRAM_FREE=$(( ${VRAM_FREE_MB:-0} / 1024 ))
if [ "${FORCE:-0}" != 1 ] && { [ "${KFD_PIDS:-0}" -gt 0 ] || [ "${GPU_USE:-0}" -gt 20 ] || [ "${VRAM_FREE:-0}" -lt 8 ]; }; then
    echo "error: GPU not clear (${KFD_PIDS} compute processes, ${GPU_USE}% used, ${VRAM_FREE} GB free)"
    exit 1
fi

"$@" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    USED=$(rocm-smi --showpids 2>/dev/null | awk -v pid="$PID" '$1 == pid {print $4; exit}')
    if [ -n "$USED" ] && [ "$USED" -gt "$MEM_CAP_BYTES" ]; then
        echo "error: process $PID exceeded ${MEM_CAP_MB} MB VRAM cap - killed" >&2
        kill -TERM "$PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
wait "$PID"
