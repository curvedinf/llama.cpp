#!/usr/bin/env bash
# Launch a single TP4 llama-server (all 4 MI100s) for the concurrency-gfx908 build.
set -euo pipefail

ROOT_DIR="/home/curved/llama.cpp-concurrency"
BIN_DIR="${ROOT_DIR}/build/bin"
MODEL="/home/curved/models/qwen3.6-27b-mtp-gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/tp4}"
HOST="127.0.0.1"
PORT="${PORT:-8080}"

N_PARALLEL="${N_PARALLEL:-8}"
CONTEXT_SIZE="${CONTEXT_SIZE:-16384}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
THREADS="${THREADS:-12}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-2}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
GDN_F16="${GDN_F16:-1}"
KV_PAGED="${KV_PAGED:-0}"
UX_DYNAMIC_BUDGET="${UX_DYNAMIC_BUDGET:-0}"
PREFILL_CHUNK="${PREFILL_CHUNK:-1024}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"

export LD_LIBRARY_PATH="/opt/rocm-7.2.0/lib:${BIN_DIR}:${LD_LIBRARY_PATH:-}"
# Prefix cache: was disabled for TP due to D4 (nr>1 gather in get/set_tensor_async).
# T1 fix may have resolved this. Test enabled.
export LLAMA_PREFIX_CACHE_DISABLE=1

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  -m "${MODEL}"
  --alias qwen36-27b-mtp
  --host "${HOST}" --port "${PORT}"
  --no-webui
  -c "${CONTEXT_SIZE}" -np "${N_PARALLEL}" --kv-unified
  -b "${BATCH_SIZE}" -ub "${UBATCH_SIZE}"
  -t "${THREADS}" -tb "${THREADS}"
  --threads-http 12
  -cb -ngl 99 -fit off --no-mmap
  -fa on -ctk "${CACHE_TYPE}" -ctv "${CACHE_TYPE}"
  --no-warmup
  --metrics
  -sm tensor -ts 1,1,1,1
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
)

if [[ "${SPEC_TYPE}" != "none" ]]; then
  COMMON_ARGS+=(
    --spec-type "${SPEC_TYPE}"
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX}"
    --spec-draft-n-min 0
    --spec-draft-p-min 0.0
    --spec-draft-type-k "${CACHE_TYPE}"
    --spec-draft-type-v "${CACHE_TYPE}"
    --spec-draft-ngl 99
    --spec-draft-backend-sampling
  )
fi

usage() { echo "usage: $0 {start|stop}"; }

case "${1:-start}" in
  start)
    pkill -f "${BIN_DIR}/llama-server" 2>/dev/null || true
    sleep 2
    log="${LOG_DIR}/server-tp4.log"
    setsid env \
      LLAMA_GDN_STATE_F16="${GDN_F16}" \
      LLAMA_KV_PAGED="${KV_PAGED}" \
      LLAMA_UX_DYNAMIC_BUDGET="${UX_DYNAMIC_BUDGET}" \
      LLAMA_PREFILL_CHUNK="${PREFILL_CHUNK}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        "${BIN_DIR}/llama-server" "${COMMON_ARGS[@]}" \
        >"${log}" 2>&1 </dev/null &
    echo "started TP4 server pid=$! port=${PORT}"
    for _ in $(seq 1 240); do
      if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
        echo "TP4 server ready"
        exit 0
      fi
      sleep 2
    done
    echo "TP4 server failed to start"; tail -20 "${log}" >&2; exit 1
    ;;
  stop) pkill -f "${BIN_DIR}/llama-server" 2>/dev/null || true; sleep 2; echo "stopped" ;;
  *) usage; exit 1 ;;
esac
