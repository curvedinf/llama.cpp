#!/usr/bin/env bash
# Launch 4x llama-server (1 per MI100) for the concurrency-gfx908 build,
# mirroring the vllm tp2-pairs benchmark topology (c=8 per endpoint).
set -euo pipefail

ROOT_DIR="/home/curved/llama.cpp-concurrency"
BIN_DIR="${ROOT_DIR}/build/bin"
MODEL="/home/curved/models/qwen3.6-27b-mtp-gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/bench}"
HOST="127.0.0.1"
PORT_BASE="${PORT_BASE:-8080}"

N_INSTANCES="${N_INSTANCES:-4}"
N_PARALLEL="${N_PARALLEL:-8}"
CONTEXT_SIZE="${CONTEXT_SIZE:-16384}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-1024}"
THREADS_PER_INSTANCE="${THREADS_PER_INSTANCE:-12}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-2}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
GDN_F16="${GDN_F16:-0}"
KV_PAGED="${KV_PAGED:-0}"

export LD_LIBRARY_PATH="/opt/rocm-7.2.0/lib:${BIN_DIR}:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"

stop_all() {
  pkill -f "${BIN_DIR}/llama-server" || true
  sleep 2
}

start_one() {
  local idx="$1"
  local port=$((PORT_BASE + idx))
  local first_core=$((idx * THREADS_PER_INSTANCE))
  local last_core=$((first_core + THREADS_PER_INSTANCE - 1))
  local log="${LOG_DIR}/server-${idx}-port-${port}.log"
  setsid env \
    HIP_VISIBLE_DEVICES="${idx}" \
    LLAMA_GDN_STATE_F16="${GDN_F16}" \
    LLAMA_KV_PAGED="${KV_PAGED}" \
    LLAMA_UX_DYNAMIC_BUDGET="${UX_DYNAMIC_BUDGET:-0}" \
    LLAMA_PREFILL_CHUNK="${PREFILL_CHUNK:-1024}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    taskset -c "${first_core}-${last_core}" \
      "${BIN_DIR}/llama-server" \
        -m "${MODEL}" \
        --alias qwen36-27b-mtp \
        --host "${HOST}" --port "${port}" \
        --no-webui \
        -c "${CONTEXT_SIZE}" -np "${N_PARALLEL}" --kv-unified \
        -b "${BATCH_SIZE}" -ub "${UBATCH_SIZE}" \
        -t "${THREADS_PER_INSTANCE}" -tb "${THREADS_PER_INSTANCE}" \
        --threads-http 12 \
        -cb -ngl 99 -fit off --no-mmap \
        -fa on -ctk "${CACHE_TYPE}" -ctv "${CACHE_TYPE}" \
        --no-warmup \
        --metrics \
$(if [[ "${SPEC_TYPE:-draft-mtp}" != "none" ]]; then printf '%s\n' \
        "--spec-type" "draft-mtp" \
        "--spec-draft-n-max" "${SPEC_DRAFT_N_MAX}" \
        "--spec-draft-n-min" "0" \
        "--spec-draft-p-min" "0.0" \
        "--spec-draft-type-k" "${CACHE_TYPE}" \
        "--spec-draft-type-v" "${CACHE_TYPE}" \
        "--spec-draft-ngl" "99" \
        "--spec-draft-backend-sampling"; fi) \
        --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
        >"${log}" 2>&1 </dev/null &
  echo "started server ${idx}: port=${port} pid=$!"
}

wait_ready() {
  local idx="$1"
  local port=$((PORT_BASE + idx))
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 2 "http://${HOST}:${port}/health" 2>/dev/null | grep -q '"ok"'; then
      echo "server ${idx} ready"
      return 0
    fi
    sleep 2
  done
  echo "server ${idx} failed to start; last log:" >&2
  tail -20 "${LOG_DIR}/server-${idx}-port-${port}.log" >&2
  return 1
}

case "${1:-start}" in
  start)
    stop_all
    for i in $(seq 0 $((N_INSTANCES - 1))); do start_one "${i}"; done
    for i in $(seq 0 $((N_INSTANCES - 1))); do wait_ready "${i}"; done
    ;;
  stop) stop_all ;;
  *) echo "usage: $0 {start|stop}"; exit 1 ;;
esac
