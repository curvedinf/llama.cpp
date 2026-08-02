#!/usr/bin/env bash
# RS divergence probe: same single request through (A) MTP server with n_rs_seq=2
# and (B) the same MTP pipeline with LLAMA_N_RS_SEQ_FORCE=0. Dumps RS_STEP /
# RSPROBE / LOGIT_DUMP per step so the first divergence point can be localized.
set -euo pipefail

ROOT="/home/curved/llama.cpp-concurrency"
BIN="${ROOT}/build/bin"
MODEL="/home/curved/models/qwen3.6-27b-mtp-gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf"
LOG_DIR="${ROOT}/logs/probe"
PORT=8091
PROMPT="Explain what happens when you boil water. Be concise."
N_PREDICT="${N_PREDICT:-16}"

export LD_LIBRARY_PATH="/opt/rocm/lib:${BIN}"
export LLAMA_PREFIX_CACHE_DISABLE=1
export LLAMA_RS_DEBUG=1
export LLAMA_LOGIT_DUMP=1
export LLAMA_GDN_STATE_F16=1
export LLAMA_KV_PAGED=0
export LLAMA_UX_DYNAMIC_BUDGET=0

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" --no-webui
  -c 4096 -np 1 --kv-unified -b 512 -ub 512 -t 12 -tb 12 --threads-http 4
  -lv 1 -cb -ngl 99 -fit off --no-mmap -fa on -ctk q8_0 -ctv q8_0
  --no-warmup --metrics -sm layer -ts 1,0,0,0
  --temp 0 --top-p 0.95 --top-k 20 --min-p 0.0
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0
  --spec-draft-p-min 0.0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
  --spec-draft-ngl 99 --spec-draft-backend-sampling
)

run_once() {
  local label="$1"; shift
  local log="${LOG_DIR}/${label}.log"
  pkill -9 -f "llama-server.*--port ${PORT}" 2>/dev/null || true
  sleep 2
  setsid env "$@" env HIP_VISIBLE_DEVICES=0 "${BIN}/llama-server" "${COMMON_ARGS[@]}" >"${log}" 2>&1 </dev/null &
  local srv=$!
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
      break
    fi
    sleep 2
  done
  curl -fsS --max-time 300 "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT},\"temperature\":0,\"cache_prompt\":true,\"stream\":false}" \
    >"${LOG_DIR}/${label}.json" 2>&1 || true
  sleep 1
  kill "${srv}" 2>/dev/null || true
  pkill -9 -f "llama-server.*--port ${PORT}" 2>/dev/null || true
  echo "--- ${label}: $(grep -c RS_STEP "${log}") RS_STEP, $(grep -c RSPROBE "${log}") RSPROBE, $(grep -c LOGIT_DUMP "${log}") LOGIT_DUMP"
}

run_once mtp_n2
run_once mtp_force0 LLAMA_N_RS_SEQ_FORCE=0

echo "=== first-divergence scan (LOGIT_DUMP pairs) ==="
python3 - "${LOG_DIR}" <<'EOF'
import sys, re
ld = re.compile(r'LOGIT_DUMP idx=(\d+): (.*)')
def load(p):
    out = []
    for line in open(p, errors='replace'):
        m = ld.search(line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out
a = load(sys.argv[1] + '/mtp_n2.log')
b = load(sys.argv[1] + '/mtp_force0.log')
n = min(len(a), len(b))
for i in range(n):
    if a[i] != b[i]:
        print(f'first LOGIT_DUMP divergence at step {i}: mtp={a[i]} force0={b[i]}')
        break
else:
    print(f'no divergence in first {n} steps')
print(f'total: mtp={len(a)} force0={len(b)}')
EOF
