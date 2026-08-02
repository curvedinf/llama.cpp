#!/usr/bin/env bash
# TP4 burst validation: start the MTP TP4 server, fire 8 concurrent requests,
# verify all complete with coherent text and no server crash.
set -euo pipefail

ROOT="/home/curved/llama.cpp-concurrency"
LOG_DIR="${ROOT}/logs/tp4"
PORT=8080

mkdir -p "${LOG_DIR}"

"${ROOT}/run_tp4_bench.sh" start
trap '"${ROOT}/run_tp4_bench.sh" stop' EXIT

PROMPTS=(
  "Explain what happens when you boil water. Be concise."
  "Explain what happens when you boil water. Be concise."
  "Write a short poem about the sea."
  "What is the capital of France and its main landmarks?"
  "Explain the difference between TCP and UDP in networking."
  "Give three tips for learning a new language quickly."
  "Describe the water cycle in simple terms."
  "What causes the seasons on Earth?"
  "Explain how a compiler works, briefly."
)

pids=()
for i in $(seq 0 7); do
  curl -fsS --max-time 600 "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"${PROMPTS[$i]}\",\"n_predict\":48,\"temperature\":0,\"cache_prompt\":true,\"stream\":false}" \
    > "${LOG_DIR}/burst_req${i}.json" 2> "${LOG_DIR}/burst_req${i}.err" &
  pids+=($!)
done

fail=0
for i in $(seq 0 7); do
  if wait "${pids[$i]}"; then
    echo "req${i}: OK"
  else
    echo "req${i}: FAILED"; fail=1
  fi
done

sleep 2
if pgrep -f "llama-server.*--port ${PORT}" >/dev/null; then
  echo "server: alive"
else
  echo "server: DEAD"; fail=1
fi

echo "=== response sanity ==="
python3 - "${LOG_DIR}" <<'EOF'
import json, glob, sys
ok = 0
for p in sorted(glob.glob(sys.argv[1] + '/burst_req*.json')):
    try:
        d = json.load(open(p))
        c = d.get('content', '')
        n = d.get('timings', {}).get('predicted_n', 0)
        ok += 1
        print(f"{p.split('/')[-1]}: {n} tokens | {c[:90]!r}")
    except Exception as e:
        print(f"{p.split('/')[-1]}: PARSE FAIL {e}")
print(f"{ok}/8 responses parsed")
EOF

grep -c "ROCm error\|SIGSEGV\|Fatal" "${LOG_DIR}/server-tp4.log" 2>/dev/null | xargs -I{} echo "server log errors: {}"

exit ${fail}
