#!/usr/bin/env bash
# Correlate input-buffer addresses (OUTIDS / IN_EMBD / IN_KQMASK / SETINP / RS_STEP)
# to find which tensors overlap the out_ids buffers (the corruption target).
# Usage: correlate_addrs.sh <logfile>
set -euo pipefail
LOG="${1:-logs/tp4/server-tp4.log}"

echo "=== all input tensor addresses (unique, near the fault) ==="
grep -E 'OUTIDS:|IN_EMBD|IN_KQMASK|SETINP:|RS_STEP' "${LOG}" | tail -40 | \
  sed -E 's/.*(buf|data|scopy)=0x([0-9a-f]+).*/0x\2/' | sort -u | head -30

echo "=== out_ids buffers ==="
grep 'OUTIDS:' "${LOG}" | grep -o 'buf=0x[0-9a-f]*' | sort -u | head -10

echo "=== other input buffers ==="
grep -E 'IN_EMBD|IN_KQMASK|SETINP:|RS_STEP' "${LOG}" | grep -oE '(buf|data|scopy)=0x[0-9a-f]+' | sort -u | head -20

echo "=== nearest CPY_ASYNC destinations (excluded if far) ==="
grep 'CPY_ASYNC' "${LOG}" | grep -o 'dst=0x[0-9a-f]*' | sort -u | head -10
