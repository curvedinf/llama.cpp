#!/usr/bin/env bash
# Correlate the OUTIDS fills with the GRV_OOB reads and the fault line.
# Usage: correlate_outids.sh <logfile>
set -euo pipefail
LOG="${1:-logs/tp4/server-tp4.log}"

ERR=$(grep -n 'hipErrorIllegalAddress' "${LOG}" | head -1 | cut -d: -f1 || true)
if [ -z "$ERR" ]; then
  ERR=$(grep -n 'Memory Fault Error' "${LOG}" | head -1 | cut -d: -f1)
fi
echo "=== fault line: ${ERR} ==="

echo "=== last OUTIDS / OUTIDS_FILLED before fault ==="
grep -n 'OUTIDS' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -12

echo "=== GRAPH_HIT / graph cache before fault ==="
grep -n 'GRAPH_HIT\|graph cache' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -8

echo "=== GRV_OOB / GRV_BIG / Q81_FIVE before fault ==="
grep -n 'GRV_OOB\|GRV_BIG\|Q81_FIVE' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -12

echo "=== last RS_STEP before fault ==="
grep -n 'RS_STEP\|RS_DIRECT' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -6

echo "=== faulting dispatch dumps ==="
grep -n -A4 'Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION' "${LOG}" | tail -22

echo "=== last kernel submissions before fault ==="
grep -n 'ShaderName' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -8 | sed -E 's/(ShaderName : void |ShaderName : )//; s/<.*//' | cut -c1-140
