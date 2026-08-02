#!/usr/bin/env bash
# Correlate the faulting HSA dispatch with the last RS_STEP / GRV_* traces in a
# TP4 burst log. Usage: correlate_fault.sh <logfile>
set -euo pipefail
LOG="${1:-logs/tp4/server-tp4.log}"

echo "=== fault markers ==="
grep -n 'Memory Fault\|hipErrorIllegalAddress\|MEMORY_APERTURE_VIOLATION\|QUEUE_ABORT' "${LOG}" | tail -5

ERR=$(grep -n 'hipErrorIllegalAddress' "${LOG}" | head -1 | cut -d: -f1)
if [ -z "$ERR" ]; then
  ERR=$(grep -n 'Memory Fault Error' "${LOG}" | head -1 | cut -d: -f1)
fi
echo "=== fault line: ${ERR} ==="

echo "=== last RS_STEP before fault ==="
grep -n 'RS_STEP\|RS_DIRECT' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -8

echo "=== GRV_OOB / GRV_BIG / Q81_FIVE before fault ==="
grep -n 'GRV_OOB\|GRV_BIG\|Q81_FIVE\|GRV_K' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -10

echo "=== faulting dispatch dumps ==="
grep -n -A4 'Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION' "${LOG}" | tail -20

echo "=== last kernel submissions before fault ==="
grep -n 'ShaderName' "${LOG}" | awk -F: -v e="${ERR}" '$1 < e' | tail -8 | sed -E 's/(ShaderName : void |ShaderName : )//; s/<.*//' | cut -c1-160
