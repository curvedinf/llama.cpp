#!/usr/bin/env bash
# vram-watchdog: immediately SIGKILL any compute process that exceeds the VRAM cap,
# and kill the largest offender if total compute VRAM threatens the card.
#
# usage: ./vram-watchdog.sh [per-process cap MB]   (default 20480 = 20 GB)
# env:   TOTAL_CAP_MB (default 22528), INTERVAL (default 0.5s),
#        WATCHDOG_LOG (default bench-results/vram-watchdog.log)
#
# Run this in the background BEFORE any GPU work and keep it running for the
# whole session. It is the last line of defense; bench-c16.sh / run-guarded.sh
# remain the first (contention guard + per-run caps).
set -uo pipefail

CAP_MB=${1:-20480}
CAP_BYTES=$(( CAP_MB * 1024 * 1024 ))
TOTAL_CAP_MB=${TOTAL_CAP_MB:-22528}
TOTAL_CAP_BYTES=$(( TOTAL_CAP_MB * 1024 * 1024 ))
INTERVAL=${INTERVAL:-0.5}
LOG=${WATCHDOG_LOG:-bench-results/vram-watchdog.log}
mkdir -p "$(dirname "$LOG")"

# single instance only
exec 9> /tmp/vram-watchdog.lock
flock -n 9 || { echo "vram-watchdog: already running"; exit 0; }

echo "$(date -Is) vram-watchdog start: per-process cap ${CAP_MB} MB, total cap ${TOTAL_CAP_MB} MB, interval ${INTERVAL}s" | tee -a "$LOG"

kill_pid() {
    local pid=$1 why=$2
    # never kill ourselves or a pid that vanished
    [ "$pid" = "$$" ] && return
    local cmd
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-200)
    [ -z "$cmd" ] && return
    echo "$(date -Is) KILL pid=$pid ($why): $cmd" | tee -a "$LOG"
    kill -KILL "$pid" 2>/dev/null || true
}

while true; do
    PIDS=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/ {print $1, $4}')
    # per-process cap
    while read -r pid used; do
        [ -z "${pid:-}" ] && continue
        if [ -n "${used:-}" ] && [ "$used" -gt "$CAP_BYTES" ]; then
            kill_pid "$pid" "process VRAM ${used} B > ${CAP_BYTES} B"
        fi
    done <<< "$PIDS"
    # total cap: many small processes can OOM the card just as well
    TOTAL=$(awk '{s += $2} END {print s+0}' <<< "$PIDS")
    if [ "${TOTAL:-0}" -gt "$TOTAL_CAP_BYTES" ]; then
        BIG=$(sort -k2 -rn <<< "$PIDS" | head -1)
        kill_pid "$(awk '{print $1}' <<< "$BIG")" "total VRAM ${TOTAL} B > ${TOTAL_CAP_BYTES} B (largest: $(awk '{print $2}' <<< "$BIG") B)"
    fi
    sleep "$INTERVAL"
done
