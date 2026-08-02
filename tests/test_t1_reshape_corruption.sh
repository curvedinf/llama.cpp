#!/usr/bin/env bash
# T1 failing test: demonstrate reshape corruption under TP4
# Runs two sequential requests and compares TP4 output vs single-GPU output
# If outputs diverge significantly, the reshape corruption persists.
set -e

BIN="/home/curved/llama.cpp-concurrency/build/bin"
MODEL="/home/curved/models/qwen3.6-27b-mtp-gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf"
export LD_LIBRARY_PATH="/opt/rocm/lib:$BIN"

PROMPT="Explain what happens when you boil water. Be concise."
N=64

echo "=== Single GPU reference ==="
HIP_VISIBLE_DEVICES=0 timeout 90 $BIN/llama-cli \
    -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 4096 \
    -p "$PROMPT" -n $N --temp 0 --no-mmap --no-warmup \
    -r "User:" </dev/null 2>/dev/null > /tmp/t1_sg.txt || true

echo "=== TP4 (4 GPUs) ==="
timeout 90 $BIN/llama-cli \
    -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 4096 \
    -sm tensor -ts 1,1,1,1 -b 4096 -ub 2048 \
    -p "$PROMPT" -n $N --temp 0 --no-mmap --no-warmup \
    -r "User:" </dev/null 2>/dev/null > /tmp/t1_tp4.txt || true

echo "=== TP4 second request (corruption manifests here) ==="
timeout 90 $BIN/llama-cli \
    -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 4096 \
    -sm tensor -ts 1,1,1,1 -b 4096 -ub 2048 \
    -p "$PROMPT" -n $N --temp 0 --no-mmap \
    -r "User:" </dev/null 2>/dev/null > /tmp/t1_tp4_2nd.txt || true

echo "=== Extracting generation output ==="
# Extract just the generated text (after the prompt line, before interactive prompts)
for f in /tmp/t1_sg.txt /tmp/t1_tp4.txt /tmp/t1_tp4_2nd.txt; do
    grep -A$N "$PROMPT" "$f" | head -$N > "${f}.gen"
    echo "--- $(basename $f) ($(wc -c < ${f}.gen) bytes):"
    head -5 "${f}.gen"
    echo "..."
done

echo "=== Diff (single vs TP4 first request) ==="
diff /tmp/t1_sg.txt.gen /tmp/t1_tp4.txt.gen > /dev/null 2>&1 && echo "IDENTICAL" || echo "DIFFERS"

echo "=== Diff (TP4 first vs TP4 second request) ==="
diff /tmp/t1_tp4.txt.gen /tmp/t1_tp4_2nd.txt.gen > /dev/null 2>&1 && echo "IDENTICAL" || echo "DIFFERS"

echo "=== Test complete ==="
