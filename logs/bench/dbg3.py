#!/usr/bin/env python3
# 3-seq concurrent divergence probe with debug server (LLAMA_RS_DEBUG=1).
# Phase 1: sequential reference (fresh server). Phase 2: N concurrent bursts.
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8080"
PROMPTS = [
    "The capital of France is",
    "Summarize the water cycle: evaporation,",
    "The primary colors are red,",
    "Explain gravity in one sentence:",
    "Write a Python lambda to square a number:",
    "The largest planet is",
]

def stream_ids(prompt, n_predict=48, timeout=300):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.0, "top_k": 1, "stream": True}).encode()
    req = urllib.request.Request(URL + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    ids = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ids.extend(obj.get("tokens") or [])
            if obj.get("stop"):
                break
    return ids

def run(prompts, label):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(stream_ids, p): i for i, p in enumerate(prompts)}
        for f in futs:
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:
                out[i] = {"error": str(e)}
    return out

def run_seq(prompts):
    out = {}
    for i, p in enumerate(prompts):
        out[i] = stream_ids(p)
        print(f"  seq prompt {i}: {len(out[i])} tokens", flush=True)
    return out

def main():
    n3 = PROMPTS[:3]
    ref = run_seq(n3)
    print("SEQ REF:", {k: len(v) for k, v in ref.items() if isinstance(v, list)}, flush=True)
    with open("/tmp/dbg3_ref.json", "w") as f:
        json.dump(ref, f)

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for rnd in range(rounds):
        time.sleep(3)
        out = run(n3, f"conc{rnd}")
        ndiv = 0
        for k in ref:
            r, t = ref[k], out[k]
            if isinstance(r, dict) or isinstance(t, dict):
                continue
            first_bad = next((j for j, (a, b) in enumerate(zip(r, t)) if a != b), None)
            if first_bad is not None:
                ndiv += 1
                print(f"  ROUND {rnd} prompt {k}: DIVERGES at token {first_bad} (ref {len(r)} vs got {len(t)})", flush=True)
        if ndiv == 0:
            print(f"  ROUND {rnd}: all 3 match (tokens {[len(out[k]) for k in out]})", flush=True)
        with open(f"/tmp/dbg3_conc_{rnd}.json", "w") as f:
            json.dump(out, f)

main()
