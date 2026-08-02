#!/usr/bin/env python3
# Compare concurrent greedy outputs vs sequential greedy outputs on the same server.
# Distinguishes a multi-sequence bug from a TP bug: if concurrent diverges from
# sequential on a SINGLE-GPU server too, the bug is in multi-seq recurrent state.
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
PROMPTS = [
    "The capital of France is",
    "Summarize the water cycle: evaporation,",
    "The primary colors are red,",
    "Explain gravity in one sentence:",
    "Write a Python lambda to square a number:",
    "The largest planet is",
    "Photosynthesis converts sunlight into",
    "The Great Wall of China was built",
    "Define recursion in programming:",
    "Water boils at sea level at",
    "The speed of light is approximately",
    "List three fruits: apple,",
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

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "seq"
    out = {}
    if mode == "seq":
        for i, p in enumerate(PROMPTS):
            try:
                out[i] = stream_ids(p)
                print(f"  seq prompt {i}: {len(out[i])} tokens", flush=True)
            except Exception as e:
                out[i] = {"error": str(e)}
                print(f"  seq prompt {i}: ERROR {e}", flush=True)
    elif mode == "conc":
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(stream_ids, p): i for i, p in enumerate(PROMPTS)}
            for f in futs:
                i = futs[f]
                try:
                    out[i] = f.result()
                    print(f"  conc prompt {i}: {len(out[i])} tokens", flush=True)
                except Exception as e:
                    out[i] = {"error": str(e)}
                    print(f"  conc prompt {i}: ERROR {e}", flush=True)
    with open(f"/tmp/{mode}_out.json", "w") as f:
        json.dump(out, f)

    # compare if both files exist
    ref_path, test_path = "/tmp/seq_out.json", "/tmp/conc_out.json"
    if os.path.exists(ref_path) and os.path.exists(test_path):
        ref = json.load(open(ref_path))
        test = json.load(open(test_path))
        nmatch = nmis = 0
        for k in sorted(ref, key=int):
            r = ref[k].get("ids", []) if isinstance(ref[k], dict) else ref[k]
            t = test[k].get("ids", []) if isinstance(test[k], dict) else test[k]
            if r == t:
                nmatch += 1
                print(f"  {k}: MATCH ({len(r)} tok)")
            else:
                nmis += 1
                div = next((pos for pos in range(min(len(r), len(t))) if r[pos] != t[pos]), "len")
                print(f"  {k}: DIVERGE at {div} (seq {len(r)} vs conc {len(t)})")
        print(f"== matched {nmatch}/{nmatch+nmis}, corruption {100*nmis/(nmatch+nmis):.0f}% ==")

if __name__ == "__main__":
    main()
