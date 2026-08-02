#!/usr/bin/env python3
# Minimal C=8 burst bench: N concurrent /completion requests, aggregate tok/s.
import argparse, json, os, threading, time, urllib.request

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")

SEEDS = [
    "The history of computing begins with abacuses and mechanical calculators.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Quantum mechanics describes the behavior of matter at atomic scales.",
    "The French Revolution began in 1789 and transformed European politics.",
    "Machine learning models learn patterns from large datasets through optimization.",
    "Ocean currents distribute heat around the planet and shape regional climates.",
    "The Roman Empire stretched from Britain to Mesopotamia at its greatest extent.",
    "DNA contains the genetic instructions used in the growth and development of living organisms.",
]
TAIL = " Explain the underlying ideas in detail, step by step."

def one_request(idx, prompt, n_predict, out):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "stream": False,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": (idx + 1) * 7919,
        "ignore_eos": True,
    }).encode()
    try:
        req = urllib.request.Request(URL + "/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            obj = json.loads(r.read())
        t = obj.get("timings", {}) or {}
        out[idx] = dict(
            prompt_n=t.get("prompt_n", 0),
            predicted_n=t.get("predicted_n", 0),
            predicted_ms=t.get("predicted_ms", 0),
        )
    except Exception as e:
        out[idx] = {"error": str(e)}

def wait_ready(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-C", "--concurrency", type=int, default=8)
    ap.add_argument("-n", "--n-predict", type=int, default=48)
    ap.add_argument("-r", "--repeats", type=int, default=1)
    args = ap.parse_args()

    if not wait_ready():
        print("server not ready")
        return 1

    prompts = [(SEEDS[i % len(SEEDS)] + " ") * args.repeats + TAIL for i in range(args.concurrency)]
    res = [None] * args.concurrency
    t0 = time.time()
    ths = [threading.Thread(target=one_request, args=(i, prompts[i], args.n_predict, res)) for i in range(args.concurrency)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0

    ok = [r for r in res if r and "error" not in r]
    if len(ok) < args.concurrency:
        print(f"{len(ok)}/{args.concurrency} ok, errors: {[r.get('error') for r in res if r and 'error' in r][:2]}")
        return 1

    tot_out = sum(r["predicted_n"] for r in ok)
    agg = tot_out / wall
    dec_ms = sum(r["predicted_ms"] for r in ok) / len(ok)
    print(f"C={args.concurrency} wall={wall:.1f}s agg={agg:.1f} tok/s (out) "
          f"per-slot={agg/args.concurrency:.1f} avg_decode_ms={dec_ms:.0f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
