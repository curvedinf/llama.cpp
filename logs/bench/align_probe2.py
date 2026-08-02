#!/usr/bin/env python3
# Full RSPROBE analysis: align rounds and find the first step + component where a
# failing round's state diverges from a passing round's, per sequence.
import re, sys

log = open(sys.argv[1] if len(sys.argv) > 1 else "logs/bench/dbg3h.log").read().splitlines()

launch_re = re.compile(r"launch_slot_: id\s+(\d+) \| task (\d+)")
probe_re  = re.compile(r"RSPROBE: n_t=(\d+) n_seqs=(\d+) seqs=\[([0-9,]*)\] (.*)")

events = []
for i, line in enumerate(log):
    m = launch_re.search(line)
    if m:
        events.append((i, "launch", int(m.group(2)), int(m.group(1))))
        continue
    m = probe_re.search(line)
    if m:
        hashes = {}
        for tok in m.group(4).split():
            tag, h = tok.split("=")
            hashes[tag] = h
        events.append((i, "probe", int(m.group(1)), int(m.group(2)), m.group(3), hashes))

# rounds: group launches by task-id proximity (a new round starts when the task
# id jumps by >= 40); the ref phase is sequential (tasks 0, 50, 100 -> own rounds)
rounds = []
cur = []
for e in events:
    if e[1] == "launch":
        if cur and abs(e[2] - cur[-1][2]) < 40:
            cur.append(e)
        else:
            if cur:
                rounds.append(cur)
            cur = [e]
    else:
        if cur:
            cur.append(e)
if cur:
    rounds.append(cur)

def round_info(rnd):
    launches = [e for e in rnd if e[1] == "launch"]
    probes   = [e for e in rnd if e[1] == "probe"]
    if not launches or not probes:
        return None
    p0_task = min(e[2] for e in launches)
    p0_slot = str(next(e[3] for e in launches if e[2] == p0_task))
    first3 = next((e for e in probes if e[3] == 3 and e[2] == 1), None)
    if not first3:
        return None
    seqs = first3[4].split(",")
    rowmap = {s: i for i, s in enumerate(seqs)}   # seq -> batch position (=row at prefill)
    # prompt 0's seq id and its row
    p0_seq = p0_slot
    p0_row = rowmap.get(p0_seq)
    return {
        "tasks": sorted(e[2] for e in launches),
        "p0_slot": int(p0_slot),
        "p0_row": p0_row,
        "probes": probes,
        "rowmap": rowmap,
        "seqs": seqs,
    }

infos = [round_info(rnd) for rnd in rounds]
for ri, info in enumerate(infos):
    if info:
        print(f"round {ri}: tasks {info['tasks']} p0_slot={info['p0_slot']} p0_row={info['p0_row']} batch={info['seqs']} probes={len(info['probes'])}")

# compare each round's prompt-0 state against the first multi-seq round (passing)
ref = infos[3]  # script round 0
if not ref or ref["p0_row"] is None:
    print("no reference round"); sys.exit(1)

for ri, info in enumerate(infos):
    if not info or ri == 3 or info["p0_row"] is None:
        continue
    # per-probe comparison: prompt 0's row hashes (r/s per layer) and its K/V hashes
    # (k/v per layer, tagged by the batch position q = p0_row)
    n = min(len(ref["probes"]), len(info["probes"]))
    div = None
    for k in range(n):
        rp, sp = ref["probes"][k], info["probes"][k]
        # recurrent rows: r<layer>_<row>, s<layer>_<row>
        for tag, hv in rp[5].items():
            if tag[0] in ("r", "s") and tag.endswith(f"_{ref['p0_row']}"):
                tag2 = tag[:-1] + str(info["p0_row"])
                if sp[5].get(tag2) != hv:
                    div = (k, tag, hv, sp[5].get(tag2))
                    break
        if div:
            break
        # K/V: k<layer>_s<q>, v<layer>_s<q>
        for tag, hv in rp[5].items():
            if tag[0] in ("k", "v") and tag.endswith(f"_s{ref['p0_row']}"):
                tag2 = tag[:-1] + str(info["p0_row"])
                if sp[5].get(tag2) != hv:
                    div = (k, tag, hv, sp[5].get(tag2))
                    break
        if div:
            break
    if div is None:
        print(f"round {ri}: prompt-0 state matches through {n} probes")
    else:
        k, tag, a, b = div
        print(f"round {ri}: prompt-0 DIVERGES at probe {k} [{tag}]: {a} vs {b}")
