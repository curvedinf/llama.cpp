#!/usr/bin/env python3
# Align RSPROBE state-row hashes across rounds and find the first step where the
# prompt-0 row diverges between a passing and a failing round.
import json, re, sys

log = open(sys.argv[1] if len(sys.argv) > 1 else "logs/bench/dbg3g.log").read().splitlines()

launch_re = re.compile(r"launch_slot_: id\s+(\d+) \| task (\d+)")
probe_re  = re.compile(r"RSPROBE: n_t=(\d+) n_seqs=(\d+) seqs=\[([0-9,]*)\] (.*)")

# rounds: launches define boundaries; task ids map to rounds via order of first launch per round
events = []
for i, line in enumerate(log):
    m = launch_re.search(line)
    if m:
        events.append((i, "launch", int(m.group(2)), int(m.group(1))))
    m = probe_re.search(line)
    if m:
        hashes = {}
        for tok in m.group(4).split():
            tag, h = tok.split("=")
            hashes[tag] = h
        events.append((i, "probe", int(m.group(1)), int(m.group(2)), m.group(3), hashes))

# rounds: group launches that occur within 15 lines of each other (same burst)
rounds = []
cur = []
last_launch_line = -1000
for e in events:
    if e[1] == "launch":
        if cur and e[0] - last_launch_line < 15:
            cur.append(e)
        else:
            if cur:
                rounds.append(cur)
            cur = [e]
        last_launch_line = e[0]
    else:
        if cur:
            cur.append(e)
if cur:
    rounds.append(cur)

# prompt 0 = the task with the smallest id in each round; its seqid = slot id
def prompt0_row(round_probes):
    # first probe with n_seqs==3 gives the batch order and the row assignment
    for e in round_probes:
        if e[1] == "probe" and e[3] == 3:
            seqs = e[4].split(",")
            # rows assigned in batch order: row i -> seqs[i]
            # prompt 0 is the seq with the smallest slot (task order maps: tasks ascending)
            return seqs
    return None

for ri, rnd in enumerate(rounds):
    launches = [e for e in rnd if e[1] == "launch"]
    probes   = [e for e in rnd if e[1] == "probe"]
    if not launches or not probes:
        continue
    tasks = sorted(e[2] for e in launches)
    first3 = next((e for e in probes if e[3] == 3 and e[2] == 1), None)
    rowmap = None
    if first3:
        seqs = first3[4].split(",")
        rowmap = {s: i for i, s in enumerate(seqs)}
    p0_task = min(e[2] for e in launches)
    p0_slot = next(e[3] for e in launches if e[2] == p0_task)
    p0_row = rowmap.get(str(p0_slot)) if rowmap else None
    print(f"round {ri}: tasks {tasks} p0_slot={p0_slot} p0_row={p0_row} probes {len(probes)}")

# compare the prompt-0 row hash sequence of each round against round 0 (passing)
def p0_seq(rnd):
    launches = [e for e in rnd if e[1] == "launch"]
    probes   = [e for e in rnd if e[1] == "probe"]
    if not launches or not probes:
        return None
    p0_task = min(e[2] for e in launches)
    p0_slot = str(next(e[3] for e in launches if e[2] == p0_task))
    first3 = next((e for e in probes if e[3] == 3 and e[2] == 1), None)
    if not first3:
        return None
    row = first3[4].split(",").index(p0_slot)
    return [(e[2], e[4], e[5].get(f"r{row}"), e[5].get(f"s{row}")) for e in probes]

seqs = [p0_seq(rnd) for rnd in rounds]
ref = seqs[3]  # script-round 0 (passing)
for ri, s in enumerate(seqs):
    if not s or ri == 3:
        continue
    n = min(len(ref), len(s))
    div = next((k for k in range(n) if ref[k][2] != s[k][2] or ref[k][3] != s[k][3]), None)
    if div is None:
        print(f"round {ri}: prompt-0 state rows match through {n} probes")
    else:
        print(f"round {ri}: prompt-0 state rows DIVERGE at probe {div}: n_t={ref[div][0]}/{s[div][0]} r={ref[div][2]}/{s[div][2]} s={ref[div][3]}/{s[div][3]}")
