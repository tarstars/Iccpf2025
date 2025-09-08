# Colab single-cell: Chalk solver + instrumentation + dump + offline analyzer
# Assumes TEAM_ID is already defined globally as "email TOKEN".
# Change PROBLEM if needed (e.g., "aleph").

import json, time, math
from typing import List, Tuple, Dict, Any
from collections import defaultdict, deque, Counter
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===================== USER CONFIG =====================
BASE     = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
PROBLEM  = "aleph"            # chalk level
ALPHA    = "012345"
CHUNK    = 600                # /explore batch size
SIGS     = ["", "0", "3", "01", "30"]   # multi-suffix signature for pruning

# Dump + stop thresholds (tune these; None disables a condition)
DUMP_PATH = "/content/icfpc_chalk_dump.json"
STOP_AFTER_TOTAL_PLANS_CACHED = 6000     # stop when cached plans >= N
STOP_AFTER_EXPLORE_CALLS      = 6        # stop after N explore batches
STOP_AFTER_EQ_TESTS           = 2500     # stop after N equality pairs tested
VERBOSE_EVERY                 = 1        # log every K explore() wrapper calls
# =======================================================

# ---- sanity: TEAM_ID must be defined outside this cell ----
if "TEAM_ID" not in globals() or not isinstance(TEAM_ID, str) or " " not in TEAM_ID:
    raise RuntimeError("TEAM_ID must be set globally before running this cell (full 'email TOKEN').")

# ---- session with retries ----
_session = requests.Session()
_retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))

def api(path: str, payload: dict, timeout: int=120) -> dict:
    r = _session.post(f"{BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ===================== GLOBAL STATE =====================
results: Dict[str, List[int]] = {}          # plan -> 2-bit outputs
_equality_memo: Dict[Tuple[str,str], bool] = {}  # (rep,cand) -> same?

# ===================== CORE I/O =====================
def explore(plans: List[str]) -> int:
    """Basic explorer (will be wrapped by instrumentation below)."""
    todo = [p for p in plans if p not in results]
    if not todo: return 0
    fetched = 0
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i+CHUNK]
        data  = api("/explore", {"id": TEAM_ID, "plans": batch})
        for p, seq in zip(batch, data["results"]):
            results[p] = [v & 3 for v in seq]
        fetched += len(batch)
        print(f"  fetched {len(batch)} (queryCount={data.get('queryCount')}) | total cached={len(results)}")
    return fetched

def label_end(p: str) -> int:
    if p not in results: explore([p])
    return results[p][-1]

# ===================== SIGNATURES =====================
def prefetch_signatures(paths: List[str]):
    need = []
    for p in paths:
        if p not in results: need.append(p)
        for s in SIGS[1:]:
            q = p + s
            if q not in results: need.append(q)
    if need:
        explore(need)

def signature_of(p: str) -> Tuple[int, ...]:
    return tuple(results[p + s][-1] for s in SIGS)

# ===================== EQUALITY (BATCHED) =====================
def batch_test_equal(pairs: List[Tuple[str,str]]) -> List[bool]:
    """Pairs are (PX, PY): plan = PY + PX + [mv] + PY; decide same/diff by last label."""
    if not pairs: return []
    fresh = [(x,y) for (x,y) in pairs if (x,y) not in _equality_memo]
    if fresh:
        need = set()
        for PX, PY in fresh:
            if PX not in results: need.add(PX)
            if PY not in results: need.add(PY)
        if need: explore(list(need))

        plans, side = [], []  # side: (PX, PY, mv, y0)
        for PX, PY in fresh:
            xlab = results[PX][-1]
            y0   = results[PY][-1]
            mv   = (xlab + 1) % 4
            plans.append(f"{PY}{PX}[{mv}]{PY}")
            side.append((PX, PY, mv, y0))

        explore(plans)

        for plan, (PX, PY, mv, y0) in zip(plans, side):
            y1 = results[plan][-1]
            same = (y1 == mv)
            if not same and y1 != y0:
                # unexpected; treat as different safely
                same = False
            _equality_memo[(PX, PY)] = same

    return [_equality_memo[(x,y)] for (x,y) in pairs]

# ===================== SEEDING =====================
def seed_short_paths() -> List[str]:
    # "" + 1-step + 2-step (43 total)
    paths = [""]
    paths += list(ALPHA)
    paths += [a+b for a in ALPHA for b in ALPHA]
    prefetch_signatures(paths)
    return paths

# ===================== MERGE (CHALK) =====================
def merge_with_chalk(paths: List[str]) -> List[str]:
    # group by multi-suffix signature
    buckets: Dict[Tuple[int,...], List[str]] = defaultdict(list)
    for p in paths:
        buckets[signature_of(p)].append(p)

    reps: List[str] = []
    repr_of: Dict[str,str] = {}
    for sig, group in buckets.items():
        group.sort(key=len)
        rep = group[0]
        reps.append(rep)
        for g in group: repr_of[g] = rep

    # inside-bucket confirmations (batched)
    pairs, back = [], []
    for sig, group in buckets.items():
        if len(group) <= 1: continue
        group.sort(key=len)
        rep = group[0]
        for g in group[1:]:
            pairs.append((rep, g))
            back.append((rep, g))
    if pairs:
        eq = batch_test_equal(pairs)
        for (rep, g), same in zip(back, eq):
            if same:
                repr_of[g] = rep
            else:
                repr_of[g] = g
                reps.append(g)

    reps = sorted(set(repr_of.values()), key=len)
    prefetch_signatures(reps)
    return reps

def remerge_reps(reps: List[str]) -> List[str]:
    # Cheap second-pass merge by signature + chalk on bucket leaders
    prefetch_signatures(reps)
    buckets: Dict[Tuple[int,...], List[str]] = defaultdict(list)
    for r in reps:
        buckets[signature_of(r)].append(r)
    keep, pairs, back = [], [], []
    for sig, group in buckets.items():
        group.sort(key=len)
        base = group[0]
        keep.append(base)
        for g in group[1:]:
            pairs.append((base, g))
            back.append((base, g))
    if pairs:
        eq = batch_test_equal(pairs)
        for (base, g), same in zip(back, eq):
            if not same:
                keep.append(g)
    return sorted(set(keep), key=len)

# ===================== MAPPING (GLOBAL-BATCH, PRUNED) =====================
def map_transitions(reps: List[str], max_cands_per_sig: int = 6) -> Tuple[List[List[int]], List[str]]:
    reps = sorted(set(reps), key=len)
    prefetch_signatures(reps)

    sig_index: Dict[Tuple[int,...], List[str]] = defaultdict(list)
    for r in reps:
        sig_index[signature_of(r)].append(r)
    for sig in sig_index: sig_index[sig].sort(key=len)

    idx = {r:i for i,r in enumerate(reps)}
    delta = [[None]*6 for _ in reps]

    while True:
        # collect all unknown successors
        unknown: List[Tuple[int,int,str]] = []
        for i, rep in enumerate(reps):
            for a in range(6):
                if delta[i][a] is None:
                    unknown.append((i, a, rep + str(a)))
        if not unknown: break

        succ_paths = [s for _,_,s in unknown]
        prefetch_signatures(succ_paths)

        # build equality pairs with signature pruning
        pair_list: List[Tuple[str,str]] = []
        succ_block: List[Tuple[int,int,str,List[str]]] = []  # (i,a,s,cands)

        for i, a, s in unknown:
            sig = signature_of(s)
            cands = sig_index.get(sig, [])
            if not cands:
                # fallback: label-only small pool
                lab = results[s][-1]
                cands = [r for r in reps if results[r][-1] == lab]
                cands.sort(key=len)
            cands = cands[:max_cands_per_sig]
            if not cands:
                succ_block.append((i,a,s,[]))
                continue
            for c in cands:
                if (c, s) in _equality_memo:  # reuse memo
                    continue
                pair_list.append((c, s))
            succ_block.append((i,a,s,cands))

        if pair_list:
            _ = batch_test_equal(pair_list)

        # assign matches; collect new reps
        new_reps: List[str] = []
        for (i, a, s, cands) in succ_block:
            hit = None
            for c in cands:
                if _equality_memo.get((c, s), False):
                    hit = c; break
            if hit is not None:
                delta[i][a] = idx[hit]
            else:
                if s not in idx:
                    new_reps.append(s)

        if new_reps:
            # merge new reps among themselves & with existing ones before extending
            new_reps = sorted(set(new_reps), key=len)
            prefetch_signatures(new_reps)
            merged_keep = []
            for nr in new_reps:
                sig = signature_of(nr)
                bucket = sig_index.get(sig, [])
                if bucket:
                    c = bucket[0]
                    if (c, nr) not in _equality_memo:
                        batch_test_equal([(c, nr)])
                    if _equality_memo.get((c, nr), False):
                        continue
                merged_keep.append(nr)

            reps.extend(merged_keep)
            reps = remerge_reps(reps)
            idx = {r:i for i,r in enumerate(reps)}
            while len(delta) < len(reps):
                delta.append([None]*6)

            # rebuild signature index
            sig_index = defaultdict(list)
            prefetch_signatures(reps)
            for r in reps:
                sig_index[signature_of(r)].append(r)
            for sig in sig_index: sig_index[sig].sort(key=len)

            # set pending edges to new indices
            for i, a, s, _ in succ_block:
                if delta[i][a] is None:
                    delta[i][a] = idx[s]
        else:
            # no new reps; next loop will terminate when all transitions filled
            pass

    return delta, reps

# ===================== CONNECTIONS & VALIDATION =====================
def build_connections(delta: List[List[int]]) -> List[dict]:
    n = len(delta)
    ports = [(i,a) for i in range(n) for a in range(6)]
    right_index = {p:k for k,p in enumerate(ports)}
    adj = [[] for _ in ports]
    for u,(i,a) in enumerate(ports):
        j = delta[i][a]
        backs = [b for b in range(6) if delta[j][b] == i]
        for b in backs:
            adj[u].append(right_index[(j,b)])

    # Hopcroft–Karp
    N = len(ports)
    pairU = [-1]*N; pairV = [-1]*N; dist = [0]*N; INF = 10**9
    def bfs():
        q = deque()
        for u in range(N):
            if pairU[u] == -1: dist[u]=0; q.append(u)
            else:               dist[u]=INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1: found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u]+1; q.append(pu)
        return found
    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu]==dist[u]+1 and dfs(pu)):
                pairU[u]=v; pairV[v]=u; return True
        dist[u]=INF; return False
    matching=0
    while bfs():
        for u in range(N):
            if pairU[u]==-1 and dfs(u): matching+=1
    assert matching==N, "Port pairing failed"

    edges=set(); connections=[]
    for u in range(N):
        v = pairU[u]
        i,a = ports[u]; j,b = ports[v]
        assert delta[i][a]==j and delta[j][b]==i
        key = tuple(sorted([(i,a),(j,b)]))
        if key not in edges:
            edges.add(key)
            connections.append({"from":{"room":i,"door":a},"to":{"room":j,"door":b}})
    return connections

def validate_map(rooms: List[int], connections: List[dict]):
    n = len(rooms)
    used=set()
    for e in connections:
        a=(e["from"]["room"],e["from"]["door"])
        b=(e["to"]["room"],e["to"]["door"])
        assert 0<=a[1]<6 and 0<=b[1]<6
        assert a not in used and b not in used, "Door reused"
        used.add(a); used.add(b)
    assert len(used)==6*n, f"Covered {len(used)} ports, expected {6*n}"
    per=defaultdict(set)
    for (r,d) in used: per[r].add(d)
    for r in range(n):
        assert per[r]==set(range(6)), f"Room {r} missing doors: {set(range(6))-per[r]}"
    print(f"Validation OK ✅ (rooms={n}, connections={len(connections)})")

# ===================== INSTRUMENTATION / DUMP =====================
_INSTR = {
    "explore_calls": 0,
    "explore_plans_sent": 0,
    "eq_pairs_requested": 0,
    "eq_pairs_new": 0,
    "eq_pairs_from_memo": 0,
    "explore_log": [],   # list of {sent, cached_before, cached_after, ts}
}

# wrap explore
_orig_explore = explore
def _should_stop() -> bool:
    plans_cached = len(results)
    if STOP_AFTER_TOTAL_PLANS_CACHED is not None and plans_cached >= STOP_AFTER_TOTAL_PLANS_CACHED:
        print(f"[Stop] Cached plans {plans_cached} >= {STOP_AFTER_TOTAL_PLANS_CACHED}"); return True
    if STOP_AFTER_EXPLORE_CALLS is not None and _INSTR["explore_calls"] >= STOP_AFTER_EXPLORE_CALLS:
        print(f"[Stop] explore() calls { _INSTR['explore_calls'] } >= {STOP_AFTER_EXPLORE_CALLS}"); return True
    if STOP_AFTER_EQ_TESTS is not None and _INSTR["eq_pairs_requested"] >= STOP_AFTER_EQ_TESTS:
        print(f"[Stop] equality tests { _INSTR['eq_pairs_requested'] } >= {STOP_AFTER_EQ_TESTS}"); return True
    return False

def dump_state(reason: str = "manual") -> str:
    # serialize equality memo as list for portability
    eq_list = [[k[0], k[1], v] for k,v in _equality_memo.items()]
    blob: Dict[str, Any] = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "explore_calls": _INSTR["explore_calls"],
            "eq_pairs_requested": _INSTR["eq_pairs_requested"],
            "eq_pairs_new": _INSTR["eq_pairs_new"],
            "eq_pairs_from_memo": _INSTR["eq_pairs_from_memo"],
            "problem": PROBLEM,
            "chunk": CHUNK,
            "sigs": SIGS,
        },
        "results": results,
        "equality_memo_list": eq_list,  # [[PX, PY, bool], ...]
        "current_reps": globals().get("reps", []),
        "current_delta": globals().get("delta", None),
        "explore_log": _INSTR["explore_log"],
        "thresholds": {
            "STOP_AFTER_TOTAL_PLANS_CACHED": STOP_AFTER_TOTAL_PLANS_CACHED,
            "STOP_AFTER_EXPLORE_CALLS": STOP_AFTER_EXPLORE_CALLS,
            "STOP_AFTER_EQ_TESTS": STOP_AFTER_EQ_TESTS,
        },
    }
    Path(DUMP_PATH).write_text(json.dumps(blob), encoding="utf-8")
    print(f"\n[Dump] Wrote solver state to: {DUMP_PATH}")
    return DUMP_PATH

def explore(plans: List[str]) -> int:
    before = len(results)
    to_send = [p for p in plans if p not in results]
    sent = len(to_send)
    fetched = _orig_explore(plans)
    after = len(results)
    _INSTR["explore_calls"] += (math.ceil(sent / CHUNK) if sent else 0)
    _INSTR["explore_plans_sent"] += sent
    _INSTR["explore_log"].append({
        "sent": sent, "cached_before": before, "cached_after": after, "ts": time.time(),
    })
    if VERBOSE_EVERY and (len(_INSTR["explore_log"]) % VERBOSE_EVERY == 0):
        print(f"[explore#{len(_INSTR['explore_log'])}] sent={sent}, cache={before}->{after}")
    if _should_stop():
        p = dump_state(reason="threshold_explore")
        raise SystemExit(f"Stopped after dump: {p}")
    return fetched

# wrap equality
_orig_batch_test_equal = batch_test_equal
def batch_test_equal(pairs: List[Tuple[str,str]]) -> List[bool]:
    _INSTR["eq_pairs_requested"] += len(pairs)
    _INSTR["eq_pairs_from_memo"] += sum(1 for t in pairs if t in _equality_memo)
    _INSTR["eq_pairs_new"] += sum(1 for t in pairs if t not in _equality_memo)
    if _should_stop():
        p = dump_state(reason="threshold_equality")
        raise SystemExit(f"Stopped after dump: {p}")
    return _orig_batch_test_equal(pairs)

print("[Instr] Instrumentation installed.")

# ===================== OFFLINE ANALYZER =====================
def analyze_dump(path: str = DUMP_PATH):
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    res: Dict[str, List[int]] = blob.get("results", {})
    eq_list = blob.get("equality_memo_list", [])
    reps: List[str] = blob.get("current_reps") or []
    delta = blob.get("current_delta")
    explore_log = blob.get("explore_log", [])
    meta = blob.get("meta", {})

    print("\n=== Dump meta ===")
    print(json.dumps(meta, indent=2))

    # cache stats
    lens = [len(p) for p in res.keys()]
    if lens:
        lens_sorted = sorted(lens)
        p50 = lens_sorted[len(lens)//2]
        p90 = lens_sorted[(len(lens)*9)//10]
        print(f"\n[Cache] plans={len(res)}  len(min/50%/90%/max)={min(lens)}/{p50}/{p90}/{max(lens)}")
    else:
        print("\n[Cache] plans=0")

    # label distribution at end
    last_labels = Counter(seq[-1] for seq in res.values() if seq)
    print(f"[Labels@end] {dict(last_labels)}")

    # short signatures
    def safe_sig(p: str) -> Tuple[int,...] | None:
        try: return tuple(res[p+s][-1] for s in SIGS)
        except KeyError: return None
    shorts = [p for p in res if len(p) <= 2]
    sigs = Counter(safe_sig(p) for p in shorts if safe_sig(p) is not None)
    print(f"[Signatures<=2] unique={len(sigs)}  top5={sigs.most_common(5)}")

    # equality memo stats
    t_true = sum(1 for _,_,b in eq_list if b is True)
    t_false = sum(1 for _,_,b in eq_list if b is False)
    print(f"[EqMemo] total={len(eq_list)}  True={t_true}  False={t_false}")

    # explore log
    if explore_log:
        avg_sent = sum(x["sent"] for x in explore_log)/len(explore_log)
        print(f"[Explore] calls={len(explore_log)}  avg_sent≈{round(avg_sent,1)}  last_cached={explore_log[-1]['cached_after']}")
        for x in explore_log[-5:]:
            print("  last:", {k:x[k] for k in ("sent","cached_before","cached_after")})

    # reps/delta
    if reps:
        print(f"[Reps] count={len(reps)}  shortest_len={len(min(reps, key=len))}  longest_len={len(max(reps, key=len))}")
    if delta:
        n = len(delta)
        filled = sum(1 for i in range(n) for a in range(6) if delta[i][a] is not None)
        print(f"[Delta] states={n}  filled_edges={filled}/{6*n}")

    # longest plans preview
    longs = sorted(res.keys(), key=len)[-10:]
    if longs:
        print("[Longest plans]")
        for p in longs:
            seq = res[p]
            print(f"  len={len(p)} tail={seq[-5:]} plan='{p[:60]}{'…' if len(p)>60 else ''}'")

    print("\n[Analysis complete] Attach this dump when you share; I can drill into signatures, equality splits, and batch sizes from it.")

# ===================== RUN =====================
try:
    print(f"Selecting {PROBLEM}")
    api("/select", {"id": TEAM_ID, "problemName": PROBLEM})

    print("Seeding short paths (≤2) & signatures…")
    paths = seed_short_paths()
    print(f"  seeded {len(paths)} paths (should be 43)")

    print("Merging with chalk (batched)…")
    reps = merge_with_chalk(paths)
    print(f"  reps after merge: {len(reps)}")

    print("Mapping transitions (signature-pruned, memoized, global-batched)…")
    delta, reps = map_transitions(reps, max_cands_per_sig=6)
    print(f"  mapped states: {len(reps)}")

    print("Building undirected connections…")
    connections = build_connections(delta)

    print("Collecting room labels…")
    # reps labels (already cached)
    rooms = [results[r][-1] for r in reps]
    start_idx = reps.index("") if "" in reps else 0

    validate_map(rooms, connections)

    final_map = {"rooms": rooms, "startingRoom": start_idx, "connections": connections}
    print("\n=== MAP (chalk) ===")
    print(json.dumps(final_map, indent=2))

    print("\nSubmitting…")
    resp = _session.post(f"{BASE}/guess", json={"id": TEAM_ID, "map": final_map}, timeout=120)
    print("Server response:", resp.status_code, resp.text)

except SystemExit as e:
    # Threshold hit: dump already written; also run a quick analyzer summary
    print(str(e))
    try:
        analyze_dump(DUMP_PATH)
    except Exception as ex:
        print(f"Analyzer error: {ex}")
