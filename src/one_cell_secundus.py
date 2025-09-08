# === SECUNDUS: discover → build → validate → submit (fixed row seeding) ===
import json, requests
from collections import defaultdict, deque

BASE    = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
TEAM_ID = "tarstars@gmail.com YJRqfaVXC0Olk5m8c_WV5g"  # full ID string

ALPHA = "012345"
E     = [""] + list(ALPHA) + [a+b for a in ALPHA for b in ALPHA]  # Σ^2 rows (plenty for probatio)
CHUNK = 200

def api(path, payload, timeout=60):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def explore(plans):
    todo = [p for p in plans if p not in results]
    if not todo: return 0
    fetched = 0
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i+CHUNK]
        data  = api("/explore", {"id": TEAM_ID, "plans": batch}, timeout=90)
        for p, seq in zip(batch, data["results"]):
            results[p] = [v & 3 for v in seq]  # mask to 2 bits (harmless)
        fetched += len(batch)
        print(f"  fetched {len(batch)} (queryCount={data.get('queryCount')}) | total={len(results)}")
    return fetched

def ensure_rows_for(prefixes):
    need = []
    for p in prefixes:
        for s in E:
            q = p + s
            if q not in results:
                need.append(q)
    if need:
        explore(need)

def has_row(p):  # after ensure_rows_for, this should be true
    return all(p + s in results for s in E)

def row_of(prefix):
    return tuple(results[p][len(p)] for p in (prefix + s for s in E))

print("Selecting secundus")
api("/select", {"id": TEAM_ID, "problemName": "secundus"}, timeout=30)

# --- Seed & guarantee base row exists ---
results = {}
explore([""])
ensure_rows_for([""])           # make "" + E exist
assert has_row(""), "Failed to seed base row"

# --- Discover states (probatio has 3) ---
rows = {}
rows[row_of("")] = ""
reps = [""]

while True:
    succs = [rep + a for rep in reps for a in ALPHA]
    explore(succs)
    ensure_rows_for(succs)
    new_found = False
    for rep in list(reps):
        for a in ALPHA:
            sp = rep + a
            if has_row(sp):
                r = row_of(sp)
                if r not in rows:
                    rows[r] = sp
                    reps.append(sp)
                    new_found = True
    if len(rows) >= 3 and not new_found:
        break

# Representatives (shortest per row)
reps_sorted = [rows[r] for r in rows]
reps_sorted.sort(key=len)
idx = {row_of(rep): i for i, rep in enumerate(reps_sorted)}
n = len(reps_sorted)
print("States discovered:", n, "Reps:", reps_sorted)

# Ensure all rep+a rows exist then build delta
needs = [rep + a for rep in reps_sorted for a in ALPHA]
ensure_rows_for(needs)

delta = [[None]*6 for _ in range(n)]
for i, rep in enumerate(reps_sorted):
    for a in range(6):
        sp = rep + str(a)
        assert has_row(sp), f"missing row for {sp}"
        delta[i][a] = idx[row_of(sp)]

filled = sum(v is not None for row in delta for v in row)
print(f"Filled transitions: {filled}/{6*n}")

# --- Pair ports with a one-to-one matching (self-loops allowed) ---
ports = [(i,a) for i in range(n) for a in range(6)]
right_index = {p:k for k,p in enumerate(ports)}
adj = [[] for _ in ports]  # left (u) -> right (v) where mutual

for u,(i,a) in enumerate(ports):
    j = delta[i][a]
    backs = [b for b in range(6) if delta[j][b] == i]
    for b in backs:
        adj[u].append(right_index[(j,b)])

# Hopcroft–Karp on 36×36
from collections import deque
N = len(ports)
pairU = [-1]*N; pairV = [-1]*N; dist = [0]*N; INF = 10**9

def bfs():
    q = deque()
    for u in range(N):
        if pairU[u] == -1:
            dist[u] = 0; q.append(u)
        else:
            dist[u] = INF
    found = False
    while q:
        u = q.popleft()
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1:
                found = True
            elif dist[pu] == INF:
                dist[pu] = dist[u] + 1; q.append(pu)
    return found

def dfs(u):
    for v in adj[u]:
        pu = pairV[v]
        if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
            pairU[u] = v; pairV[v] = u
            return True
    dist[u] = INF
    return False

matching = 0
while bfs():
    for u in range(N):
        if pairU[u] == -1 and dfs(u):
            matching += 1
print(f"Port matching size: {matching}/{N}")
assert matching == N, "Could not find a full pairing of ports."

# Emit each undirected edge once (loops allowed)
edges=set(); connections=[]
for u in range(N):
    v = pairU[u]
    i,a = ports[u]
    j,b = ports[v]
    # sanity: mutual
    assert delta[i][a] == j and delta[j][b] == i
    key = tuple(sorted([(i,a),(j,b)]))
    if key not in edges:
        edges.add(key)
        connections.append({"from":{"room":i,"door":a},"to":{"room":j,"door":b}})

print("Connections:", len(connections))

# Build rooms + starting
rooms = [row_of(rep)[0] for rep in reps_sorted]
start_idx = idx[row_of("")]
final_map = {"rooms": rooms, "startingRoom": start_idx, "connections": connections}

print("\n=== MAP (probatio) ===")
print(json.dumps(final_map, indent=2))

# --- Validate (cover all 6*n ports; allow self-loops) ---
def validate_map_general(m):
    used=set()
    for e in m["connections"]:
        a=(e["from"]["room"],e["from"]["door"])
        b=(e["to"]["room"],e["to"]["door"])
        assert 0<=a[1]<6 and 0<=b[1]<6
        assert a not in used, f"Port reused: {a}"
        assert b not in used, f"Port reused: {b}"
        used.add(a); used.add(b)
    assert len(used)==6*n, f"Covered {len(used)} ports, expected {6*n}"
    per=defaultdict(set)
    for (r,d) in used: per[r].add(d)
    for r in range(n):
        assert per[r]==set(range(6)), f"Room {r} missing doors: {set(range(6)) - per[r]}"
    print(f"Validation OK ✅ (rooms={n}, connections={len(m['connections'])})")

validate_map_general(final_map)

# --- Submit ---
resp = requests.post(f"{BASE}/guess", json={"id": TEAM_ID, "map": final_map}, timeout=60)
print("Server response:", resp.status_code, resp.text)
