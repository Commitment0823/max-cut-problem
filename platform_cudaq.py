"""
platform_cudaq.py  —  CUDA-Q worker (optimized)
All params injected via --config JSON + --p int
Prints one JSON line to stdout on completion.
"""
import os, sys, json, time, itertools, argparse
import numpy as np
import networkx as nx
import cudaq
from cudaq import spin
from typing import List
from scipy.optimize import minimize
import cvxpy as cp

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--p",      required=True, type=int)
args = parser.parse_args()
cfg  = json.loads(args.config)

N       = cfg["n"]
D       = cfg["d"]
P       = args.p
SHOTS   = cfg["shots"]
SEED    = cfg["seed"]
GWR     = cfg.get("gw_rounds", 300)
MAXITER = cfg.get("maxiter", 10000)
RHOBEG  = cfg.get("rhobeg", 0.5)
RHOEND  = cfg.get("rhoend", 1e-4)

# ── target ────────────────────────────────────────────────────────────────────
try:
    cudaq.set_target("nvidia")
    BACKEND = "nvidia (GPU)"
except Exception:
    cudaq.set_target("qpp-cpu")
    BACKEND = "qpp-cpu"

cudaq.set_random_seed(SEED)
np.random.seed(SEED)

# ── graph ─────────────────────────────────────────────────────────────────────
G     = nx.random_regular_graph(d=D, n=N, seed=SEED)
EDGES = list(G.edges())
EU    = [int(u) for u, v in EDGES]
EV    = [int(v) for u, v in EDGES]
NE    = len(EDGES)

def cut(bits) -> int:
    return sum(1 for u, v in EDGES if bits[u] != bits[v])

# ── baseline ──────────────────────────────────────────────────────────────────
def brute_force():
    return max(cut(list(b)) for b in itertools.product([0,1], repeat=N))

def gw():
    X = cp.Variable((N, N), symmetric=True)
    cons = [X >> 0] + [X[i,i] == 1 for i in range(N)]
    cp.Problem(cp.Maximize(0.5 * sum(1 - X[u,v] for u,v in G.edges())), cons).solve(solver=cp.SCS, verbose=False)
    mat = X.value; mat = (mat + mat.T) / 2
    e   = np.linalg.eigvalsh(mat).min()
    if e < 0: mat += (-e + 1e-8) * np.eye(N)
    L = np.linalg.cholesky(mat)
    np.random.seed(SEED)
    best = 0
    for _ in range(GWR):
        bits = (L @ np.random.randn(N) >= 0).astype(int)
        best = max(best, cut(bits))
    return best

if N <= 20:
    BASE = brute_force(); BKEY = "optimal_cut"
else:
    BASE = gw();          BKEY = "gw_cut"

# ── hamiltonian ───────────────────────────────────────────────────────────────
H = sum(0.5 * (spin.i(u) * spin.i(v) - spin.z(u) * spin.z(v)) for u, v in EDGES)

# ── kernel ────────────────────────────────────────────────────────────────────
@cudaq.kernel
def qaoa(n: int, p: int, ne: int, eu: List[int], ev: List[int], th: List[float]):
    q = cudaq.qvector(n)
    h(q)
    for l in range(p):
        g = th[l]; b = th[l + p]
        for e in range(ne):
            x.ctrl(q[eu[e]], q[ev[e]])
            rz(2.0 * g, q[ev[e]])
            x.ctrl(q[eu[e]], q[ev[e]])
        for i in range(n):
            rx(2.0 * b, q[i])

# ── optimize ──────────────────────────────────────────────────────────────────
def objective(params):
    return -cudaq.observe(qaoa, H, N, P, NE, EU, EV, params.tolist()).expectation()

np.random.seed(SEED)
init   = np.random.uniform(0, np.pi, 2 * P)
t0     = time.time()
result = minimize(objective, init, method="COBYLA",
                  options={"maxiter": MAXITER, "rhobeg": RHOBEG, "catol": RHOEND})

raw    = cudaq.sample(qaoa, N, P, NE, EU, EV, result.x.tolist(), shots_count=SHOTS)
counts = {bs: raw[bs] for bs in raw}
total  = sum(counts.values())
mc     = sum(cut([int(b) for b in bs]) * cnt for bs, cnt in counts.items()) / total
bc     = max(cut([int(b) for b in bs]) for bs in counts)

print(json.dumps({
    "platform": "CUDA-Q", "backend": BACKEND,
    "p": P, "n": N, "d": D,
    "mean_cut": round(mc, 4), "best_cut": int(bc),
    BKEY: BASE, "approximation_ratio": round(mc / BASE, 4),
    "total_shots": total, "runtime_sec": round(time.time() - t0, 2),
    "cobyla_nfev": int(result.nfev), "cobyla_success": bool(result.success),
    "optimal_params": result.x.tolist(),
}))
sys.stdout.flush()