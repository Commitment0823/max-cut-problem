"""
platform_cudaq.py  —  CUDA-Q worker (optimized)
All params injected via --config JSON + --p int
Prints one JSON line to stdout on completion.
"""
import os, sys, json, time, argparse
import numpy as np
import networkx as nx
import cudaq
from typing import List
from scipy.optimize import minimize

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
MAXITER = cfg.get("maxiter", 10000)
RHOBEG  = cfg.get("rhobeg", 0.5)
RHOEND  = cfg.get("rhoend", 1e-4)
REQUIRE_GPU = bool(cfg.get("require_gpu", True))

# ── target ────────────────────────────────────────────────────────────────────
try:
    cudaq.set_target("nvidia")
    BACKEND = "nvidia (GPU)"
except Exception as e:
    if REQUIRE_GPU:
        print(json.dumps({
            "status": "ERROR",
            "error": f"CUDA-Q GPU backend unavailable: {e}",
            "platform": "CUDA-Q",
            "p": P,
            "n": N,
            "d": D,
        }))
        sys.exit(1)
    cudaq.set_target("qpp-cpu")
    BACKEND = "qpp-cpu"

cudaq.set_random_seed(SEED)
np.random.seed(SEED)

# ── graph ─────────────────────────────────────────────────────────────────────
if "graph_edges" in cfg:
    EDGES = [(int(u), int(v)) for u, v in cfg["graph_edges"]]
else:
    # Backward-compatible fallback for older analyzer payloads.
    G = nx.random_regular_graph(d=D, n=N, seed=SEED)
    EDGES = list(G.edges())
EU    = [int(u) for u, v in EDGES]
EV    = [int(v) for u, v in EDGES]
NE    = len(EDGES)

def cut(bits) -> int:
    return sum(1 for u, v in EDGES if bits[u] != bits[v])

# ── baseline (pre-computed by analyzer) ──────────────────────────────────────
if "baseline_value" in cfg:
    BASE = cfg["baseline_value"]
    BKEY = cfg.get("baseline_key", "optimal_cut")
elif "optimal_cut" in cfg:
    BASE = cfg["optimal_cut"]; BKEY = "optimal_cut"
elif "gw_cut" in cfg:
    BASE = cfg["gw_cut"];      BKEY = "gw_cut"
else:
    sys.stderr.write("ERROR: no baseline in config\n"); sys.exit(1)

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
def sample_counts(params):
    raw = cudaq.sample(qaoa, N, P, NE, EU, EV, params.tolist(), shots_count=SHOTS)
    return {bs: raw[bs] for bs in raw}


def objective(params):
    # Match PennyLane/Qiskit: optimize sampled mean-cut under finite shots.
    cudaq.set_random_seed(SEED)
    counts = sample_counts(params)
    return -sum(cut([int(b) for b in bs]) * cnt for bs, cnt in counts.items()) / SHOTS

np.random.seed(SEED)
init   = np.random.uniform(0, np.pi, 2 * P)
t0     = time.time()
result = minimize(objective, init, method="COBYLA",
                  options={"maxiter": MAXITER, "rhobeg": RHOBEG, "catol": RHOEND})

counts = sample_counts(result.x)
total  = sum(counts.values())
mc     = sum(cut([int(b) for b in bs]) * cnt for bs, cnt in counts.items()) / total
bc     = max(cut([int(b) for b in bs]) for bs in counts)

print(json.dumps({
    "platform": "CUDA-Q", "backend": BACKEND,
    "status": "OK",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "p": P, "n": N, "d": D,
    "mean_cut": round(mc, 4), "best_cut": int(bc),
    BKEY: BASE, "approximation_ratio": round(mc / BASE, 4),
    "total_shots": total, "runtime_sec": round(time.time() - t0, 2),
    "cobyla_nfev": int(result.nfev), "cobyla_success": bool(result.success),
    "optimal_params": result.x.tolist(),
}))
sys.stdout.flush()