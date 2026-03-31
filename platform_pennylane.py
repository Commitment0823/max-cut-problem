"""
platform_pennylane.py  —  PennyLane worker (GPU)
Requires: pennylane-lightning-gpu
All params injected via --config JSON + --p int
Prints one JSON line to stdout on completion.
"""
import os, sys, json, time, argparse
import numpy as np
import networkx as nx
import pennylane as qml
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
GWR     = cfg.get("gw_rounds", 300)
MAXITER = cfg.get("maxiter", 10000)
RHOBEG  = cfg.get("rhobeg", 0.5)
RHOEND  = cfg.get("rhoend", 1e-4)

np.random.seed(SEED)

# ── GPU backend (hard require) ────────────────────────────────────────────────
try:
    dev_test = qml.device("lightning.gpu", wires=2, shots=2)
    dev_test.execute([qml.tape.QuantumScript(
        [qml.Hadamard(0)], [qml.sample(wires=[0, 1])], shots=2
    )])
    BACKEND = "lightning.gpu"
except Exception as e:
    print(json.dumps({"status": "ERROR", "error": f"lightning.gpu unavailable: {e}",
                      "platform": "PennyLane", "p": P, "n": N, "d": D}))
    sys.exit(1)

# ── graph ─────────────────────────────────────────────────────────────────────
G     = nx.random_regular_graph(d=D, n=N, seed=SEED)
EDGES = list(G.edges())

def cut(bits) -> int:
    return sum(1 for u, v in EDGES if bits[u] != bits[v])

# ── baseline (pre-computed by analyzer) ──────────────────────────────────────
if "optimal_cut" in cfg:
    BASE = cfg["optimal_cut"]; BKEY = "optimal_cut"
elif "gw_cut" in cfg:
    BASE = cfg["gw_cut"];      BKEY = "gw_cut"
else:
    sys.stderr.write("ERROR: no baseline in config\n"); sys.exit(1)

# ── circuit ───────────────────────────────────────────────────────────────────
dev = qml.device(BACKEND, wires=N)

@qml.qnode(dev)
def circuit(params):
    for i in range(N):
        qml.Hadamard(wires=i)
    for l in range(P):
        g = params[l]; b = params[l + P]
        for u, v in EDGES:
            qml.CNOT(wires=[u, v])
            qml.RZ(2 * g, wires=v)
            qml.CNOT(wires=[u, v])
        for i in range(N):
            qml.RX(2 * b, wires=i)
    return qml.sample(wires=range(N))

circuit_with_shots = qml.set_shots(circuit, shots=SHOTS)

# ── optimize ──────────────────────────────────────────────────────────────────
def cost_fn(params):
    np.random.seed(SEED)
    samples = circuit_with_shots(params)
    return -np.mean([cut(s) for s in samples])

np.random.seed(SEED)
init   = np.random.uniform(0, np.pi, 2 * P)
t0     = time.time()
result = minimize(cost_fn, init, method="COBYLA",
                  options={"maxiter": MAXITER, "rhobeg": RHOBEG, "catol": RHOEND})

np.random.seed(SEED)
final  = circuit_with_shots(result.x)
counts = {}
for s in final:
    bs = "".join(map(str, s))
    counts[bs] = counts.get(bs, 0) + 1

total = sum(counts.values())
mc    = sum(cut([int(b) for b in bs]) * cnt for bs, cnt in counts.items()) / total
bc    = max(cut([int(b) for b in bs]) for bs in counts)

print(json.dumps({
    "platform": "PennyLane", "backend": BACKEND,
    "p": P, "n": N, "d": D,
    "mean_cut": round(mc, 4), "best_cut": int(bc),
    BKEY: BASE, "approximation_ratio": round(mc / BASE, 4),
    "total_shots": total, "runtime_sec": round(time.time() - t0, 2),
    "cobyla_nfev": int(result.nfev), "cobyla_success": bool(result.success),
    "optimal_params": result.x.tolist(),
}))
sys.stdout.flush()