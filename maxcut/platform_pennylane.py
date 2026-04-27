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
if os.path.isfile(args.config):
    with open(args.config, "r") as f:
        cfg = json.load(f)
else:
    cfg = json.loads(args.config)

N       = cfg["n"]
D       = cfg["d"]
P       = args.p
SHOTS   = cfg["shots"]
SEED    = cfg["seed"]
MAXITER = cfg.get("maxiter", 10000)
RHOBEG  = cfg.get("rhobeg", 0.5)
RHOEND  = cfg.get("rhoend", 1e-4)
WARMUP = bool(cfg.get("warmup", True))
setup_t0 = time.time()

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
if "graph_edges" in cfg:
    EDGES = [(int(u), int(v)) for u, v in cfg["graph_edges"]]
else:
    # Backward-compatible fallback for older analyzer payloads.
    G = nx.random_regular_graph(d=D, n=N, seed=SEED)
    EDGES = list(G.edges())

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
warmup_shots = SHOTS if WARMUP else 0
warmup_sec = 0.0
if WARMUP:
    tw = time.time()
    np.random.seed(SEED)
    _ = circuit_with_shots(init)
    warmup_sec = time.time() - tw

compile_sec = time.time() - setup_t0
opt_t0  = time.time()
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
optimize_sec = time.time() - opt_t0
runtime = compile_sec + optimize_sec
nfev = int(result.nfev)
optimization_shots = int(SHOTS * nfev)
effective_total_shots = int(optimization_shots + total)
runtime_per_1k_shots = round(runtime / (effective_total_shots / 1000.0), 6) if effective_total_shots > 0 else None
effective_total_shots_with_warmup = int(effective_total_shots + warmup_shots)
runtime_per_1k_shots_with_warmup = round(runtime / (effective_total_shots_with_warmup / 1000.0), 6) if effective_total_shots_with_warmup > 0 else None

print(json.dumps({
    "platform": "PennyLane", "backend": BACKEND,
    "status": "OK",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "p": P, "n": N, "d": D,
    "mean_cut": round(mc, 4), "best_cut": int(bc),
    BKEY: BASE, "approximation_ratio": round(mc / BASE, 4),
    "total_shots": total,
    "final_sampling_shots": total,
    "shots_per_eval": SHOTS,
    "objective_shots_per_eval": SHOTS,
    "objective_evals": nfev,
    "optimizer_objective_evals": nfev,
    "optimization_shots": optimization_shots,
    "optimizer_total_shots": optimization_shots,
    "effective_total_shots": effective_total_shots,
    "experiment_total_shots": effective_total_shots,
    "warmup_enabled": WARMUP,
    "warmup_shots": warmup_shots,
    "effective_total_shots_with_warmup": effective_total_shots_with_warmup,
    "compile_sec": round(compile_sec, 4),
    "warmup_sec": round(warmup_sec, 4),
    "optimize_sec": round(optimize_sec, 4),
    "runtime_sec": round(runtime, 2),
    "runtime_per_1k_shots": runtime_per_1k_shots,
    "runtime_per_1k_shots_with_warmup": runtime_per_1k_shots_with_warmup,
    "cobyla_nfev": nfev, "cobyla_success": bool(result.success),
    "optimal_params": result.x.tolist(),
}))
sys.stdout.flush()