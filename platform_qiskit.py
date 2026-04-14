"""
platform_qiskit.py  —  Qiskit worker (GPU via qiskit-aer-gpu)
All params injected via --config JSON + --p int
Prints one JSON line to stdout on completion.
Debug output goes to stderr (visible in terminal, not captured by analyzer).
"""
import os, sys, json, time, argparse
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ParameterVector

def dbg(msg):
    sys.stderr.write(f"[qiskit debug] {msg}\n")
    sys.stderr.flush()

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
QISKIT_PREFLIGHT = bool(cfg.get("qiskit_preflight", False))

dbg(f"config loaded: n={N} d={D} p={P} shots={SHOTS} seed={SEED}")
np.random.seed(SEED)

# ── GPU backend ───────────────────────────────────────────────────────────────
dbg("importing qiskit_aer...")
try:
    from qiskit_aer import AerSimulator
    dbg("AerSimulator imported OK")
    simulator = AerSimulator(method="statevector", device="GPU")
    dbg(f"AerSimulator created, available_devices={simulator.available_devices()}")
    available = simulator.available_devices()
    if "GPU" not in available:
        raise RuntimeError(f"GPU not in available devices: {available}")
    BACKEND = "AerSimulator (GPU)"
    dbg("GPU confirmed")
except Exception as e:
    dbg(f"FATAL: {e}")
    import traceback; traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# ── graph ─────────────────────────────────────────────────────────────────────
dbg("building graph...")
if "graph_edges" in cfg:
    EDGES = [(int(u), int(v)) for u, v in cfg["graph_edges"]]
else:
    # Backward-compatible fallback for older analyzer payloads.
    G = nx.random_regular_graph(d=D, n=N, seed=SEED)
    EDGES = list(G.edges())
dbg(f"graph: {N} nodes, {len(EDGES)} edges")

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
dbg(f"baseline: {BKEY}={BASE}")

# ── circuit ───────────────────────────────────────────────────────────────────
dbg("building circuit...")
params = ParameterVector("θ", 2 * P)
qc = QuantumCircuit(N)
qc.h(range(N))
for l in range(P):
    g = params[l]; b = params[l + P]
    for u, v in EDGES:
        qc.cx(u, v)
        qc.rz(2 * g, v)
        qc.cx(u, v)
    for i in range(N):
        qc.rx(2 * b, i)
qc.measure_all()
dbg(f"circuit built: depth={qc.depth()} gates={qc.size()}")

dbg("transpiling...")
qc_t = transpile(qc, simulator)
dbg(f"transpiled: depth={qc_t.depth()} gates={qc_t.size()}")

# ── run helper ────────────────────────────────────────────────────────────────
def run_circuit(x):
    bound = qc_t.assign_parameters(dict(zip(params, x)))
    job   = simulator.run(bound, shots=SHOTS, seed_simulator=SEED)
    result = job.result()
    return result.get_counts()

if QISKIT_PREFLIGHT:
    # Optional preflight can help debugging but changes warm-start behavior.
    dbg("test run with init params...")
    try:
        np.random.seed(SEED)
        test_params = np.random.uniform(0, np.pi, 2 * P)
        test_counts = run_circuit(test_params)
        sample_bs   = next(iter(test_counts))
        dbg(f"test OK: got {len(test_counts)} unique bitstrings, sample='{sample_bs}' len={len(sample_bs.replace(' ',''))}")
    except Exception as e:
        dbg(f"FATAL test run: {e}")
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)

# ── optimize ──────────────────────────────────────────────────────────────────
def cost_fn(x):
    raw = run_circuit(x)
    return -sum(cut([int(b) for b in bs.replace(" ", "")[::-1]]) * cnt
                for bs, cnt in raw.items()) / SHOTS

dbg("starting COBYLA optimization...")
np.random.seed(SEED)
init   = np.random.uniform(0, np.pi, 2 * P)
t0     = time.time()
result = minimize(cost_fn, init, method="COBYLA",
                  options={"maxiter": MAXITER, "rhobeg": RHOBEG, "catol": RHOEND})
dbg(f"optimization done: nfev={result.nfev} success={result.success} fun={result.fun:.4f}")

raw    = run_circuit(result.x)
counts = {}
for bs, cnt in raw.items():
    bs = bs.replace(" ", "").zfill(N)
    counts[bs] = counts.get(bs, 0) + cnt

total = sum(counts.values())
mc    = sum(cut([int(b) for b in bs[::-1]]) * cnt for bs, cnt in counts.items()) / total
bc    = max(cut([int(b) for b in bs[::-1]]) for bs in counts)
dbg(f"results: mc={mc:.4f} bc={bc} ar={mc/BASE:.4f}")

print(json.dumps({
    "platform": "Qiskit", "backend": BACKEND,
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
dbg("done")