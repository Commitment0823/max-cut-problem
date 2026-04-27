"""
platform_pennylane.py  —  PennyLane worker for MaxSAT
All params injected via --config JSON + --p int.
Prints one JSON line to stdout on completion.
"""
import os, sys, json, time, argparse

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--p", required=True, type=int)
args = parser.parse_args()
if os.path.isfile(args.config):
    with open(args.config, "r") as f:
        cfg = json.load(f)
else:
    cfg = json.loads(args.config)

N = int(cfg["n"])
P = int(args.p)
SHOTS = int(cfg.get("shots", 1024))
SEED = int(cfg.get("seed", 42))
MAXITER = int(cfg.get("maxiter", 10000))
RHOBEG = float(cfg.get("rhobeg", 0.5))
RHOEND = float(cfg.get("rhoend", 1e-4))
CLAUSES = [[int(lit) for lit in clause] for clause in (cfg.get("clauses") or [])]
WARMUP = bool(cfg.get("warmup", True))
setup_t0 = time.time()

if not CLAUSES:
    print(json.dumps({"status": "ERROR", "error": "Missing clauses in config", "platform": "PennyLane", "p": P, "n": N}))
    sys.exit(1)

BASE = float(cfg.get("baseline_value", cfg.get("optimal_satisfied", cfg.get("sampled_satisfied", 1))))
BKEY = cfg.get("baseline_key", "baseline_value")
if BASE <= 0:
    BASE = 1.0

np.random.seed(SEED)

try:
    dev_test = qml.device("lightning.gpu", wires=2, shots=2)
    dev_test.execute([qml.tape.QuantumScript([qml.Hadamard(0)], [qml.sample(wires=[0, 1])], shots=2)])
    BACKEND = "lightning.gpu"
except Exception as e:
    print(json.dumps({"status": "ERROR", "error": f"lightning.gpu unavailable: {e}", "platform": "PennyLane", "p": P, "n": N}))
    sys.exit(1)


def satisfied_clause(bits: list[int], clause: list[int]) -> bool:
    for lit in clause:
        idx = abs(lit) - 1
        if idx < 0 or idx >= len(bits):
            continue
        val = bits[idx]
        if (lit > 0 and val == 1) or (lit < 0 and val == 0):
            return True
    return False


def satisfied_count(bits: list[int]) -> int:
    return sum(1 for clause in CLAUSES if satisfied_clause(bits, clause))


dev = qml.device(BACKEND, wires=N)

@qml.qnode(dev)
def circuit(params):
    for i in range(N):
        qml.Hadamard(wires=i)
    for l in range(P):
        g = params[l]
        b = params[l + P]
        for i in range(N):
            qml.RY(2 * g, wires=i)
            qml.RZ(2 * b, wires=i)
        for i in range(N - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(g, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
    return qml.sample(wires=range(N))

circuit_with_shots = qml.set_shots(circuit, shots=SHOTS)


def cost_fn(params: np.ndarray) -> float:
    np.random.seed(SEED)
    samples = circuit_with_shots(params)
    return -np.mean([satisfied_count([int(x) for x in sample]) for sample in samples])


init = np.random.uniform(0, np.pi, 2 * P)
warmup_shots = SHOTS if WARMUP else 0
warmup_sec = 0.0
if WARMUP:
    tw = time.time()
    np.random.seed(SEED)
    _ = circuit_with_shots(init)
    warmup_sec = time.time() - tw

compile_sec = time.time() - setup_t0
opt_t0 = time.time()
result = minimize(cost_fn, init, method="COBYLA", options={"maxiter": MAXITER, "rhobeg": RHOBEG, "catol": RHOEND})

np.random.seed(SEED)
final = circuit_with_shots(result.x)
counts = {}
for sample in final:
    bs = "".join(map(str, [int(x) for x in sample]))
    counts[bs] = counts.get(bs, 0) + 1

total = sum(counts.values())
mean_sat = sum(satisfied_count([int(b) for b in bs]) * cnt for bs, cnt in counts.items()) / total
best_sat = max(satisfied_count([int(b) for b in bs]) for bs in counts)
optimize_sec = time.time() - opt_t0
runtime = compile_sec + optimize_sec
nfev = int(result.nfev)
optimization_shots = int(SHOTS * nfev)
effective_total_shots = int(optimization_shots + total)
runtime_per_1k_shots = round(runtime / (effective_total_shots / 1000.0), 6) if effective_total_shots > 0 else None
effective_total_shots_with_warmup = int(effective_total_shots + warmup_shots)
runtime_per_1k_shots_with_warmup = round(runtime / (effective_total_shots_with_warmup / 1000.0), 6) if effective_total_shots_with_warmup > 0 else None

print(json.dumps({
    "platform": "PennyLane",
    "backend": BACKEND,
    "status": "OK",
    "problem_type": "maxsat",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "p": P,
    "n": N,
    "mean_satisfied": round(mean_sat, 4),
    "best_satisfied": int(best_sat),
    BKEY: BASE,
    "approximation_ratio": round(mean_sat / BASE, 4),
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
    "cobyla_nfev": nfev,
    "cobyla_success": bool(result.success),
    "optimal_params": result.x.tolist(),
}))
sys.stdout.flush()
