"""
analyzer.py  —  QAOA Benchmark Unified Controller (optimized)
==============================================================
Edit CONFIG + EXPERIMENT below, then:
    python analyzer.py

Features:
  - All params here, zero hardcode in platform files
  - Baseline (brute-force or GW) computed ONCE here, passed to all platforms
  - Per-p timeout kill point
  - Per-p checkpoint (partial results saved immediately)
  - Outputs timestamped JSON + prints summary table
  - cobyla_nfev recorded (how many iterations COBYLA actually ran)
"""

import subprocess, json, os, sys, time, itertools
from datetime import datetime
from pathlib import Path
import numpy as np
import networkx as nx
import cvxpy as cp

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG  —  edit everything here
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "n"        : 20,       # nodes
    "d"        : 4,        # graph regularity
    "shots"    : 1024,     # measurement shots
    "seed"     : 42,
    "gw_rounds": 300,      # GW rounding iterations (used when n > 20)
    "maxiter"  : 10000,    # COBYLA max iterations (exits early on convergence)
    "rhobeg"   : 0.5,      # COBYLA initial trust region radius
    "rhoend"   : 1e-4,     # COBYLA convergence threshold (catol internally)
    "gpu_index": 0,        # force all workers to this visible GPU index
    "require_gpu": True,   # fail a platform run if GPU backend is unavailable
}

P_DEPTHS = [1, 2, 3]

TIMEOUT_PER_P = 900        # seconds per (platform, p). None = no limit.

PLATFORMS = [
    "cudaq",
    "pennylane",
    "qiskit",
]
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPTS = {
    "cudaq"    : "platform_cudaq.py",
    "pennylane": "platform_pennylane.py",
    "qiskit"   : "platform_qiskit.py",
}
LABELS = {
    "cudaq": "CUDA-Q", "pennylane": "PennyLane", "qiskit": "Qiskit"
}

DIR       = Path(__file__).parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def build_dataset_edges(n: int, d: int, seed: int) -> list[list[int]]:
    """Build a canonical graph instance once in analyzer."""
    G = nx.random_regular_graph(d=d, n=n, seed=seed)
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in G.edges())
    return [[u, v] for u, v in edges]

# ── baseline (computed once) ──────────────────────────────────────────────────
def compute_baseline(n: int, edges: list[list[int]], seed: int, gw_rounds: int):
    edge_tuples = [(int(u), int(v)) for u, v in edges]

    def cut(bits):
        return sum(1 for u, v in edge_tuples if bits[u] != bits[v])

    if n <= 20:
        print(f"  [baseline] brute-force (n={n}, 2^{n}={2**n:,} combinations)...")
        t0  = time.time()
        val = max(cut(list(b)) for b in itertools.product([0, 1], repeat=n))
        print(f"  [baseline] optimal_cut = {val}  ({time.time()-t0:.1f}s)")
        return val, "optimal_cut"
    else:
        print(f"  [baseline] Goemans-Williamson SDP (n={n}, rounds={gw_rounds})...")
        t0 = time.time()
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edge_tuples)
        X  = cp.Variable((n, n), symmetric=True)
        cons = [X >> 0] + [X[i, i] == 1 for i in range(n)]
        cp.Problem(cp.Maximize(0.5 * sum(1 - X[u, v] for u, v in edge_tuples)),
                   cons).solve(solver=cp.SCS, verbose=False)
        mat = X.value; mat = (mat + mat.T) / 2
        e   = np.linalg.eigvalsh(mat).min()
        if e < 0: mat += (-e + 1e-8) * np.eye(n)
        L   = np.linalg.cholesky(mat)
        np.random.seed(seed)
        best = 0
        for _ in range(gw_rounds):
            bits = (L @ np.random.randn(n) >= 0).astype(int)
            best = max(best, cut(bits))
        print(f"  [baseline] gw_cut = {best}  ({time.time()-t0:.1f}s)")
        return best, "gw_cut"


def parse_worker_json(stdout_text: str) -> dict:
    text = (stdout_text or "").strip()
    if not text:
        raise ValueError("empty worker stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise
        return json.loads(lines[-1])


def get_gpu_info(gpu_index: int) -> dict:
    """Best-effort query for one GPU row via nvidia-smi."""
    query = "index,name,uuid,pci.bus_id,memory.total"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]
    try:
        rows = subprocess.check_output(cmd, text=True).strip().splitlines()
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            if idx == gpu_index:
                return {
                    "gpu_index": idx,
                    "gpu_name": parts[1],
                    "gpu_uuid": parts[2],
                    "gpu_pci_bus_id": parts[3],
                    "gpu_memory_total": parts[4],
                }
    except Exception:
        pass
    return {
        "gpu_index": gpu_index,
        "gpu_name": "unknown",
        "gpu_uuid": "unknown",
        "gpu_pci_bus_id": "unknown",
        "gpu_memory_total": "unknown",
    }

# ── runner ────────────────────────────────────────────────────────────────────
def run_p(platform: str, p: int, cfg_str: str) -> dict:
    cmd = [sys.executable, str(DIR / SCRIPTS[platform]),
           "--config", cfg_str, "--p", str(p)]

    print(f"    p={p} ...", end="", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(CONFIG.get("gpu_index", 0))

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env)
        try:
            out, err = proc.communicate(timeout=TIMEOUT_PER_P)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate()
            rt = round(time.time() - t0, 1)
            print(f"  TIMEOUT ({rt}s > {TIMEOUT_PER_P}s)")
            return {"platform": LABELS[platform], "p": p, "n": CONFIG["n"],
                    "d": CONFIG["d"], "status": "TIMEOUT",
                    "timeout_sec": TIMEOUT_PER_P, "runtime_sec": rt}

        if proc.returncode != 0:
            rt = round(time.time() - t0, 1)
            print(f"  ERROR (exit {proc.returncode})")
            if out and out.strip():
                try:
                    r = parse_worker_json(out)
                    r.setdefault("platform", LABELS[platform])
                    r.setdefault("p", p)
                    r.setdefault("n", CONFIG["n"])
                    r.setdefault("d", CONFIG["d"])
                    r.setdefault("status", "ERROR")
                    r["runtime_sec"] = rt
                    return r
                except Exception:
                    pass
            if err: print(f"      {err.strip()[:1000]}")
            return {"platform": LABELS[platform], "p": p, "n": CONFIG["n"],
                    "d": CONFIG["d"], "status": "ERROR",
                    "error": err.strip()[:1000], "runtime_sec": rt}

        r = parse_worker_json(out)
        r.setdefault("status", "OK")
        r.setdefault("platform", LABELS[platform])
        r.setdefault("p", p)
        r.setdefault("n", CONFIG["n"])
        r.setdefault("d", CONFIG["d"])
        ar   = r.get("approximation_ratio", "?")
        bc   = r.get("best_cut", "?")
        rt   = r.get("runtime_sec", "?")
        nfev = r.get("cobyla_nfev", "?")
        ok   = r.get("cobyla_success", "?")
        print(f"  AR={ar}  best={bc}  nfev={nfev}  converged={ok}  ({rt}s)")
        return r

    except Exception as e:
        rt = round(time.time() - t0, 1)
        print(f"  EXCEPTION: {e}")
        return {"platform": LABELS[platform], "p": p, "n": CONFIG["n"],
                "d": CONFIG["d"], "status": "ERROR",
                "error": str(e), "runtime_sec": rt}


def run_cudaq(p: int, cfg_str: str) -> dict:
    return run_p("cudaq", p, cfg_str)


def run_pennylane(p: int, cfg_str: str) -> dict:
    return run_p("pennylane", p, cfg_str)


def run_qiskit(p: int, cfg_str: str) -> dict:
    return run_p("qiskit", p, cfg_str)


RUNNERS = {
    "cudaq": run_cudaq,
    "pennylane": run_pennylane,
    "qiskit": run_qiskit,
}

# ── checkpoint ────────────────────────────────────────────────────────────────
def save(platform: str, results: list):
    path = DIR / f"results_{platform}_{TIMESTAMP}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def save_combined(all_results: dict):
    combined = [r for rs in all_results.values() for r in rs]
    path = DIR / f"results_combined_{TIMESTAMP}.json"
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    return path

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  QAOA Benchmark — Unified Analyzer (optimized)")
    print("=" * 65)
    print(f"  n={CONFIG['n']}  d={CONFIG['d']}  shots={CONFIG['shots']}  seed={CONFIG['seed']}")
    print(f"  maxiter={CONFIG['maxiter']}  rhobeg={CONFIG['rhobeg']}  rhoend={CONFIG['rhoend']}")
    print(f"  gpu_index : {CONFIG['gpu_index']} (forced by analyzer)")
    print(f"  p depths  : {P_DEPTHS}")
    print(f"  platforms : {PLATFORMS}")
    print(f"  timeout   : {TIMEOUT_PER_P}s per p")
    print(f"  timestamp : {TIMESTAMP}")
    print("=" * 65)

    # Build one dataset instance and compute one baseline for all platforms.
    gpu_info = get_gpu_info(int(CONFIG.get("gpu_index", 0)))
    print(f"  gpu_name  : {gpu_info['gpu_name']}")
    print(f"  gpu_uuid  : {gpu_info['gpu_uuid']}")

    dataset_edges = build_dataset_edges(CONFIG["n"], CONFIG["d"], CONFIG["seed"])
    baseline_val, baseline_key = compute_baseline(
        CONFIG["n"], dataset_edges, CONFIG["seed"], CONFIG["gw_rounds"]
    )
    cfg = {
        **CONFIG,
        "graph_edges": dataset_edges,
        **gpu_info,
        "baseline_key": baseline_key,
        "baseline_value": baseline_val,
        baseline_key: baseline_val,
    }
    cfg_str = json.dumps(cfg)

    print(f"  dataset   : {len(dataset_edges)} edges (owned by analyzer)")
    print(f"\n  baseline  : {baseline_key} = {baseline_val}")
    print("=" * 65)

    all_results = {}
    t_wall = time.time()

    for platform in PLATFORMS:
        print(f"\n  ── {LABELS[platform]} ──")
        results   = []
        skip_rest = False

        for p in P_DEPTHS:
            if skip_rest:
                print(f"    p={p}  SKIPPED (previous p timed out or errored)")
                results.append({"platform": LABELS[platform], "p": p,
                                 "n": CONFIG["n"], "d": CONFIG["d"],
                                 "status": "SKIPPED"})
                continue

            r = RUNNERS[platform](p, cfg_str)
            r.setdefault("gpu_index", gpu_info["gpu_index"])
            r.setdefault("gpu_name", gpu_info["gpu_name"])
            r.setdefault("gpu_uuid", gpu_info["gpu_uuid"])
            results.append(r)
            save(platform, results)

            if r.get("status") in ("TIMEOUT", "ERROR"):
                skip_rest = True

        all_results[platform] = results

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {'Platform':<12} {'Backend':<22} {'p':<4} {'Status':<9} "
          f"{'AR':<8} {'Best':<6} {'Baseline':<10} {'nfev':<7} {'Runtime'}")
    print(f"  {'-' * 62}")

    for platform in PLATFORMS:
        for r in all_results.get(platform, []):
            status   = r.get("status", "?")
            ar       = f"{r['approximation_ratio']:.4f}" if "approximation_ratio" in r else "—"
            best     = str(r.get("best_cut", "—"))
            baseline = str(r.get("optimal_cut", r.get("gw_cut", "—")))
            nfev     = str(r.get("cobyla_nfev", "—"))
            rt       = f"{r['runtime_sec']:.1f}s" if "runtime_sec" in r else "—"
            backend  = r.get("backend", "—")[:20]
            print(f"  {r.get('platform','?'):<12} {backend:<22} "
                  f"{r.get('p','?'):<4} {status:<9} {ar:<8} {best:<6} {baseline:<10} {nfev:<7} {rt}")

    combined = save_combined(all_results)
    total    = round(time.time() - t_wall, 1)
    print(f"\n  Total runtime : {total}s")
    print(f"  Output        : {combined.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()