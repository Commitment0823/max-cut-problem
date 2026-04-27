"""
analyzer.py  —  QAOA Benchmark Unified Controller (optimized)
==============================================================
Edit CONFIG + EXPERIMENT below, then:
    python analyzer.py

Features:
    - All params here, zero hardcode in platform files
    - Gset datasets loaded from maxcut/datasets
    - Known benchmark baseline passed to all platforms
    - Per-p timeout kill point
    - Per-p checkpoint (partial results saved immediately)
    - Outputs timestamped JSON + prints summary table
    - cobyla_nfev recorded (how many iterations COBYLA actually ran)
"""

import subprocess, json, os, sys, time, tempfile
import itertools
from datetime import datetime
from pathlib import Path
import networkx as nx

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG  —  edit everything here
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "shots"    : 1024,         # measurement shots
    "seed"     : 42,           # random seed (used for baseline computation)
    "maxiter"  : 10000,        # COBYLA max iterations (exits early on convergence)
    "rhobeg"   : 0.5,          # COBYLA initial trust region radius
    "rhoend"   : 1e-4,         # COBYLA convergence threshold (catol internally)
    "gpu_index": 0,            # force all workers to this visible GPU index
    "require_gpu": True,       # fail a platform run if GPU backend is unavailable
}

P_DEPTHS = [1, 2, 3, 4]
MAX_EXACT_N = 30
DATASET_NAMES = ["n4", "n15", "n20"]
SEEDS = [42, 123, 999]

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
DATASETS_DIR = DIR / "datasets"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# Module-level variable set by main() and used by run_p()
DATASET_N = None


def load_dataset(dataset_name: str) -> dict:
    """Load a Max-Cut Gset .txt dataset from datasets/ directory."""
    dataset_path = DATASETS_DIR / f"{dataset_name}.txt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Empty maxcut dataset: {dataset_path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid Gset header in {dataset_path}")

    n = int(header[0])
    edge_count = int(header[1])
    edges = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        if u == v:
            continue
        edges.append((min(u, v), max(u, v)))
    edges = sorted(set(edges))
    avg_degree = (2 * len(edges)) / n if n else 0.0

    metadata = {
        "source_format": "gset_txt",
        "declared_edges": edge_count,
        "avg_degree": round(avg_degree, 2),
    }

    print(f"  [dataset] Loaded '{dataset_name}' type=maxcut n={n} edges={len(edges)} avg_degree={avg_degree:.2f}")
    return {
        "name": dataset_name,
        "type": "maxcut",
        "n": n,
        "edges": [[u, v] for u, v in edges],
        "regularity": int(round(avg_degree)),
        "metadata": metadata,
    }

def baseline_for_dataset(dataset: dict):
    n = int(dataset["n"])
    edges = [(int(u), int(v)) for u, v in dataset["edges"]]

    def cut_value(bits):
        return sum(1 for u, v in edges if bits[u] != bits[v])

    if n <= 20:
        print(f"  [baseline] brute-force optimal cut (n={n})")
        best = 0
        for bits in itertools.product([0, 1], repeat=n):
            best = max(best, cut_value(bits))
        return int(best), "optimal_cut"

    # For medium-size toy datasets, use a fast classical heuristic baseline.
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    _, best = nx.algorithms.approximation.maxcut.one_exchange(g)
    print(f"  [baseline] heuristic reference cut = {best}")
    return int(best), "reference_cut"


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
def run_p(platform: str, p: int, cfg_ref: str) -> dict:
    cmd = [sys.executable, str(DIR / SCRIPTS[platform]),
           "--config", cfg_ref, "--p", str(p)]

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
            return {"platform": LABELS[platform], "p": p, "n": DATASET_N,
                    "status": "TIMEOUT",
                    "timeout_sec": TIMEOUT_PER_P, "runtime_sec": rt}

        if proc.returncode != 0:
            rt = round(time.time() - t0, 1)
            print(f"  ERROR (exit {proc.returncode})")
            if out and out.strip():
                try:
                    r = parse_worker_json(out)
                    r.setdefault("platform", LABELS[platform])
                    r.setdefault("p", p)
                    r.setdefault("n", DATASET_N)
                    r.setdefault("status", "ERROR")
                    r["runtime_sec"] = rt
                    r["worker_wall_sec"] = rt
                    return r
                except Exception:
                    pass
            if err: print(f"      {err.strip()[:1000]}")
            return {"platform": LABELS[platform], "p": p, "n": DATASET_N,
                    "status": "ERROR",
                    "error": err.strip()[:1000], "runtime_sec": rt,
                    "worker_wall_sec": rt}

        r = parse_worker_json(out)
        r.setdefault("status", "OK")
        r.setdefault("platform", LABELS[platform])
        r.setdefault("p", p)
        r.setdefault("n", DATASET_N)
        r.setdefault("worker_wall_sec", round(time.time() - t0, 2))
        ar   = r.get("approximation_ratio", "?")
        bc   = r.get("best_cut", "?")
        rt   = r.get("runtime_sec", "?")
        nfev = r.get("cobyla_nfev", "?")
        ok   = r.get("cobyla_success", "?")
        csec = r.get("compile_sec", "?")
        osec = r.get("optimize_sec", "?")
        wsec = r.get("worker_wall_sec", "?")
        print(f"  AR={ar}  best={bc}  nfev={nfev}  converged={ok}  (run={rt}s compile={csec}s opt={osec}s wall={wsec}s)")
        return r

    except Exception as e:
        rt = round(time.time() - t0, 1)
        print(f"  EXCEPTION: {e}")
        return {"platform": LABELS[platform], "p": p, "n": DATASET_N,
            "status": "ERROR",
                "error": str(e), "runtime_sec": rt,
                "worker_wall_sec": rt}


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
def save(dataset_name: str, platform: str, results: list):
    path = DIR / f"results_{dataset_name}_{platform}_{TIMESTAMP}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def save_combined(all_results: dict):
    combined = [r for platform_results in all_results.values() for rs in platform_results.values() for r in rs]
    path = DIR / f"results_combined_{TIMESTAMP}.json"
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    return path


def baseline_value_from_result(r: dict) -> str:
    return str(r.get("optimal_cut", r.get("reference_cut", "—")))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global DATASET_N
    
    print("=" * 65)
    print("  QAOA Benchmark — Max-Cut Gset Analyzer")
    print("=" * 65)
    
    gpu_info = get_gpu_info(int(CONFIG.get("gpu_index", 0)))
    print(f"  gpu_name  : {gpu_info['gpu_name']}")
    print(f"  gpu_uuid  : {gpu_info['gpu_uuid']}")
    print(f"  shots     : {CONFIG['shots']}")
    print(f"  seeds     : {SEEDS}")
    print(f"  maxiter   : {CONFIG['maxiter']}  rhobeg={CONFIG['rhobeg']}  rhoend={CONFIG['rhoend']}")
    print(f"  gpu_index : {CONFIG['gpu_index']} (forced by analyzer)")
    print(f"  p depths  : {P_DEPTHS}")
    print(f"  platforms : {PLATFORMS}")
    print(f"  timeout   : {TIMEOUT_PER_P}s per p")
    print(f"  timestamp : {TIMESTAMP}")
    print("=" * 65)

    all_results = {}
    t_wall = time.time()
    temp_cfg_paths = []

    try:
        for dataset_name in DATASET_NAMES:
            print(f"\n  ===== Dataset {dataset_name} =====")
            dataset = load_dataset(dataset_name)
            n = dataset["n"]
            dataset_edges = dataset["edges"]
            regularity = dataset["regularity"]
            DATASET_N = n

            seed_cfg_paths = {}
            seed_baselines = {}
            for seed in SEEDS:
                baseline_val, baseline_key = baseline_for_dataset(dataset)
                cfg = {
                    **CONFIG,
                    "seed": seed,
                    "n": n,
                    **gpu_info,
                    "baseline_key": baseline_key,
                    "baseline_value": baseline_val,
                    baseline_key: baseline_val,
                    "d": int(regularity),
                    "graph_edges": dataset_edges,
                }

                cfg_file = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=DIR, prefix=f"cfg_{dataset_name}_s{seed}_", suffix=".json")
                json.dump(cfg, cfg_file)
                cfg_file.flush()
                cfg_file.close()
                cfg_path = cfg_file.name
                temp_cfg_paths.append(cfg_path)
                seed_cfg_paths[seed] = cfg_path
                seed_baselines[seed] = (baseline_key, baseline_val)

            print(f"  dataset   : {dataset_name}")
            print(f"  n         : {n}")
            print(f"  d         : {regularity}")
            print(f"  edges     : {len(dataset_edges)}")
            for seed in SEEDS:
                baseline_key, baseline_val = seed_baselines[seed]
                print(f"  baseline  : seed={seed} {baseline_key} = {baseline_val}")
            print("=" * 65)

            if n > MAX_EXACT_N:
                print(f"  [skip] {dataset_name} has n={n}, which exceeds the exact-simulation cap of {MAX_EXACT_N}")
                dataset_results = {}
                for platform in PLATFORMS:
                    dataset_results[platform] = [
                        {
                            "platform": LABELS[platform],
                            "p": p,
                            "n": n,
                            "seed": seed,
                            "dataset": dataset_name,
                            "status": "UNSUPPORTED",
                            "error": f"n={n} exceeds exact-simulation cap of {MAX_EXACT_N}",
                            seed_baselines[seed][0]: seed_baselines[seed][1],
                        }
                        for seed in SEEDS
                        for p in P_DEPTHS
                    ]
                    save(dataset_name, platform, dataset_results[platform])
                all_results[dataset_name] = dataset_results
                continue

            dataset_results = {}
            for platform in PLATFORMS:
                print(f"\n  ── {LABELS[platform]} ──")
                results = []
                for seed in SEEDS:
                    print(f"    [seed={seed}]")
                    skip_rest = False

                    for p in P_DEPTHS:
                        if skip_rest:
                            print(f"      p={p}  SKIPPED (previous p timed out or errored)")
                            results.append({"platform": LABELS[platform], "p": p, "n": n, "seed": seed, "dataset": dataset_name, "status": "SKIPPED"})
                            continue

                        r = RUNNERS[platform](p, seed_cfg_paths[seed])
                        r.setdefault("dataset", dataset_name)
                        r.setdefault("seed", seed)
                        r.setdefault("gpu_index", gpu_info["gpu_index"])
                        r.setdefault("gpu_name", gpu_info["gpu_name"])
                        r.setdefault("gpu_uuid", gpu_info["gpu_uuid"])
                        results.append(r)
                        save(dataset_name, platform, results)

                        if r.get("status") in ("TIMEOUT", "ERROR"):
                            skip_rest = True

                dataset_results[platform] = results
            all_results[dataset_name] = dataset_results
    finally:
        for cfg_path in temp_cfg_paths:
            try:
                os.unlink(cfg_path)
            except FileNotFoundError:
                pass

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {'Dataset':<8} {'Seed':<6} {'Platform':<12} {'Backend':<22} {'p':<4} {'Status':<9} "
        f"{'AR':<8} {'Best':<6} {'Baseline':<10} {'nfev':<7} {'Run':<8} {'Compile':<9} {'Optimize':<9} {'Wall'}")
    print(f"  {'-' * 62}")

    for dataset_name, platform_results in all_results.items():
        for platform in PLATFORMS:
            for r in platform_results.get(platform, []):
                status   = r.get("status", "?")
                ar       = f"{r['approximation_ratio']:.4f}" if "approximation_ratio" in r else "—"
                best     = str(r.get("best_cut", "—"))
                baseline = baseline_value_from_result(r)
                nfev     = str(r.get("cobyla_nfev", "—"))
                rt       = f"{r['runtime_sec']:.1f}s" if "runtime_sec" in r else "—"
                csec     = f"{r['compile_sec']:.3f}s" if "compile_sec" in r else "—"
                osec     = f"{r['optimize_sec']:.3f}s" if "optimize_sec" in r else "—"
                wsec     = f"{r['worker_wall_sec']:.1f}s" if "worker_wall_sec" in r else "—"
                seed     = str(r.get("seed", "—"))
                backend  = r.get("backend", "—")[:20]
                print(f"  {dataset_name:<8} {seed:<6} {r.get('platform','?'):<12} {backend:<22} "
                      f"{r.get('p','?'):<4} {status:<9} {ar:<8} {best:<6} {baseline:<10} {nfev:<7} {rt:<8} {csec:<9} {osec:<9} {wsec}")

    combined = save_combined(all_results)
    total    = round(time.time() - t_wall, 1)
    print(f"\n  Total runtime : {total}s")
    print(f"  Output        : {combined.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()