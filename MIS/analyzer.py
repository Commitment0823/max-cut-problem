"""
analyzer.py  —  MIS benchmark controller (aligned with Max-Cut)
"""

import json
import itertools
import os
import random
import subprocess
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path


CONFIG = {
    "shots": 1024,
    "seed": 42,
    "maxiter": 10000,
    "rhobeg": 0.5,
    "rhoend": 1e-4,
    "gpu_index": 0,
    "require_gpu": True,
    "warmup": True,
}

P_DEPTHS = [1, 2, 3, 4]
MAX_EXACT_N = 20
DATASET_NAMES = ["n4", "n15", "n20"]
SEEDS = [42, 123, 999]
TIMEOUT_PER_P = 900

PLATFORMS = ["cudaq", "pennylane", "qiskit"]

SCRIPTS = {
    "cudaq": "platform_cudaq.py",
    "pennylane": "platform_pennylane.py",
    "qiskit": "platform_qiskit.py",
}
LABELS = {
    "cudaq": "CUDA-Q",
    "pennylane": "PennyLane",
    "qiskit": "Qiskit",
}

DIR = Path(__file__).parent
DATASETS_DIR = DIR / "datasets"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(dataset_name: str) -> dict:
    dataset_path = DATASETS_DIR / f"{dataset_name}.txt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Empty MIS dataset: {dataset_path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid MIS header in {dataset_path}")

    n = int(header[0])
    declared_edges = int(header[1])
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

    metadata = {
        "source_format": "txt",
        "declared_edges": declared_edges,
        "loaded_edges": len(edges),
    }

    print(f"  [dataset] Loaded '{dataset_name}' type=mis n={n} edges={len(edges)}")
    return {
        "name": dataset_name,
        "type": "mis",
        "n": n,
        "edges": [[u, v] for u, v in edges],
        "metadata": metadata,
    }


def is_independent(bits: list[int], edges: list[list[int]]) -> bool:
    selected = {index for index, value in enumerate(bits) if value == 1}
    for u, v in edges:
        if u in selected and v in selected:
            return False
    return True


def compute_baseline(dataset: dict, seed: int) -> tuple[int, str]:
    n = dataset["n"]
    edges = dataset["edges"]

    if n <= MAX_EXACT_N:
        print(f"  [baseline] MIS brute-force optimal independent set (n={n})")
        t0 = time.time()
        best = 0
        for bits in itertools.product([0, 1], repeat=n):
            if is_independent(list(bits), edges):
                best = max(best, sum(bits))
        print(f"  [baseline] optimal_independent_set = {best}  ({time.time() - t0:.1f}s)")
        return int(best), "optimal_independent_set"

    print(f"  [baseline] MIS greedy sampling (n={n})")
    rng = random.Random(seed)
    t0 = time.time()
    best = 0
    adjacency = {i: set() for i in range(n)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    for _ in range(1000):
        remaining = set(range(n))
        chosen = set()
        order = list(range(n))
        rng.shuffle(order)
        for node in order:
            if node in remaining:
                chosen.add(node)
                remaining.discard(node)
                remaining.difference_update(adjacency[node])
        best = max(best, len(chosen))
    print(f"  [baseline] sampled_independent_set = {best}  ({time.time() - t0:.1f}s)")
    return int(best), "sampled_independent_set"


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


def run_p(platform: str, p: int, cfg_ref: str, n: int) -> dict:
    cmd = [sys.executable, str(DIR / SCRIPTS[platform]), "--config", cfg_ref, "--p", str(p)]

    print(f"    p={p} ...", end="", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(CONFIG.get("gpu_index", 0))

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            out, err = proc.communicate(timeout=TIMEOUT_PER_P)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            rt = round(time.time() - t0, 1)
            print(f"  TIMEOUT ({rt}s > {TIMEOUT_PER_P}s)")
            return {"platform": LABELS[platform], "p": p, "n": n, "status": "TIMEOUT", "runtime_sec": rt, "worker_wall_sec": rt}

        if proc.returncode != 0:
            rt = round(time.time() - t0, 1)
            print(f"  ERROR (exit {proc.returncode})")
            if out and out.strip():
                try:
                    r = parse_worker_json(out)
                    r.setdefault("platform", LABELS[platform])
                    r.setdefault("p", p)
                    r.setdefault("n", n)
                    r.setdefault("status", "ERROR")
                    r["runtime_sec"] = rt
                    r["worker_wall_sec"] = rt
                    return r
                except Exception:
                    pass
            if err:
                print(f"      {err.strip()[:1000]}")
            return {"platform": LABELS[platform], "p": p, "n": n, "status": "ERROR", "error": (err or "").strip()[:1000], "runtime_sec": rt, "worker_wall_sec": rt}

        r = parse_worker_json(out)
        r.setdefault("status", "OK")
        r.setdefault("platform", LABELS[platform])
        r.setdefault("p", p)
        r.setdefault("n", n)
        r.setdefault("worker_wall_sec", round(time.time() - t0, 2))
        ms = r.get("mean_independent_size", "?")
        bs = r.get("best_independent_size", "?")
        ff = r.get("feasible_fraction", "?")
        rt = r.get("runtime_sec", "?")
        nfev = r.get("cobyla_nfev", "?")
        csec = r.get("compile_sec", "?")
        osec = r.get("optimize_sec", "?")
        wsec = r.get("worker_wall_sec", "?")
        print(f"  mean_size={ms}  best_size={bs}  feasible={ff}  nfev={nfev}  (run={rt}s compile={csec}s opt={osec}s wall={wsec}s)")
        return r

    except Exception as e:
        rt = round(time.time() - t0, 1)
        print(f"  EXCEPTION: {e}")
        return {"platform": LABELS[platform], "p": p, "n": n, "status": "ERROR", "error": str(e), "runtime_sec": rt, "worker_wall_sec": rt}


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
    return str(r.get("optimal_independent_set", r.get("sampled_independent_set", "—")))


def main():
    print("=" * 65)
    print("  QAOA Benchmark — MIS Controller")
    print("=" * 65)

    gpu_info = {"gpu_index": CONFIG["gpu_index"], "gpu_name": "unknown", "gpu_uuid": "unknown"}
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
            edges = dataset["edges"]

            seed_cfg_paths = {}
            seed_baselines = {}
            for seed in SEEDS:
                baseline_val, baseline_key = compute_baseline(dataset, seed)
                cfg = {
                    **CONFIG,
                    "seed": seed,
                    "problem_type": "mis",
                    "n": n,
                    "edges": edges,
                    "baseline_key": baseline_key,
                    "baseline_value": baseline_val,
                    baseline_key: baseline_val,
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
            print(f"  edges     : {len(edges)}")
            for seed in SEEDS:
                baseline_key, baseline_val = seed_baselines[seed]
                print(f"  baseline  : seed={seed} {baseline_key} = {baseline_val}")
            print("=" * 65)

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

                        r = run_p(platform, p, seed_cfg_paths[seed], n)
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

    print(f"\n{'=' * 65}")
    print(f"  {'Dataset':<8} {'Seed':<6} {'Platform':<12} {'Backend':<22} {'p':<4} {'Status':<9} {'MeanSize':<8} {'Best':<6} {'Feasible':<9} {'Baseline':<10} {'nfev':<7} {'Run':<8} {'Compile':<9} {'Optimize':<9} {'Wall'}")
    print(f"  {'-' * 62}")

    for dataset_name, platform_results in all_results.items():
        for platform in PLATFORMS:
            for r in platform_results.get(platform, []):
                status = r.get("status", "?")
                ms = f"{r['mean_independent_size']:.4f}" if "mean_independent_size" in r else "—"
                bs = str(r.get("best_independent_size", "—"))
                ff = f"{r['feasible_fraction']:.4f}" if "feasible_fraction" in r else "—"
                baseline = baseline_value_from_result(r)
                nfev = str(r.get("cobyla_nfev", "—"))
                rt = f"{r['runtime_sec']:.1f}s" if "runtime_sec" in r else "—"
                csec = f"{r['compile_sec']:.3f}s" if "compile_sec" in r else "—"
                osec = f"{r['optimize_sec']:.3f}s" if "optimize_sec" in r else "—"
                wsec = f"{r['worker_wall_sec']:.1f}s" if "worker_wall_sec" in r else "—"
                seed = str(r.get("seed", "—"))
                backend = r.get("backend", "—")[:20]
                print(f"  {dataset_name:<8} {seed:<6} {r.get('platform','?'):<12} {backend:<22} {r.get('p','?'):<4} {status:<9} {ms:<8} {bs:<6} {ff:<9} {baseline:<10} {nfev:<7} {rt:<8} {csec:<9} {osec:<9} {wsec}")

    combined = save_combined(all_results)
    total = round(time.time() - t_wall, 1)
    print(f"\n  Total runtime : {total}s")
    print(f"  Output        : {combined.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
