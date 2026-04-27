"""
analyzer.py  —  MaxSAT benchmark controller (aligned with Max-Cut)
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
        raise ValueError(f"Empty maxSAT dataset: {dataset_path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid MaxSAT header in {dataset_path}")

    n = int(header[0])
    declared_clauses = int(header[1])
    clauses = []
    for ln in lines[1:]:
        clause = [int(x) for x in ln.split() if int(x) != 0]
        if clause:
            clauses.append(clause)

    metadata = {
        "source_format": "txt",
        "declared_clauses": declared_clauses,
        "loaded_clauses": len(clauses),
    }

    print(f"  [dataset] Loaded '{dataset_name}' type=maxsat n={n} clauses={len(clauses)}")
    return {
        "name": dataset_name,
        "type": "maxsat",
        "n": n,
        "clauses": clauses,
        "metadata": metadata,
    }


def clause_satisfied(bits: list[int], clause: list[int]) -> bool:
    for lit in clause:
        idx = abs(lit) - 1
        if idx < 0 or idx >= len(bits):
            continue
        value = bits[idx]
        if (lit > 0 and value == 1) or (lit < 0 and value == 0):
            return True
    return False


def compute_baseline(dataset: dict, seed: int) -> tuple[int, str]:
    n = dataset["n"]
    clauses = dataset["clauses"]

    if n <= MAX_EXACT_N:
        print(f"  [baseline] maxSAT brute-force optimal satisfied clauses (n={n})")
        t0 = time.time()
        best = 0
        for bits in itertools.product([0, 1], repeat=n):
            satisfied = sum(1 for clause in clauses if clause_satisfied(list(bits), clause))
            best = max(best, satisfied)
        print(f"  [baseline] optimal_satisfied = {best}  ({time.time() - t0:.1f}s)")
        return int(best), "optimal_satisfied"

    print(f"  [baseline] maxSAT random sampling (n={n})")
    rng = random.Random(seed)
    t0 = time.time()
    best = 0
    for _ in range(5000):
        bits = [rng.randint(0, 1) for _ in range(n)]
        satisfied = sum(1 for clause in clauses if clause_satisfied(bits, clause))
        best = max(best, satisfied)
    print(f"  [baseline] sampled_satisfied = {best}  ({time.time() - t0:.1f}s)")
    return int(best), "sampled_satisfied"


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
        ms = r.get("mean_satisfied", "?")
        bs = r.get("best_satisfied", "?")
        ar = r.get("approximation_ratio", "?")
        rt = r.get("runtime_sec", "?")
        nfev = r.get("cobyla_nfev", "?")
        csec = r.get("compile_sec", "?")
        osec = r.get("optimize_sec", "?")
        wsec = r.get("worker_wall_sec", "?")
        print(f"  mean_sat={ms}  best_sat={bs}  AR={ar}  nfev={nfev}  (run={rt}s compile={csec}s opt={osec}s wall={wsec}s)")
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
    return str(r.get("optimal_satisfied", r.get("sampled_satisfied", "—")))


def main():
    print("=" * 65)
    print("  QAOA Benchmark — MaxSAT Controller")
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
            clauses = dataset["clauses"]

            seed_cfg_paths = {}
            seed_baselines = {}
            for seed in SEEDS:
                baseline_val, baseline_key = compute_baseline(dataset, seed)
                cfg = {
                    **CONFIG,
                    "seed": seed,
                    "problem_type": "maxsat",
                    "n": n,
                    "clauses": clauses,
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
            print(f"  clauses   : {len(clauses)}")
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
    print(f"  {'Dataset':<8} {'Seed':<6} {'Platform':<12} {'Backend':<22} {'p':<4} {'Status':<9} {'MeanSat':<8} {'Best':<6} {'Baseline':<10} {'nfev':<7} {'Run':<8} {'Compile':<9} {'Optimize':<9} {'Wall'}")
    print(f"  {'-' * 62}")

    for dataset_name, platform_results in all_results.items():
        for platform in PLATFORMS:
            for r in platform_results.get(platform, []):
                status = r.get("status", "?")
                ms = f"{r['mean_satisfied']:.4f}" if "mean_satisfied" in r else "—"
                bs = str(r.get("best_satisfied", "—"))
                baseline = baseline_value_from_result(r)
                nfev = str(r.get("cobyla_nfev", "—"))
                rt = f"{r['runtime_sec']:.1f}s" if "runtime_sec" in r else "—"
                csec = f"{r['compile_sec']:.3f}s" if "compile_sec" in r else "—"
                osec = f"{r['optimize_sec']:.3f}s" if "optimize_sec" in r else "—"
                wsec = f"{r['worker_wall_sec']:.1f}s" if "worker_wall_sec" in r else "—"
                seed = str(r.get("seed", "—"))
                backend = r.get("backend", "—")[:20]
                print(f"  {dataset_name:<8} {seed:<6} {r.get('platform','?'):<12} {backend:<22} {r.get('p','?'):<4} {status:<9} {ms:<8} {bs:<6} {baseline:<10} {nfev:<7} {rt:<8} {csec:<9} {osec:<9} {wsec}")

    combined = save_combined(all_results)
    total = round(time.time() - t_wall, 1)
    print(f"\n  Total runtime : {total}s")
    print(f"  Output        : {combined.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
