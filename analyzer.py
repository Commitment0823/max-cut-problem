"""
analyzer.py  —  QAOA Benchmark Unified Controller (optimized)
==============================================================
Edit CONFIG + EXPERIMENT below, then:
    python analyzer.py

Features:
  - All params here, zero hardcode in platform files
  - Per-p timeout kill point
  - Per-p checkpoint (partial results saved immediately)
  - Auto-detects best backend per platform at runtime
  - Outputs timestamped JSON + prints summary table
  - cobyla_nfev recorded (tells you how many iterations COBYLA actually ran)
"""

import subprocess, json, os, sys, time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG  —  edit everything here
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "n"        : 20,       # nodes
    "d"        : 4,        # graph regularity
    "shots"    : 1024,     # measurement shots
    "seed"     : 42,
    "gw_rounds": 300,      # GW rounding iterations (n > 20 only)
    "maxiter"  : 10000,    # COBYLA max iterations (exits early on convergence)
    "rhobeg"   : 0.5,      # COBYLA initial trust region radius
    "rhoend"   : 1e-4,     # COBYLA convergence threshold (catol internally)
}

P_DEPTHS = [1, 2, 3]

# Per-p timeout in seconds. None = no limit.
# Recommendation: set high enough for CUDA-Q to finish, kill Qiskit if it crawls
TIMEOUT_PER_P = 900        # 15 min per p

# Platforms to run (comment out any you want to skip)
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
CFG_STR   = json.dumps(CONFIG)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── runner ────────────────────────────────────────────────────────────────────
def run_p(platform: str, p: int) -> dict:
    cmd = [sys.executable, str(DIR / SCRIPTS[platform]),
           "--config", CFG_STR, "--p", str(p)]

    print(f"    p={p} ...", end="", flush=True)
    t0 = time.time()

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
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
            if err: print(f"      {err.strip()[:300]}")
            return {"platform": LABELS[platform], "p": p, "n": CONFIG["n"],
                    "d": CONFIG["d"], "status": "ERROR",
                    "error": err.strip()[:300], "runtime_sec": rt}

        r = json.loads(out.strip())
        r["status"] = "OK"
        ar  = r.get("approximation_ratio", "?")
        bc  = r.get("best_cut", "?")
        rt  = r.get("runtime_sec", "?")
        nfev = r.get("cobyla_nfev", "?")
        ok  = r.get("cobyla_success", "?")
        print(f"  AR={ar}  best={bc}  nfev={nfev}  converged={ok}  ({rt}s)")
        return r

    except Exception as e:
        rt = round(time.time() - t0, 1)
        print(f"  EXCEPTION: {e}")
        return {"platform": LABELS[platform], "p": p, "n": CONFIG["n"],
                "d": CONFIG["d"], "status": "ERROR",
                "error": str(e), "runtime_sec": rt}

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
    print(f"  p depths  : {P_DEPTHS}")
    print(f"  platforms : {PLATFORMS}")
    print(f"  timeout   : {TIMEOUT_PER_P}s per p")
    print(f"  timestamp : {TIMESTAMP}")
    print("=" * 65)

    all_results = {}
    t_wall = time.time()

    for platform in PLATFORMS:
        print(f"\n  ── {LABELS[platform]} ──")
        results     = []
        skip_rest   = False

        for p in P_DEPTHS:
            if skip_rest:
                print(f"    p={p}  SKIPPED (previous p timed out or errored)")
                results.append({"platform": LABELS[platform], "p": p,
                                 "n": CONFIG["n"], "d": CONFIG["d"],
                                 "status": "SKIPPED"})
                continue

            r = run_p(platform, p)
            results.append(r)
            save(platform, results)   # checkpoint after every p

            if r.get("status") in ("TIMEOUT", "ERROR"):
                skip_rest = True

        all_results[platform] = results

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {'Platform':<12} {'Backend':<22} {'p':<4} {'Status':<9} "
          f"{'AR':<8} {'Best':<6} {'nfev':<7} {'Runtime'}")
    print(f"  {'-' * 62}")

    for platform in PLATFORMS:
        for r in all_results.get(platform, []):
            status  = r.get("status", "?")
            ar      = f"{r['approximation_ratio']:.4f}" if "approximation_ratio" in r else "—"
            best    = str(r.get("best_cut", "—"))
            nfev    = str(r.get("cobyla_nfev", "—"))
            rt      = f"{r['runtime_sec']:.1f}s" if "runtime_sec" in r else "—"
            backend = r.get("backend", "—")[:20]
            print(f"  {r.get('platform','?'):<12} {backend:<22} "
                  f"{r.get('p','?'):<4} {status:<9} {ar:<8} {best:<6} {nfev:<7} {rt}")

    combined = save_combined(all_results)
    total    = round(time.time() - t_wall, 1)
    print(f"\n  Total runtime : {total}s")
    print(f"  Output        : {combined.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()