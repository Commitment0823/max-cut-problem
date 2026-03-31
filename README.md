# QAOA Max Cut — Cross-Platform GPU-Accelerated Benchmark

A systematic benchmark of the Quantum Approximate Optimization Algorithm (QAOA) for the Max Cut problem across three quantum computing frameworks: **CUDA-Q**, **PennyLane**, and **Qiskit**. All three platforms run on GPU, controlled by a single unified analyzer.

---

## Overview

Max Cut is a canonical NP-hard combinatorial optimization problem. This project benchmarks QAOA across platforms under identical conditions — same graph, same seed, same optimizer, same shots — so results are directly comparable.

**Key features:**
- Single `analyzer.py` controls all parameters — zero hardcode in platform files
- All three platforms GPU-accelerated (CUDA-Q `nvidia`, PennyLane `lightning.gpu`, Qiskit `AerSimulator GPU`)
- Per-depth kill point: configurable timeout per `p`, auto-skips remaining depths on timeout
- Per-depth checkpoint: results saved after every `p` completes, nothing lost on interrupt
- COBYLA optimizer runs to natural convergence (`catol` threshold) instead of hard iteration cap
- Outputs timestamped JSON per platform + combined JSON for analysis

---

## Repository Structure

```
├── analyzer.py              # Unified controller — edit this to configure experiments
├── platform_cudaq.py        # CUDA-Q worker (GPU)
├── platform_pennylane.py    # PennyLane worker (GPU via lightning.gpu)
├── platform_qiskit.py       # Qiskit worker (GPU via qiskit-aer-gpu)
└── qaoa_visualize.py        # HTML report generator (reads output JSONs)
```

---

## Requirements

### Hardware
- NVIDIA GPU (tested on RTX 3060 and above)
- CUDA driver ≥ 525 (supports CUDA 12)

### Software
- Python 3.10–3.12
- pip ≥ 24.0

---

## Installation

### Linux (Ubuntu/Debian)

```bash
# 1. Create virtual environment
python3 -m venv ~/venv/qaoa
source ~/venv/qaoa/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install all dependencies
pip install numpy scipy networkx cvxpy
pip install pennylane pennylane-lightning pennylane-lightning-gpu
pip install qiskit qiskit-aer-gpu
pip install cudaq
```

> **Note:** `qiskit-aer-gpu` replaces `qiskit-aer`. Do not install both.
> If you previously installed `qiskit-aer`, uninstall it first:
> ```bash
> pip uninstall qiskit-aer -y
> pip install qiskit-aer-gpu
> ```

Activate the environment before every session:
```bash
source ~/venv/qaoa/bin/activate
```

---

### Windows

GPU simulation on Windows requires WSL2 (Windows Subsystem for Linux). Native Windows is not supported for `cudaq` or `pennylane-lightning-gpu`.

**Step 1 — Install WSL2:**
```powershell
wsl --install
```
Restart your machine, then open Ubuntu from the Start menu.

**Step 2 — Inside WSL2, follow the Linux instructions above exactly.**

**Step 3 — Verify GPU is visible inside WSL2:**
```bash
nvidia-smi
```
If this fails, update your NVIDIA Windows driver to the latest version (WSL2 shares the Windows driver).

> **Note:** `cudaq` GPU support (`nvidia` target) requires Linux x86_64. On WSL2 this works correctly.

---

## Configuration

Open `analyzer.py` and edit the `CONFIG` block at the top:

```python
CONFIG = {
    "n"        : 20,       # number of graph nodes
    "d"        : 4,        # graph regularity (d-regular)
    "shots"    : 1024,     # measurement shots per circuit evaluation
    "seed"     : 42,       # random seed (graph, params, sampling — all unified)
    "gw_rounds": 300,      # Goemans-Williamson rounding iterations (n > 20 only)
    "maxiter"  : 10000,    # COBYLA max iterations (exits early on convergence)
    "rhobeg"   : 0.5,      # COBYLA initial trust region radius
    "rhoend"   : 1e-4,     # COBYLA convergence threshold
}

P_DEPTHS = [1, 2, 3]       # QAOA depths to run

TIMEOUT_PER_P = 900        # seconds per (platform, p) — None to disable

PLATFORMS = [
    "cudaq",
    "pennylane",
    "qiskit",
]
```

**Baseline selection is automatic:**
- `n ≤ 20` → brute-force exact optimal (stored as `optimal_cut`)
- `n > 20` → Goemans-Williamson SDP approximation (stored as `gw_cut`)

---

## Running

```bash
source ~/venv/qaoa/bin/activate   # Linux / WSL2
python analyzer.py
```

The analyzer runs platforms sequentially. Live output:

```
=================================================================
  QAOA Benchmark — Unified Analyzer (optimized)
=================================================================
  n=20  d=4  shots=1024  seed=42
  ...
  ── CUDA-Q ──
    p=1 ...  AR=0.7794  best=34  nfev=36  converged=True  (1.3s)
    p=2 ...  AR=0.8404  best=34  nfev=394  converged=True  (20.1s)
    p=3 ...  AR=0.8644  best=34  nfev=162  converged=True  (11.0s)
  ── PennyLane ──
    p=1 ...  AR=0.7732  best=34  nfev=35  converged=True  (0.8s)
    ...
```

---

## Output

After each run, three files are written to the same directory:

| File | Contents |
|------|----------|
| `results_cudaq_<timestamp>.json` | CUDA-Q results, all depths |
| `results_pennylane_<timestamp>.json` | PennyLane results, all depths |
| `results_qiskit_<timestamp>.json` | Qiskit results, all depths |
| `results_combined_<timestamp>.json` | All platforms merged |

Each record contains:

```json
{
  "platform": "CUDA-Q",
  "backend": "nvidia (GPU)",
  "p": 2,
  "n": 20,
  "d": 4,
  "mean_cut": 28.5723,
  "best_cut": 34,
  "optimal_cut": 34,
  "approximation_ratio": 0.8404,
  "total_shots": 1024,
  "runtime_sec": 20.05,
  "cobyla_nfev": 394,
  "cobyla_success": true,
  "optimal_params": [1.915, 2.994, 1.929, 2.375]
}
```

---

## Visualization

Feed the combined JSON into the visualizer to generate an HTML report:

```bash
python qaoa_visualize.py
```

Opens `qaoa_report.html` in your browser with:
- Approximation ratio vs depth charts per experiment
- Runtime comparison charts (log scale)
- GPU scaling analysis across problem sizes
- Full results tables

---

## Benchmark Design

### Fairness guarantees

All three platforms use:
- Identical graph: `networkx.random_regular_graph(d, n, seed=42)`
- Identical initial parameters: `np.random.uniform(0, π, 2p)` with `seed=42`
- Identical optimizer: `scipy COBYLA` with `maxiter=10000`, `rhobeg=0.5`, `catol=1e-4`
- Identical shots: 1024
- Identical metric: `approximation_ratio = mean_cut / baseline_cut`

### What differs between platforms

| | CUDA-Q | PennyLane | Qiskit |
|--|--------|-----------|--------|
| GPU backend | cuQuantum (native) | cuQuantum via lightning.gpu | cuStateVec via AerSimulator |
| Optimization cost function | `cudaq.observe` (expectation value, no shot noise) | shot sampling | shot sampling |
| Circuit API | JIT-compiled kernel (`@cudaq.kernel`) | `@qml.qnode` decorator | `QuantumCircuit` + `transpile` |

### Why CUDA-Q tends to get higher AR

CUDA-Q's `cudaq.observe` computes the Hamiltonian expectation value analytically from the statevector — no shot noise. PennyLane and Qiskit optimize using noisy sample averages, which makes COBYLA's gradient estimation less reliable at high depths. This is a fundamental difference in optimization method, not just platform performance.

---

## Troubleshooting

**`cudaq` GPU not detected:**
```bash
nvidia-smi   # confirm driver is working
python -c "import cudaq; cudaq.set_target('nvidia'); print('OK')"
```

**`lightning.gpu` import error:**
```bash
pip install --upgrade pennylane-lightning-gpu
python -c "import pennylane as qml; dev = qml.device('lightning.gpu', wires=2); print('OK')"
```

**`qiskit-aer-gpu` GPU not found:**
```bash
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"
# should print ['CPU', 'GPU']
# if only ['CPU'], your qiskit-aer-gpu installation doesn't have CUDA support
```

**COBYLA `rhoend` warning:** Ignore it — we use `catol` which is the correct scipy parameter name. `rhoend` is the underlying Fortran parameter name and appears in some warning messages but does not affect results.

---

## Citation

If you use this benchmark, please cite the original QAOA paper:

> Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. *arXiv:1411.4028*.

And the Goemans-Williamson baseline:

> Goemans, M. X., & Williamson, D. P. (1995). Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming. *Journal of the ACM, 42*(6), 1115–1145.