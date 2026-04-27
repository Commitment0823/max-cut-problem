# QAOA Cross-Platform Benchmark for NP-hard Optimization

This project studies one unified research question:

**How do three quantum computing platforms (CUDA-Q, PennyLane, Qiskit) compare when implementing QAOA as a heuristic solver for NP-hard combinatorial optimization problems?**

The benchmark currently covers three problems:
- MaxCut
- MaxSAT
- MIS (Maximum Independent Set)

## Research Scope

### Title Alignment
This repository is aligned with the topic:

"Exploring performance and outcome comparison of implementing QAOA on three quantum computing platforms for NP-hard combinatorial optimization via heuristic algorithms"

### Comparison Dimensions
All problem folders use the same core comparison dimensions:
- solution quality: approximation ratio
- optimization behavior: objective evaluations (`objective_evals` / `cobyla_nfev`)
- runtime behavior: compile, optimize, and total runtime
- sampling cost: effective total shots
- robustness: timeout/error/skip status handling

## Repository Structure

```text
projects/cpr/
├── README.md
├── analyze_cross_problem.py      # cross-problem integrated analysis (new)
├── maxcut/
│   ├── analyzer.py
│   ├── platform_cudaq.py
│   ├── platform_pennylane.py
│   ├── platform_qiskit.py
│   └── datasets/
├── maxSAT/
│   ├── analyzer.py
│   ├── platform_cudaq.py
│   ├── platform_pennylane.py
│   ├── platform_qiskit.py
│   └── datasets/
└── MIS/
    ├── analyzer.py
    ├── platform_cudaq.py
    ├── platform_pennylane.py
    ├── platform_qiskit.py
    └── datasets/
```

All benchmark runners are now problem-local:
- `maxcut/analyzer.py` + `maxcut/platform_*.py`
- `maxSAT/analyzer.py` + `maxSAT/platform_*.py`
- `MIS/analyzer.py` + `MIS/platform_*.py`

The repository root keeps only cross-problem integration utilities (for example `analyze_cross_problem.py`).

## Environment

### Hardware
- NVIDIA GPU
- CUDA driver compatible with your platform stack

### Software
- Python 3.10+
- Linux or WSL2 recommended for GPU simulation stacks

Install baseline dependencies:

```bash
pip install --upgrade pip
pip install numpy scipy networkx cvxpy
pip install pennylane pennylane-lightning pennylane-lightning-gpu
pip install qiskit qiskit-aer-gpu
pip install cudaq
```

## Unified Experimental Design

To ensure fair cross-platform comparisons, each problem analyzer controls:
- same dataset instance
- same depth schedule (`P_DEPTHS`)
- same shot count (`shots`)
- same optimizer family (COBYLA)
- same timeout policy per `(platform, p)`
- same seed list (`SEEDS`) for repeated runs

### Multi-seed Design
Each problem analyzer now supports repeated runs by:

```python
SEEDS = [42, 123, 999]
```

For each dataset and platform, the analyzer runs all configured seeds and writes the seed into every record (`"seed": ...`).

## Running Experiments

Run each problem benchmark independently:

```bash
cd projects/cpr/maxcut && python analyzer.py
cd projects/cpr/maxSAT && python analyzer.py
cd projects/cpr/MIS && python analyzer.py
```

Each run writes:
- `results_<dataset>_<platform>_<timestamp>.json`
- `results_combined_<timestamp>.json`

## Cross-Problem Integrated Analysis

After producing combined files for the three problems, run:

```bash
cd projects/cpr
python analyze_cross_problem.py
```

The script auto-detects the latest `results_combined_*.json` in:
- `maxcut/`
- `maxSAT/`
- `MIS/`

It outputs:
- `cross_problem_summary_<timestamp>.json`
- `cross_problem_summary_<timestamp>_by_seed.csv`
- `cross_problem_summary_<timestamp>_seed_aggregate.csv`
- `cross_problem_summary_<timestamp>_platform_overall.csv`

Optional explicit input paths:

```bash
python analyze_cross_problem.py \
  --maxcut maxcut/results_combined_YYYYMMDD_HHMMSS.json \
  --maxsat maxSAT/results_combined_YYYYMMDD_HHMMSS.json \
  --mis MIS/results_combined_YYYYMMDD_HHMMSS.json
```

## Result Interpretation Notes

### Why use multiple seeds?
QAOA optimization is sensitive to random initialization and sampling noise. Multiple seeds help estimate:
- stability (variance across seeds)
- best-case potential
- average expected behavior

### Typical seed effects
Different seeds may change:
- initial QAOA parameters
- sampled bitstring distribution under finite shots
- optimizer trajectory and local minima reached

Thus, a single seed can be misleading. Reporting mean and standard deviation across seeds is recommended.

## Current Benchmark Coverage

Current default datasets are toy scales in each problem folder:
- `n4`
- `n15`
- `n20`

These are good for controlled comparison and methodology validation. For stronger external claims, add larger or standard benchmark instances.

Current dataset names use the shorter `n4`, `n15`, and `n20` form throughout the problem folders.

## Troubleshooting

- If GPU backend is unavailable, verify installation and CUDA visibility (`nvidia-smi`).
- If one platform times out at a depth, subsequent depths for that seed are marked `SKIPPED`.
- Combined JSON keeps partial progress, so interrupted runs still retain completed records.

## Suggested Reporting Template

For a final presentation, include:
- problem definition and QAOA formulation per problem
- fairness constraints (same settings across platforms)
- per-problem platform ranking by quality and runtime
- cross-seed mean +- std tables
- key takeaways and limitations
