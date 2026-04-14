# QAOA Benchmark Collaboration Contract

## Goal
Make analyzer the only file edited for experiments, while workers remain stable executors.

## Controller and Workers
- Controller: projects/cpr/analyzer.py
- Workers:
  - projects/cpr/platform_pennylane.py
  - projects/cpr/platform_qiskit.py
  - projects/cpr/platform_cudaq.py

## Required Operating Model
1. Analyzer is the single source of truth for config, dataset, baseline, schedule, and outputs.
2. Workers are stateless executors for one (platform, p) run.
3. Worker stdout must emit exactly one JSON object.
4. Worker stderr may include debug logs.

## Analyzer to Worker Protocol
### Input
- --config JSON string
- --p integer

### Required config keys
- n, d, shots, seed
- maxiter, rhobeg, rhoend
- graph_edges
- baseline_key, baseline_value

### Required result keys
- platform, backend, status
- p, n, d
- mean_cut, best_cut, approximation_ratio
- runtime_sec
- cobyla_nfev, cobyla_success

## Reliability Rules
1. Dataset identity is owned by analyzer (workers do not create independent datasets).
2. Analyzer captures timeout/error per p and checkpoints partial outputs.
3. Analyzer remains robust if worker emits extra lines by parsing final JSON line.
4. Legacy compatibility is preserved for old payloads where possible.

## Acceptance Criteria
1. Benchmark changes require analyzer edits only.
2. All platforms run on the same graph instance per benchmark.
3. Combined output is directly comparable across platforms.
4. Failed/timeout runs are recorded without crashing full benchmark.
