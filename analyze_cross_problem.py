#!/usr/bin/env python3
"""
Cross-problem QAOA analysis for MaxCut, MaxSAT, and MIS.

Features:
- Auto-detect latest results_combined_*.json for each problem folder
- Aggregate by (problem, dataset, platform, seed)
- Aggregate across seeds for fair platform comparison
- Export JSON + CSV summaries
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent
DEFAULT_PATHS = {
    "maxcut": ROOT / "maxcut",
    "maxsat": ROOT / "maxSAT",
    "mis": ROOT / "MIS",
}


def latest_combined_file(problem_dir: Path) -> Path | None:
    files = sorted(problem_dir.glob("results_combined_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_records(problem_name: str, file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []

    records = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        row = {
            "problem": problem_name,
            "dataset": rec.get("dataset"),
            "platform": rec.get("platform"),
            "seed": rec.get("seed"),
            "p": rec.get("p"),
            "status": rec.get("status"),
            "approximation_ratio": safe_float(rec.get("approximation_ratio")),
            "runtime_sec": safe_float(rec.get("runtime_sec")),
            "cobyla_nfev": safe_float(rec.get("cobyla_nfev")),
            "objective_evals": safe_float(rec.get("objective_evals")),
            "source_file": str(file_path),
        }
        records.append(row)
    return records


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.fmean(vals))


def std(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(statistics.pstdev(vals))


def aggregate_by_seed(ok_records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in ok_records:
        key = (r["problem"], r["dataset"], r["platform"], r["seed"])
        groups[key].append(r)

    rows = []
    for (problem, dataset, platform, seed), items in sorted(groups.items()):
        best_row = max(items, key=lambda x: (x["approximation_ratio"] if x["approximation_ratio"] is not None else float("-inf")))
        rows.append(
            {
                "problem": problem,
                "dataset": dataset,
                "platform": platform,
                "seed": seed,
                "num_points": len(items),
                "best_p": best_row.get("p"),
                "best_ar": best_row.get("approximation_ratio"),
                "mean_ar": mean([x.get("approximation_ratio") for x in items]),
                "total_runtime_sec": sum(x.get("runtime_sec") or 0.0 for x in items),
                "mean_runtime_sec": mean([x.get("runtime_sec") for x in items]),
                "total_nfev": sum(x.get("cobyla_nfev") or 0.0 for x in items),
                "total_objective_evals": sum(x.get("objective_evals") or 0.0 for x in items),
            }
        )
    return rows


def aggregate_across_seeds(seed_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in seed_rows:
        key = (r["problem"], r["dataset"], r["platform"])
        groups[key].append(r)

    rows = []
    for (problem, dataset, platform), items in sorted(groups.items()):
        rows.append(
            {
                "problem": problem,
                "dataset": dataset,
                "platform": platform,
                "seeds": len(items),
                "best_ar_mean": mean([x.get("best_ar") for x in items]),
                "best_ar_std": std([x.get("best_ar") for x in items]),
                "mean_ar_mean": mean([x.get("mean_ar") for x in items]),
                "total_runtime_mean": mean([x.get("total_runtime_sec") for x in items]),
                "total_runtime_std": std([x.get("total_runtime_sec") for x in items]),
                "total_nfev_mean": mean([x.get("total_nfev") for x in items]),
            }
        )
    return rows


def platform_overall(seed_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in seed_rows:
        groups[(r["problem"], r["platform"])].append(r)

    rows = []
    for (problem, platform), items in sorted(groups.items()):
        rows.append(
            {
                "problem": problem,
                "platform": platform,
                "num_dataset_seed_groups": len(items),
                "best_ar_mean": mean([x.get("best_ar") for x in items]),
                "best_ar_std": std([x.get("best_ar") for x in items]),
                "total_runtime_mean": mean([x.get("total_runtime_sec") for x in items]),
                "total_runtime_std": std([x.get("total_runtime_sec") for x in items]),
            }
        )

    rows.sort(key=lambda x: (x["problem"], -(x["best_ar_mean"] or -1), x["total_runtime_mean"] or float("inf")))
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-problem QAOA result analyzer")
    parser.add_argument("--maxcut", help="Path to maxcut combined json")
    parser.add_argument("--maxsat", help="Path to maxsat combined json")
    parser.add_argument("--mis", help="Path to MIS combined json")
    parser.add_argument("--out-prefix", default=f"cross_problem_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Output file prefix")
    return parser.parse_args()


def resolve_inputs(args) -> dict[str, Path]:
    resolved = {}

    input_map = {
        "maxcut": args.maxcut,
        "maxsat": args.maxsat,
        "mis": args.mis,
    }

    for key, given in input_map.items():
        if given:
            resolved[key] = Path(given)
            continue
        detected = latest_combined_file(DEFAULT_PATHS[key])
        if detected is None:
            raise FileNotFoundError(f"No results_combined_*.json found for {key} in {DEFAULT_PATHS[key]}")
        resolved[key] = detected

    return resolved


def main():
    args = parse_args()
    inputs = resolve_inputs(args)

    all_records = []
    for problem, path in inputs.items():
        records = load_records(problem, path)
        all_records.extend(records)

    ok_records = [r for r in all_records if r.get("status") == "OK"]
    seed_rows = aggregate_by_seed(ok_records)
    seed_agg_rows = aggregate_across_seeds(seed_rows)
    platform_rows = platform_overall(seed_rows)

    out_json = ROOT / f"{args.out_prefix}.json"
    out_seed_csv = ROOT / f"{args.out_prefix}_by_seed.csv"
    out_seed_agg_csv = ROOT / f"{args.out_prefix}_seed_aggregate.csv"
    out_platform_csv = ROOT / f"{args.out_prefix}_platform_overall.csv"

    payload = {
        "inputs": {k: str(v) for k, v in inputs.items()},
        "total_records": len(all_records),
        "ok_records": len(ok_records),
        "by_seed": seed_rows,
        "seed_aggregate": seed_agg_rows,
        "platform_overall": platform_rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    write_csv(out_seed_csv, seed_rows)
    write_csv(out_seed_agg_csv, seed_agg_rows)
    write_csv(out_platform_csv, platform_rows)

    print("=" * 70)
    print("Cross-Problem QAOA Summary")
    print("=" * 70)
    print(f"inputs        : {inputs}")
    print(f"records       : total={len(all_records)} ok={len(ok_records)}")
    print(f"json output   : {out_json.name}")
    print(f"csv output    : {out_seed_csv.name}")
    print(f"csv output    : {out_seed_agg_csv.name}")
    print(f"csv output    : {out_platform_csv.name}")
    print("=" * 70)

    print("Top platform summary by problem (higher best_ar_mean is better):")
    for row in platform_rows:
        print(
            f"  {row['problem']:<7} {row['platform']:<10} "
            f"best_ar_mean={row['best_ar_mean']:.4f} "
            f"runtime_mean={row['total_runtime_mean']:.3f}s"
        )


if __name__ == "__main__":
    main()
