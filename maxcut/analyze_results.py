import json
import pandas as pd

# Load the data
file_path = '/home/ugs4/projects/cpr/maxcut/results_combined_20260420_210519.json'
with open(file_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Create a unique identifier for dataset/platform
# Assuming 'n' and 'd' define the dataset, though the prompt says dataset/platform.
# Based on the head output, we have platform and backend.
# Let's use (platform, backend, n, d) as the group identifier.
df['dataset_id'] = df.apply(lambda row: f"n{row['n']}_d{row['d']}_{row['platform']}", axis=1)

# Group by dataset_id and p
grouped = df.sort_values(['dataset_id', 'p']).groupby('dataset_id')

results = []

for name, group in grouped:
    g = group.set_index('p')
    
    # Try to get p=1, 2, 3 values
    res = {'dataset_id': name}
    
    for p_val in [1, 2, 3]:
        if p_val in g.index:
            res[f'AR_p{p_val}'] = g.loc[p_val, 'approximation_ratio']
            res[f'evals_p{p_val}'] = g.loc[p_val, 'objective_evals']
            res[f'runtime_p{p_val}'] = g.loc[p_val, 'runtime_sec']
            res[f'shots_p{p_val}'] = g.loc[p_val, 'effective_total_shots']
        else:
            res[f'AR_p{p_val}'] = None
            res[f'evals_p{p_val}'] = None
            res[f'runtime_p{p_val}'] = None
            res[f'shots_p{p_val}'] = None

    # Calculations
    if res['AR_p1'] is not None and res['AR_p3'] is not None:
        res['delta_AR_p1_p3'] = res['AR_p3'] - res['AR_p1']
    else:
        res['delta_AR_p1_p3'] = None

    # delta_AR_per_eval = (AR_p3-AR_p1)/(nfev_p1+nfev_p2+nfev_p3)
    total_evals = sum(res[f'evals_p{p}'] for p in [1, 2, 3] if res[f'evals_p{p}'] is not None)
    if res['delta_AR_p1_p3'] is not None and total_evals > 0:
        res['delta_AR_per_eval'] = res['delta_AR_p1_p3'] / total_evals
    else:
        res['delta_AR_per_eval'] = None

    # shot-efficiency proxy at p=3: AR_p3 / effective_total_shots_p3
    if res['AR_p3'] is not None and res['shots_p3'] is not None and res['shots_p3'] > 0:
        res['shot_efficiency_p3'] = res['AR_p3'] / res['shots_p3']
    else:
        res['shot_efficiency_p3'] = None

    results.append(res)

res_df = pd.DataFrame(results)

# 1) Table for AR
print("--- 1) AR and Delta AR (p1->p3) ---")
ar_cols = ['dataset_id', 'AR_p1', 'AR_p2', 'AR_p3', 'delta_AR_p1_p3']
print(res_df[ar_cols].to_string(index=False))

# 2) Table for Evals and Runtime
print("\n--- 2) Objective Evals and Runtime (sec) ---")
eval_cols = ['dataset_id', 'evals_p1', 'evals_p2', 'evals_p3', 'runtime_p1', 'runtime_p2', 'runtime_p3']
print(res_df[eval_cols].to_string(index=False))

# 3) & 4) Proxies
print("\n--- 3) & 4) Convergence speed and Shot-efficiency ---")
proxy_cols = ['dataset_id', 'delta_AR_per_eval', 'shot_efficiency_p3']
print(res_df[proxy_cols].to_string(index=False))
