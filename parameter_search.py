#!/usr/bin/env python3
"""
Automated grid + random search over selected constants.
Edit `grid` or `rand_ranges` to explore other parameters.
"""

import itertools, json, random, pandas as pd
from calculate_reimbursement import calculate_reimbursement as CR

# ------------------------------------------------ load public cases
with open("public_cases.json") as f:
    data = json.load(f)
df = pd.json_normalize(data)
df.rename(columns={
    "input.trip_duration_days":     "d",
    "input.miles_traveled":         "m",
    "input.total_receipts_amount":  "r",
    "expected_output":              "exp"
}, inplace=True)

def score(cfg):
    pred = df.apply(lambda row: CR(row.d, row.m, row.r, cfg=cfg), axis=1)
    return (pred - df.exp).abs().mean()

# ------------------------------------------------ GRID SEARCH (coarse / targeted)
grid = {
    # existing short‑trip receipt bands
    "SHORT_RATE3":  [0.40, 0.45, 0.50],
    "SHORT_RATE4":  [0.28, 0.30, 0.32],
    "SHORT_RATE5":  [0.12, 0.15, 0.18],

    # moderate‑length bonus cap
    "MOD_RPD_MAX":  [300, 350, 400],

    # efficiency bonus amount
    "EFF_BONUS":    [110, 125, 140],

    # NEW: tune mid‑length receipt rates & long‑trip bonuses / penalties
    "MID_RATE3":    [0.40, 0.42, 0.45],
    "MID_RATE4":    [0.28, 0.30, 0.32],
    "LONG_BONUS":   [60, 80, 100],
    "LL_MULT":      [0.75, 0.80],
}

best_mae, best_cfg = 1e9, None
print("Grid search …")
for combo in itertools.product(*grid.values()):
    trial = dict(zip(grid.keys(), combo))
    mae = score(trial)
    if mae < best_mae:
        best_mae, best_cfg = mae, trial
        print(f"  MAE {mae:.2f}  cfg={trial}")

print(f"\nBest from grid: MAE {best_mae:.2f} cfg={best_cfg}")

# ------------------------------------------------ RANDOM SEARCH (wider)
rand_ranges = {
    "SHORT_RATE3":  (0.40, 0.55),
    "SHORT_RATE4":  (0.28, 0.45),
    "SHORT_RATE5":  (0.10, 0.22),
    "MOD_RPD_MAX":  (250, 450),
    "EFF_BONUS":    ( 80, 160),
}

best_mae_rand, best_cfg_rand = best_mae, best_cfg
print("\nRandom search …")
for _ in range(4000):
    trial = {k: random.uniform(*v) for k, v in rand_ranges.items()}
    mae = score(trial)
    if mae < best_mae_rand:
        best_mae_rand, best_cfg_rand = mae, trial
        print(f"  MAE {mae:.2f} cfg={trial}")

print(f"\nFINAL best MAE {best_mae_rand:.2f} cfg={best_cfg_rand}")