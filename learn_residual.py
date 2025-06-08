#!/usr/bin/env python3
import json, joblib, pandas as pd, lightgbm as lgb
from calculate_reimbursement import calculate_reimbursement as CR

# 1. load public cases
with open("public_cases.json") as f:
    raw = json.load(f)
df = pd.json_normalize(raw).rename(columns={
    "input.trip_duration_days": "days",
    "input.miles_traveled": "miles",
    "input.total_receipts_amount": "receipts",
    "expected_output": "expected"
})

# 2. model input features
df["pred"] = df.apply(lambda r: CR(r.days, r.miles, r.receipts), axis=1)
df["err"]  = df["pred"] - df["expected"]
df["rpd"]  = df["receipts"] / df["days"]
df["mpd"]  = df["miles"]    / df["days"]
df["quarter"] = (df.index // 250)

X = df[["days", "miles", "receipts", "rpd", "mpd", "quarter"]]
y = df["err"]

# 3. train a tiny model
gbm = lgb.LGBMRegressor(max_depth=3, n_estimators=30, learning_rate=0.1)
gbm.fit(X, y)
print("MAE on residuals:", abs(gbm.predict(X) - y).mean())
joblib.dump(gbm, "residual.gbm")
print("✔ residual.gbm saved")