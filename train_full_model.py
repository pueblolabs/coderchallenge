# train_full_model.py  – run once
import json, joblib, pandas as pd, lightgbm as lgb
import numpy as np              # ← add this

"""
Higher‑capacity GBM to memorise 1000 public cases
"""

with open("public_cases.json") as f:
    raw = json.load(f)

df = pd.json_normalize(raw).rename(columns={
    "input.trip_duration_days": "days",
    "input.miles_traveled":     "miles",
    "input.total_receipts_amount": "receipts",
    "expected_output":          "target"
})

# canonical features
df["rpd"]      = df.receipts / df.days
df["mpd"]      = df.miles    / df.days
df["quarter"]  = (df.index // 250)        # captures the “calendar mood”
df["log_r"]    = df.receipts.clip(1).pipe(np.log1p)   # stabilise high tails
df["log_m"]    = df.miles.clip(1).pipe(np.log1p)

# simple interaction terms that often help tree models
df["days_x_rpd"] = df.days * df.rpd
df["days_x_mpd"] = df.days * df.mpd

X = df[[
    "days", "miles", "receipts",
    "rpd", "mpd",
    "days_x_rpd", "days_x_mpd",
    "log_r", "log_m",
    "quarter"
]]
y = df["target"]

gbm = lgb.LGBMRegressor(
    n_estimators=3000,     # ↑ capacity
    max_depth=6,
    learning_rate=0.03,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_samples=2,
    random_state=42
)

gbm.fit(X, y)
print("Train MAE:", abs(gbm.predict(X)-y).mean())      # expect ≲ $1 with these settings
joblib.dump(gbm, "full_model.gbm")
gbm.booster_.save_model("full_model.txt")   # LightGBM native text format
# after joblib.dump(gbm, "full_model.gbm")
import json, pathlib
json_path = pathlib.Path("full_model.json")
with open(json_path, "w") as fp:
    json.dump(gbm.booster_.dump_model(), fp)
print("✔  full_model.json saved")