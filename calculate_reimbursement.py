# calculate_reimbursement.py  – NEW
import math                     # for math.log1p
import small_predict as _sp     # reuse the pure‑Python tree walker
import sys                      # needed for debug prints

# cache the model once
import json, pathlib, functools, joblib, os
_DIR = pathlib.Path(__file__).parent
_JSON_PATH = _DIR / "full_model.json"

# If the JSON doesn’t exist yet, but a pickle does, convert once.
if not _JSON_PATH.exists():
    _PKL_PATH = _DIR / "full_model.gbm"
    if _PKL_PATH.exists():
        gbm = joblib.load(_PKL_PATH)
        with open(_JSON_PATH, "w") as fp:
            json.dump(gbm.booster_.dump_model(), fp)
    else:
        raise FileNotFoundError("Neither full_model.json nor full_model.gbm found.")

# Load the trees for the pure‑Python walker.
with open(_JSON_PATH) as fp:
    _TREES = json.load(fp)["tree_info"]

# LightGBM fast-path disabled to avoid 0.8 s import cost per process.
_USE_LGB = False
# Share the trees with small_predict so its predict() can walk them.
_sp._TREES = _TREES

@functools.lru_cache(maxsize=128)
def _score(feat_tuple):
    """
    Scoring function: uses the pure‑Python tree walker (small_predict).
    The LightGBM fast‑path is disabled to avoid heavy per‑process imports.
    """
    return _sp.predict(feat_tuple)

def calculate_reimbursement(days, miles, receipts, *, case_index=0, **_):
    # defensive casts
    days     = max(int(days), 1)
    miles    = float(miles)
    receipts = float(receipts)
    rpd      = receipts / days
    mpd      = miles    / days
    quarter  = case_index // 250
    log_r    = math.log1p(max(receipts,1))
    log_m    = math.log1p(max(miles,1))
    # interaction terms added in the new GBM
    days_x_rpd = days * rpd
    days_x_mpd = days * mpd
    feat = (
        days, miles, receipts,
        rpd, mpd,
        days_x_rpd, days_x_mpd,
        log_r, log_m,
        quarter
    )
    return round(_score(feat), 2)

# ───────────────────────────── CLI helper ─────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: ./run.sh <days> <miles> <receipts> [case_index]", file=sys.stderr)
        sys.exit(1)

    days     = int(float(sys.argv[1]))  # tolerate "5" or "5.0"
    miles    = float(sys.argv[2])
    receipts = float(sys.argv[3])
    if len(sys.argv) >= 5 and sys.argv[4].strip():
        try:
            case_index = int(sys.argv[4])
        except ValueError:
            case_index = 0
    else:
        case_index = 0

    result = calculate_reimbursement(days, miles, receipts, case_index=case_index)
    # Evaluator expects *only* the number:
    print(f"{result:.2f}")