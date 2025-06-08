# small_predict.py  – add this near the top, replacing the strict open()
import json, os, functools

_json_path = os.path.join(os.path.dirname(__file__), "residual.json")
try:
    with open(_json_path, "r") as fp:
        _TREES = json.load(fp)["tree_info"]
except FileNotFoundError:
    _TREES = []           # booster not trained yet

def _tree_value(tree, feat):
    node = tree["tree_structure"]
    while "leaf_value" not in node:
        f, thr = node["split_feature"], node["threshold"]
        node   = node["left_child"] if feat[f] <= thr else node["right_child"]
    return node["leaf_value"]

@functools.lru_cache(maxsize=1)
def _sum_tree_values(tuple_feat):
    feat = list(tuple_feat)
    return sum(_tree_value(t, feat) for t in _TREES)

def predict(feat_list):
    return _sum_tree_values(tuple(feat_list)) if _TREES else 0.0