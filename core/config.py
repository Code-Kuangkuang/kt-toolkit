# core/config.py
import json, yaml
def load_cfg(path):
    if path.endswith(".json"):
        return json.load(open(path, "r", encoding="utf-8"))
    return yaml.safe_load(open(path, "r", encoding="utf-8"))
