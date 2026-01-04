# 清洗调度器
from cleaning.adapters import ADAPTERS

def run_cleaning(cfg):
    name = cfg["dataset_name"]
    if name not in ADAPTERS:
        raise ValueError(f"Unknown dataset name: {name}")
    return ADAPTERS[name](cfg)