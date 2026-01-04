import os

from torch.utils.data import DataLoader

from core.registry import DATASET_REGISTRY
from .kt_dataset import KTDataset


@DATASET_REGISTRY.register("kt_default")
def build_dataloaders(dataset_name, data_config, fold, batch_size, num_workers=0, **kwargs):
    if dataset_name in data_config:
        cfg = data_config[dataset_name]
    else:
        cfg = data_config

    train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])
    all_folds = set(cfg["folds"])
    train_ds = KTDataset(train_valid_path, cfg["input_type"], all_folds - {fold})
    valid_ds = KTDataset(train_valid_path, cfg["input_type"], {fold})

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader
