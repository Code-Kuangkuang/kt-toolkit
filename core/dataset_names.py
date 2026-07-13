DATASET_NAME_ALIASES = {
    "peiyou": "aaai2023",
}

HIDDEN_LABEL_DATASETS = {
    "aaai2023",
}


def normalize_dataset_name(name):
    normalized = str(name).strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def is_hidden_label_dataset(name):
    return normalize_dataset_name(name) in HIDDEN_LABEL_DATASETS
