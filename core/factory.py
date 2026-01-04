from .registry import MODEL_REGISTRY, DATASET_REGISTRY, TRAINER_REGISTRY

def build_model(name, **kwargs):
    return MODEL_REGISTRY.get(name)(**kwargs)

def build_dataset(name, **kwargs):
    return DATASET_REGISTRY.get(name)(**kwargs)

def build_trainer(name, **kwargs):
    return TRAINER_REGISTRY.get(name)(**kwargs)