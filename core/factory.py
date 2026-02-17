import inspect

from .registry import MODEL_REGISTRY, DATASET_REGISTRY, TRAINER_REGISTRY

def build_model(name, **kwargs):
    model_cls = MODEL_REGISTRY.get(name)
    try:
        sig = inspect.signature(model_cls.__init__)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            filtered = kwargs
        else:
            allowed = set(params.keys())
            allowed.discard("self")
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        filtered = kwargs
    return model_cls(**filtered)

def build_dataset(name, **kwargs):
    return DATASET_REGISTRY.get(name)(**kwargs)

def build_trainer(name, **kwargs):
    return TRAINER_REGISTRY.get(name)(**kwargs)