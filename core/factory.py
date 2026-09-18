import inspect

from .registry import MODEL_REGISTRY, DATASET_REGISTRY, TRAINER_REGISTRY


def _filter_to_signature(cls, kwargs):
    """Keep only the keyword arguments `cls.__init__` can accept.

    Model and trainer constructors take different subsets of the same config
    block, so the caller passes a superset and this narrows it. A class taking
    `**kwargs` receives everything, since it has chosen to decide for itself.

    This is a quiet filter by design, and that is its hazard: a key the
    constructor does not name is dropped without a word. IEKT lost its `device`
    argument that way for as long as its `__init__` omitted the parameter, which
    also meant `--gpu_id` did nothing for it. The guard against a mistyped or
    renamed config key is not here but in
    tests/test_config_keys_are_consumed.py, which requires every key in every
    config block to reach a consumer.

    Signature inspection failure is not caught. It used to fall back to passing
    every argument through, which only converts a clear error into a confusing
    TypeError from inside the constructor.
    """
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    allowed = set(params) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def build_model(name, **kwargs):
    return MODEL_REGISTRY.get(name)(**_filter_to_signature(MODEL_REGISTRY.get(name), kwargs))


def build_dataset(name, **kwargs):
    return DATASET_REGISTRY.get(name)(**kwargs)


def build_trainer(name, **kwargs):
    trainer_cls = TRAINER_REGISTRY.get(name)
    return trainer_cls(**_filter_to_signature(trainer_cls, kwargs))
