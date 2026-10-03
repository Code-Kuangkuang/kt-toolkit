"""Validate public model overrides against the actual registered constructors."""
import inspect

from core.registry import MODEL_REGISTRY


def validate_model_overrides(model_name, overrides, defaults):
    import models  # noqa: F401
    from core.train_runner import NON_MODEL_CONFIG_KEYS
    cls = MODEL_REGISTRY.get(model_name)
    allowed = set(defaults) | NON_MODEL_CONFIG_KEYS | {"weight_decay"}
    pending, visited = [cls], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(getattr(current, "composed_of", ()))
        for base in current.__mro__:
            allowed.update(name for name, param in inspect.signature(base.__init__).parameters.items()
                           if name != "self" and param.kind not in
                           (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL))
    allowed -= {"num_c", "num_q", "device", "seq_len", "dpath"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported parameters for {model_name}: {sorted(unknown)}")
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "dropout" and (not isinstance(value, (int, float)) or not 0 <= value < 1):
            raise ValueError("dropout must be in [0, 1).")
        if key == "learning_rate" and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError("learning_rate must be positive.")
