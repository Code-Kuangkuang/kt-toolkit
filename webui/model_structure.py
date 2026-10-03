import copy
from pathlib import Path

from core.config import load_cfg
from core.factory import build_model


from core.train_runner import NON_MODEL_CONFIG_KEYS, resolve_dataset_mode
from core.model_inputs import RunContext, spec_for
from core.model_build import prepare_model_inputs, construct_prepared_model
from core.registry import MODEL_REGISTRY

OTHER_CONFIG_KEYS = NON_MODEL_CONFIG_KEYS

def build_model_structure(root, request):
    import torch
    import torch.nn as nn
    import models  # noqa: F401 - import side effects register models

    root = Path(root)
    kt_cfg = load_cfg(str(root / "configs" / "kt_config.json"))
    data_cfg = load_cfg(str(root / "configs" / "data_config.json"))

    dataset_name = request["dataset_name"]
    model_name = request["model_name"]
    if dataset_name not in data_cfg:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if model_name not in kt_cfg:
        raise ValueError(f"Unknown model: {model_name}")

    train_cfg = copy.deepcopy(kt_cfg.get("train_config", {}))
    model_cfg = copy.deepcopy(kt_cfg.get(model_name, {}))
    from core.config_schema import validate_model_overrides
    overrides = request.get("model_config") or {}
    validate_model_overrides(model_name, overrides, model_cfg)
    model_cfg.update({k: v for k, v in overrides.items() if v is not None and v != ""})
    emb_type = request.get("emb_type") or model_cfg.get("emb_type", "qid")
    model_cfg["emb_type"] = emb_type

    dataset_cfg = copy.deepcopy(data_cfg[dataset_name])
    dpath = Path(dataset_cfg.get("dpath", ""))
    dataset_cfg["dpath"] = str(dpath if dpath.is_absolute() else root / dpath)
    spec = spec_for(MODEL_REGISTRY.get(model_name))
    mode = resolve_dataset_mode(model_name, train_cfg, model_cfg, spec=spec)
    train_cfg["dataset_mode"] = model_cfg["dataset_mode"] = mode
    def resolve_file(primary, fallback):
        for key in (primary, fallback):
            value = dataset_cfg.get(key)
            if value and (Path(dataset_cfg["dpath"]) / value).is_file():
                return value
        return dataset_cfg.get(primary) or dataset_cfg.get(fallback)
    ctx = RunContext(model_name, dataset_name, int(request.get("fold", 0)), mode,
                     model_cfg, dataset_cfg, train_cfg, str(root), resolve_file)
    # Preview must not perturb training RNGs in this process.
    from core.checkpoint import preserve_rng
    with preserve_rng():
        inputs = prepare_model_inputs(ctx)
        model = construct_prepared_model(ctx, inputs, "cpu")
    model.eval()

    rows = []
    for name, module in model.named_modules():
        own_params = sum(p.numel() for p in module.parameters(recurse=False))
        own_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        rows.append(
            {
                "path": name or "(root)",
                "depth": 0 if not name else name.count(".") + 1,
                "type": module.__class__.__name__,
                "own_params": own_params,
                "own_trainable_params": own_trainable,
                "param_shapes": _param_shapes(module),
                "btd": _shape_hint(module, nn),
                "children": len(list(module.children())),
            }
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "emb_type": emb_type,
        "seq_len": train_cfg.get("seq_len"),
        "num_q": dataset_cfg.get("num_q"),
        "num_c": dataset_cfg.get("num_c"),
        "max_concepts": dataset_cfg.get("max_concepts"),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "modules": rows,
        "repr": repr(model),
    }


def _param_shapes(module):
    shapes = []
    for name, param in module.named_parameters(recurse=False):
        shapes.append({"name": name, "shape": list(param.shape), "trainable": bool(param.requires_grad)})
    return shapes


def _shape_hint(module, nn):
    if isinstance(module, nn.Embedding):
        return {
            "input": "ids [B, T]",
            "output": f"[B, T, {module.embedding_dim}]",
            "detail": f"D={module.embedding_dim}, vocab={module.num_embeddings}",
        }
    if isinstance(module, nn.Linear):
        return {
            "input": f"[B, T, {module.in_features}]",
            "output": f"[B, T, {module.out_features}]",
            "detail": f"D: {module.in_features} -> {module.out_features}",
        }
    if isinstance(module, (nn.Dropout, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.LayerNorm)):
        return {"input": "[B, T, D]", "output": "[B, T, D]", "detail": "shape preserved"}
    if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
        direction = 2 if getattr(module, "bidirectional", False) else 1
        return {
            "input": f"[B, T, {module.input_size}]",
            "output": f"[B, T, {module.hidden_size * direction}]",
            "detail": f"hidden={module.hidden_size}, layers={module.num_layers}",
        }
    if isinstance(module, nn.MultiheadAttention):
        return {
            "input": f"[B, T, {module.embed_dim}]",
            "output": f"[B, T, {module.embed_dim}]",
            "detail": f"heads={module.num_heads}",
        }

    cls_name = module.__class__.__name__
    if cls_name == "QueEmb" and hasattr(module, "emb_size"):
        return {
            "input": "q,c ids [B, T]",
            "output": f"[B, T, {int(module.emb_size) * 2}]",
            "detail": f"question/concept embedding, emb_type={getattr(module, 'emb_type', '')}",
        }
    if cls_name in {"funcs", "funcsgru"} and hasattr(module, "out"):
        out = module.out
        return {
            "input": f"[B, T, {out.in_features}]",
            "output": f"[B, T, {out.out_features}]",
            "detail": f"MLP D: {out.in_features} -> {out.out_features}",
        }
    if cls_name == "mygru" and hasattr(module, "g_ir") and hasattr(module.g_ir, "out"):
        out = module.g_ir.out
        return {
            "input": f"[B, {out.in_features}]",
            "output": f"[B, {out.out_features}]",
            "detail": "custom GRU state update",
        }
    return {"input": "", "output": "", "detail": ""}
