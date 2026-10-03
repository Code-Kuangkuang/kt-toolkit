"""The shared InputSpec preparation and construction path."""
from core.factory import build_model
from core.model_inputs import spec_for
from core.registry import MODEL_REGISTRY


def prepare_model_inputs(ctx):
    spec = spec_for(MODEL_REGISTRY.get(ctx.model_name))
    spec.validate(ctx)
    inputs = spec.prepare(ctx)
    ctx.model_cfg.update(inputs.model_cfg_updates)
    ctx.dataset_cfg.update(inputs.dataset_cfg_updates)
    return inputs


def construct_prepared_model(ctx, inputs, device):
    from core.train_runner import NON_MODEL_CONFIG_KEYS
    kwargs = {k: v for k, v in ctx.model_cfg.items() if k not in NON_MODEL_CONFIG_KEYS}
    kwargs.update(inputs.model_kwargs)
    model = build_model(
        ctx.model_name, num_c=ctx.dataset_cfg["num_c"], num_q=ctx.dataset_cfg["num_q"],
        emb_type=ctx.model_cfg.get("emb_type", "qid"), seq_len=ctx.train_cfg.get("seq_len"),
        device=str(device), dpath=ctx.dataset_cfg.get("dpath", ""),
        num_at=ctx.model_cfg.get("num_at"), num_it=ctx.model_cfg.get("num_it"), **kwargs,
    ).to(device)
    return spec_for(MODEL_REGISTRY.get(ctx.model_name)).post_build(model, ctx)
