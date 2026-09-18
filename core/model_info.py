"""What was actually built, recorded next to what it scored.

`run_config.json` records the hyperparameters a run was *asked* for. It does not
say what came out: how many parameters the model ended up with, how they are
distributed, or how much of it is frozen. That gap matters here for a specific
reason -- this repository compares a backbone against the same backbone plus a
plugin, and if the plugin adds 20% more parameters then the comparison is
measuring capacity alongside whatever the plugin does. Nothing surfaced that.

So every run writes `model_info.json` beside its metrics, and the breakdown is
by top-level module, which is exactly the granularity that separates a
backbone's parameters from a plugin's.

Three things a naive `sum(p.numel())` misses, all of which occur in this
repository:

* **Buffers.** GKT's adjacency matrix and LPKT's Q-matrix are registered
  buffers, not parameters. They are real state, they take real memory, and they
  do not appear in `parameters()`.
* **Frozen parameters.** `dkt_pebg` loads a pretrained question embedding and
  sets `requires_grad=False`. Counting it as capacity overstates what the
  optimiser can use; not counting it at all understates what the model knows.
* **Where the mass is.** On assist2009 a question embedding table is
  17,737 x d, which dwarfs everything else. A "model size" that does not say so
  invites tuning the wrong knob.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _module_rows(model):
    """One row per top-level child, plus a row for parameters held directly.

    Top level, not recursive: it is the level at which a composed model splits
    into backbone and plugin, and a full tree is unreadable for a transformer.
    """
    rows = {}
    claimed = set()
    for name, child in model.named_children():
        params = list(child.named_parameters())
        buffers = list(child.named_buffers())
        for pname, _ in params:
            claimed.add(f"{name}.{pname}")
        rows[name] = {
            "type": type(child).__name__,
            "parameters": sum(p.numel() for _, p in params),
            "trainable": sum(p.numel() for _, p in params if p.requires_grad),
            "parameter_bytes": sum(_tensor_bytes(p) for _, p in params),
            "buffers": sum(b.numel() for _, b in buffers),
            "buffer_bytes": sum(_tensor_bytes(b) for _, b in buffers),
        }

    direct = [
        (n, p) for n, p in model.named_parameters() if n not in claimed
    ]
    if direct:
        # e.g. LPKT's `initial_knowledge`, which is a Parameter on the model
        # itself rather than inside a submodule, and would otherwise vanish
        # from a per-child breakdown while still counting in the total.
        rows["(direct)"] = {
            "type": "Parameter",
            "parameters": sum(p.numel() for _, p in direct),
            "trainable": sum(p.numel() for _, p in direct if p.requires_grad),
            "parameter_bytes": sum(_tensor_bytes(p) for _, p in direct),
            "buffers": 0,
            "buffer_bytes": 0,
        }
    return rows


def collect_model_info(model, device=None, top_tensors=8):
    """Everything about the constructed model that the config cannot tell you."""
    params = list(model.named_parameters())
    buffers = list(model.named_buffers())

    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    frozen = sum(p.numel() for _, p in params if not p.requires_grad)
    param_bytes = sum(_tensor_bytes(p) for _, p in params)
    buffer_bytes = sum(_tensor_bytes(b) for _, b in buffers)

    by_dtype = defaultdict(int)
    for _, p in params:
        by_dtype[str(p.dtype)] += p.numel()

    info = {
        "class": type(model).__name__,
        "model_name": getattr(model, "model_name", None),
        "device": str(device) if device is not None else None,
        "total_parameters": trainable + frozen,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "parameter_tensors": len(params),
        "parameter_bytes": param_bytes,
        "buffer_elements": sum(b.numel() for _, b in buffers),
        "buffer_bytes": buffer_bytes,
        "total_bytes": param_bytes + buffer_bytes,
        "total_mb": round((param_bytes + buffer_bytes) / 2 ** 20, 3),
        "parameters_by_dtype": dict(by_dtype),
        "by_module": _module_rows(model),
        "largest_tensors": [
            {"name": n, "shape": list(p.shape), "parameters": p.numel(),
             "trainable": bool(p.requires_grad)}
            for n, p in sorted(params, key=lambda kv: -kv[1].numel())[:top_tensors]
        ],
    }

    # A composed model's whole point is "backbone plus something". Saying how
    # much the something costs is the number an ablation table needs.
    if hasattr(model, "backbone") and hasattr(model, "plugin"):
        backbone = sum(p.numel() for p in model.backbone.parameters())
        plugin = sum(p.numel() for p in model.plugin.parameters())
        info["composition"] = {
            "backbone_parameters": backbone,
            "plugin_parameters": plugin,
            "plugin_share": round(plugin / (backbone + plugin), 4) if backbone + plugin else 0.0,
            # How many times the size of the bare backbone the composed model
            # is. An ablation row that reads "backbone vs backbone+plugin" is
            # only about the plugin if this is near 1.
            "size_vs_backbone": round((backbone + plugin) / backbone, 3) if backbone else None,
        }
    return info


def format_model_info(info, max_modules=12):
    """A compact block for the console and the run log."""
    lines = [
        f"Model: {info['class']}"
        + (f" ({info['model_name']})" if info.get("model_name") else ""),
        f"  parameters   {info['total_parameters']:>12,}"
        f"   trainable {info['trainable_parameters']:,}"
        + (f"   frozen {info['frozen_parameters']:,}" if info["frozen_parameters"] else ""),
        f"  memory       {info['total_mb']:>12.2f} MB"
        f"   (params {info['parameter_bytes'] / 2**20:.2f} MB"
        f" + buffers {info['buffer_bytes'] / 2**20:.2f} MB)",
    ]

    comp = info.get("composition")
    if comp:
        lines.append(
            f"  composition  backbone {comp['backbone_parameters']:>12,}"
            f"   plugin {comp['plugin_parameters']:,}"
            f"  ({comp['plugin_share']:.1%} of the model)"
        )
        lines.append(
            f"               {'this model is':<12s} {comp['size_vs_backbone']:>12.2f}x"
            f" the size of the bare backbone"
        )

    total = max(info["total_parameters"], 1)
    rows = sorted(
        info["by_module"].items(), key=lambda kv: -kv[1]["parameters"]
    )
    lines.append("  by module:")
    for name, row in rows[:max_modules]:
        share = row["parameters"] / total
        extra = f"  +{row['buffers']:,} buffer" if row["buffers"] else ""
        lines.append(
            f"    {name:<24s} {row['parameters']:>12,}  {share:6.1%}"
            f"  {row['type']}{extra}"
        )
    if len(rows) > max_modules:
        rest = sum(r["parameters"] for _, r in rows[max_modules:])
        lines.append(f"    {'(' + str(len(rows) - max_modules) + ' more)':<24s} {rest:>12,}")

    lines.append("  largest tensors:")
    for t in info["largest_tensors"][:5]:
        shape = "x".join(str(d) for d in t["shape"])
        flag = "" if t["trainable"] else "  [frozen]"
        lines.append(f"    {t['name']:<40s} {shape:<18s} {t['parameters']:>12,}{flag}")
    return "\n".join(lines)


def save_model_info(path, info):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=True)


#: Fields that decide whether two folds built the same model.
_SIGNATURE_FIELDS = (
    "class", "total_parameters", "trainable_parameters", "frozen_parameters",
    "buffer_elements",
)


def _signature(info):
    sig = {k: info.get(k) for k in _SIGNATURE_FIELDS}
    sig["by_module"] = {
        name: row["parameters"] for name, row in info["by_module"].items()
    }
    return sig


def save_model_info_once(save_root, info, fold_id):
    """Write `<save_root>/model_info.json` for the first fold, not every fold.

    A cross-validation run calls `train_one_fold` five times and the model is
    usually identical each time, so five copies would be five identical files.

    Usually, not always. `dkt_forget` sizes its gap tables from the current
    fold's training rows, and `lpkt`/`hdkt` size their answer-time and
    interval-time vocabularies the same way, so those models genuinely have a
    fold-dependent parameter count. Overwriting or skipping silently would hide
    that, and it is the kind of thing you want to know before reading a
    per-fold spread. A fold that built a different model writes its own file and
    says which part changed.

    Returns the path written, or None when this fold matched the first one.
    """
    path = os.path.join(save_root, "model_info.json")
    payload = dict(info, fold=fold_id)
    if not os.path.exists(path):
        save_model_info(path, payload)
        return path

    try:
        with open(path, encoding="utf-8") as f:
            first = json.load(f)
    except (OSError, json.JSONDecodeError):
        save_model_info(path, payload)
        return path

    if _signature(first) == _signature(info):
        return None

    changed = [
        k for k in _SIGNATURE_FIELDS if first.get(k) != info.get(k)
    ]
    fold_path = os.path.join(save_root, f"model_info_fold{fold_id}.json")
    save_model_info(fold_path, payload)
    print(
        f"Note: fold {fold_id} built a different model from fold "
        f"{first.get('fold')} ({', '.join(changed) or 'module sizes'}). "
        "That is expected for models whose tables are fitted per fold "
        f"(dkt_forget, lpkt, hdkt); written to {fold_path}."
    )
    return fold_path
