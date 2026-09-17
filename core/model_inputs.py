"""Model-side declaration of the extra inputs a model needs.

`core/train_runner.py` grew a `model_name` check for every model that needs data
the standard loaders do not produce -- a graph, a difficulty map, precomputed
statistics. That is the coupling the registry was introduced to remove, and it
scales with the number of models rather than staying fixed.

A model declares its requirements here instead, as a nested `Inputs` class:

    @MODEL_REGISTRY.register("gkt")
    class GKT(nn.Module):
        class Inputs(InputSpec):
            dataset_mode = "one_by_one"

            @classmethod
            def prepare(cls, ctx):
                return ModelInputs(model_kwargs={"graph": load_graph(ctx)})

The runner then calls `validate` and `prepare` and stops caring which model it
is holding.

Three rules this design exists to enforce, each of which the `model_name` chain
got wrong at least once:

1. `prepare` RETURNS its effects. The old blocks mutated `model_cfg_local` and
   `dataset_cfg_local` in place and later code read those mutations back, so
   what a block actually did could only be established by reading the rest of
   the file.

2. `prepare` runs before `set_seed`. Anything that consumes the RNG after the
   seed is set shifts model initialisation, which changes the metrics without
   changing the algorithm.

3. Heavy imports go inside the method body, not at module top. `models/` does
   not currently import `datasets/`; keeping it that way means `import models`
   stays cheap and a future `datasets` -> `models` import cannot create a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class RunContext:
    """Everything a model may consult when deciding what it needs.

    `dataset_mode` is already resolved by the runner; a spec must not re-derive
    it, or CLI and config overrides stop working.

    `model_cfg` and `dataset_cfg` are this fold's private copies. They are
    readable, but a spec should report changes through `ModelInputs` rather than
    writing to them, so that what it changed stays visible in one place.
    """

    model_name: str
    dataset_name: str
    fold_id: int
    dataset_mode: str
    model_cfg: Dict[str, Any]
    dataset_cfg: Dict[str, Any]
    train_cfg: Dict[str, Any]
    root_dir: str
    resolve_file: Callable[[str, str], Optional[str]]

    def train_folds(self):
        """Folds used for fitting, i.e. every configured fold but this one.

        Statistics computed for a fold must not see that fold, so several specs
        need this and previously each spelled it out.
        """
        return sorted(set(self.dataset_cfg.get("folds", [])) - {int(self.fold_id)})

    def quelevel_key(self, base_key: str) -> str:
        """`train_valid_file` -> `train_valid_file_quelevel` under all_in_one."""
        return f"{base_key}_quelevel" if self.dataset_mode == "all_in_one" else base_key


@dataclass
class ModelInputs:
    """What a spec asks the runner to do. Empty means "nothing special"."""

    # Extra keyword arguments for the model constructor.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Extra keyword arguments for every dataloader built in this run.
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Merged into the fold's model config, and so into run_config.json.
    model_cfg_updates: Dict[str, Any] = field(default_factory=dict)
    # Merged into the fold's dataset config.
    dataset_cfg_updates: Dict[str, Any] = field(default_factory=dict)
    # Recorded verbatim in run_config.json, e.g. dkt_pebg's booster choice.
    run_config_extras: Dict[str, Any] = field(default_factory=dict)


class InputSpec:
    """Default behaviour: a model that needs nothing beyond the standard batch.

    Subclass it as a nested `Inputs` class on the model. Every member is
    optional; override only what applies.
    """

    #: "all_in_one", "one_by_one", or None to take the training config's default.
    dataset_mode: Optional[str] = None

    #: Fail early, with the dataset named, rather than at an embedding lookup.
    requires_question_ids: bool = False

    #: Models whose constructor takes the question count as `num_pid`.
    needs_num_pid: bool = False

    @classmethod
    def validate(cls, ctx: RunContext) -> None:
        """Raise if this dataset cannot support the model. Runs before any work.

        The base implementation covers the common case; override and call
        `super().validate(ctx)` to add model-specific checks.
        """
        if cls.requires_question_ids:
            input_type = ctx.dataset_cfg.get("input_type", [])
            num_q = ctx.dataset_cfg.get("num_q", 0)
            if "questions" not in input_type or num_q <= 0:
                raise ValueError(
                    f"{ctx.model_name} requires question ids, but dataset "
                    f"{ctx.dataset_name} has input_type={input_type} and num_q={num_q}."
                )

    @classmethod
    def prepare(cls, ctx: RunContext) -> ModelInputs:
        """Compute the extra inputs. Called once per fold, before `set_seed`."""
        inputs = ModelInputs()
        if cls.needs_num_pid:
            inputs.model_kwargs["num_pid"] = ctx.dataset_cfg.get("num_q", 0)
        return inputs

    @classmethod
    def post_build(cls, model, ctx: RunContext):
        """Adjust the constructed model. Return the model to use.

        For the one case that needs it: Hawkes applies its own weight init and
        runs in double precision.
        """
        return model


def spec_for(model_cls) -> type:
    """The model's `Inputs` spec, or the default when it declares none."""
    spec = getattr(model_cls, "Inputs", None)
    if spec is None:
        return InputSpec
    if not (isinstance(spec, type) and issubclass(spec, InputSpec)):
        raise TypeError(
            f"{model_cls.__name__}.Inputs must subclass core.model_inputs.InputSpec, "
            f"got {spec!r}."
        )
    return spec
