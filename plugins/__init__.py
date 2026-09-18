"""Plugins: things that modify a backbone without being a model.

A plugin changes one intermediate tensor inside a backbone and may add a term
to its loss. It is registered against a backbone by name, producing a model the
registry serves like any other -- `hd_akt` is `akt` plus `HDPlugin`, not a
separate implementation of AKT.

The machinery lives in `core/plugin.py` and the seam it acts on in
`core/backbone.py`; this package holds the concrete plugins. Importing it
registers every composition, which is why `models/__init__.py` imports it last:
`register_plugged` looks its backbone up in `MODEL_REGISTRY`, so the backbones
must already be there.

Sibling of `strategies/`, which resolves *which artifact* a model loads. This
one changes what the model computes.
"""

from . import hd  # noqa: F401  -- registers hd_dkt / hd_akt / hd_simplekt

__all__ = ["hd"]
