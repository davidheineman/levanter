# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for MuP-style learning rate scaling."""

from __future__ import annotations

from typing import Any

import jax
from optax import GradientTransformation
from optax._src.base import init_empty_state

from haliax.nn.mup import ReparamEnabled

from levanter.utils.jax_utils import is_inexact_arrayish


def _scale_module_updates(updates: Any, lr_scale: float) -> Any:
    """Apply ``lr_scale`` to all inexact leaves of ``updates``.

    ``optax`` maintains the PyTree structure of the parameters inside the updates, so
    the incoming ``updates`` value is typically another :class:`ReparamEnabled`
    module. We descend into the module to scale its array leaves.

    Note: We don't use ``is_leaf`` here because we want to reach the actual array
    leaves inside the module. Nested modules will be handled by the outer tree_map
    in ``scale_by_mup_lr``.
    """

    def _maybe_scale(leaf: Any) -> Any:
        if is_inexact_arrayish(leaf):
            return leaf * lr_scale
        return leaf

    return jax.tree_util.tree_map(_maybe_scale, updates)


def scale_by_mup_lr() -> GradientTransformation:
    """Scale updates by the MuP per-layer learning rate multipliers.

    ``ReparamEnabled`` modules are expected to provide an ``lr_scale`` attribute that
    contains the layer-specific learning rate multiplier. This transformation simply
    multiplies the updates associated with those modules by their ``lr_scale`` while
    leaving other updates untouched.
    """

    def update_fn(updates, state, params=None, **extra_args):
        del extra_args
        if params is None:
            raise ValueError("scale_by_mup_lr requires params to be passed to update_fn.")

        def _apply_scale(update, param):
            if isinstance(param, ReparamEnabled):
                lr_scale = param.reparam.lr_scale
                return _scale_module_updates(update, lr_scale)
            return update

        scaled_updates = jax.tree_util.tree_map(
            _apply_scale,
            updates,
            params,
            is_leaf=lambda x: isinstance(x, ReparamEnabled),
        )
        return scaled_updates, state

    return GradientTransformation(init_empty_state, update_fn)
