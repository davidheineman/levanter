# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Collapse training configuration and utilities.

This module re-exports Collapse training utilities from haliax and adds
Levanter-specific helpers.

Collapse enables predictable loss curves across model scales by fixing:
- TPP (tokens-per-parameter): determines training duration
- τ (tau): AdamW timescale, determines weight decay

Usage with train_lm.py:
    from levanter.optim.collapse import CollapseConfig

    collapse = CollapseConfig(tpp=20.0, tau=0.05)

    # Compute training parameters
    total_tokens = collapse.compute_total_tokens(num_params)
    num_steps = collapse.compute_total_steps(num_params, batch_size, seq_len)
    weight_decay = collapse.compute_weight_decay(learning_rate, batch_size * seq_len, total_tokens)

    # Use these in your TrainerConfig and OptimizerConfig
"""

import logging
from typing import NamedTuple

# Re-export from haliax
from haliax.nn.collapse import (
    CollapseConfig,
    CollapseTracker,
    NormalizedSchedule,
    make_lr_schedule,
    validate_collapse_config,
)

__all__ = [
    # From haliax
    "CollapseConfig",
    "CollapseTracker",
    "NormalizedSchedule",
    "make_lr_schedule",
    "validate_collapse_config",
    # Levanter helpers
    "CollapseTrainingParams",
    "compute_collapse_training_params",
    "estimate_params_from_config",
]

logger = logging.getLogger(__name__)


class CollapseTrainingParams(NamedTuple):
    """Result of computing Collapse training parameters."""

    num_train_steps: int
    weight_decay: float
    total_tokens: int


def compute_collapse_training_params(
    collapse: CollapseConfig,
    num_params: int,
    batch_size: int,
    seq_len: int,
    learning_rate: float,
) -> CollapseTrainingParams:
    """Compute all training parameters from Collapse config.

    This is a convenience function that calls CollapseConfig methods
    and returns a NamedTuple with all the values you need.

    Args:
        collapse: Collapse configuration with tpp and tau.
        num_params: Number of trainable parameters.
        batch_size: Batch size in sequences.
        seq_len: Sequence length.
        learning_rate: Peak learning rate.

    Returns:
        CollapseTrainingParams with num_train_steps, weight_decay, total_tokens.

    Example:
        collapse = CollapseConfig(tpp=20.0, tau=0.05)
        params = compute_collapse_training_params(
            collapse, num_params=125_000_000, batch_size=256,
            seq_len=2048, learning_rate=1e-3
        )
        # params.num_train_steps, params.weight_decay, params.total_tokens
    """
    batch_size_tokens = batch_size * seq_len
    total_tokens = collapse.compute_total_tokens(num_params)
    num_train_steps = collapse.compute_total_steps(num_params, batch_size, seq_len)
    weight_decay = collapse.compute_weight_decay(learning_rate, batch_size_tokens, total_tokens)

    return CollapseTrainingParams(
        num_train_steps=num_train_steps,
        weight_decay=weight_decay,
        total_tokens=total_tokens,
    )


def estimate_params_from_config(model_config, vocab_size: int) -> int:
    """Estimate parameter count from model config.

    Args:
        model_config: Model configuration (e.g., LlamaConfig).
        vocab_size: Vocabulary size.

    Returns:
        Estimated number of trainable parameters.
    """
    if hasattr(model_config, "total_trainable_params"):
        return model_config.total_trainable_params(vocab_size)

    # Rough estimate for transformer: 12 * n_layers * hidden_dim^2
    if hasattr(model_config, "num_layers") and hasattr(model_config, "hidden_dim"):
        estimate = 12 * model_config.num_layers * model_config.hidden_dim**2
        logger.warning(f"Using rough parameter estimate: {estimate:,}")
        return estimate

    raise ValueError(
        "Cannot estimate parameters. Model config should have "
        "total_trainable_params() method or num_layers/hidden_dim attributes."
    )


def log_collapse_params(
    collapse: CollapseConfig,
    params: CollapseTrainingParams,
    num_params: int,
) -> dict:
    """Log Collapse training parameters and return dict for tracking.

    Args:
        collapse: Collapse configuration.
        params: Computed training parameters.
        num_params: Number of model parameters.

    Returns:
        Dict suitable for levanter.tracker.log_hyperparameters()
    """
    # Compute actual values for validation
    actual_tpp = params.total_tokens / num_params

    info = {
        "collapse/tpp": collapse.tpp,
        "collapse/tau": collapse.tau,
        "collapse/warmup_fraction": collapse.warmup_fraction,
        "collapse/cooldown_fraction": collapse.cooldown_fraction,
        "collapse/num_params": num_params,
        "collapse/total_tokens": params.total_tokens,
        "collapse/num_train_steps": params.num_train_steps,
        "collapse/weight_decay": params.weight_decay,
        "collapse/actual_tpp": actual_tpp,
    }

    logger.info("Collapse Training Parameters:")
    logger.info(f"  TPP: {actual_tpp:.2f} (target: {collapse.tpp})")
    logger.info(f"  Parameters: {num_params:,}")
    logger.info(f"  Total tokens: {params.total_tokens:,}")
    logger.info(f"  Training steps: {params.num_train_steps:,}")
    logger.info(f"  Weight decay: {params.weight_decay:.6f}")

    return info
