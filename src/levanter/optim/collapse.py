# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Collapse training configuration and utilities.

This module provides configuration for "Scaling with Collapse" training,
which enables predictable loss curves across model scales by enforcing:
- Fixed tokens-per-parameter (TPP) ratio
- Fixed AdamW timescale (τ) 
- Normalized LR scheduling

Combined with CompleteP parameterization, this enables hyperparameter
transfer across both width and depth.
"""

from dataclasses import dataclass, field
from typing import Optional

import draccus

from haliax.nn.collapse import CollapseConfig, NormalizedSchedule, validate_collapse_config
from haliax.nn.mup import CompletePConfig


@dataclass(frozen=True)
class CollapseOptimizerConfig:
    """Optimizer configuration derived from Collapse constraints.

    Instead of specifying weight_decay directly, it is derived from the
    AdamW timescale τ. This ensures consistent optimization dynamics
    across model scales.

    Attributes:
        collapse: Core Collapse configuration (TPP, tau).
        base_lr: Peak learning rate.
        beta1: Adam beta1.
        beta2: Adam beta2.
        epsilon: Adam epsilon (will be scaled by width/depth if using CompleteP).
        max_grad_norm: Gradient clipping norm.
        use_mup: Enable MuP LR scaling.
    """

    collapse: CollapseConfig = field(default_factory=CollapseConfig)

    base_lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    use_mup: bool = True

    def compute_weight_decay(
        self,
        batch_size_tokens: int,
        total_tokens: int,
    ) -> float:
        """Derive weight decay from AdamW timescale τ.

        Args:
            batch_size_tokens: Batch size in tokens.
            total_tokens: Total training tokens.

        Returns:
            Weight decay coefficient.
        """
        return self.collapse.compute_weight_decay(
            learning_rate=self.base_lr,
            batch_size=batch_size_tokens,
            total_tokens=total_tokens,
        )

    def compute_epsilon(
        self,
        completep_config: Optional[CompletePConfig] = None,
        width_multiplier: float = 1.0,
    ) -> float:
        """Compute scaled Adam epsilon for CompleteP.

        Args:
            completep_config: Optional CompleteP config for depth scaling.
            width_multiplier: Width multiplier for MuP scaling.

        Returns:
            Scaled epsilon value.
        """
        if completep_config is None:
            return self.epsilon
        return self.epsilon * completep_config.adam_epsilon_scale(width_multiplier)


@dataclass
class CollapseScheduleConfig:
    """Learning rate schedule configuration for Collapse training.

    Uses normalized time (t̂ = step / total_steps) to ensure identical
    schedule shapes across model scales.
    """

    warmup_fraction: float = 0.01
    cooldown_fraction: float = 0.2
    min_lr_fraction: float = 0.1

    def to_normalized_schedule(self) -> NormalizedSchedule:
        """Convert to NormalizedSchedule."""
        return NormalizedSchedule(
            warmup_fraction=self.warmup_fraction,
            cooldown_fraction=self.cooldown_fraction,
            min_lr_fraction=self.min_lr_fraction,
        )

    @classmethod
    def from_collapse_config(cls, config: CollapseConfig) -> "CollapseScheduleConfig":
        """Create from CollapseConfig defaults."""
        return cls(
            warmup_fraction=config.warmup_fraction,
            cooldown_fraction=config.cooldown_fraction,
        )


@dataclass
class CollapseTrainingBudget:
    """Computed training budget from Collapse constraints.

    This is the result of applying CollapseConfig to a specific model size.
    All derived values are computed here.
    """

    num_params: int
    total_tokens: int
    total_steps: int
    batch_size_tokens: int
    weight_decay: float
    base_lr: float

    # Collapse diagnostics
    actual_tpp: float
    actual_tau: float

    @classmethod
    def compute(
        cls,
        num_params: int,
        batch_size: int,
        seq_len: int,
        optimizer_config: CollapseOptimizerConfig,
    ) -> "CollapseTrainingBudget":
        """Compute training budget from model size and Collapse config.

        Args:
            num_params: Number of trainable parameters.
            batch_size: Batch size in sequences.
            seq_len: Sequence length.
            optimizer_config: Collapse optimizer configuration.

        Returns:
            Fully computed training budget.
        """
        collapse = optimizer_config.collapse
        batch_size_tokens = batch_size * seq_len

        total_tokens = collapse.compute_total_tokens(num_params)
        total_steps = collapse.compute_total_steps(num_params, batch_size, seq_len)
        weight_decay = collapse.compute_weight_decay(
            learning_rate=optimizer_config.base_lr,
            batch_size=batch_size_tokens,
            total_tokens=total_tokens,
        )

        # Compute actual values for validation
        actual_tpp = total_tokens / num_params
        actual_tau = batch_size_tokens / (optimizer_config.base_lr * weight_decay * total_tokens)

        return cls(
            num_params=num_params,
            total_tokens=total_tokens,
            total_steps=total_steps,
            batch_size_tokens=batch_size_tokens,
            weight_decay=weight_decay,
            base_lr=optimizer_config.base_lr,
            actual_tpp=actual_tpp,
            actual_tau=actual_tau,
        )

    def validate(self, config: CollapseConfig, tolerance: float = 0.01) -> bool:
        """Validate that computed budget matches Collapse constraints.

        Args:
            config: Expected Collapse configuration.
            tolerance: Relative tolerance for matching.

        Returns:
            True if budget is consistent with Collapse constraints.
        """
        is_valid, _ = validate_collapse_config(
            num_params=self.num_params,
            batch_size=self.batch_size_tokens,
            learning_rate=self.base_lr,
            weight_decay=self.weight_decay,
            total_tokens=self.total_tokens,
            config=config,
            tolerance=tolerance,
        )
        return is_valid

    def to_hyperparameters(self) -> dict:
        """Return dict suitable for logging as hyperparameters."""
        return {
            "collapse/num_params": self.num_params,
            "collapse/total_tokens": self.total_tokens,
            "collapse/total_steps": self.total_steps,
            "collapse/batch_size_tokens": self.batch_size_tokens,
            "collapse/weight_decay": self.weight_decay,
            "collapse/base_lr": self.base_lr,
            "collapse/actual_tpp": self.actual_tpp,
            "collapse/actual_tau": self.actual_tau,
        }

