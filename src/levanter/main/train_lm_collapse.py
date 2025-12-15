# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Training script for Collapse + CompleteP scaling.

This script extends train_lm.py with Collapse training discipline:
- Tokens-per-parameter (TPP) determines training duration
- Weight decay is derived from AdamW timescale τ
- LR schedule uses normalized time for consistent shapes

Combined with CompleteP (enabled via use_completep), this enables
hyperparameter transfer across both width and depth.

Usage:
    python -m levanter.main.train_lm_collapse --config config/llama_collapse.yaml

Example config:
    collapse:
      tpp: 20.0
      tau: 0.05
    completep:
      depth_multiplier: 2.0  # if 24 layers vs 12 base
      depth_alpha_exp: 0.5
    model:
      type: llama
      use_mup: true
      ...
"""

import dataclasses
import functools
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.nn.collapse import CollapseConfig, CollapseTracker, NormalizedSchedule, make_lr_schedule
from haliax.nn.mup import CompletePConfig
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
import levanter.eval
import levanter.eval_harness
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfig, UrlSingleDatasetLMConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.optim.collapse import CollapseOptimizerConfig, CollapseScheduleConfig, CollapseTrainingBudget
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainLmCollapseConfig:
    """Training configuration for Collapse + CompleteP scaling.

    This config automatically derives training parameters from Collapse constraints:
    - num_train_steps = total_tokens / (batch_size × seq_len)
    - weight_decay = batch_size / (τ × lr × total_tokens)

    Attributes:
        collapse: Core Collapse configuration (TPP, tau).
        completep: Optional CompleteP configuration for depth scaling.
        schedule: LR schedule configuration.
        data: Dataset configuration.
        trainer: Trainer configuration (num_train_steps will be overridden).
        model: Model configuration.
        base_lr: Peak learning rate.
    """

    # Collapse-specific configs
    collapse: CollapseConfig = field(default_factory=CollapseConfig)
    completep: Optional[CompletePConfig] = None
    schedule: CollapseScheduleConfig = field(default_factory=CollapseScheduleConfig)

    # Standard configs
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    # Optimizer settings (weight_decay will be computed)
    base_lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    # HF checkpointing
    initialize_from_hf: Union[bool, str] = False
    use_hf_model_config: bool = False
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    # Other settings
    z_loss_weight: float = 0.0
    data_seed: Optional[int] = None
    initialize_from_checkpoint_path: Optional[str] = None
    epoch: int = 0
    eval_harness: Optional[LmEvalHarnessConfig] = None
    eval_harness_steps: int = 10000

    def __post_init__(self):
        # Sync schedule with collapse config
        if self.schedule.warmup_fraction != self.collapse.warmup_fraction:
            self.schedule = dataclasses.replace(
                self.schedule,
                warmup_fraction=self.collapse.warmup_fraction,
                cooldown_fraction=self.collapse.cooldown_fraction,
            )

    def compute_training_budget(self, num_params: int) -> CollapseTrainingBudget:
        """Compute full training budget from model size.

        Args:
            num_params: Number of trainable parameters in the model.

        Returns:
            Fully computed training budget with all derived values.
        """
        batch_size = self.trainer.train_batch_size
        seq_len = self.model.Pos.size

        optimizer_config = CollapseOptimizerConfig(
            collapse=self.collapse,
            base_lr=self.base_lr,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            max_grad_norm=self.max_grad_norm,
            use_mup=getattr(self.model, "use_mup", False),
        )

        return CollapseTrainingBudget.compute(
            num_params=num_params,
            batch_size=batch_size,
            seq_len=seq_len,
            optimizer_config=optimizer_config,
        )

    def build_optimizer(self, budget: CollapseTrainingBudget):
        """Build optimizer with Collapse-derived parameters.

        Args:
            budget: Computed training budget.

        Returns:
            Configured optimizer.
        """
        import optax

        from levanter.optim.mup import scale_by_mup_lr

        # Build LR schedule using normalized time
        lr_schedule = make_lr_schedule(
            base_lr=self.base_lr,
            total_steps=budget.total_steps,
            schedule=self.schedule.to_normalized_schedule(),
        )

        # Compute epsilon scaling for CompleteP
        epsilon = self.epsilon
        if self.completep is not None:
            width_mult = getattr(self.model, "hidden_dim", 768) / 768.0  # Assume base width 768
            epsilon = self.completep.adam_epsilon_scale(width_mult) * self.epsilon

        # Build optimizer components
        components = []

        if self.max_grad_norm:
            components.append(optax.clip_by_global_norm(self.max_grad_norm))

        components.append(optax.scale_by_adam(self.beta1, self.beta2, epsilon))

        if budget.weight_decay > 0:
            # Use decoupled weight decay for stability
            components.append(optax.add_decayed_weights(budget.weight_decay / self.base_lr))

        # Apply MuP LR scaling if enabled
        if getattr(self.model, "use_mup", False):
            components.append(scale_by_mup_lr())

        # Apply LR schedule
        def lr_schedule_fn(step):
            return -lr_schedule(step)  # Negative for descent

        components.append(optax.scale_by_schedule(lr_schedule_fn))

        return optax.chain(*components)


def main(config: TrainLmCollapseConfig):
    tokenizer = config.data.the_tokenizer

    # Handle HF initialization
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    # Enable MuP on model if not already set
    if hasattr(config.model, "use_mup") and not config.model.use_mup:
        logger.info("Enabling MuP on model for Collapse training")
        config.model = dataclasses.replace(config.model, use_mup=True)

    levanter.initialize(config)

    # We need to know parameter count to compute budget, but we don't have the model yet.
    # Use the model config's total_trainable_params method if available.
    vocab_size = len(tokenizer)
    if hasattr(config.model, "total_trainable_params"):
        estimated_params = config.model.total_trainable_params(vocab_size)
    else:
        # Rough estimate for transformer: 12 * n_layers * hidden_dim^2
        estimated_params = 12 * config.model.num_layers * config.model.hidden_dim ** 2
        logger.warning(f"Using rough parameter estimate: {estimated_params:,}")

    # Compute training budget from Collapse constraints
    budget = config.compute_training_budget(estimated_params)

    # Log Collapse hyperparameters
    levanter.tracker.log_hyperparameters({
        **budget.to_hyperparameters(),
        "collapse/tpp": config.collapse.tpp,
        "collapse/tau": config.collapse.tau,
        "collapse/warmup_fraction": config.collapse.warmup_fraction,
        "collapse/cooldown_fraction": config.collapse.cooldown_fraction,
    })

    if config.completep is not None:
        levanter.tracker.log_hyperparameters({
            "completep/depth_multiplier": config.completep.depth_multiplier,
            "completep/depth_alpha_exp": config.completep.depth_alpha_exp,
            "completep/residual_scale": config.completep.residual_scale,
            "completep/depth_lr_scale": config.completep.depth_lr_scale,
        })

    logger.info(f"Collapse Training Budget:")
    logger.info(f"  Parameters: {budget.num_params:,}")
    logger.info(f"  Total tokens: {budget.total_tokens:,}")
    logger.info(f"  Total steps: {budget.total_steps:,}")
    logger.info(f"  Weight decay: {budget.weight_decay:.6f}")
    logger.info(f"  TPP: {budget.actual_tpp:.2f} (target: {config.collapse.tpp})")
    logger.info(f"  τ: {budget.actual_tau:.4f} (target: {config.collapse.tau})")

    # Override trainer num_train_steps with computed value
    config.trainer = dataclasses.replace(config.trainer, num_train_steps=budget.total_steps)

    # Build optimizer with Collapse-derived parameters
    optimizer = config.build_optimizer(budget)

    loss_function = functools.partial(compute_next_token_loss, logsumexp_weight=config.z_loss_weight)

    # Initialize Collapse tracker for diagnostics
    collapse_tracker = CollapseTracker()

    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos

        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        train_dataset = config.data.train_set(
            Pos,
            config.trainer.batch_schedule,
            key=data_key,
            epochs=config.epoch,
        )

        tagged_eval_datasets = config.data.tagged_eval_sets(Pos)

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        # Update budget with actual parameter count
        actual_params = parameter_count(state.model)
        if abs(actual_params - budget.num_params) / budget.num_params > 0.05:
            logger.warning(
                f"Actual parameter count ({actual_params:,}) differs from estimate ({budget.num_params:,}) by >5%. "
                "Consider re-running with accurate estimate for proper Collapse scaling."
            )

        if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
            state = load_checkpoint(state, config.initialize_from_checkpoint_path)

        if int(state.step) == 0:
            if config.initialize_from_hf:
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config=config.model if not config.use_hf_model_config else None,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        max_eval_examples_per_ds = config.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= config.trainer.eval_batch_size

        if len(tagged_eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch,
                tagged_eval_datasets,
                tokenizer,
                trainer.device_mesh,
                compute_axis_mapping,
                max_eval_examples_per_ds,
                mp=config.trainer.mp,
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        flops_per_token = config.model.flops_per_token(vocab_size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )

        # Add Collapse tracking callback
        def collapse_tracking_callback(step_info):
            t_hat = config.collapse.normalized_step(step_info.step, budget.total_steps)
            metrics = collapse_tracker.update(t_hat, float(step_info.loss))
            levanter.tracker.log({
                "collapse/t_hat": t_hat,
                "collapse/ema_loss": metrics.get("ema_loss", 0),
            }, step=step_info.step)

        trainer.add_hook(collapse_tracking_callback, every=10)

        if config.hf_save_path is not None and config.hf_save_steps is not None:
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            save_dtype: Optional[jnp.dtype] = None
            if config.hf_save_dtype is not None:
                try:
                    save_dtype = jnp.dtype(config.hf_save_dtype)
                except TypeError:
                    logger.warning(f"Invalid hf_save_dtype: {config.hf_save_dtype}. Defaulting to None.")

            trainer.add_hook(
                save_hf_checkpoint_callback(
                    full_save_path, converter, upload_to_hf=config.hf_upload or False, save_dtype=save_dtype
                ),
                every=config.hf_save_steps,
            )

        if config.eval_harness is not None:
            eval_harness = config.eval_harness
            trainer.add_hook(
                levanter.eval_harness.lm_eval_harness(
                    eval_harness, tokenizer, EvalBatch, compute_axis_mapping, trainer.mp
                ),
                every=config.eval_harness_steps,
            )

        train_loader = trainer.data_loader(train_dataset)
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        # Train!
        last_info = trainer.train(state, train_loader)

        # Check collapse health at end of training
        if not collapse_tracker.is_collapse_healthy():
            logger.warning("Collapse health check failed! Loss curve may have diverged from expected trajectory.")

        if trainer.config.checkpointer is not None and config.epoch > 0:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()

