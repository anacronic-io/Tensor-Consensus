"""Training script for Tensor-Consensus models.

Implements training loop with support for multiple environments
and baselines for comparison.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import chex
from tqdm import tqdm
import wandb


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Environment
    env_name: str = "smac"
    scenario: str = "3s_vs_5z"
    n_agents: int = 100

    # Training
    num_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps_per_episode: int = 200

    # Model
    model_type: str = "tensor_consensus"  # or "transformer", "mamba"
    d_model: int = 64
    complexity_dim: int = 8
    consensus_sigma: float = 1.0

    # Optimization
    optimizer: str = "adam"
    grad_clip: float = 1.0
    weight_decay: float = 1e-5

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    use_wandb: bool = False
    wandb_project: str = "tensor-consensus"

    # Hardware
    device: str = "tpu"  # or "gpu", "cpu"
    seed: int = 42


class TrainState(train_state.TrainState):
    """Extended train state with additional metrics."""

    batch_stats: Optional[Any] = None
    epoch: int = 0


class Trainer:
    """Main trainer class for Tensor-Consensus models."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize environment
        self.env = self._create_environment()

        # Initialize model
        self.model = self._create_model()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize train state
        self.state = None

        # Initialize logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=vars(config)
            )

    def _create_environment(self):
        """Create training environment."""
        if self.config.env_name == "smac":
            from ..environments.smac_integration import SMACEnvironment
            return SMACEnvironment(
                map_name=self.config.scenario,
                batch_size=self.config.batch_size,
            )
        elif self.config.env_name == "mpe":
            from ..environments.mpe_integration import MPEEnvironment
            return MPEEnvironment(
                scenario=self.config.scenario,
                n_agents=self.config.n_agents,
                batch_size=self.config.batch_size,
            )
        elif self.config.env_name == "football":
            from ..environments.football_integration import FootballEnvironment
            return FootballEnvironment(
                scenario=self.config.scenario,
                num_players=self.config.n_agents,
                batch_size=self.config.batch_size,
            )
        else:
            raise ValueError(f"Unknown environment: {self.config.env_name}")

    def _create_model(self):
        """Create model based on configuration."""
        if self.config.model_type == "tensor_consensus":
            from ..operators.pipeline import TensorConsensusPipeline
            return TensorConsensusPipeline(
                complexity_dim=self.config.complexity_dim,
                embedding_dim=self.config.d_model,
                consensus_sigma=self.config.consensus_sigma,
            )
        elif self.config.model_type == "transformer":
            from ..models.transformer import MultiAgentTransformer
            return MultiAgentTransformer(
                d_model=self.config.d_model,
                num_layers=4,
            )
        elif self.config.model_type == "mamba":
            from ..models.mamba_ssm import MambaSSM
            return MambaSSM(
                d_model=self.config.d_model,
                num_layers=4,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _create_optimizer(self):
        """Create optimizer."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=100,
            decay_steps=self.config.num_epochs * 100,
            end_value=self.config.learning_rate * 0.1,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(learning_rate=schedule, weight_decay=self.config.weight_decay),
        )

        return optimizer

    def init_train_state(self, rng: chex.PRNGKey) -> TrainState:
        """Initialize training state.

        Args:
            rng: Random key

        Returns:
            Initialized train state
        """
        # Get sample input
        states, _ = self.env.reset()

        # Initialize model
        variables = self.model.init(rng, states)

        # Create train state
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.optimizer,
            batch_stats=variables.get('batch_stats'),
            epoch=0,
        )

        return state

    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, chex.Array],
        rng: chex.PRNGKey,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Single training step.

        Args:
            state: Current train state
            batch: Batch of data
            rng: Random key

        Returns:
            Updated state and metrics
        """
        def loss_fn(params):
            # Forward pass
            if state.batch_stats is not None:
                variables = {'params': params, 'batch_stats': state.batch_stats}
                (output, metrics), updates = state.apply_fn(
                    variables,
                    batch['states'],
                    training=True,
                    mutable=['batch_stats']
                )
                new_batch_stats = updates['batch_stats']
            else:
                output, metrics = state.apply_fn(
                    {'params': params},
                    batch['states'],
                    training=True,
                )
                new_batch_stats = None

            # Compute loss (simplified - in practice would use proper RL loss)
            # For now, use mean squared error to target rewards
            if 'target_values' in batch:
                loss = jnp.mean((output - batch['target_values']) ** 2)
            else:
                # Simple exploration loss
                loss = -jnp.mean(jnp.sum(output ** 2, axis=-1))

            return loss, (metrics, new_batch_stats)

        # Compute gradients
        (loss, (metrics, new_batch_stats)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        # Update parameters
        state = state.apply_gradients(grads=grads)

        if new_batch_stats is not None:
            state = state.replace(batch_stats=new_batch_stats)

        # Compile metrics
        train_metrics = {
            'loss': loss,
            'grad_norm': optax.global_norm(grads),
        }

        if metrics:
            # Add pipeline metrics if available
            if 'projection' in metrics and metrics['projection']:
                train_metrics.update({
                    f'projection/{k}': v for k, v in metrics['projection'].items()
                })

        return state, train_metrics

    def collect_batch(self, state: TrainState) -> Dict[str, chex.Array]:
        """Collect a batch of experience.

        Args:
            state: Current train state

        Returns:
            Batch dictionary
        """
        env_states, _ = self.env.reset()

        batch = {
            'states': env_states,
        }

        # In a full implementation, would collect trajectories
        # and compute returns for RL training

        return batch

    def train(self):
        """Main training loop."""
        # Initialize
        rng = jax.random.PRNGKey(self.config.seed)
        rng, init_rng = jax.random.split(rng)

        state = self.init_train_state(init_rng)

        # Training loop
        pbar = tqdm(range(self.config.num_epochs), desc="Training")

        for epoch in pbar:
            rng, step_rng = jax.random.split(rng)

            # Collect batch
            batch = self.collect_batch(state)

            # Training step
            state, metrics = self.train_step(state, batch, step_rng)

            # Update progress bar
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})

            # Logging
            if epoch % self.config.log_interval == 0:
                if self.config.use_wandb:
                    wandb.log({'epoch': epoch, **metrics})

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(state, epoch)

            state = state.replace(epoch=epoch)

        return state

    def save_checkpoint(self, state: TrainState, epoch: int):
        """Save checkpoint.

        Args:
            state: Train state
            epoch: Current epoch
        """
        # In practice, would use orbax or similar for checkpointing
        pass


def create_trainer(config_dict: Dict[str, Any]) -> Trainer:
    """Factory function to create trainer from config dict.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configured trainer
    """
    config = TrainingConfig(**config_dict)
    return Trainer(config)
