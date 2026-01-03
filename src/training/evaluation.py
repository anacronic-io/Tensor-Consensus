"""Evaluation utilities for Tensor-Consensus models."""

from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import chex


class Evaluator:
    """Model evaluator for benchmarking."""

    def __init__(self, env, model, params):
        """Initialize evaluator.

        Args:
            env: Environment instance
            model: Model instance
            params: Model parameters
        """
        self.env = env
        self.model = model
        self.params = params

    def evaluate(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy

        Returns:
            Evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        win_rates = []

        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            states, _ = self.env.reset()

            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < 500:
                # Get actions from model
                output, _ = self.model.apply(
                    {'params': self.params},
                    states,
                    training=False
                )

                # Convert to discrete actions (simplified)
                actions = jnp.argmax(output.squeeze(1), axis=-1)

                # Step environment
                next_states, rewards, dones, info = self.env.step(actions)

                episode_reward += jnp.mean(rewards)
                episode_length += 1
                done = jnp.all(dones)

                states = next_states

            episode_returns.append(float(episode_reward))
            episode_lengths.append(episode_length)

            # Track wins if available
            if hasattr(info, 'get') and 'battle_won' in info:
                win_rates.append(float(jnp.mean(info['battle_won'])))

        # Compute statistics
        metrics = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }

        if win_rates:
            metrics['win_rate'] = np.mean(win_rates)

        return metrics


def evaluate_model(
    model,
    params,
    env_name: str,
    scenario: str,
    num_episodes: int = 100,
) -> Dict[str, float]:
    """Convenience function for model evaluation.

    Args:
        model: Model instance
        params: Model parameters
        env_name: Environment name
        scenario: Scenario name
        num_episodes: Number of episodes

    Returns:
        Evaluation metrics
    """
    # Create environment
    if env_name == "smac":
        from ..environments.smac_integration import SMACEnvironment
        env = SMACEnvironment(map_name=scenario, batch_size=16)
    elif env_name == "mpe":
        from ..environments.mpe_integration import MPEEnvironment
        env = MPEEnvironment(scenario=scenario, batch_size=16)
    elif env_name == "football":
        from ..environments.football_integration import FootballEnvironment
        env = FootballEnvironment(scenario=scenario, batch_size=16)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Evaluate
    evaluator = Evaluator(env, model, params)
    metrics = evaluator.evaluate(num_episodes=num_episodes)

    env.close()

    return metrics
