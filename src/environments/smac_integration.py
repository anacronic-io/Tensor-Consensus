"""SMAC (StarCraft Multi-Agent Challenge) Environment Integration.

Provides integration with the SMAC benchmark for cooperative multi-agent
reinforcement learning in StarCraft II micromanagement scenarios.
"""

from typing import Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np
import chex


class SMACEnvironment:
    """SMAC environment wrapper for Tensor-Consensus.

    This wrapper converts SMAC observations and actions to/from
    batched tensor representations suitable for TPU execution.

    Attributes:
        map_name: SMAC map name (e.g., '3s_vs_5z', '27m_vs_30m')
        batch_size: Number of parallel environments
        state_dim: Dimension of agent state representation
        action_dim: Dimension of action space
    """

    def __init__(
        self,
        map_name: str = "3s_vs_5z",
        batch_size: int = 32,
        state_dim: int = 64,
        difficulty: str = "7",
        seed: Optional[int] = None,
    ):
        """Initialize SMAC environment.

        Args:
            map_name: SMAC scenario name
            batch_size: Number of parallel environments
            state_dim: State representation dimension
            difficulty: Difficulty level ("1" to "9")
            seed: Random seed
        """
        self.map_name = map_name
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.difficulty = difficulty
        self.seed = seed

        # Initialize environments (lazy import to avoid dependency issues)
        self._init_environments()

        # Get environment specs
        self.n_agents = self.envs[0].n_agents
        self.n_enemies = self.envs[0].n_enemies
        self.obs_dim = self.envs[0].get_obs_size()
        self.action_dim = self.envs[0].n_actions

    def _init_environments(self):
        """Initialize batch of SMAC environments."""
        try:
            from smac.env import StarCraft2Env
        except ImportError:
            raise ImportError(
                "SMAC not installed. Install with: pip install git+https://github.com/oxwhirl/smac.git"
            )

        self.envs = []
        for i in range(self.batch_size):
            env = StarCraft2Env(
                map_name=self.map_name,
                difficulty=self.difficulty,
                seed=self.seed + i if self.seed is not None else None,
            )
            self.envs.append(env)

    def reset(self) -> Tuple[chex.Array, Dict[str, Any]]:
        """Reset all environments.

        Returns:
            states: Agent states [B, N, state_dim]
            info: Additional information
        """
        all_obs = []
        all_states = []

        for env in self.envs:
            env.reset()
            obs = env.get_obs()
            state = env.get_state()

            all_obs.append(obs)
            all_states.append(state)

        # Convert to tensors
        states_tensor = self._process_observations(all_obs)

        info = {
            'raw_states': all_states,
            'n_agents': self.n_agents,
            'n_enemies': self.n_enemies,
        }

        return states_tensor, info

    def step(
        self,
        actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Execute actions in all environments.

        Args:
            actions: Agent actions [B, N] (discrete action indices)

        Returns:
            next_states: Next agent states [B, N, state_dim]
            rewards: Rewards [B]
            dones: Done flags [B]
            info: Additional information
        """
        all_obs = []
        all_rewards = []
        all_dones = []
        all_info = []

        # Convert JAX array to numpy if needed
        if isinstance(actions, jnp.ndarray):
            actions = np.array(actions)

        for i, env in enumerate(self.envs):
            env_actions = actions[i].tolist() if len(actions.shape) > 1 else actions.tolist()

            reward, done, info = env.step(env_actions)
            obs = env.get_obs()

            all_obs.append(obs)
            all_rewards.append(reward)
            all_dones.append(done)
            all_info.append(info)

        # Convert to tensors
        next_states = self._process_observations(all_obs)
        rewards = jnp.array(all_rewards, dtype=jnp.float32)
        dones = jnp.array(all_dones, dtype=jnp.bool_)

        info = {
            'battle_won': jnp.array([i.get('battle_won', False) for i in all_info]),
            'dead_allies': jnp.array([i.get('dead_allies', 0) for i in all_info]),
            'dead_enemies': jnp.array([i.get('dead_enemies', 0) for i in all_info]),
        }

        return next_states, rewards, dones, info

    def _process_observations(self, observations: list) -> chex.Array:
        """Process raw observations into tensor format.

        Args:
            observations: List of observations from each environment

        Returns:
            Processed states [B, N, state_dim]
        """
        # Convert list of agent observations to tensor
        batch_obs = []

        for env_obs in observations:
            # env_obs is a list of observations, one per agent
            if isinstance(env_obs, list):
                agent_obs = np.array(env_obs, dtype=np.float32)
            else:
                agent_obs = env_obs

            # Pad or project to target state_dim
            if agent_obs.shape[-1] < self.state_dim:
                padding = np.zeros((agent_obs.shape[0], self.state_dim - agent_obs.shape[-1]))
                agent_obs = np.concatenate([agent_obs, padding], axis=-1)
            elif agent_obs.shape[-1] > self.state_dim:
                agent_obs = agent_obs[:, :self.state_dim]

            batch_obs.append(agent_obs)

        return jnp.array(batch_obs, dtype=jnp.float32)

    def get_avail_actions(self) -> chex.Array:
        """Get available actions for each agent.

        Returns:
            Available actions mask [B, N, action_dim]
        """
        all_avail = []

        for env in self.envs:
            avail_actions = env.get_avail_actions()
            all_avail.append(avail_actions)

        return jnp.array(all_avail, dtype=jnp.float32)

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def get_stats(self) -> Dict[str, float]:
        """Get environment statistics.

        Returns:
            Statistics dictionary
        """
        stats = {}

        for i, env in enumerate(self.envs):
            env_stats = env.get_stats()
            for key, value in env_stats.items():
                if key not in stats:
                    stats[key] = []
                stats[key].append(value)

        # Average across environments
        return {key: np.mean(values) for key, values in stats.items()}


class SMACTensorWrapper:
    """High-level wrapper that integrates SMAC with Tensor-Consensus pipeline.

    This provides a complete interface for training and evaluation.
    """

    def __init__(
        self,
        map_name: str = "3s_vs_5z",
        batch_size: int = 32,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SMAC tensor wrapper.

        Args:
            map_name: SMAC scenario name
            batch_size: Number of parallel environments
            pipeline_config: Configuration for Tensor-Consensus pipeline
        """
        self.env = SMACEnvironment(map_name=map_name, batch_size=batch_size)

        # Import pipeline
        from ..operators.pipeline import TensorConsensusPipeline

        pipeline_config = pipeline_config or {}
        self.pipeline = TensorConsensusPipeline(**pipeline_config)

        # Initialize pipeline parameters
        self.params = None
        self.optimizer = None

    def init_params(self, rng: chex.PRNGKey):
        """Initialize pipeline parameters.

        Args:
            rng: Random key
        """
        states, _ = self.env.reset()
        self.params = self.pipeline.init(rng, states)

    def select_actions(
        self,
        params: Any,
        states: chex.Array,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        """Select actions using Tensor-Consensus pipeline.

        Args:
            params: Pipeline parameters
            states: Agent states [B, N, d]
            deterministic: Whether to use deterministic policy

        Returns:
            actions: Selected actions [B, N]
            metrics: Pipeline metrics
        """
        # Forward pass through pipeline
        consensus_output, metrics = self.pipeline.apply(params, states)

        # Convert consensus output to action logits
        # This is a simple projection - in practice, would use policy network
        action_logits = consensus_output.squeeze(1)  # [B, action_dim]

        if deterministic:
            actions = jnp.argmax(action_logits, axis=-1)
        else:
            # Sample from categorical distribution
            actions = jax.random.categorical(
                jax.random.PRNGKey(0),  # In practice, use proper RNG
                action_logits,
                axis=-1
            )

        return actions, metrics

    def collect_episode(
        self,
        params: Any,
        max_steps: int = 200,
    ) -> Dict[str, Any]:
        """Collect a full episode of experience.

        Args:
            params: Pipeline parameters
            max_steps: Maximum steps per episode

        Returns:
            Episode data including states, actions, rewards, metrics
        """
        states, info = self.env.reset()

        episode_data = {
            'states': [states],
            'actions': [],
            'rewards': [],
            'dones': [],
            'metrics': [],
        }

        for step in range(max_steps):
            # Select actions
            actions, metrics = self.select_actions(params, states)

            # Environment step
            next_states, rewards, dones, step_info = self.env.step(actions)

            # Store data
            episode_data['actions'].append(actions)
            episode_data['rewards'].append(rewards)
            episode_data['dones'].append(dones)
            episode_data['metrics'].append(metrics)
            episode_data['states'].append(next_states)

            states = next_states

            # Check if all environments are done
            if jnp.all(dones):
                break

        return episode_data
