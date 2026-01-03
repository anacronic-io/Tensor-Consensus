"""Google Research Football Environment Integration.

Provides integration with the Google Research Football environment
for strategic team play and coordination.
"""

from typing import Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np
import chex


class FootballEnvironment:
    """Google Research Football environment wrapper.

    Supports various football scenarios with 11v11 or smaller teams.
    """

    def __init__(
        self,
        scenario: str = "academy_3_vs_1_with_keeper",
        num_players: int = 3,
        batch_size: int = 16,
        representation: str = "simple115v2",
        rewards: str = "scoring,checkpoints",
        seed: Optional[int] = None,
    ):
        """Initialize Football environment.

        Args:
            scenario: Football scenario name
            num_players: Number of controlled players
            batch_size: Number of parallel environments
            representation: Observation representation type
            rewards: Reward type
            seed: Random seed
        """
        self.scenario = scenario
        self.num_players = num_players
        self.batch_size = batch_size
        self.representation = representation
        self.rewards = rewards
        self.seed = seed

        self._init_environments()

    def _init_environments(self):
        """Initialize batch of Football environments."""
        try:
            import gfootball.env as football_env
        except ImportError:
            raise ImportError(
                "Google Research Football not installed. "
                "Install with: pip install gfootball"
            )

        self.envs = []
        for i in range(self.batch_size):
            env = football_env.create_environment(
                env_name=self.scenario,
                representation=self.representation,
                rewards=self.rewards,
                number_of_left_players_agent_controls=self.num_players,
                stacked=False,
            )
            self.envs.append(env)

        # Get observation dimensions
        sample_obs = self.envs[0].reset()
        if isinstance(sample_obs, list):
            self.obs_dim = sample_obs[0].shape[0]
        else:
            self.obs_dim = sample_obs.shape[0]

        self.action_dim = 19  # Football has 19 discrete actions

    def reset(self) -> Tuple[chex.Array, Dict[str, Any]]:
        """Reset all environments.

        Returns:
            states: Agent observations [B, N, obs_dim]
            info: Additional information
        """
        all_obs = []

        for env in self.envs:
            obs = env.reset()

            # Convert to array [N, obs_dim]
            if isinstance(obs, list):
                obs_array = np.array(obs)
            else:
                obs_array = np.array([obs])  # Single agent case

            # Ensure correct shape
            if len(obs_array.shape) == 1:
                obs_array = obs_array[None, :]

            # Pad to num_players if needed
            if obs_array.shape[0] < self.num_players:
                padding = np.zeros(
                    (self.num_players - obs_array.shape[0], self.obs_dim)
                )
                obs_array = np.concatenate([obs_array, padding], axis=0)

            all_obs.append(obs_array)

        states = jnp.array(all_obs, dtype=jnp.float32)

        info = {
            'n_players': self.num_players,
            'obs_dim': self.obs_dim,
        }

        return states, info

    def step(
        self,
        actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Execute actions in all environments.

        Args:
            actions: Agent actions [B, N] (discrete action indices)

        Returns:
            next_states: Next observations [B, N, obs_dim]
            rewards: Rewards [B]
            dones: Done flags [B]
            info: Additional information
        """
        all_obs = []
        all_rewards = []
        all_dones = []
        all_scores = []

        # Convert to numpy
        if isinstance(actions, jnp.ndarray):
            actions = np.array(actions, dtype=np.int32)

        for i, env in enumerate(self.envs):
            # Get actions for this environment
            env_actions = actions[i].tolist()

            # Step environment
            obs, reward, done, info = env.step(env_actions)

            # Process observation
            if isinstance(obs, list):
                obs_array = np.array(obs)
            else:
                obs_array = np.array([obs])

            if len(obs_array.shape) == 1:
                obs_array = obs_array[None, :]

            if obs_array.shape[0] < self.num_players:
                padding = np.zeros(
                    (self.num_players - obs_array.shape[0], self.obs_dim)
                )
                obs_array = np.concatenate([obs_array, padding], axis=0)

            all_obs.append(obs_array)
            all_rewards.append(reward)
            all_dones.append(done)
            all_scores.append(info.get('score_reward', 0))

        next_states = jnp.array(all_obs, dtype=jnp.float32)
        rewards = jnp.array(all_rewards, dtype=jnp.float32)
        dones = jnp.array(all_dones, dtype=jnp.bool_)

        info = {
            'scores': jnp.array(all_scores, dtype=jnp.float32),
        }

        return next_states, rewards, dones, info

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class FootballTensorWrapper:
    """High-level wrapper for Football with Tensor-Consensus."""

    def __init__(
        self,
        scenario: str = "academy_3_vs_1_with_keeper",
        num_players: int = 3,
        batch_size: int = 16,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Football tensor wrapper."""
        self.env = FootballEnvironment(
            scenario=scenario,
            num_players=num_players,
            batch_size=batch_size
        )

        from ..operators.pipeline import TensorConsensusPipeline

        pipeline_config = pipeline_config or {}
        self.pipeline = TensorConsensusPipeline(**pipeline_config)

        self.params = None

    def init_params(self, rng: chex.PRNGKey):
        """Initialize pipeline parameters."""
        states, _ = self.env.reset()
        self.params = self.pipeline.init(rng, states)

    def collect_episode(
        self,
        params: Any,
        max_steps: int = 300,
    ) -> Dict[str, Any]:
        """Collect a full episode."""
        states, _ = self.env.reset()

        episode_data = {
            'states': [states],
            'actions': [],
            'rewards': [],
            'dones': [],
            'scores': [],
        }

        for step in range(max_steps):
            # Get consensus and convert to actions
            consensus_output, _ = self.pipeline.apply(params, states)

            # Simple action selection (argmax over action space)
            # In practice, would use proper policy network
            action_logits = consensus_output.squeeze(1)
            actions = jnp.argmax(action_logits[:, :, :self.env.action_dim], axis=-1)

            # Step
            next_states, rewards, dones, info = self.env.step(actions)

            episode_data['actions'].append(actions)
            episode_data['rewards'].append(rewards)
            episode_data['dones'].append(dones)
            episode_data['scores'].append(info['scores'])
            episode_data['states'].append(next_states)

            states = next_states

            if jnp.all(dones):
                break

        return episode_data
