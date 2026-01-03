"""MPE (Multi-Particle Environment) Integration.

Provides integration with PettingZoo's MPE for cooperative and competitive
multi-agent scenarios.
"""

from typing import Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np
import chex


class MPEEnvironment:
    """MPE environment wrapper for Tensor-Consensus.

    Supports various MPE scenarios:
    - simple_spread: Cooperative navigation
    - simple_adversary: Competitive pursuit-evasion
    - simple_tag: Tag game
    - simple_push: Object manipulation
    """

    def __init__(
        self,
        scenario: str = "simple_spread",
        n_agents: int = 5,
        batch_size: int = 32,
        max_cycles: int = 25,
        continuous_actions: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize MPE environment.

        Args:
            scenario: MPE scenario name
            n_agents: Number of agents
            batch_size: Number of parallel environments
            max_cycles: Maximum cycles per episode
            continuous_actions: Whether to use continuous action space
            seed: Random seed
        """
        self.scenario = scenario
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.seed = seed

        self._init_environments()

    def _init_environments(self):
        """Initialize batch of MPE environments."""
        try:
            from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3
        except ImportError:
            raise ImportError(
                "PettingZoo not installed. Install with: pip install pettingzoo[mpe]"
            )

        # Map scenario names to constructors
        scenario_map = {
            'simple_spread': simple_spread_v3,
            'simple_adversary': simple_adversary_v3,
        }

        if self.scenario not in scenario_map:
            raise ValueError(f"Unknown MPE scenario: {self.scenario}")

        env_fn = scenario_map[self.scenario]

        self.envs = []
        for i in range(self.batch_size):
            env = env_fn.parallel_env(
                N=self.n_agents,
                max_cycles=self.max_cycles,
                continuous_actions=self.continuous_actions,
            )
            if self.seed is not None:
                env.reset(seed=self.seed + i)
            else:
                env.reset()
            self.envs.append(env)

        # Get specs from first environment
        sample_env = self.envs[0]
        agent_name = sample_env.possible_agents[0]
        self.obs_dim = sample_env.observation_space(agent_name).shape[0]
        self.action_dim = (
            sample_env.action_space(agent_name).shape[0]
            if self.continuous_actions
            else sample_env.action_space(agent_name).n
        )

    def reset(self) -> Tuple[chex.Array, Dict[str, Any]]:
        """Reset all environments.

        Returns:
            states: Agent observations [B, N, obs_dim]
            info: Additional information
        """
        all_obs = []

        for env in self.envs:
            obs_dict, _ = env.reset()

            # Convert dict to array [N, obs_dim]
            obs_array = np.array([
                obs_dict[agent] for agent in env.possible_agents
            ])
            all_obs.append(obs_array)

        states = jnp.array(all_obs, dtype=jnp.float32)

        info = {
            'n_agents': self.n_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }

        return states, info

    def step(
        self,
        actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Execute actions in all environments.

        Args:
            actions: Agent actions [B, N, action_dim] or [B, N] for discrete

        Returns:
            next_states: Next observations [B, N, obs_dim]
            rewards: Rewards [B, N]
            dones: Done flags [B, N]
            info: Additional information
        """
        all_obs = []
        all_rewards = []
        all_dones = []

        # Convert to numpy
        if isinstance(actions, jnp.ndarray):
            actions = np.array(actions)

        for i, env in enumerate(self.envs):
            # Convert actions array to dict
            action_dict = {
                agent: actions[i, j]
                for j, agent in enumerate(env.possible_agents)
            }

            # Step environment
            obs_dict, reward_dict, done_dict, trunc_dict, info_dict = env.step(action_dict)

            # Convert to arrays
            obs_array = np.array([
                obs_dict[agent] for agent in env.possible_agents
            ])
            reward_array = np.array([
                reward_dict[agent] for agent in env.possible_agents
            ])
            done_array = np.array([
                done_dict[agent] or trunc_dict[agent]
                for agent in env.possible_agents
            ])

            all_obs.append(obs_array)
            all_rewards.append(reward_array)
            all_dones.append(done_array)

        next_states = jnp.array(all_obs, dtype=jnp.float32)
        rewards = jnp.array(all_rewards, dtype=jnp.float32)
        dones = jnp.array(all_dones, dtype=jnp.bool_)

        info = {}

        return next_states, rewards, dones, info

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class MPETensorWrapper:
    """High-level wrapper for MPE with Tensor-Consensus."""

    def __init__(
        self,
        scenario: str = "simple_spread",
        n_agents: int = 5,
        batch_size: int = 32,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MPE tensor wrapper."""
        self.env = MPEEnvironment(
            scenario=scenario,
            n_agents=n_agents,
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
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """Collect a full episode."""
        states, _ = self.env.reset()

        episode_data = {
            'states': [states],
            'actions': [],
            'rewards': [],
            'dones': [],
        }

        for step in range(max_steps):
            # Get consensus actions
            consensus_output, _ = self.pipeline.apply(params, states)

            # Convert to actions (simplified)
            if self.env.continuous_actions:
                actions = consensus_output.squeeze(1)[:, :self.env.n_agents, :self.env.action_dim]
            else:
                actions = jnp.argmax(consensus_output.squeeze(1), axis=-1)

            # Step
            next_states, rewards, dones, _ = self.env.step(actions)

            episode_data['actions'].append(actions)
            episode_data['rewards'].append(rewards)
            episode_data['dones'].append(dones)
            episode_data['states'].append(next_states)

            states = next_states

            if jnp.all(dones):
                break

        return episode_data
