"""Adaptive Resource Projection Operator.

This module implements the adaptive projection operator that allocates computational
resources based on task complexity. The operator is fully differentiable and optimized
for TPU execution with O(N) complexity.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex


@jax.vmap  # Parallelize over batches
@jax.vmap  # Parallelize over agents
def adaptive_projection(
    state: chex.Array,
    Wc: chex.Array,
    bc: chex.Array,
    resource_budget: chex.Array,
    epsilon: float = 1e-8,
) -> chex.Array:
    """Adaptive projection for dynamic resource allocation.

    Args:
        state: Agent state [d]
        Wc: Complexity projection weights [d, k]
        bc: Complexity bias [k]
        resource_budget: Maximum resources per complexity [k]
        epsilon: Small constant for numerical stability

    Returns:
        Allocated resources [k]
    """
    # Normalize state
    Q = jax.nn.standardize(state, epsilon=epsilon)

    # Project to complexity space with temperature scaling
    logits = (Q @ Wc + bc) / jnp.sqrt(state.shape[-1])

    # Softmax to get complexity distribution
    complexities = jax.nn.softmax(logits)

    # Allocate resources proportionally
    resources = complexities * resource_budget

    return resources


class AdaptiveProjection(nn.Module):
    """Flax module for adaptive resource projection.

    Attributes:
        complexity_dim: Number of complexity levels (k)
        hidden_dim: Hidden dimension for projection
        resource_budget: Maximum resources per complexity level
    """

    complexity_dim: int = 8
    hidden_dim: Optional[int] = None
    resource_budget: float = 100.0
    epsilon: float = 1e-8

    @nn.compact
    def __call__(self, states: chex.Array) -> Tuple[chex.Array, dict]:
        """Forward pass of adaptive projection.

        Args:
            states: Agent states [B, N, d] (batch, agents, dimension)

        Returns:
            resources: Allocated resources [B, N, k]
            metrics: Dictionary with diagnostic metrics
        """
        batch_size, n_agents, state_dim = states.shape
        hidden_dim = self.hidden_dim or state_dim

        # Learnable complexity projection
        Wc = self.param(
            'complexity_weights',
            nn.initializers.lecun_normal(),
            (state_dim, self.complexity_dim)
        )
        bc = self.param(
            'complexity_bias',
            nn.initializers.zeros,
            (self.complexity_dim,)
        )

        # Resource budget (learnable or fixed)
        R_max = self.param(
            'resource_budget',
            lambda key, shape: jnp.full(shape, self.resource_budget),
            (self.complexity_dim,)
        )

        # Apply projection (vmapped over batches and agents)
        resources = jax.vmap(jax.vmap(adaptive_projection, in_axes=(0, None, None, None, None)),
                            in_axes=(0, None, None, None, None))(
            states, Wc, bc, R_max, self.epsilon
        )

        # Compute diagnostics
        metrics = {
            'mean_resource': jnp.mean(jnp.sum(resources, axis=-1)),
            'std_resource': jnp.std(jnp.sum(resources, axis=-1)),
            'complexity_entropy': jnp.mean(
                -jnp.sum(resources / (jnp.sum(resources, axis=-1, keepdims=True) + self.epsilon)
                        * jnp.log(resources / (jnp.sum(resources, axis=-1, keepdims=True) + self.epsilon) + self.epsilon),
                        axis=-1)
            ),
        }

        return resources, metrics


def project_with_attention(
    states: chex.Array,
    query_proj: chex.Array,
    key_proj: chex.Array,
    value_proj: chex.Array,
    num_heads: int = 4,
    epsilon: float = 1e-8,
) -> chex.Array:
    """Multi-head attention-based projection (alternative approach).

    Args:
        states: Agent states [B, N, d]
        query_proj: Query projection weights [d, d_k]
        key_proj: Key projection weights [d, d_k]
        value_proj: Value projection weights [d, d_v]
        num_heads: Number of attention heads
        epsilon: Small constant for numerical stability

    Returns:
        Projected states [B, N, d_v]
    """
    B, N, d = states.shape
    d_k = query_proj.shape[-1] // num_heads

    # Project to Q, K, V
    Q = jnp.reshape(states @ query_proj, (B, N, num_heads, d_k))
    K = jnp.reshape(states @ key_proj, (B, N, num_heads, d_k))
    V = jnp.reshape(states @ value_proj, (B, N, num_heads, -1))

    # Compute attention scores
    scores = jnp.einsum('bnhd,bmhd->bhnm', Q, K) / jnp.sqrt(d_k)
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention
    out = jnp.einsum('bhnm,bmhd->bnhd', attn_weights, V)
    out = jnp.reshape(out, (B, N, -1))

    return out


class HierarchicalProjection(nn.Module):
    """Hierarchical projection for very large agent counts (N > 1000).

    This splits agents into groups and applies projection hierarchically
    to maintain O(N) complexity even for very large systems.

    Attributes:
        group_size: Number of agents per group
        complexity_dim: Number of complexity levels
        num_levels: Number of hierarchical levels
    """

    group_size: int = 100
    complexity_dim: int = 8
    num_levels: int = 2

    @nn.compact
    def __call__(self, states: chex.Array) -> Tuple[chex.Array, dict]:
        """Hierarchical projection forward pass.

        Args:
            states: Agent states [B, N, d]

        Returns:
            resources: Allocated resources [B, N, k]
            metrics: Diagnostic metrics
        """
        B, N, d = states.shape

        # Reshape into groups
        n_groups = (N + self.group_size - 1) // self.group_size
        padded_n = n_groups * self.group_size

        # Pad if necessary
        if padded_n > N:
            padding = jnp.zeros((B, padded_n - N, d))
            states_padded = jnp.concatenate([states, padding], axis=1)
        else:
            states_padded = states

        # Reshape to groups
        states_grouped = jnp.reshape(states_padded, (B, n_groups, self.group_size, d))

        # Apply projection at group level
        group_proj = AdaptiveProjection(
            complexity_dim=self.complexity_dim,
            name='group_projection'
        )

        resources_list = []
        all_metrics = []

        for level in range(self.num_levels):
            if level == 0:
                # First level: project within groups
                resources_grouped = jax.vmap(
                    lambda g: group_proj(g[None, ...])[0][0]
                )(states_grouped.reshape(-1, self.group_size, d))
                resources_grouped = resources_grouped.reshape(B, n_groups, self.group_size, -1)
            else:
                # Higher levels: aggregate group representations
                group_repr = jnp.mean(states_grouped, axis=2)  # [B, n_groups, d]
                global_resources, metrics = group_proj(group_repr)
                all_metrics.append(metrics)

                # Broadcast back to agents
                resources_grouped = jnp.repeat(
                    global_resources[:, :, None, :],
                    self.group_size,
                    axis=2
                )

            resources_list.append(resources_grouped)

        # Combine resources from all levels
        final_resources = jnp.mean(jnp.stack(resources_list, axis=0), axis=0)

        # Remove padding
        final_resources = final_resources.reshape(B, padded_n, -1)[:, :N, :]

        metrics = {
            'num_groups': n_groups,
            'group_size': self.group_size,
        }
        if all_metrics:
            metrics.update(all_metrics[0])

        return final_resources, metrics
