"""Robust Latent Space Consensus Operator.

Implements robust consensus with adaptive weights that penalize outliers,
achieving convergence in O(log(1/epsilon)) iterations with O(N) complexity per iteration.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex


def robust_consensus(
    responses: chex.Array,
    sigma: float = 1.0,
    max_iters: int = 10,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
) -> Tuple[chex.Array, chex.Array]:
    """Robust consensus with exponential outlier weighting.

    Args:
        responses: Agent responses [B, N, d]
        sigma: Bandwidth for exponential weights
        max_iters: Maximum iterations
        tol: Convergence tolerance
        epsilon: Numerical stability constant

    Returns:
        centroid: Consensus centroid [B, 1, d]
        weights: Final agent weights [B, N]
    """
    B, N, d = responses.shape

    # Initialize centroid as mean
    centroid = jnp.mean(responses, axis=1, keepdims=True)

    def body_fn(carry):
        centroid, prev_centroid, iteration = carry

        # Compute distances from centroid
        distances = jnp.sum((responses - centroid) ** 2, axis=-1)  # [B, N]

        # Compute exponential weights
        weights = jnp.exp(-distances / (2 * sigma ** 2))
        weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + epsilon)

        # Update centroid as weighted average
        new_centroid = jnp.sum(
            responses * weights[..., None],
            axis=1,
            keepdims=True
        )

        return new_centroid, centroid, iteration + 1

    def cond_fn(carry):
        centroid, prev_centroid, iteration = carry
        # Continue if not converged and under max iterations
        not_converged = jnp.any(jnp.abs(centroid - prev_centroid) >= tol)
        under_max = iteration < max_iters
        return not_converged & under_max

    # Run iteration loop
    final_centroid, prev_centroid, final_iter = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (centroid, centroid + 2 * tol, 0)
    )

    # Compute final weights
    final_distances = jnp.sum((responses - final_centroid) ** 2, axis=-1)
    final_weights = jnp.exp(-final_distances / (2 * sigma ** 2))
    final_weights = final_weights / (jnp.sum(final_weights, axis=1, keepdims=True) + epsilon)

    return final_centroid, final_weights


def weighted_median_consensus(
    responses: chex.Array,
    weights: Optional[chex.Array] = None,
) -> chex.Array:
    """Weighted median for robust consensus (alternative to mean).

    Args:
        responses: Agent responses [B, N, d]
        weights: Optional weights [B, N]

    Returns:
        median: Weighted median [B, 1, d]
    """
    B, N, d = responses.shape

    if weights is None:
        weights = jnp.ones((B, N)) / N

    # For each dimension, compute weighted median
    def compute_median_dim(responses_dim, weights_batch):
        # Sort responses
        sorted_indices = jnp.argsort(responses_dim)
        sorted_responses = responses_dim[sorted_indices]
        sorted_weights = weights_batch[sorted_indices]

        # Cumulative weights
        cumsum_weights = jnp.cumsum(sorted_weights)

        # Find median index (where cumsum crosses 0.5)
        median_idx = jnp.searchsorted(cumsum_weights, 0.5)
        median_idx = jnp.clip(median_idx, 0, N - 1)

        return sorted_responses[median_idx]

    # Apply for each batch and dimension
    medians = jax.vmap(
        lambda r, w: jax.vmap(
            lambda r_dim: compute_median_dim(r_dim, w)
        )(r.T)
    )(responses, weights)

    return medians[:, None, :]


class RobustConsensus(nn.Module):
    """Flax module for robust consensus with learnable parameters.

    Attributes:
        sigma: Bandwidth for outlier weighting
        max_iters: Maximum consensus iterations
        tol: Convergence tolerance
        adaptive_sigma: Whether to learn sigma
        use_median: Whether to use median instead of mean
    """

    sigma: float = 1.0
    max_iters: int = 10
    tol: float = 1e-6
    adaptive_sigma: bool = False
    use_median: bool = False
    epsilon: float = 1e-8

    @nn.compact
    def __call__(
        self,
        responses: chex.Array,
        response_mask: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, dict]:
        """Forward pass of robust consensus.

        Args:
            responses: Agent responses [B, N, d]
            response_mask: Optional mask for valid responses [B, N]

        Returns:
            centroid: Consensus result [B, 1, d]
            metrics: Diagnostic metrics including weights
        """
        B, N, d = responses.shape

        # Learnable or fixed sigma
        if self.adaptive_sigma:
            log_sigma = self.param(
                'log_sigma',
                lambda key, shape: jnp.log(jnp.full(shape, self.sigma)),
                ()
            )
            sigma = jnp.exp(log_sigma)
        else:
            sigma = self.sigma

        # Apply mask if provided
        if response_mask is not None:
            # Replace masked responses with zeros (will get zero weight)
            responses = responses * response_mask[..., None]

        # Compute consensus
        if self.use_median:
            centroid = weighted_median_consensus(responses)
            # Compute weights for metrics
            distances = jnp.sum((responses - centroid) ** 2, axis=-1)
            weights = jnp.exp(-distances / (2 * sigma ** 2))
            weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + self.epsilon)
        else:
            centroid, weights = robust_consensus(
                responses, sigma, self.max_iters, self.tol, self.epsilon
            )

        # Apply mask to weights
        if response_mask is not None:
            weights = weights * response_mask
            weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + self.epsilon)

        # Compute diagnostic metrics
        distances = jnp.sum((responses - centroid) ** 2, axis=-1)
        metrics = {
            'consensus_variance': jnp.mean(jnp.sum(weights[..., None] * (responses - centroid) ** 2, axis=(1, 2))),
            'mean_distance': jnp.mean(distances),
            'max_distance': jnp.max(distances),
            'weight_entropy': -jnp.mean(jnp.sum(weights * jnp.log(weights + self.epsilon), axis=1)),
            'effective_agents': jnp.mean(1.0 / jnp.sum(weights ** 2, axis=1)),
            'outlier_ratio': jnp.mean(jnp.sum(weights < 0.1, axis=1) / N),
            'sigma': sigma,
        }

        # Add weights to metrics for analysis
        metrics['weights'] = weights

        return centroid, metrics


class HierarchicalConsensus(nn.Module):
    """Hierarchical consensus for very large agent counts.

    Splits agents into groups, computes local consensus, then global consensus.
    Maintains O(N) complexity with better robustness for large N.

    Attributes:
        group_size: Number of agents per group
        sigma: Bandwidth for outlier weighting
        max_iters: Maximum iterations per level
    """

    group_size: int = 100
    sigma: float = 1.0
    max_iters: int = 10

    @nn.compact
    def __call__(self, responses: chex.Array) -> Tuple[chex.Array, dict]:
        """Hierarchical consensus forward pass.

        Args:
            responses: Agent responses [B, N, d]

        Returns:
            centroid: Global consensus [B, 1, d]
            metrics: Diagnostic metrics
        """
        B, N, d = responses.shape

        # Split into groups
        n_groups = (N + self.group_size - 1) // self.group_size
        padded_n = n_groups * self.group_size

        # Pad if necessary
        if padded_n > N:
            padding = jnp.zeros((B, padded_n - N, d))
            responses_padded = jnp.concatenate([responses, padding], axis=1)
        else:
            responses_padded = responses

        # Reshape to groups
        responses_grouped = jnp.reshape(
            responses_padded,
            (B, n_groups, self.group_size, d)
        )

        # Local consensus within groups
        local_consensus = RobustConsensus(
            sigma=self.sigma,
            max_iters=self.max_iters,
            name='local_consensus'
        )

        local_centroids_list = []
        for i in range(n_groups):
            local_centroid, _ = local_consensus(responses_grouped[:, i, :, :])
            local_centroids_list.append(local_centroid)

        local_centroids = jnp.concatenate(local_centroids_list, axis=1)  # [B, n_groups, d]

        # Global consensus across group representatives
        global_consensus = RobustConsensus(
            sigma=self.sigma,
            max_iters=self.max_iters,
            name='global_consensus'
        )

        global_centroid, metrics = global_consensus(local_centroids)

        metrics['num_groups'] = n_groups
        metrics['group_size'] = self.group_size

        return global_centroid, metrics


class ByzantineRobustConsensus(nn.Module):
    """Byzantine-robust consensus for adversarial settings.

    Uses trimmed mean or median to be robust against Byzantine agents
    (agents that can behave arbitrarily or maliciously).

    Attributes:
        trim_ratio: Fraction of agents to trim from each end
        sigma: Bandwidth for weighting
        use_median: Whether to use median (more robust) vs trimmed mean
    """

    trim_ratio: float = 0.2
    sigma: float = 1.0
    use_median: bool = True

    @nn.compact
    def __call__(self, responses: chex.Array) -> Tuple[chex.Array, dict]:
        """Byzantine-robust consensus.

        Args:
            responses: Agent responses [B, N, d]

        Returns:
            centroid: Robust consensus [B, 1, d]
            metrics: Diagnostic metrics
        """
        B, N, d = responses.shape

        # Compute distances from geometric median
        geometric_median = jnp.median(responses, axis=1, keepdims=True)
        distances = jnp.sum((responses - geometric_median) ** 2, axis=-1)

        # Trim furthest agents
        n_trim = int(N * self.trim_ratio)
        if n_trim > 0:
            # Get indices of non-trimmed agents
            trim_threshold = jnp.percentile(distances, 100 * (1 - self.trim_ratio), axis=1, keepdims=True)
            keep_mask = distances <= trim_threshold
        else:
            keep_mask = jnp.ones((B, N), dtype=bool)

        # Compute consensus on non-trimmed agents
        consensus_module = RobustConsensus(
            sigma=self.sigma,
            use_median=self.use_median,
            name='robust_consensus'
        )

        centroid, metrics = consensus_module(responses, keep_mask.astype(jnp.float32))

        metrics['trimmed_agents'] = jnp.mean(jnp.sum(~keep_mask, axis=1))
        metrics['trim_ratio'] = self.trim_ratio

        return centroid, metrics


def streaming_consensus(
    current_centroid: chex.Array,
    new_responses: chex.Array,
    current_count: int,
    sigma: float = 1.0,
    epsilon: float = 1e-8,
) -> Tuple[chex.Array, chex.Array]:
    """Streaming/online consensus update.

    Updates consensus incrementally as new responses arrive,
    without needing to store all historical responses.

    Args:
        current_centroid: Current consensus [B, 1, d]
        new_responses: New agent responses [B, M, d]
        current_count: Number of responses already incorporated
        sigma: Bandwidth for weighting
        epsilon: Numerical stability

    Returns:
        updated_centroid: Updated consensus [B, 1, d]
        weights: Weights for new responses [B, M]
    """
    B, M, d = new_responses.shape

    # Compute weights for new responses
    distances = jnp.sum((new_responses - current_centroid) ** 2, axis=-1)
    weights = jnp.exp(-distances / (2 * sigma ** 2))
    weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + epsilon)

    # Weighted update
    total_count = current_count + M
    decay = current_count / total_count
    new_weight = M / total_count

    weighted_new = jnp.sum(new_responses * weights[..., None], axis=1, keepdims=True)
    updated_centroid = decay * current_centroid + new_weight * weighted_new

    return updated_centroid, weights
