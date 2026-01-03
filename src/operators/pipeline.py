"""Tensor-Consensus Pipeline - Integrated Coordination System.

This module combines projection, assignment, and consensus into a unified
differentiable pipeline for multi-agent coordination.
"""

from typing import Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex

from .projection import AdaptiveProjection
from .assignment import DifferentiableAssignment
from .consensus import RobustConsensus


class TensorConsensusPipeline(nn.Module):
    """Complete Tensor-Consensus coordination pipeline.

    This integrates all three fundamental operators:
    1. Adaptive Projection (O(N))
    2. Differentiable Assignment (O(N log N))
    3. Robust Consensus (O(N))

    Attributes:
        complexity_dim: Number of complexity levels for projection
        embedding_dim: Dimension for assignment embeddings
        consensus_sigma: Bandwidth for consensus outlier weighting
        temperature: Temperature for assignment
        num_sinkhorn_iters: Iterations for Sinkhorn algorithm
        num_consensus_iters: Iterations for consensus
    """

    complexity_dim: int = 8
    embedding_dim: int = 64
    consensus_sigma: float = 1.0
    temperature: float = 1.0
    num_sinkhorn_iters: int = 10
    num_consensus_iters: int = 10
    use_hierarchical: bool = False
    hierarchical_group_size: int = 100

    @nn.compact
    def __call__(
        self,
        agent_states: chex.Array,
        task_states: Optional[chex.Array] = None,
        agent_mask: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        """Forward pass of complete pipeline.

        Args:
            agent_states: Agent states [B, N, d_s]
            task_states: Optional task states [B, M, d_t]
            agent_mask: Optional mask for valid agents [B, N]

        Returns:
            coordinated_actions: Coordinated outputs [B, N, d_out] or [B, 1, d_out]
            metrics: Comprehensive diagnostic metrics
        """
        B, N, d_s = agent_states.shape

        all_metrics = {}

        # Step 1: Adaptive Projection
        projection = AdaptiveProjection(
            complexity_dim=self.complexity_dim,
            name='projection'
        )
        resources, proj_metrics = projection(agent_states)
        all_metrics['projection'] = proj_metrics

        # Modulate states by resources
        resource_weights = jnp.sum(resources, axis=-1, keepdims=True) / self.complexity_dim
        modulated_states = agent_states * (1.0 + 0.1 * resource_weights)

        # Step 2: Differentiable Assignment (if tasks provided)
        if task_states is not None:
            assignment = DifferentiableAssignment(
                embedding_dim=self.embedding_dim,
                temperature=self.temperature,
                num_iterations=self.num_sinkhorn_iters,
                name='assignment'
            )
            assignment_matrix, assign_metrics = assignment(modulated_states, task_states)
            all_metrics['assignment'] = assign_metrics

            # Weight states by assignment
            # For each agent, aggregate information from assigned tasks
            task_weighted = jnp.einsum('bnm,bmd->bnd', assignment_matrix, task_states)
            combined_states = jnp.concatenate([modulated_states, task_weighted], axis=-1)

            # Project back to original dimension
            combined_proj = nn.Dense(d_s, name='combination_projection')
            consensus_inputs = combined_proj(combined_states)
        else:
            consensus_inputs = modulated_states
            all_metrics['assignment'] = None

        # Step 3: Robust Consensus
        consensus = RobustConsensus(
            sigma=self.consensus_sigma,
            max_iters=self.num_consensus_iters,
            name='consensus'
        )
        consensus_output, consensus_metrics = consensus(consensus_inputs, agent_mask)
        all_metrics['consensus'] = consensus_metrics

        # Final output projection
        output_proj = nn.Dense(d_s, name='output_projection')
        final_output = output_proj(consensus_output)

        # Compute overall metrics
        all_metrics['pipeline'] = {
            'num_agents': N,
            'batch_size': B,
        }

        return final_output, all_metrics


class MultiScalePipeline(nn.Module):
    """Multi-scale Tensor-Consensus for handling varying agent counts.

    Dynamically switches between standard and hierarchical approaches
    based on the number of agents.

    Attributes:
        threshold: Agent count threshold for hierarchical mode
        standard_config: Config for standard pipeline
        hierarchical_config: Config for hierarchical pipeline
    """

    threshold: int = 500
    complexity_dim: int = 8
    embedding_dim: int = 64
    consensus_sigma: float = 1.0

    @nn.compact
    def __call__(
        self,
        agent_states: chex.Array,
        task_states: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        """Forward pass with dynamic scaling.

        Args:
            agent_states: Agent states [B, N, d_s]
            task_states: Optional task states [B, M, d_t]

        Returns:
            coordinated_actions: Coordinated outputs
            metrics: Diagnostic metrics
        """
        B, N, d_s = agent_states.shape

        # Choose pipeline based on agent count
        if N < self.threshold:
            pipeline = TensorConsensusPipeline(
                complexity_dim=self.complexity_dim,
                embedding_dim=self.embedding_dim,
                consensus_sigma=self.consensus_sigma,
                use_hierarchical=False,
                name='standard_pipeline'
            )
        else:
            pipeline = TensorConsensusPipeline(
                complexity_dim=self.complexity_dim,
                embedding_dim=self.embedding_dim,
                consensus_sigma=self.consensus_sigma,
                use_hierarchical=True,
                hierarchical_group_size=100,
                name='hierarchical_pipeline'
            )

        output, metrics = pipeline(agent_states, task_states)
        metrics['pipeline_type'] = 'hierarchical' if N >= self.threshold else 'standard'

        return output, metrics


class ParallelPipeline(nn.Module):
    """Parallel execution of multiple pipelines with ensembling.

    Runs multiple independent pipelines and combines their outputs
    for improved robustness and diversity.

    Attributes:
        num_pipelines: Number of parallel pipelines
        combination_method: How to combine outputs ('mean', 'weighted', 'vote')
    """

    num_pipelines: int = 3
    combination_method: str = 'weighted'
    complexity_dim: int = 8
    embedding_dim: int = 64

    @nn.compact
    def __call__(
        self,
        agent_states: chex.Array,
        task_states: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        """Forward pass with parallel pipelines.

        Args:
            agent_states: Agent states [B, N, d_s]
            task_states: Optional task states [B, M, d_t]

        Returns:
            combined_output: Combined coordinated outputs
            metrics: Aggregated metrics from all pipelines
        """
        outputs = []
        all_metrics = []

        for i in range(self.num_pipelines):
            pipeline = TensorConsensusPipeline(
                complexity_dim=self.complexity_dim,
                embedding_dim=self.embedding_dim,
                consensus_sigma=0.8 + 0.4 * i / max(1, self.num_pipelines - 1),  # Vary sigma
                temperature=0.5 + 1.0 * i / max(1, self.num_pipelines - 1),  # Vary temperature
                name=f'pipeline_{i}'
            )

            output, metrics = pipeline(agent_states, task_states)
            outputs.append(output)
            all_metrics.append(metrics)

        # Combine outputs
        if self.combination_method == 'mean':
            combined_output = jnp.mean(jnp.stack(outputs, axis=0), axis=0)
        elif self.combination_method == 'weighted':
            # Weight by consensus confidence (inverse variance)
            variances = jnp.array([
                m['consensus']['consensus_variance']
                for m in all_metrics
            ])
            weights = 1.0 / (variances + 1e-8)
            weights = weights / jnp.sum(weights)

            combined_output = sum(
                w * out for w, out in zip(weights, outputs)
            )
        else:
            # Default to mean
            combined_output = jnp.mean(jnp.stack(outputs, axis=0), axis=0)

        # Aggregate metrics
        aggregated_metrics = {
            'num_pipelines': self.num_pipelines,
            'combination_method': self.combination_method,
            'individual_metrics': all_metrics,
        }

        return combined_output, aggregated_metrics


def create_pipeline(
    config: Dict[str, Any],
    mode: str = 'standard',
) -> nn.Module:
    """Factory function to create pipeline based on configuration.

    Args:
        config: Configuration dictionary
        mode: Pipeline mode ('standard', 'multiscale', 'parallel')

    Returns:
        Configured pipeline module
    """
    if mode == 'standard':
        return TensorConsensusPipeline(**config)
    elif mode == 'multiscale':
        return MultiScalePipeline(**config)
    elif mode == 'parallel':
        return ParallelPipeline(**config)
    else:
        raise ValueError(f"Unknown pipeline mode: {mode}")
