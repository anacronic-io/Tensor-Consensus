"""Tests for tensor operators."""

import pytest
import jax
import jax.numpy as jnp
from src.operators.projection import AdaptiveProjection, adaptive_projection
from src.operators.assignment import DifferentiableAssignment, sinkhorn_assignment
from src.operators.consensus import RobustConsensus, robust_consensus


class TestAdaptiveProjection:
    """Tests for adaptive projection operator."""

    def test_projection_shape(self):
        """Test output shape is correct."""
        batch_size, n_agents, d = 4, 10, 64
        complexity_dim = 8

        model = AdaptiveProjection(complexity_dim=complexity_dim)
        states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))

        params = model.init(jax.random.PRNGKey(1), states)
        resources, metrics = model.apply(params, states)

        assert resources.shape == (batch_size, n_agents, complexity_dim)

    def test_projection_values(self):
        """Test projected values are in valid range."""
        batch_size, n_agents, d = 4, 10, 64

        model = AdaptiveProjection(complexity_dim=8, resource_budget=100.0)
        states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))

        params = model.init(jax.random.PRNGKey(1), states)
        resources, metrics = model.apply(params, states)

        # Resources should be non-negative
        assert jnp.all(resources >= 0)

        # Sum should be close to budget
        assert jnp.all(jnp.sum(resources, axis=-1) <= 120.0)

    def test_projection_deterministic(self):
        """Test projection is deterministic."""
        batch_size, n_agents, d = 4, 10, 64

        model = AdaptiveProjection(complexity_dim=8)
        states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))

        params = model.init(jax.random.PRNGKey(1), states)

        resources1, _ = model.apply(params, states)
        resources2, _ = model.apply(params, states)

        assert jnp.allclose(resources1, resources2)


class TestDifferentiableAssignment:
    """Tests for differentiable assignment operator."""

    def test_assignment_shape(self):
        """Test assignment matrix has correct shape."""
        batch_size, n_agents, n_tasks, d = 4, 10, 8, 64

        model = DifferentiableAssignment(embedding_dim=32)
        agent_states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))
        task_states = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_tasks, d))

        params = model.init(jax.random.PRNGKey(2), agent_states, task_states)
        assignment, metrics = model.apply(params, agent_states, task_states)

        assert assignment.shape == (batch_size, n_agents, n_tasks)

    def test_assignment_doubly_stochastic(self):
        """Test assignment is approximately doubly stochastic."""
        batch_size, n_agents, n_tasks, d = 4, 10, 10, 64

        model = DifferentiableAssignment(embedding_dim=32, num_iterations=20)
        agent_states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))
        task_states = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_tasks, d))

        params = model.init(jax.random.PRNGKey(2), agent_states, task_states)
        assignment, metrics = model.apply(params, agent_states, task_states)

        # Row sums should be close to 1
        row_sums = jnp.sum(assignment, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=0.1)

        # Column sums should be close to 1 (when n_agents == n_tasks)
        col_sums = jnp.sum(assignment, axis=1)
        assert jnp.allclose(col_sums, 1.0, atol=0.1)

    def test_sinkhorn_convergence(self):
        """Test Sinkhorn algorithm converges."""
        n, m = 10, 10
        utility_matrix = jax.random.normal(jax.random.PRNGKey(0), (n, m))

        assignment = sinkhorn_assignment(utility_matrix, temperature=1.0, num_iterations=20)

        # Check row and column sums
        row_sums = jnp.sum(assignment, axis=1)
        col_sums = jnp.sum(assignment, axis=0)

        assert jnp.allclose(row_sums, 1.0, atol=0.01)
        assert jnp.allclose(col_sums, 1.0, atol=0.01)


class TestRobustConsensus:
    """Tests for robust consensus operator."""

    def test_consensus_shape(self):
        """Test consensus output has correct shape."""
        batch_size, n_agents, d = 4, 10, 64

        model = RobustConsensus(sigma=1.0)
        responses = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))

        params = model.init(jax.random.PRNGKey(1), responses)
        centroid, metrics = model.apply(params, responses)

        assert centroid.shape == (batch_size, 1, d)

    def test_consensus_convergence(self):
        """Test consensus converges to approximate mean."""
        batch_size, n_agents, d = 4, 10, 64

        model = RobustConsensus(sigma=1.0, max_iters=50)
        responses = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))

        params = model.init(jax.random.PRNGKey(1), responses)
        centroid, metrics = model.apply(params, responses)

        # Centroid should be close to mean
        mean = jnp.mean(responses, axis=1, keepdims=True)
        assert jnp.allclose(centroid, mean, atol=0.5)

    def test_consensus_outlier_robustness(self):
        """Test consensus is robust to outliers."""
        batch_size, n_agents, d = 1, 10, 2

        # Create responses with one outlier
        responses = jnp.zeros((batch_size, n_agents, d))
        responses = responses.at[:, -1, :].set(100.0)  # Outlier

        centroid, weights = robust_consensus(responses, sigma=1.0, max_iters=20)

        # Outlier should have low weight
        assert weights[0, -1] < 0.1

        # Centroid should be close to zero (majority)
        assert jnp.allclose(centroid, 0.0, atol=5.0)


class TestPipeline:
    """Integration tests for full pipeline."""

    def test_pipeline_end_to_end(self):
        """Test complete pipeline runs without errors."""
        from src.operators.pipeline import TensorConsensusPipeline

        batch_size, n_agents, d = 4, 10, 64

        model = TensorConsensusPipeline(
            complexity_dim=8,
            embedding_dim=32,
            consensus_sigma=1.0,
        )

        states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_agents, d))

        params = model.init(jax.random.PRNGKey(1), states)
        output, metrics = model.apply(params, states)

        # Check output shape
        assert output.shape == (batch_size, 1, d)

        # Check metrics are present
        assert 'projection' in metrics
        assert 'consensus' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
