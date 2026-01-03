"""Mamba State Space Model Integration.

Implements Mamba SSM for efficient sequence modeling in multi-agent systems,
providing linear-time complexity for long sequences.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex


class MambaBlock(nn.Module):
    """Single Mamba block with selective state space.

    This implements a simplified version of Mamba that integrates
    with Tensor-Consensus operators.

    Attributes:
        d_model: Model dimension
        d_state: State space dimension
        d_conv: Convolution kernel size
        expand_factor: Expansion factor for hidden dimension
    """

    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """Forward pass of Mamba block.

        Args:
            x: Input sequence [B, L, d_model]

        Returns:
            Output sequence [B, L, d_model]
        """
        B, L, D = x.shape
        d_inner = self.d_model * self.expand_factor

        # Input projection
        x_proj = nn.Dense(d_inner * 2, name='in_proj')(x)
        x_z, gate = jnp.split(x_proj, 2, axis=-1)

        # Depthwise convolution
        x_conv = nn.Conv(
            features=d_inner,
            kernel_size=(self.d_conv,),
            feature_group_count=d_inner,
            padding='SAME',
            name='conv1d'
        )(x_z)

        # Activation
        x_conv = nn.silu(x_conv)

        # SSM parameters (simplified)
        A = self.param(
            'A',
            nn.initializers.lecun_normal(),
            (d_inner, self.d_state)
        )
        B = nn.Dense(self.d_state, name='B_proj')(x_conv)
        C = nn.Dense(self.d_state, name='C_proj')(x_conv)
        D = self.param('D', nn.initializers.ones, (d_inner,))

        # Discretization (simplified Euler method)
        dt = nn.Dense(d_inner, name='dt_proj')(x_conv)
        dt = nn.softplus(dt)

        # State space computation (simplified)
        # In full Mamba, this uses selective scan
        dA = jnp.exp(dt[..., None] * A[None, None, :, :])  # [B, L, d_inner, d_state]
        dB = dt[..., None] * B[..., None, :]  # [B, L, d_inner, d_state]

        # Simplified scan (not selective)
        def scan_fn(h, inputs):
            x_t, dA_t, dB_t, C_t = inputs
            h_new = dA_t * h + dB_t * x_t[..., None]
            y_t = jnp.sum(h_new * C_t[..., None, :], axis=-1)
            return h_new, y_t

        # Initialize hidden state
        h0 = jnp.zeros((B, d_inner, self.d_state))

        # Scan over sequence
        inputs = (x_conv, dA, dB, C)
        inputs_transposed = jax.tree_map(lambda t: jnp.transpose(t, (1, 0, 2)), inputs)

        _, y = jax.lax.scan(
            scan_fn,
            h0,
            inputs_transposed
        )

        # Transpose back
        y = jnp.transpose(y, (1, 0, 2))  # [B, L, d_inner]

        # Add skip connection
        y = y + x_conv * D[None, None, :]

        # Gating
        y = y * nn.silu(gate)

        # Output projection
        output = nn.Dense(self.d_model, name='out_proj')(y)

        return output


class MambaSSM(nn.Module):
    """Multi-layer Mamba SSM for sequence modeling.

    Stacks multiple Mamba blocks for deep sequence processing.

    Attributes:
        num_layers: Number of Mamba blocks
        d_model: Model dimension
        d_state: State space dimension
        dropout_rate: Dropout rate
    """

    num_layers: int = 4
    d_model: int = 64
    d_state: int = 16
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        training: bool = True
    ) -> Tuple[chex.Array, dict]:
        """Forward pass through Mamba SSM.

        Args:
            x: Input sequence [B, L, d_in]
            training: Whether in training mode

        Returns:
            output: Output sequence [B, L, d_model]
            metrics: Diagnostic metrics
        """
        B, L, d_in = x.shape

        # Input embedding
        if d_in != self.d_model:
            x = nn.Dense(self.d_model, name='input_embedding')(x)

        # Apply Mamba blocks
        for i in range(self.num_layers):
            # Mamba block
            residual = x
            x_norm = nn.LayerNorm(name=f'norm_{i}')(x)
            x = MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                name=f'mamba_{i}'
            )(x_norm)

            # Dropout
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

            # Residual connection
            x = x + residual

        # Final norm
        x = nn.LayerNorm(name='final_norm')(x)

        metrics = {
            'sequence_length': L,
            'num_layers': self.num_layers,
        }

        return x, metrics


class MambaWithConsensus(nn.Module):
    """Mamba SSM integrated with Tensor-Consensus.

    Combines sequence modeling via Mamba with multi-agent coordination.

    Attributes:
        mamba_config: Configuration for Mamba SSM
        consensus_config: Configuration for Tensor-Consensus
    """

    num_mamba_layers: int = 4
    d_model: int = 64
    complexity_dim: int = 8
    consensus_sigma: float = 1.0

    @nn.compact
    def __call__(
        self,
        agent_sequences: chex.Array,
        training: bool = True
    ) -> Tuple[chex.Array, dict]:
        """Forward pass with Mamba + Consensus.

        Args:
            agent_sequences: Agent observation sequences [B, N, L, d]
            training: Whether in training mode

        Returns:
            consensus_output: Coordinated output [B, 1, d_model]
            metrics: Combined metrics
        """
        B, N, L, d = agent_sequences.shape

        # Process each agent's sequence with Mamba
        mamba = MambaSSM(
            num_layers=self.num_mamba_layers,
            d_model=self.d_model,
            name='mamba'
        )

        # Reshape for processing
        sequences_flat = agent_sequences.reshape(B * N, L, d)
        processed_flat, mamba_metrics = mamba(sequences_flat, training)

        # Take final timestep representation
        agent_representations = processed_flat[:, -1, :]  # [B*N, d_model]
        agent_representations = agent_representations.reshape(B, N, self.d_model)

        # Apply Tensor-Consensus
        from ..operators.pipeline import TensorConsensusPipeline

        consensus_pipeline = TensorConsensusPipeline(
            complexity_dim=self.complexity_dim,
            consensus_sigma=self.consensus_sigma,
            name='consensus'
        )

        consensus_output, consensus_metrics = consensus_pipeline(agent_representations)

        # Combine metrics
        all_metrics = {
            'mamba': mamba_metrics,
            'consensus': consensus_metrics,
        }

        return consensus_output, all_metrics
