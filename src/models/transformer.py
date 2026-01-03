"""Transformer Models for Multi-Agent Systems.

Implements standard Transformer and multi-agent specific variants
for comparison with Tensor-Consensus.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    Attributes:
        num_heads: Number of attention heads
        d_model: Model dimension
        dropout_rate: Dropout rate
    """

    num_heads: int = 8
    d_model: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
        training: bool = True,
    ) -> chex.Array:
        """Multi-head attention forward pass.

        Args:
            query: Query tensor [B, L_q, d_model]
            key: Key tensor [B, L_k, d_model]
            value: Value tensor [B, L_v, d_model]
            mask: Optional attention mask [B, L_q, L_k]
            training: Whether in training mode

        Returns:
            Attention output [B, L_q, d_model]
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        d_k = self.d_model // self.num_heads

        # Project Q, K, V
        Q = nn.Dense(self.d_model, name='q_proj')(query)
        K = nn.Dense(self.d_model, name='k_proj')(key)
        V = nn.Dense(self.d_model, name='v_proj')(value)

        # Reshape for multi-head
        Q = Q.reshape(B, L_q, self.num_heads, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, L_k, self.num_heads, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, L_k, self.num_heads, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)

        if mask is not None:
            scores = jnp.where(mask[:, None, :, :], scores, -1e9)

        attn_weights = nn.softmax(scores, axis=-1)

        if training:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=not training
            )

        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, V)

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L_q, self.d_model)
        output = nn.Dense(self.d_model, name='out_proj')(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout_rate: Dropout rate
    """

    d_model: int = 64
    num_heads: int = 8
    d_ff: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
        training: bool = True,
    ) -> chex.Array:
        """Transformer block forward pass.

        Args:
            x: Input tensor [B, L, d_model]
            mask: Optional attention mask [B, L, L]
            training: Whether in training mode

        Returns:
            Output tensor [B, L, d_model]
        """
        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name='self_attn'
        )(x, x, x, mask, training)

        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        x = x + residual

        # Feedforward
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.d_ff)(x)
        x = nn.gelu(x)

        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        x = nn.Dense(self.d_model)(x)

        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        x = x + residual

        return x


class TransformerModel(nn.Module):
    """Multi-layer Transformer model.

    Attributes:
        num_layers: Number of Transformer blocks
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout_rate: Dropout rate
    """

    num_layers: int = 6
    d_model: int = 64
    num_heads: int = 8
    d_ff: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
        training: bool = True,
    ) -> Tuple[chex.Array, dict]:
        """Transformer forward pass.

        Args:
            x: Input tensor [B, L, d_in]
            mask: Optional attention mask [B, L, L]
            training: Whether in training mode

        Returns:
            output: Output tensor [B, L, d_model]
            metrics: Diagnostic metrics
        """
        B, L, d_in = x.shape

        # Input embedding
        if d_in != self.d_model:
            x = nn.Dense(self.d_model, name='input_embedding')(x)

        # Positional encoding
        pos_encoding = self.param(
            'pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (1, L, self.d_model)
        )
        x = x + pos_encoding

        # Apply Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'block_{i}'
            )(x, mask, training)

        # Final layer norm
        x = nn.LayerNorm()(x)

        metrics = {
            'sequence_length': L,
            'num_layers': self.num_layers,
        }

        return x, metrics


class MultiAgentTransformer(nn.Module):
    """Multi-agent Transformer with agent-specific attention.

    This implements a Transformer variant specifically designed for
    multi-agent coordination, similar to HAMT from the paper.

    Attributes:
        num_layers: Number of layers
        d_model: Model dimension
        num_heads: Number of attention heads
        use_agent_specific: Whether to use agent-specific projections
    """

    num_layers: int = 4
    d_model: int = 64
    num_heads: int = 8
    use_agent_specific: bool = True

    @nn.compact
    def __call__(
        self,
        agent_states: chex.Array,
        training: bool = True,
    ) -> Tuple[chex.Array, dict]:
        """Multi-agent Transformer forward pass.

        Args:
            agent_states: Agent states [B, N, d]
            training: Whether in training mode

        Returns:
            agent_outputs: Per-agent outputs [B, N, d_model]
            metrics: Diagnostic metrics
        """
        B, N, d = agent_states.shape

        # Agent embeddings
        if self.use_agent_specific:
            agent_ids = jnp.arange(N)[None, :, None]  # [1, N, 1]
            agent_id_embeddings = nn.Embed(
                num_embeddings=N,
                features=self.d_model,
                name='agent_id_embed'
            )(agent_ids)

            x = nn.Dense(self.d_model, name='state_embed')(agent_states)
            x = x + agent_id_embeddings
        else:
            x = nn.Dense(self.d_model, name='state_embed')(agent_states)

        # Process with Transformer
        transformer = TransformerModel(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            name='transformer'
        )

        agent_outputs, metrics = transformer(x, training=training)

        return agent_outputs, metrics


class HierarchicalMultiAgentTransformer(nn.Module):
    """Hierarchical Multi-Agent Transformer (HAMT-like).

    Implements hierarchical attention for better scalability.

    Attributes:
        num_layers: Number of layers
        d_model: Model dimension
        num_heads: Number of attention heads
        group_size: Size of agent groups for hierarchical attention
    """

    num_layers: int = 4
    d_model: int = 64
    num_heads: int = 8
    group_size: int = 10

    @nn.compact
    def __call__(
        self,
        agent_states: chex.Array,
        training: bool = True,
    ) -> Tuple[chex.Array, dict]:
        """Hierarchical attention forward pass.

        Args:
            agent_states: Agent states [B, N, d]
            training: Whether in training mode

        Returns:
            agent_outputs: Per-agent outputs [B, N, d_model]
            metrics: Diagnostic metrics
        """
        B, N, d = agent_states.shape

        # Embed states
        x = nn.Dense(self.d_model, name='state_embed')(agent_states)

        # Group agents
        n_groups = (N + self.group_size - 1) // self.group_size
        padded_n = n_groups * self.group_size

        if padded_n > N:
            padding = jnp.zeros((B, padded_n - N, self.d_model))
            x_padded = jnp.concatenate([x, padding], axis=1)
        else:
            x_padded = x

        x_grouped = x_padded.reshape(B, n_groups, self.group_size, self.d_model)

        # Local attention within groups
        local_transformer = TransformerModel(
            num_layers=self.num_layers // 2,
            d_model=self.d_model,
            num_heads=self.num_heads,
            name='local_transformer'
        )

        # Process each group
        x_local_list = []
        for i in range(n_groups):
            x_group, _ = local_transformer(x_grouped[:, i, :, :], training=training)
            x_local_list.append(x_group)

        x_local = jnp.stack(x_local_list, axis=1)  # [B, n_groups, group_size, d_model]

        # Global attention across group representatives
        group_repr = jnp.mean(x_local, axis=2)  # [B, n_groups, d_model]

        global_transformer = TransformerModel(
            num_layers=self.num_layers // 2,
            d_model=self.d_model,
            num_heads=self.num_heads,
            name='global_transformer'
        )

        group_repr_updated, metrics = global_transformer(group_repr, training=training)

        # Broadcast back to agents
        group_repr_broadcast = jnp.repeat(
            group_repr_updated[:, :, None, :],
            self.group_size,
            axis=2
        )

        # Combine local and global
        x_combined = x_local + group_repr_broadcast

        # Reshape back
        x_output = x_combined.reshape(B, padded_n, self.d_model)[:, :N, :]

        metrics['num_groups'] = n_groups
        metrics['group_size'] = self.group_size

        return x_output, metrics
