from typing import Optional
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from PoroX.models.components.fc import FFNSwiGLU
from PoroX.models.defaults import DEFAULT_DTYPE

@struct.dataclass
class EncoderConfig:
    # Number of EncoderBlocks
    num_blocks: int = 2
    # Number of heads per EncoderBlock
    num_heads: int = 4
    
    # Project the hidden dimension to a smaller size
    project_dim: Optional[int] = None
    # How many blocks encode at the projected dimension
    project_blocks: int = 1
    # Sometimes the project_dim won't be divisible by num_heads
    project_num_heads: Optional[int] = None
    
    # Layer size of the QKV linear layers
    qkv_features: Optional[int] = None
    # Hidden dimension of the FFN
    hidden_dim: Optional[int] = None
    
    dtype: jnp.dtype = DEFAULT_DTYPE

# Transformer Encoder
class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block Using:
    1. LayerNorm
    2. MHSA
    3. Residual Connection
    4. LayerNorm
    5. FFN
    6. Residual Connection
    """
    num_heads: int
    qkv_features: Optional[int] = None
    hidden_dim: Optional[int] = None
    
    dtype: jnp.dtype = DEFAULT_DTYPE
    
    @nn.compact
    def __call__(self, x):
        y = nn.RMSNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=self.dtype
        )(y)
        x = x + y
        
        y = nn.RMSNorm(dtype=self.dtype)(x)
        y = FFNSwiGLU(
            hidden_dim=self.hidden_dim,
        )(y)
        x = x + y
        
        return x
    
class ProjectionBlock(nn.Module):
    out_dim: int
    dtype: jnp.dtype = DEFAULT_DTYPE
    
    @nn.compact
    def __call__(self, x):
        x = FFNSwiGLU(
            out_dim=self.out_dim,
            dtype=self.dtype
        )(x)

        return x
    
class Encoder(nn.Module):
    """
    Simple Transformer Encoder
    """
    config: EncoderConfig

    def setup(self):
        if self.config.project_dim is not None:
            assert self.config.project_blocks <= self.config.num_blocks, "project_blocks must be less than or equal to num_blocks"
    
    @nn.compact
    def __call__(self, x):
        # No projection
        if self.config.project_dim is None:
            for layer in range(self.config.num_blocks):
                x = EncoderBlock(
                    num_heads=self.config.num_heads,
                    qkv_features=self.config.qkv_features,
                    hidden_dim=self.config.hidden_dim,
                    dtype=self.config.dtype
                )(x)

            x = nn.RMSNorm(dtype=self.config.dtype)(x)
            
            return x
        
        # Project down to project_dima after N blocks specified by project_ratio
        else:
            num_blocks = self.config.num_blocks - self.config.project_blocks
            proj_num_heads = self.config.project_num_heads or self.config.num_heads
            
            for layer in range(num_blocks):
                x = EncoderBlock(
                    num_heads=self.config.num_heads,
                    qkv_features=self.config.qkv_features,
                    hidden_dim=self.config.hidden_dim,
                    dtype=self.config.dtype
                )(x)
                
            x = ProjectionBlock(
                out_dim=self.config.project_dim,
                dtype=self.config.dtype
            )(x)
            
            for layer in range(self.config.project_blocks):
                x = EncoderBlock(
                    num_heads=proj_num_heads,
                    qkv_features=self.config.qkv_features,
                    hidden_dim=self.config.hidden_dim,
                    dtype=self.config.dtype
                )(x)
                
            x = nn.RMSNorm(dtype=self.config.dtype)(x)

            return x
    
class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block
    
    Params:
    x: The tensor to be used as query for cross attention
    context: The tensor to be used for the key and value for cross attention

    """
    config: EncoderConfig
    
    @nn.compact
    def __call__(self, x, context):
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_features,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=self.config.dtype
        )(x, inputs_k=context, inputs_v=context)
        y = nn.RMSNorm(dtype=self.config.dtype)(y)
        x = x + y
        
        y = FFNSwiGLU(
            hidden_dim=self.config.hidden_dim,
            dtype=self.config.dtype
        )(x)
        y = nn.RMSNorm(dtype=self.config.dtype)(y)
        x = x + y

        return x
        
class CrossAttentionEncoder(nn.Module):
    """
    Cross Attention Encoder
    """
    config: EncoderConfig
    
    @nn.compact
    def __call__(self, x, context):
        for layer in range(self.config.num_blocks):
            x = CrossAttentionBlock(self.config)(x, context)
        
        x = nn.RMSNorm(dtype=self.config.dtype)(x)
        
        return x