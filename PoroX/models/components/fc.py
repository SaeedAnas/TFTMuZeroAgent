import jax.numpy as jnp
from typing import Sequence, Callable, Optional
from flax import linen as nn
from PoroX.models.defaults import DEFAULT_DTYPE

"""
The fc module is a collection of useful components for fully connected networks
"""

class MLP(nn.Module):
    features: Sequence[int]
    act: Callable = nn.gelu
    dtype: jnp.dtype = DEFAULT_DTYPE
    
    def setup(self):
        layers = []
        for i, size in enumerate(self.features):
            layers.append(nn.Dense(size, dtype=self.dtype))

            if i != len(self.features) - 1:
                layers.append(self.act)

        self.mlp = nn.Sequential(layers=layers)
    
    def __call__(self, x):
        return self.mlp(x)
    
class FFNSwiGLU(nn.Module):
    """
    Feedforward Network + The fabled SwiGLU actiation function

    Taken directly from the official LLaMA implementation:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py
    """
    hidden_dim: Optional[int] = None
    out_dim: Optional[int] = None
    dtype: jnp.dtype = DEFAULT_DTYPE
    
    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        hidden_dim = self.hidden_dim or 2 * input_dim
        hidden_dim = int(2 * hidden_dim / 3)
        out_dim = self.out_dim or input_dim

        def dense_fn(dim):
            return nn.DenseGeneral(features=dim, use_bias=False, dtype=self.dtype)
        
        w1 = dense_fn(hidden_dim)(x)
        w1 = nn.silu(w1)
        w3 = dense_fn(hidden_dim)(x)
        w2 = dense_fn(out_dim)(w1 * w3)
        
        return w2
