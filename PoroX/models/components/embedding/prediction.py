from flax import linen as nn
import jax.numpy as jnp

from PoroX.models.defaults import DEFAULT_DTYPE

class PolicyActionTokens(nn.Module):
    dtype: jnp.dtype = DEFAULT_DTYPE
    """
    Action tokens for the policy
    0: Pass
    1: Refresh
    2: Level
    """
    @nn.compact
    def __call__(self, x):
        """
        Essentially return the three tokens broadcasted to the shape of x
        
        x: (..., sequence, embedding)
        
        tokens: (3, embedding)
        """
        
        embedding = nn.Embed(num_embeddings=3, features=x.shape[-1], dtype=self.dtype)(jnp.arange(3))

        tokens =  jnp.broadcast_to(embedding, x.shape[:-2] + (3, x.shape[-1]))
        
        # Prepend tokens to x
        # Shape: (..., 3 + sequence, embedding)
        return jnp.concatenate([tokens, x], axis=-2)
    
class ValueToken(nn.Module):
    dtype: jnp.dtype = DEFAULT_DTYPE

    """
    Value token to be used in the to pay attention to the entire sequence
    """
    @nn.compact
    def __call__(self, x):
        """
        Concatenate a single token to the front of the sequence
        """
        
        embedding = nn.Embed(num_embeddings=1, features=x.shape[-1], dtype=self.dtype)(jnp.array([0]))
        
        embedding_expanded = jnp.expand_dims(embedding, axis=-2)
        value_token = jnp.broadcast_to(embedding_expanded, x.shape[:-2] + (1, x.shape[-1]))
        
        return jnp.concatenate([value_token, x], axis=-2)
    
class PrependTokens(nn.Module):
    """
    Helper module to prepend N tokens to the front of the sequence
    """
    num_tokens: int
    dtype: jnp.dtype = DEFAULT_DTYPE
    
    @nn.compact
    def __call__(self, x):
        """
        Prepend N tokens to the front of the sequence
        """
        
        embedding = nn.Embed(num_embeddings=self.num_tokens, features=x.shape[-1], dtype=self.dtype)(jnp.arange(self.num_tokens))
        
        tokens = jnp.broadcast_to(embedding, x.shape[:-2] + (self.num_tokens, x.shape[-1]))
        
        return jnp.concatenate([tokens, x], axis=-2)
        