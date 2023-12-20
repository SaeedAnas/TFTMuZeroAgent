from flax import linen as nn
import jax.numpy as jnp

class PolicyActionTokens(nn.Module):
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
        
        embedding = nn.Embed(num_embeddings=3, features=x.shape[-1])(jnp.arange(3))

        tokens =  jnp.broadcast_to(embedding, x.shape[:-2] + (3, x.shape[-1]))
        
        # Prepend tokens to x
        # Shape: (..., 3 + sequence, embedding)
        return jnp.concatenate([tokens, x], axis=-2)
    
class ValueToken(nn.Module):
    """
    Value token to be used in the value head
    """
    @nn.compact
    def __call__(self, x):
        """
        Just add a <value> token to the front of the sequence
        """
        
        embedding = nn.Embed(num_embeddings=1, features=x.shape[-1])(jnp.array([0]))
        
        value_token = jnp.broadcast_to(embedding, x.shape[:-2] + (1, x.shape[-1]))
        
        return jnp.concatenate([value_token, x], axis=-2)
        