import jax
from flax import linen as nn
import jax.numpy as jnp

from PoroX.models.components.fc import FFNSwiGLU

from PoroX.models.defaults import DEFAULT_DTYPE

@jax.jit
def action_space_to_action(action):
    col, index = action // 38, action % 38

    def pass_fn():
        return jnp.array([0, 0, 0])
    
    def level_fn():
        return jnp.array([1, 0, 0])
    
    def refresh_fn():
        return jnp.array([2, 0, 0])
    
    def board_bench_fn(col, index):
        from_loc = col - 3
        to_loc = index
        
        return jnp.where(
            index == 37,
            jnp.array([4, from_loc, 0]),
            jnp.array([5, from_loc, to_loc])
        )
        
    def shop_fn(col):
        return jnp.array([3, col - 40, 0])
    
    def item_bench_fn(col, index):
        from_loc = col - 45
        to_loc = index
        
        return jnp.array([6, from_loc, to_loc])
    
    return jnp.select(
        condlist=[
            col == 0,
            col == 1,
            col == 2,
            col < 39,
            col < 45,
            col < 55,
        ],
        choicelist=[
            pass_fn(),
            level_fn(),
            refresh_fn(),
            board_bench_fn(col, index),
            shop_fn(col),
            item_bench_fn(col, index),
        ],
        default=jnp.array([0, 0, 0])
    )
    
class ActionEmbedding(nn.Module):
    dtype: jnp.dtype = DEFAULT_DTYPE
    """
    Take an action in the form of int and embed it into a vector
    """
    @nn.compact
    def __call__(self, action):
        action = action.astype(jnp.int32)

        action_vector = jax.vmap(action_space_to_action)(action)
        
        action_type = action_vector[:, 0]
        from_loc = action_vector[:, 1]
        to_loc = action_vector[:, 2]
        
        # Create one-hot vectors for each action type and concatenate
        action_type_one_hot = nn.one_hot(action_type, num_classes=7, dtype=self.dtype)
        from_loc_one_hot = nn.one_hot(from_loc, num_classes=38, dtype=self.dtype)
        to_loc_one_hot = nn.one_hot(to_loc, num_classes=38, dtype=self.dtype)
        
        action_vector_one_hot = jnp.concatenate([
            action_type_one_hot,
            from_loc_one_hot,
            to_loc_one_hot
        ], axis=-1)
        
        return action_vector_one_hot
    
class ActionGlobalToken(nn.Module):
    dtype: jnp.dtype = DEFAULT_DTYPE
    """
    Use a single action token instead of concatenating to the back of each hidden state
    """
    @nn.compact
    def __call__(self, x, action):
        action_embedding = ActionEmbedding(dtype=self.dtype)(action)

        action_token = FFNSwiGLU(out_dim=x.shape[-1], dtype=self.dtype)(action_embedding)
        
        action_token_expanded = jnp.expand_dims(action_token, axis=-2)
        action_token = jnp.broadcast_to(
            action_token_expanded,
            x.shape[:-2] + (1, x.shape[-1])
        )
        
        return jnp.concatenate([action_token, x], axis=-2)
