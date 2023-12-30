import chex
import jax
import jax.numpy as jnp

"""
Replay Buffer for storing experiences.
Inspired by:
https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/trajectory_buffer.py
"""
    
@chex.dataclass(frozen=True)
class ReplayBufferState:
    buffer: chex.ArrayTree
    base: chex.ArrayTree
    unroll_steps: int
    current_index: int
    trajectory_len: int
    
def init(experience: chex.ArrayTree, unroll_steps: int = 6):
    """Initialize the replay buffer using an initial experience.
    
    Creates a trajectory base object and expands it to the unroll_steps.
    This will speed up the replay buffer by avoiding expand_dims, which is very costly.
    
    Args:
        experience: chex.ArrayTree
        unroll_steps: int
    
    Returns:
        state: ReplayBufferState
    """

    # Create the initial trajectory object
    # The base object will have shape (1, unroll_steps, ...)
    def create_base(experience, unroll_steps):
        # zeros_like trajectory
        experience_zeros = jax.tree_map(
            lambda x: jnp.zeros_like(x),
            experience
        )
        # Expand dims to (1, ...)
        experience_expanded = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0),
            experience_zeros
        )
        # Broadcast to (unroll_steps, ...)
        experience_broadcast = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (unroll_steps,) + x.shape[1:]),
            experience_expanded
        )
        # One final expand dims to (1, unroll_steps, ...)
        return jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0),
            experience_broadcast
        )
        
    base = create_base(experience, unroll_steps)
        
    return ReplayBufferState(
        buffer=base,
        base=base,
        unroll_steps=unroll_steps,
        current_index=0,
        trajectory_len=0,
    )
    
# --- Utility functions ---
@jax.jit
def insert(buffer, experience, index):
    return jax.tree_map(
        lambda x, t: x.at[-1, index, ...].set(t),
        buffer,
        experience
    )
    
# Concatenate a new trajectory base to the buffer
@jax.jit
def concat(buffer, trajectory_base):
    return jax.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=0),
        buffer,
        trajectory_base
    )

@jax.jit
def concat_many(buffers):
    return jax.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *buffers
    )
        
def add(state: ReplayBufferState, experience: chex.ArrayTree):
    """Add an experience to the replay buffer.
    
    To speed up the replay buffer, we insert the experience into the last index of the buffer.
    Once the latest trajectory is full, we concatenate a new trajectory base to the buffer.
    
    Args:
        state: ReplayBufferState
        experience: chex.ArrayTree
    
    Returns:
        state: ReplayBufferState
    """
        
    buffer = insert(state.buffer, experience, state.current_index)
    current_index = state.current_index + 1

    # If we have reached the end of the buffer, we need to concatenate a new trajectory base
    if current_index == state.unroll_steps:
        buffer = concat(buffer, state.base)
        current_index = 0
        trajectory_len = state.trajectory_len + 1
        
        return ReplayBufferState(
            buffer=buffer,
            base=state.base,
            unroll_steps=state.unroll_steps,
            current_index=current_index,
            trajectory_len=trajectory_len,
        )
        
    return ReplayBufferState(
        buffer=buffer,
        base=state.base,
        unroll_steps=state.unroll_steps,
        current_index=current_index,
        trajectory_len=state.trajectory_len,
    )
    
def can_sample(state: ReplayBufferState, batch_size: int):
    """Checks if the replay buffer can sample a batch of trajectories"""
    return state.trajectory_len >= batch_size

def sample(state: ReplayBufferState, key: jax.random.PRNGKey, batch_size: int):
    """Sample a batch of trajectories from the replay buffer
    
    Replay Buffer must have at least batch_size trajectories to sample from.
    
    Args:
        state: ReplayBufferState
        batch_size: int
        
    Returns:
        sample_buffer: chex.ArrayTree
        state: ReplayBufferState
    """
    if not can_sample(state, batch_size):
        raise ValueError("Cannot sample from replay buffer. Not enough trajectories.")

    # First we need to create a list of indices to sample from
    # Indices need to be unique
    sample_indices = jnp.arange(state.trajectory_len)
    sample_indices = jax.random.shuffle(key, sample_indices)
    sample_indices = sample_indices[:batch_size]
    
    # Now we can sample from the buffer
    sample_buffer = jax.tree_map(
        lambda x: x[sample_indices],
        state.buffer,
    )
    
    # Now we need to remove the sampled indices from the original buffer
    buffer = jax.tree_map(
        lambda x: jnp.delete(x, sample_indices, axis=0),
        state.buffer,
    )
    
    # Update the trajectory_len
    trajectory_len = state.trajectory_len - batch_size
    
    return sample_buffer, ReplayBufferState(
        buffer=buffer,
        base=state.base,
        unroll_steps=state.unroll_steps,
        current_index=state.current_index,
        trajectory_len=trajectory_len,
    )

def clear(state: ReplayBufferState):
    """Clears the replay buffer"""
    del state.buffer

    buffer = state.base
    current_index = 0
    trajectory_len = 0
    
    return ReplayBufferState(
        buffer=buffer,
        base=state.base,
        unroll_steps=state.unroll_steps,
        current_index=current_index,
        trajectory_len=trajectory_len,
    )
    
def combine(states: list[ReplayBufferState]):
    """
    Combine a list of replay buffer states into a single replay buffer state.
    This will usually be used to combine replay buffer states from different players and different games.
    """
    if len(states) == 0:
        raise ValueError("Cannot combine empty list of replay buffer states")

    buffer = concat_many([state.buffer for state in states])
    trajectory_len = sum([state.trajectory_len for state in states])

    base = states[0].base
    unroll_steps = states[0].unroll_steps
    current_index = 0
    
    return ReplayBufferState(
        buffer=buffer,
        base=base,
        unroll_steps=unroll_steps,
        current_index=current_index,
        trajectory_len=trajectory_len,
    )