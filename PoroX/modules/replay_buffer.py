import time
import chex
import jax
import jax.numpy as jnp

from PoroX.modules.observation import BatchedObservation, compress_observation, PlayerObservation
import PoroX.modules.batch_utils as batch_utils

from collections import defaultdict


"""
Replay Buffer for storing experiences
"""

@chex.dataclass
class Trajectory:
    observation: BatchedObservation
    action: chex.ArrayDevice
    policy_logits: chex.ArrayDevice
    value: chex.ArrayDevice
    reward: chex.ArrayDevice


# First we need a local buffer for each game

# Lets make a prototype store function that stores a single game
def create_trajectories(output, obs, reward, mapping):
    """
    Store the output of an agent in the replay buffer.
    
    output: AgentOutput
    - Contains policy_logits, action, and value
    
    obs: Dict Observation
    
    reward: jnp.ndarray
    """
    list_obs, list_mapping = batch_utils.collect_list_obs(obs)

    trajectories = {}
    
    for player_id in mapping.player_ids:
        player_id = int(player_id)
        player_idx = list_mapping[player_id]

        player_observation = list_obs[player_idx]
        player_output = batch_utils.map_collection(output, lambda x: x[player_idx])
        player_reward = reward[player_idx]
        
        trajectory = Trajectory(
            observation=compress_observation(player_observation),
            action=player_output.action,
            policy_logits=player_output.action_weights,
            value=player_output.value,
            reward=player_reward,
        )
        # We need to expand the trajectory
        # So we can concatenate it with other trajectories
        expanded_trajectory = batch_utils.map_collection(trajectory, lambda x: jnp.expand_dims(x, axis=0))

        trajectories[player_id] = expanded_trajectory
    
    return trajectories

def calculate_max_len(current_len, unroll_steps):
    """
    We need to make max_len be a multiple of unroll_steps.
    Ex. current_len = 10, unroll_steps = 4 -> max_len = 12
    """
    max_len = current_len + (-current_len % unroll_steps) # This was copilot
    return max_len
    
def pad_trajectories(trajectories, unroll_steps=6):
    """
    Pad trajectories to be a multiple of unroll_steps.
    """
    return batch_utils.map_collection(
        trajectories,
        lambda x: batch_utils.pad_array(
            x,
            max_length=calculate_max_len(len(x), unroll_steps)
        )
    )
    
def reshape_trajectories(trajectories, unroll_steps=6):
    """
    Reshape trajectories to be a multiple of unroll_steps.
    Ex. trajectory shape: (12, ...), unroll_steps = 4 -> shape = (3, 4, ...)
    """
    return batch_utils.map_collection(
        trajectories,
        lambda x: jnp.reshape(
            x,
            newshape=(-1, unroll_steps) + x.shape[1:]
        )
    )


def combine_game_trajectories(trajectories):
    """
    Combine dict of trajectories into a single trajectory.
    """
    return batch_utils.concat(list(trajectories.values()))
    

def sample_trajectory(trajectory, key, batch_size=64):
    """
    Samples trajectories from a single trajectory.
    """
    trajectory_len = len(trajectory.action)

    # First need to create a list of indices to sample from
    # Indices need to be unique
    sample_indices = jnp.arange(trajectory_len)
    sample_indices = jax.random.shuffle(key, sample_indices)
    sample_indices = sample_indices[:batch_size]
    
    print(sample_indices)

    # Now we can sample from the trajectory
    sample_trajectory = batch_utils.map_collection(
        trajectory,
        lambda x: x[sample_indices]
    )
    
    # Now we need to remove the sampled indices from the original trajectory
    new_trajectory = batch_utils.map_collection(
        trajectory,
        lambda x: jnp.delete(x, sample_indices, axis=0)
    )
    
    return sample_trajectory, new_trajectory
    


def trajectory_analytics(trajectory):
    print("Trajectory size:")
    p_champ = trajectory.observation.players.champions.nbytes
    p_item = trajectory.observation.players.items.nbytes
    p_scalar = trajectory.observation.players.scalars.nbytes
    p_trait = trajectory.observation.players.traits.nbytes
    o_champ = trajectory.observation.opponents.champions.nbytes
    o_item = trajectory.observation.opponents.items.nbytes
    o_scalar = trajectory.observation.opponents.scalars.nbytes
    o_trait = trajectory.observation.opponents.traits.nbytes
    mask = trajectory.observation.action_mask.nbytes
    action = trajectory.action.nbytes
    policy = trajectory.policy_logits.nbytes
    value = trajectory.value.nbytes
    reward = trajectory.reward.nbytes
    
    bytes = jnp.array([
        p_champ,
        p_item,
        p_scalar,
        p_trait,
        o_champ,
        o_item,
        o_scalar,
        o_trait,
        mask,
        action,
        policy,
        value,
        reward
    ])
    
    print(f"trajectory size: {jnp.sum(bytes) / 1e6} mb, for {len(trajectory.action)} steps")

class LocalBuffer:
    """
    Store game trajectories for each agent.
    """
    player_trajectories: dict[int, Trajectory]
    
    def __init__(self):
        self.player_trajectories = {}
        
    def store_trajectory(self, player_id, trajectory):
        """
        Concat for memory efficiency.
        """
        if player_trajectory := self.player_trajectories.get(player_id):
            self.player_trajectories[player_id] = batch_utils.concat(
                [player_trajectory, trajectory]
            )
        else:
            self.player_trajectories[player_id] = trajectory
    
    def store_batch_trajectories(self, trajectories):
        for player_id, trajectory in trajectories.items():
            self.store_trajectory(player_id, trajectory)
            
    def get_trajectories(self):
        return self.player_trajectories