import chex
import jax
import jax.numpy as jnp

from PoroX.modules.observation import BatchedObservation, compress_observation
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

def reshape_trajectories(trajectory, unroll_steps):
    """
    Trajectories will be of the entire game.
    We need to reshape the trajectories into unroll steps.
    The trajectories will already be padding to be a multiple of unroll_steps.
    """
    def reshape(x):
        return jnp.reshape(x, (unroll_steps, -1) + x.shape[1:])
    
    return batch_utils.map_collection(trajectory, reshape)
    
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
            
    def calculate_max_len(self, current_len, unroll_steps):
        """
        We need to make max_len be a multiple of unroll_steps.
        Ex. current_len = 10, unroll_steps = 4 -> max_len = 12
        """
        max_len = current_len + (current_len % unroll_steps)
        return max_len

    def collect_trajectories(self, unroll_steps=6):
        collected_trajectories = self.player_trajectories
        
        
        padded_trajectories = {
            id: batch_utils.pad_collection(
                trajectory,
                max_length=self.calculate_max_len(len(trajectory.action), unroll_steps)
            )
            for id, trajectory in collected_trajectories.items()
        }
        
        return padded_trajectories