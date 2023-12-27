import chex
import jax
import jax.numpy as jnp

from PoroX.modules.observation import BatchedObservation
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
    
    player_len = len(list_mapping)
    mapping = mapping[:player_len]
    
    trajectories = {}
    
    for player_id in mapping:
        player_id = int(player_id)
        idx = list_mapping[player_id]
        
        trajectory = Trajectory(
            observation=list_obs[idx],
            action=output.action[idx],
            policy_logits=output.action_weights[idx],
            value=output.value[idx],
            reward=reward[idx]
        )
        
        trajectories[player_id] = trajectory
    
    return trajectories

def reshape_trajectories(trajectory, unroll_steps):
    """
    Trajectories will be of the entire game.
    We need to reshape the trajectories into unroll steps.
    The trajectories will already be padding to be a multiple of unroll_steps.
    """
    
class LocalBuffer:
    """
    Store game trajectories for each agent.
    """
    dict_trajectories: defaultdict[int, Trajectory]
    
    def __init__(self):
        self.dict_trajectories = defaultdict(list)
        
    def store_trajectory(self, player_id, trajectory):
        self.dict_trajectories[player_id].append(trajectory)
        
    def store_batch_trajectories(self, trajectories):
        for player_id, trajectory in trajectories.items():
            self.store_trajectory(player_id, trajectory)
            
    def collect_trajectories(self, unroll_steps=6):
        collected_trajectories = {
            id: batch_utils.collect(trajectories)
            for id, trajectories in self.dict_trajectories.items()
        }
        
        def calculate_max_len(current_len, unroll_steps):
            """
            We need to make max_len be a multiple of unroll_steps.
            Ex. current_len = 10, unroll_steps = 4 -> max_len = 12
            """
            max_len = current_len + (current_len % unroll_steps)
            print(max_len, current_len)
            return max_len
        
        padded_trajectories = {
            id: batch_utils.pad_collection(
                trajectory,
                max_length=calculate_max_len(len(trajectory.action), unroll_steps)
            )
            for id, trajectory in collected_trajectories.items()
        }
        
        return padded_trajectories