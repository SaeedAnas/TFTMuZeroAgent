import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

# from PoroX.models.mctx_agent import MuZeroAgent, RepresentationNetwork, PredictionNetwork, DynamicsNetwork
from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.components.embedding.dynamics import action_space_to_action
from PoroX.models.config import muzero_config, mctx_config, PoroXConfig, MCTXConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.replay_buffer import create_trajectories, LocalBuffer, trajectory_analytics, pad_trajectories, reshape_trajectories, combine_game_trajectories, sample_trajectory
from PoroX.test.utils import profile

def test_store_trajectories(first_obs, key):
    batched_obs, mapping = batch_utils.collect_obs(first_obs)
    
    local_buffer = LocalBuffer()
    

    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
            num_simulations=4,
            max_num_considered_actions=4,
        )
    )
    
    agent = PoroXV1(config, key, batched_obs)
    output = agent.act(batched_obs)
    # Dummy reward
    reward = jnp.zeros_like(output.value, dtype=jnp.float16)
    print(output.action)
    
    def apply(output, first_obs, reward, mapping):
        trajectories = create_trajectories(output, first_obs, reward, mapping)
        local_buffer.store_batch_trajectories(trajectories)
        
    N = 200
    profile(N, apply, output, first_obs, reward, mapping)
    
    trajectories = local_buffer.get_trajectories()

    for id, trajectory in trajectories.items():
        print(f"Original player_{id} trajectory analytics: ")
        print(f"Trajectory shape: {trajectory.action.shape}")
        trajectory_analytics(trajectory)

    profile(1, pad_trajectories, trajectories)
    padded_trajectories = pad_trajectories(trajectories)
        
    for id, trajectory in padded_trajectories.items():
        print(f"Padded player_{id} trajectory analytics: ")
        print(f"Trajectory shape: {trajectory.action.shape}")
        trajectory_analytics(trajectory)

    profile(1, reshape_trajectories, padded_trajectories)
    reshaped_trajectories = reshape_trajectories(padded_trajectories)

    for id, trajectory in reshaped_trajectories.items():
        print(f"Reshaped player_{id} trajectory analytics: ")
        print(f"Trajectory shape: {trajectory.action.shape}")
        trajectory_analytics(trajectory)
    
    profile(1, combine_game_trajectories, reshaped_trajectories)
    combined_trajectories = combine_game_trajectories(reshaped_trajectories)
    
    print("Combined trajectories Before Sampling: ")
    print(f"Combined trajectories shape: {combined_trajectories.action.shape}")
    trajectory_analytics(combined_trajectories)

    profile(1, sample_trajectory, combined_trajectories, key, 12)
    sampled_trajectory, combined_trajectories = sample_trajectory(combined_trajectories, key, batch_size=256)

    print("Combined trajectories After Sampling: ")
    print(f"Combined trajectories shape: {combined_trajectories.action.shape}")
    trajectory_analytics(combined_trajectories)
    
    print("Sampled trajectory: ")
    print(f"Sampled trajectory shape: {sampled_trajectory.action.shape}")
    trajectory_analytics(sampled_trajectory)
    