import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

# from PoroX.models.mctx_agent import MuZeroAgent, RepresentationNetwork, PredictionNetwork, DynamicsNetwork
from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.components.embedding.dynamics import action_space_to_action
from PoroX.models.config import muzero_config, mctx_config, PoroXConfig, MCTXConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.replay_buffer import create_trajectories, LocalBuffer, trajectory_analytics
from PoroX.test.utils import profile

def test_store_trajectories(first_obs, key):
    batched_obs, mapping = batch_utils.collect_obs(first_obs)
    
    local_buffer = LocalBuffer()
    

    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
            num_simulations=2,
            max_num_considered_actions=1,
        )
    )
    
    agent = PoroXV1(config, key, batched_obs)
    output = agent.act(batched_obs)
    # Dummy reward
    reward = jnp.zeros_like(output.value, dtype=jnp.float16)
    trajectories = create_trajectories(output, first_obs, reward, mapping)
    
    N = 5000
    profile(N, local_buffer.store_batch_trajectories, trajectories)
    
    # Test collection 
    collected = local_buffer.collect_trajectories(unroll_steps=6)
    
    for id, trajectory in collected.items():
        print(f"player_{id} trajectory analytics: ")
        trajectory_analytics(trajectory)