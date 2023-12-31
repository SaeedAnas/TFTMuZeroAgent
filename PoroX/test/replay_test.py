import jax.numpy as jnp

from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.config import muzero_config, PoroXConfig, MCTXConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.game import GameBuffer
from PoroX.modules.trajectory import create_trajectories, trajectory_size
import PoroX.modules.replay_buffer as rpb
from PoroX.test.utils import profile

def test_store_trajectories(first_obs, key):
    batched_obs, mapping = batch_utils.collect_obs(first_obs)
    
    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
            num_simulations=4,
            max_num_considered_actions=4,
        )
    )

    game_buffer = GameBuffer(unroll_steps=6)
    
    agent = PoroXV1(config, key, batched_obs)
    output = agent.act(batched_obs)

    # Dummy reward
    reward = jnp.zeros_like(output.value, dtype=jnp.float16)
    print(output.action)
    
    def apply(output, first_obs, reward, mapping):
        trajectories = create_trajectories(output, first_obs, reward, mapping)
        game_buffer.store_batch_experience(trajectories)
        
    N = 390
    profile(N, apply, output, first_obs, reward, mapping)
    
    buffers = game_buffer.get_buffers()

    for player_id, buffer in buffers.items():
        print(player_id, buffer.buffer.action.shape)

    combined_buffers = rpb.combine(list(buffers.values()))
    
    print('combined', combined_buffers.buffer.action.shape)
    print(f'size: {trajectory_size(combined_buffers.buffer) / 1e6} MB')
    
    sampled_trajectory, combined_buffers = rpb.sample(combined_buffers, key, batch_size=64)
    
    print('sampled', sampled_trajectory.action.shape)
    print('new combined', combined_buffers.buffer.action.shape)