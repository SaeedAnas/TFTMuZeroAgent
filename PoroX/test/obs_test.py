import jax.numpy as jnp

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_batch_obs(first_obs):
    # Test Logic
    obs = batch_utils.collect_obs(first_obs)
    print(obs.players.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)
    print(obs.player_ids)
    print(obs.opponents.scalars[..., 0])
    print(obs.player_len)
    
    # Profile
    N = 1000
    profile(N, batch_utils.collect_obs, first_obs)
    
def test_flattened_game_obs(first_batched_obs):
    obs = batch_utils.collect_multi_game_obs(first_batched_obs)
    obs, original_shape = batch_utils.flatten_multi_game_obs(obs)
    print(obs.players.champions.shape)
    print(obs.opponents.champions.shape)
    
    unflattened_players = batch_utils.unflatten(obs.players.champions, original_shape)
    print(unflattened_players.shape)