import jax.numpy as jnp

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_batch_obs(first_obs):
    # Test Logic
    obs = batch_utils.collect_obs(first_obs)
    print(obs.players.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)
    
    # Profile
    N = 1000
    profile(N, batch_utils.collect_obs, first_obs)
    
def test_expand():
    # Test Logic
    x = jnp.ones((8, 23, 40))
    y = jnp.ones((8, 28, 10))
    z = jnp.ones((8, 23, 2))
    collection = [x, y, z]
    collection = batch_utils.expand(collection, axis=-3)
    print(collection[0].shape)
    
def test_game_batch_obs(first_batched_obs):
    print(len(first_batched_obs))
    print(first_batched_obs[0].keys())
    obs = batch_utils.collect_multi_game_obs(first_batched_obs)
    print(obs.players.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)
    print(obs.num_players)

    # Profile
    N = 100
    profile(N, batch_utils.collect_multi_game_obs, first_batched_obs)
    
def test_padded_game_obs(first_obs):
    first_obs.pop("player_1")
    obs = batch_utils.collect_obs(first_obs)
    print(obs.players.champions.shape)
    
def test_flattened_game_obs(first_batched_obs):
    obs = batch_utils.collect_multi_game_obs(first_batched_obs)
    obs, original_shape = batch_utils.flatten_multi_game_obs(obs)
    print(obs.players.champions.shape)
    print(obs.opponents.champions.shape)
    
    unflattened_players = batch_utils.unflatten(obs.players.champions, original_shape)
    print(unflattened_players.shape)