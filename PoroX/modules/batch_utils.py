import time
from functools import partial
import jax
import jax.numpy as jnp
from PoroX.modules.observation import BatchedObservation, PlayerObservation, ObservationMapping

@jax.jit
def flatten(x):
    """
    x will be given as (Game, Player, ...)
    We want to flatten to (Game * Player, ...)
    Return flattened_x
    """
    
    flattened_shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
    flattened_x = jnp.reshape(x, flattened_shape) 
    
    return flattened_x

def unflatten(x, original_shape):
    """
    x will be given as (Game * Player, ...)
    We want to unflatten to (Game, Player, ...)
    
    The ... will most likely be different than the original shape,
    We only want to reshape the first two dimensions.
    """
    
    x_shape = x.shape
    unflattened_shape = original_shape[:2] + x_shape[1:]
    unflattened_x = jnp.reshape(x, unflattened_shape)
    
    return unflattened_x

@jax.jit
def collect(collection):
    return jax.tree_map(lambda *xs: jnp.stack(xs), *collection)

@jax.jit
def concat(collection, axis=0):
    return jax.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *collection)

def pad_array(array, max_length=8, axis=0):
    """
    Pad observation arrays to a max_length.
    For example: 
    if array is of shape (7, 42, 30) and max_length is 8,
    then return array of shape (8, 42, 30)
    """
    pad_length = max_length - array.shape[axis]
    padded_array = jnp.pad(array, [(0, pad_length)] + [(0, 0)] * (array.ndim - 1))
    
    return padded_array

def pad_collection(collection, max_length):
    """
    Pad a collection of arrays to a max_length.
    Every item in the collection must be an array.
    """

    return jax.tree_map(
        lambda x: pad_array(x, max_length=max_length),
        collection
    )

def collect_obs(obs: dict):
    # Ensure same order
    values = list(obs.values())

    players = [
        player_obs["player"]
        for player_obs in values
    ]
    
    action_mask = [
        player_obs["action_mask"]
        for player_obs in values
    ]
    
    opponents = [
        collect(player_obs["opponents"])
        for player_obs in values
    ]
    
    players = collect(players)
    action_mask = collect(action_mask)
    opponents = collect(opponents)

    # Make action mask compatible with mctx
    action_mask = invert_and_flatten_action_mask(action_mask)
    
    batched_obs = BatchedObservation(
        players=players,
        action_mask=action_mask,
        opponents=opponents,
    )
    
    player_ids = players.scalars[:, 0]
    mapping = ObservationMapping(
        player_ids=jnp.array(player_ids).astype(jnp.int8),
        player_len=jnp.array(len(values)).astype(jnp.int8)
    )
    
    padded_obs = jax.tree_map(
        lambda x: pad_array(x, max_length=8),
        batched_obs,
    )
    
    return padded_obs, mapping

def collect_list_obs(obs: dict):
    """
    Return obs as a list of BatchedObservations.
    """
    values = list(obs.values())

    batched_obs = [
        BatchedObservation(
            players=player_obs["player"],
            action_mask=player_obs["action_mask"],
            opponents=collect(player_obs["opponents"]),
        )
        
        for player_obs in values
    ]
    mapping = {
        player_obs["player"].scalars[0].astype(int): idx
        for idx, player_obs in enumerate(values)
    }
    
    return batched_obs, mapping

@jax.jit
def invert_and_flatten_action_mask(action_mask):
    """Utility to make action mask be compatible with mctx"""
    
    # Flatten last two dimensions
    action_shape = action_mask.shape
    mask_flattened = jnp.reshape(action_mask, action_shape[:-2] + (-1,))
    
    # Invert mask
    inverted_mask = 1 - mask_flattened
    
    return inverted_mask

    
def collect_multi_game_obs(obs):
    """
    Collect a batched_obs for multiple games.
    """
    
    batched_obs = []
    batched_mapping = []
    
    for game_obs in obs:
        game_obs, mapping = collect_obs(game_obs)
        batched_obs.append(game_obs)
        batched_mapping.append(mapping)
        
    return collect(batched_obs), batched_mapping

@jax.jit
def flatten_obs(obs):
    """
    Flatten a batched_obs for multiple games and return the original shape.
    """
    
    original_shape = obs.players.champions.shape
    
    return jax.tree_map(
        lambda x: flatten(x),
        obs
    ), original_shape
    
def batch_map_actions(actions: jnp.ndarray, mapping: list[ObservationMapping]):
    """
    Actions are in shape (Game, Player, Action)
    
    Need to convert it into a list of actions for each game
    [
        {
            "player_{id}": action
        }
    ]
    
    obs contains a mapping that links each action to a player_id
    """
    
    return [
        map_actions(
            actions[i],
            mapping[i],
        )
        
        for i in range(len(actions))
    ]
    
def map_actions(actions: jnp.ndarray, mapping: ObservationMapping):
    actions = actions[:mapping.player_len]
    player_ids = mapping.player_ids[:mapping.player_len]
    
    return {
        f"player_{player_id}": action
        for player_id, action in zip(player_ids, actions)
    }