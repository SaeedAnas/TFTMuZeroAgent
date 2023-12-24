import jax
import jax.numpy as jnp
from PoroX.modules.observation import BatchedObservation, PlayerObservation, to_fp16

@jax.jit
def flatten(x):
    """
    x will be given as (Game, Player, ...)
    We want to flatten to (Game * Player, ...)
    Return flattened_x and the original shape
    """
    
    original_shape = x.shape
    flattened_shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
    flattened_x = jnp.reshape(x, flattened_shape) 
    
    return flattened_x, original_shape

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

@jax.jit
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
    
    player_ids = players.scalars[:, 0]
    
    # Pad to have 8 players
    players = create_padded_player(players)
    action_mask = pad_array(action_mask)
    opponents = create_padded_player(opponents)
    player_ids = pad_array(player_ids)
    
    # Make action mask compatible with mctx
    action_mask = invert_and_flatten_action_mask(action_mask)

    return BatchedObservation(
        players=to_fp16(players),
        action_mask=action_mask,
        opponents=to_fp16(opponents),

        player_ids=jnp.array(player_ids).astype(jnp.int8),
        player_len=jnp.array(len(values)).astype(jnp.int8)
    )
    


def create_padded_player(players: PlayerObservation):
    """
    Creates a padded observation for a single game.
    """
    # Pad Champions
    champions = pad_array(players.champions)
    
    # Pad Items
    items = pad_array(players.items)
    
    # Pad Traits
    traits = pad_array(players.traits)
    
    # Pad Scalars
    scalars = pad_array(players.scalars)
    
    return PlayerObservation(
        champions=champions,
        items=items,
        traits=traits,
        scalars=scalars
    )
    
@jax.jit
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

@jax.jit
def invert_and_flatten_action_mask(action_mask):
    """Utility to make action mask be compatible with mctx"""
    
    # Flatten last two dimensions
    action_shape = action_mask.shape
    mask_flattened = jnp.reshape(action_mask, action_shape[:-2] + (-1,))
    
    # Invert mask
    inverted_mask = 1 - mask_flattened
    
    return inverted_mask

    
@jax.jit
def collect_multi_game_obs(obs):
    """
    Collect a batched_obs for multiple games.
    """
    
    batched_obs = [
        collect_obs(game_obs)
        for game_obs in obs
    ]

    return collect(batched_obs)

@jax.jit
def flatten_multi_game_obs(obs):
    """
    Flatten a batched_obs for multiple games and return the original shape.
    """
    
    players, original_shape = flatten_player_obs(obs.players)
    action_mask, _ = flatten(obs.action_mask)
    opponents, _ = flatten_player_obs(obs.opponents)
    
    return BatchedObservation(
        players=players,
        action_mask=action_mask,
        opponents=opponents,
        
        player_ids=obs.player_ids,
        player_len=obs.player_len
    ), original_shape
    
@jax.jit
def flatten_player_obs(player):
    """
    Flatten a batched_obs for a single player and return the original shape.
    """
    
    champions, original_shape = flatten(player.champions)
    items, _ = flatten(player.items)
    traits, _ = flatten(player.traits)
    scalars, _ = flatten(player.scalars)
    
    return PlayerObservation(
        champions=champions,
        items=items,
        traits=traits,
        scalars=scalars
    ), original_shape
    
def batch_map_actions(actions: jnp.ndarray, obs: BatchedObservation):
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
    
    action_dicts = []
    
    for i in range(len(actions)):
        action_dict = map_actions(
            actions[i],
            obs.player_ids[i],
            obs.player_len[i]
        )
        
        action_dicts.append(action_dict)
        
    return action_dicts
    
def map_actions(actions: jnp.ndarray, player_ids: jnp.ndarray, player_len: jnp.int32):
    actions = actions[:player_len]
    player_ids = player_ids[:player_len]
    
    return {
        f"player_{player_id}": action
        for player_id, action in zip(player_ids, actions)
    }