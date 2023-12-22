import jax
import jax.numpy as jnp
from PoroX.modules.observation import BatchedObservation, PlayerObservation

# Broadcast Player State to [...B, P, S1, L]
@jax.jit
def expand_and_get_shape(player_state, opponent_state):
    broadcasted_shape = player_state.shape[:-2] + opponent_state.shape[-3:-2] + player_state.shape[-2:]
    expanded_player_state = jnp.expand_dims(player_state, axis=-3)
    return broadcasted_shape, expanded_player_state

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
def split(collection, schema):
    """
    Splits a batched collection back into a list of collections based on a schema.
    """
    # TODO: Figure out how to do this with jax.tree_map
    ...

@jax.jit
def collect_obs(obs: dict):
    players = [
        player_obs["player"]
        for player_obs in obs.values()
    ]
    
    action_mask = [
        player_obs["action_mask"]
        for player_obs in obs.values()
    ]
    
    opponents = [
        collect(player_obs["opponents"])
        for player_obs in obs.values()
    ]
    
    players = collect(players)
    action_mask = collect(action_mask)
    opponents = collect(opponents)
    
    # Pad to have 8 players
    players = create_padded_player(players)
    action_mask = pad_array(action_mask)
    opponents = create_padded_player(opponents)

    return BatchedObservation(
        players=players,
        action_mask=action_mask,
        opponents=opponents,
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