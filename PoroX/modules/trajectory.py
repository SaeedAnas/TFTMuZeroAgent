import jax
import jax.numpy as jnp
import chex

from PoroX.modules.observation import BatchedObservation, PlayerObservation, ObservationMapping
import PoroX.modules.batch_utils as batch_utils
from PoroX.models.mctx_agent import PoroXOutput

@chex.dataclass(frozen=True)
class CompressedPlayerObservation:
    """
    Split the champions vector into two parts: one for ids and scalars, and one for the normalized stats.
    """
    champions: chex.ArrayDevice
    stats: chex.ArrayDevice
    scalars: chex.ArrayDevice
    items: chex.ArrayDevice
    traits: chex.ArrayDevice
    
@chex.dataclass(frozen=True)
class CompressedObservation:
    player: CompressedPlayerObservation
    action_mask: chex.ArrayDevice
    opponents: CompressedPlayerObservation

@chex.dataclass(frozen=True)
class Trajectory:
    observation: CompressedObservation
    action: chex.ArrayDevice
    policy_logits: chex.ArrayDevice
    value: chex.ArrayDevice
    reward: chex.ArrayDevice
    
def create_trajectories(
        output: PoroXOutput,
        obs: BatchedObservation,
        reward: dict[str, float],
        mapping: ObservationMapping
        ) -> dict[str, Trajectory]:

    obs = batch_utils.split_obs(obs, mapping)
    
    trajectories = {}
    
    for idx in range(len(mapping.player_ids)):
        player_id = f'player_{mapping.player_ids[idx]}'

        player_output = batch_utils.index(output, idx)
        player_obs = obs[player_id]
        player_reward = reward[player_id]
        
        trajectory = Trajectory(
            observation=compress_observation(player_obs), # Saves memory by converting observation to int8
            action=player_output.action,
            policy_logits=player_output.action_weights,
            value=player_output.value,
            reward=player_reward,
        )
        
        trajectories[player_id] = trajectory
        
    return trajectories

def create_batch_trajectories(
        output: PoroXOutput,
        obs: BatchedObservation,
        reward: list[dict[str, float]],
        mapping: list[ObservationMapping]
        ) -> list[dict[str, Trajectory]]:

    return [
        create_trajectories(
            batch_utils.index(output, idx),
            batch_utils.index(obs, idx),
            reward[idx],
            mapping[idx]
        )
        for idx in range(len(mapping))
    ]

@jax.jit
def compress_player(player: PlayerObservation):
    """
    Compress player
    We can compute the stats later.
    
    Champion IDs and Scalars:
    1. ChampionID (1)
    2,3. Stars, Cost (2)
    4-6. Items (3)
    7-13. Traits (7)
    
    Champion Stats:
    14-end. Stats
    """

    # Compress champions by removing stats and all traits except chosen
    champions = player.champions[..., :13].astype(jnp.int8)
    stats = player.champions[..., 13:]
    scalars = player.scalars.astype(jnp.int8)
    items = player.items.astype(jnp.int8)
    traits = player.traits.astype(jnp.int8)
    
    return CompressedPlayerObservation(
        champions=champions,
        stats=stats,
        scalars=scalars,
        items=items,
        traits=traits
    )
    
@jax.jit
def compress_observation(observation: BatchedObservation):
    return BatchedObservation(
        players=compress_player(observation.players),
        action_mask=observation.action_mask.astype(jnp.int8),
        opponents=compress_player(observation.opponents),
    )
    
@jax.jit
def decompress_player(player: CompressedPlayerObservation):
    """
    Re-combine the champion IDs and stats.
    """
    champions = player.champions.astype(jnp.float16)
    
    champions = jnp.concatenate([
        champions,
        player.stats
    ], axis=-1)
    
    return PlayerObservation(
        champions=champions,
        scalars=player.scalars.astype(jnp.float16),
        items=player.items.astype(jnp.float16),
        traits=player.traits.astype(jnp.float16),
    )

@jax.jit    
def decompress_observation(observation: BatchedObservation):
    return BatchedObservation(
        players=decompress_player(observation.players),
        action_mask=observation.action_mask.astype(jnp.float16),
        opponents=decompress_player(observation.opponents),
    )
    
def trajectory_size(trajectory):
    """Compute the bytes of a trajectory."""
    nbytes = jax.tree_map(lambda x: x.nbytes, trajectory)
    nbytes, _ = jax.tree_flatten(nbytes)
    return sum(nbytes)