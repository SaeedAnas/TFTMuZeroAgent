import jax
import jax.numpy as jnp
import numpy as np
import chex
from Simulator.porox.observation import ObservationVector
from PoroX.modules.observation import BatchedObservation, PlayerObservation
from PoroX.modules import batch_utils

@chex.dataclass(frozen=True)
class Trajectory:
    observation: BatchedObservation
    action: chex.ArrayDevice
    policy_logits: chex.ArrayDevice
    value: chex.ArrayDevice
    reward: chex.ArrayDevice
    
def create_trajectories(output, obs, reward, mapping):
    """
    Store the output of an agent in the replay buffer.
    
    output: AgentOutput
    - Contains policy_logits, action, and value
    
    obs: Dict Observation
    
    reward: jnp.ndarray
    """
    list_obs, list_mapping = batch_utils.collect_list_obs(obs)

    trajectories = {}
    
    for player_id in mapping.player_ids:
        player_id = int(player_id)
        player_idx = list_mapping[player_id]

        player_observation = list_obs[player_idx]
        player_output = jax.tree_map(lambda x: x[player_idx], output)
        player_reward = reward[player_idx]
        
        trajectory = Trajectory(
            observation=compress_observation(player_observation), # Saves memory by converting observation to int8
            action=player_output.action,
            policy_logits=player_output.action_weights,
            value=player_output.value,
            reward=player_reward,
        )

        trajectories[player_id] = trajectory
    
    return trajectories

@jax.jit
def compress_observation(observation: BatchedObservation):
    
    def compress_player(player: PlayerObservation):
        """
        Compress player
        We can compute the stats later.
        """

        # Compress champions by removing stats and all traits except chosen
        champions = player.champions[..., :7].astype(jnp.int8)
        scalars = player.scalars.astype(jnp.int8)
        items = player.items.astype(jnp.int8)
        traits = player.traits.astype(jnp.int8)
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=items,
            traits=traits
        )
        
    def compress_action_mask(action_mask):
        """
        Compress action mask
        """
        return action_mask.astype(jnp.int8)
    
    return BatchedObservation(
        players=compress_player(observation.players),
        action_mask=compress_action_mask(observation.action_mask),
        opponents=compress_player(observation.opponents),
    )