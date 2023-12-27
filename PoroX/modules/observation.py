import jax
import jax.numpy as jnp
import numpy as np
import chex
from Simulator.porox.observation import ObservationVector, ActionVector

@chex.dataclass(frozen=True)
class PlayerObservation:
    champions: chex.ArrayDevice
    scalars: chex.ArrayDevice
    items: chex.ArrayDevice
    traits: chex.ArrayDevice
    
@chex.dataclass(frozen=True)
class BatchedObservation:
    players: PlayerObservation
    action_mask: chex.ArrayDevice
    opponents: PlayerObservation
    
@chex.dataclass(frozen=True)
class ObservationMapping:
    player_ids: chex.ArrayDevice
    player_len: chex.ArrayDevice
    
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

class PoroXObservation(ObservationVector):
    def __init__(self, player):
        super().__init__(player)
        
        self.board_zeros = np.zeros_like(self.board_vector)
        self.bench_zeros = np.zeros_like(self.bench_vector)
        self.shop_zeros = np.zeros_like(self.shop_vector)
        self.item_bench_zeros = np.zeros_like(self.item_bench_vector)
        self.trait_zeros = np.zeros_like(self.trait_vector)
        self.public_zeros = np.zeros_like(self.public_scalars)
        self.private_zeros = np.zeros_like(self.private_scalars)
        self.game_zeros = np.zeros_like(self.game_scalars)
        
        self.public_zeros[0] = player.player_num


    def fetch_player_observation(self):
        """Fetch Public Observation."""
        champions = np.concatenate([
            self.board_vector,
            self.bench_vector,
            self.shop_vector
        ])
        scalars = np.concatenate([
            self.public_scalars,
            self.private_scalars,
            self.game_scalars
        ])
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_vector,
            traits=self.trait_vector
        )
        
    def fetch_public_observation(self):
        """Fetch Public Observation."""
        champions = np.concatenate([
            self.board_vector,
            self.bench_vector,
            self.shop_zeros # MASK
        ])
        scalars = np.concatenate([
            self.public_scalars,
            self.private_zeros, # MASK
            self.game_zeros # MASK
        ])
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_vector,
            traits=self.trait_vector
        )

    def fetch_dead_observation(self):
        """Fetch Dead Observation."""
        champions = np.concatenate([
            self.board_zeros,
            self.bench_zeros,
            self.shop_zeros
        ])
        scalars = np.concatenate([
            self.public_zeros,
            self.private_zeros,
            self.game_zeros
        ])
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_zeros,
            traits=self.trait_zeros
        )