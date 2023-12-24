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
    
    # Mapping to keep track of which player is which
    player_ids: chex.ArrayDevice
    player_len: chex.ArrayDevice
    
@jax.jit
def to_fp16(x: PlayerObservation):
    return PlayerObservation(
        champions=x.champions.astype(jnp.float16),
        scalars=x.scalars.astype(jnp.float16),
        items=x.items.astype(jnp.float16),
        traits=x.traits.astype(jnp.float16)
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
        
class PoroXAction(ActionVector):
    def fetch_action_mask(self):
        """
        The ActionVector action mask gives valid actions 1 and invalid actions 0.
        MCTX on the other hand excpects valid actions to be 0 and invalid actions to be 1.
        
        So we need to flip the mask.
        """
        mask = super().fetch_action_mask()
        mask = 1 - mask
        
        # Flatten mask on last two dimensions
        # action_shape = mask.shape
        # mask_flattened = np.reshape(mask, action_shape[:-2] + (-1,))
        
        # I flatten just so its easier to apply the mask with mctx
        
        return mask