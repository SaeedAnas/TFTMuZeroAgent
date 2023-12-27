from typing import Any, Callable, Optional

from flax import linen as nn
from flax import struct
import jax.numpy as jnp

from PoroX.modules.observation import PlayerObservation

from PoroX.models.components.scalar_encoder import ScalarEncoder
from PoroX.models.components.fc import MLP, FFNSwiGLU

from PoroX.models.defaults import DEFAULT_DTYPE

# -- Config -- #
@struct.dataclass
class EmbeddingConfig:
    # Champion Embedding
    champion_embedding_size: int = 30
    item_embedding_size: int = 10
    trait_embedding_size: int = 8
    stats_size: int = 12 + 4 + 15
    
    # Scalar Config
    scalar_min_value: int = 0
    scalar_max_value: int = 200

    # General Embedding Config
    num_champions: int = 60
    num_items: int = 60
    num_traits: int = 27
    
    dtype = DEFAULT_DTYPE
        
def vector_size(config: EmbeddingConfig):
    return (
        config.champion_embedding_size +
        config.item_embedding_size * 3 +
        config.trait_embedding_size * 7 +
        config.stats_size
    )
        
# -- Champion Embedding -- #

class ChampionEmbedding(nn.Module):
    """
    Embeds a champion vector into a latent space

    Champion Vector:
    0: championID
    1: chosen (0 or 1)
    2: stars (0-3)
    3: cost
    4-6: itemID
    7-13: traitID
    13-24: stats
    """
    config: EmbeddingConfig
    
    @nn.compact
    def __call__(self, x):
        dtype = self.config.dtype

        championID = x[..., 0].astype(jnp.int16)
        stars = x[..., 1]
        cost = x[..., 2]
        itemIDs = x[..., 3:6].astype(jnp.int16)
        traitIDs = x[..., 6:13].astype(jnp.int16)
        stats = x[..., 13:]
        
        champion_embedding = nn.Embed(
            num_embeddings=self.config.num_champions,
            features=self.config.champion_embedding_size,
            param_dtype=dtype)(championID)
        item_embedding = nn.Embed(
            num_embeddings=self.config.num_items,
            features=self.config.item_embedding_size,
            param_dtype=dtype)(itemIDs)
        trait_embedding = nn.Embed(
            num_embeddings=self.config.num_traits,
            features=self.config.trait_embedding_size,
            param_dtype=dtype)(traitIDs)
        
        batch_shape = item_embedding.shape[:-2]

        item_embedding = jnp.reshape(
            item_embedding,
            newshape=(*batch_shape, self.config.item_embedding_size * 3)
        )

        trait_embedding = jnp.reshape(
            trait_embedding,
            newshape=(*batch_shape, self.config.trait_embedding_size * 7)
        )
        
        # one-hot encode stars
        stars_one_hot = nn.one_hot(stars.astype(jnp.int16), num_classes=4, dtype=dtype)
        
        # one-hot encode cost
        # Actual would be 5 * 9 but i'll be damned if an agent ever gets to a 3 star 5 cost
        # Just use 2 star 5 cost as the max
        cost_one_hot = nn.one_hot(cost.astype(jnp.int16), num_classes=15, dtype=dtype)

        # Expand cost to match the shape of the other embeddings
        cost = jnp.expand_dims(cost, axis=-1)
        
        stats = stats.astype(dtype)
        
        return jnp.concatenate([
            champion_embedding,
            item_embedding,
            trait_embedding,
            stars_one_hot,
            cost_one_hot,
            stats
        ], axis=-1)
        
# -- Item Embedding -- #
        
class ItemBenchEmbedding(nn.Module):
    """
    Embeds item bench ids to a latent space
    """
    config: EmbeddingConfig

    @nn.compact
    def __call__(self, x):
        ids = jnp.int16(x)

        bench_item_embedding = nn.Embed(
            num_embeddings=self.config.num_items,
            features=vector_size(self.config),
            param_dtype=self.config.dtype)(ids)

        return bench_item_embedding

# -- Trait Embedding -- #
    
class TraitEmbedding(nn.Module):
    """
    Embeds trait ids to a latent space
    """
    config: EmbeddingConfig
    
    @nn.compact
    def __call__(self, x):
        return MLP(features=[
            self.config.num_traits,
            vector_size(self.config)
            ], dtype=self.config.dtype)(x)
        
class ScalarEmbedding(nn.Module):
    """
    Compress the scalar values into fewer tokens to save compute
    
    # TODO: Make the scalar vector just match these tokens

    Convert to:
    - id token
        0: playerID
        13: opponent1
        14: opponent2
        15: opponent3
    
    - actions remaining
        12: actions remaining

    - health token
        1: health
    
    - gold token
        7: economy
        8: gold

    - level token
        2: level
        5: max units
        6: available units
        9: exp
        10: exp to next level
    
    - game token
        11: round
        3: win streak
        4: loss streak

    """

    config: EmbeddingConfig
    
    def setup(self):
        self.scalar_encoder = ScalarEncoder(
            min_value=self.config.scalar_min_value,
            max_value=self.config.scalar_max_value,
            num_steps=50
        )

    @nn.compact
    def __call__(self, x):
        # --- ID Token --- #
        playerID = x[..., 0:1]
        opponents = x[..., 13:16]

        id_token = jnp.concatenate([
            playerID,
            opponents
        ], axis=-1)
        
        # --- Actions Remaining --- #
        action_token = x[..., 12:13]

        # --- Health Token --- #
        health_token = x[..., 1:2]

        # --- Gold Token --- #
        gold_token = x[..., 7:9]
        
        # --- Level Token --- #
        level = x[..., 2:3]
        units = x[..., 5:7]
        exp = x[..., 9:11]

        level_token = jnp.concatenate([
            level,
            units,
            exp
        ], axis=-1)
        
        # --- Game Token --- #
        round = x[..., 11:12]
        streak = x[..., 3:5]

        game_token = jnp.concatenate([
            round,
            streak
        ], axis=-1)
        
        # Run through FFN
        def ffn():
            return FFNSwiGLU(out_dim=vector_size(self.config), dtype=self.config.dtype)
        
        def encode_and_concat(x):
            """Convert the scalars into vectors of the same size and concatenate"""
            encoded_x = self.scalar_encoder.encode(x)
            flattened_x = jnp.reshape(encoded_x, (encoded_x.shape[:-2] + (-1,)))
            expanded_x = jnp.expand_dims(flattened_x, axis=-2)
            return expanded_x
            
        id_token = ffn()(encode_and_concat(id_token))
        action_token = ffn()(encode_and_concat(action_token))
        health_token = ffn()(encode_and_concat(health_token))
        gold_token = ffn()(encode_and_concat(gold_token))
        level_token = ffn()(encode_and_concat(level_token))
        game_token = ffn()(encode_and_concat(game_token))
        
        embeddings = jnp.concatenate([
            id_token,
            action_token,
            health_token,
            gold_token,
            level_token,
            game_token
        ], axis=-2)
        
        return embeddings

    
# -- Player Embedding -- #
        
class PlayerEmbedding(nn.Module):
    config: EmbeddingConfig
    
    @nn.compact
    def __call__(self, x: PlayerObservation):
        champion_embeddings = ChampionEmbedding(self.config)(x.champions)
        
        item_bench_embeddings = ItemBenchEmbedding(self.config)(x.items)

        trait_embeddings = TraitEmbedding(self.config)(x.traits)
        trait_embeddings = jnp.expand_dims(trait_embeddings, axis=-2)
        
        scalar_embeddings = ScalarEmbedding(self.config)(x.scalars)

        player_embedding = jnp.concatenate([
            champion_embeddings,
            item_bench_embeddings,
            trait_embeddings,
            scalar_embeddings,
        ], axis=-2)
        
        return player_embedding