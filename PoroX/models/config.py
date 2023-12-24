from typing import Optional
import jax.numpy as jnp
from flax import struct

from PoroX.models.components.embedding import (
    EmbeddingConfig,
    GlobalPlayerSegmentFFN, SegmentConfig, expand_segments,
)
from PoroX.models.components.transformer import EncoderConfig
from PoroX.models.player_encoder import PlayerConfig

@struct.dataclass
class MCTXConfig:
    policy_type: str = "gumbel" # "muzero" or "gumbel
    discount: float = 0.997
    num_simulations: int = 128
    
    # Defaults from MCTX repo
    max_depth: Optional[int] = None
    
    # Regular MuZero Policy
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    pb_c_init: float = 1.25
    pb_c_base: float = 19652
    temperature: float = 1.0
    
    # Gumbel MuZero Policy
    max_num_considered_actions: int = 32
    gumbel_scale: float = 1.0

@struct.dataclass
class MuZeroConfig:
    # Representation Network
    player_encoder: PlayerConfig
    cross_encoder: EncoderConfig
    merge_encoder: EncoderConfig
    global_encoder: EncoderConfig
    
    # Prediction Network
    policy_head: EncoderConfig
    value_head: EncoderConfig 
    
    # Dynamics Network
    dynamics_head: EncoderConfig
    reward_head: EncoderConfig
    
@struct.dataclass
class PoroXConfig:
    muzero: MuZeroConfig
    mctx: MCTXConfig
    
muzero_config = MuZeroConfig(
    player_encoder= PlayerConfig(
        embedding=EmbeddingConfig( # Hidden state of 256
            champion_embedding_size=40,
            item_embedding_size=20,
            trait_embedding_size=20,
        ),
        # Total: 75
        # 28 board, 9 bench, 5 shop, 10 items, 1 trait, 6 scalar tokens
        segment=SegmentConfig(
            segments=expand_segments([28, 9, 5, 10, 1, 6]),
            out_dim=192, # Reduce to 192
        ),
        segment_ffn=GlobalPlayerSegmentFFN,
        encoder=EncoderConfig(
            num_blocks=4,
            num_heads=4,
        ),
    ),
    cross_encoder=EncoderConfig(
        num_blocks=3,
        num_heads=2,
    ),
    merge_encoder=EncoderConfig(
        num_blocks=4,
        num_heads=4,
    ),
    global_encoder=EncoderConfig(
        num_blocks=4,
        num_heads=4,
    ),
    # V1 Policy Head
    # policy_head=EncoderConfig(
    #     num_blocks=4,
    #     num_heads=2,
    #     project_dim=38,
    #     project_blocks=2,
    # ),
    # V2 Policy Head
    # policy_head=EncoderConfig(
    #     num_blocks=4,
    #     num_heads=2,
    # ),
    # V3 Policy Head
    policy_head=EncoderConfig(
        num_blocks=6,
        num_heads=4,
    ),
    value_head=EncoderConfig(
        num_blocks=2,
        num_heads=2,
    ),
    # V1 Dynamics Head
    # dynamics_head=EncoderConfig(
    #     num_blocks=4,
    #     num_heads=5,
    #     project_dim=192, # Ensure hidden state matches segment hidden state
    #     project_blocks=2,
    #     project_num_heads=4,
    # ),
    # V2 Dynamics Head
    dynamics_head=EncoderConfig(
        num_blocks=4,
        num_heads=4,
    ),
    reward_head=EncoderConfig(
        num_blocks=2,
        num_heads=2,
    )
)

mctx_config = MCTXConfig()

porox_config = PoroXConfig(
    muzero=muzero_config,
    mctx=mctx_config,
)