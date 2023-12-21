from typing import Optional
from flax import struct

from PoroX.models.components.embedding import (
    EmbeddingConfig,
    GlobalPlayerSegmentFFN, SegmentConfig, expand_segments,
)
from PoroX.models.components.transformer import EncoderConfig
from PoroX.models.player_encoder import PlayerConfig

@struct.dataclass
class MCTXConfig:
    discount: float = 0.997
    num_simulations: int = 50
    
    # Defaults from MCTX repo
    max_depth: Optional[int] = None
    
    # Regular MuZero Policy
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    pb_c_init: float = 1.25
    pb_c_base: float = 19652
    temperature: float = 1.0
    
    # Gumbel MuZero Policy
    max_num_considered_actions: int = 60
    gumbel_scale: float = 1.0

@struct.dataclass
class MuZeroConfig:
    mctx_config: MCTXConfig
    player_encoder: PlayerConfig
    cross_encoder: EncoderConfig
    merge_encoder: EncoderConfig
    policy_head: EncoderConfig
    value_head: EncoderConfig 
    dynamics_head: EncoderConfig
    reward_head: EncoderConfig

test_config = MuZeroConfig(
    mctx_config=MCTXConfig(),

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
            num_blocks=8,
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
    policy_head=EncoderConfig(
        num_blocks=4,
        num_heads=2,
        project_dim=38,
        project_blocks=2,
    ),
    value_head=EncoderConfig(
        num_blocks=2,
        num_heads=2,
    ),
    dynamics_head=EncoderConfig(
        num_blocks=4,
        num_heads=5,
        project_dim=192, # Ensure hidden state matches segment hidden state
        project_blocks=2,
        project_num_heads=4,
    ),
    reward_head=EncoderConfig(
        num_blocks=2,
        num_heads=2,
    )
)