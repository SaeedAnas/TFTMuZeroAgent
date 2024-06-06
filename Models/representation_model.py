import torch
import torch.nn as nn

import config

from Models.abstract_model import AbstractNetwork
from Models.MuZero_torch_agent import RepNetwork
from Simulator.config import MAX_CHAMPION_IN_SET, MAX_ITEMS_IN_SET, BOARD_SIZE, BENCH_SIZE, SHOP_SIZE, ITEM_BENCH_SIZE
from Simulator.origin_class_stats import tiers
from Models.torch_layers import MultiMlp, ResidualBlock, TransformerEncoder, mlp


class RepresentationTesting(AbstractNetwork):
    def __init__(self, model_config):
        super().__init__()
        self.full_support_size = model_config.ENCODER_NUM_STEPS

        # self.representation_network = RepNetwork(
        #     [model_config.LAYER_HIDDEN_SIZE // 2] * model_config.N_HEAD_HIDDEN_LAYERS,
        #     model_config.HIDDEN_STATE_SIZE // 2,
        #     model_config
        # )

        self.representation_network = GeminiRepresentation(model_config)

        self.prediction_network = PredNetwork(model_config)

        self.model_config = model_config

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).float().to(
            config.DEVICE) for label, value in observation.items()}
        return self.representation_network(observation)

    def prediction(self, encoded_state):
        # comp, champ, shop, item, scalar = self.prediction_network(encoded_state)
        # return comp, champ, shop, item, scalar
        comp = self.prediction_network(encoded_state)
        return comp

    def forward(self, observation):
        hidden_state = self.representation(observation)
        # comp, champ, shop, item, scalar = self.prediction(hidden_state)
        #
        # outputs = {
        #     "comp": comp,
        #     "champ": champ,
        #     "shop": shop,
        #     "item": item,
        #     "scalar": scalar
        # }
        comp = self.prediction(hidden_state)

        outputs = {
            "comp": comp,
        }

        return outputs


class PredNetwork(torch.nn.Module):
    def __init__(self, model_config) -> torch.nn.Module:
        super().__init__()
        self.comp_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                               [model_config.LAYER_HIDDEN_SIZE] *
                                               model_config.N_HEAD_HIDDEN_LAYERS,
                                               config.TEAM_TIERS_VECTOR)

        # self.champ_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                         [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                         config.CHAMPION_LIST_DIM)
        #
        # self.shop_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                        [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                        [63, 63, 63, 63, 63])
        #
        # self.item_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                        [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                        [60 for _ in range(10)])
        #
        # self.scalar_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                          [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                          [100 for _ in range(3)])

    def forward(self, hidden_state):
        comp = self.comp_predictor_network(hidden_state)
        # champ = self.champ_predictor_network(hidden_state)
        # shop = self.shop_predictor_network(hidden_state)
        # item = self.item_predictor_network(hidden_state)
        # scalar = self.scalar_predictor_network(hidden_state)
        # return comp, champ, shop, item, scalar
        return comp


class GeminiRepresentation(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()

        # Input size from create_champion_vector
        self.champion_embedding = torch.nn.Linear(
            MAX_CHAMPION_IN_SET + 1 + MAX_ITEMS_IN_SET *
            3 + 2 + 1 + len(list(tiers.keys())),
            model_config.CHAMPION_EMBEDDING_DIM
        )

        self.shop_embedding = torch.nn.Linear(
            model_config.SHOP_EMBEDDING_DIM,
            model_config.CHAMPION_EMBEDDING_DIM
        )

        self.item_embedding = torch.nn.Linear(
            MAX_ITEMS_IN_SET + 1, model_config.CHAMPION_EMBEDDING_DIM)

        self.board_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.LAYER_HIDDEN_SIZE,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )
        self.bench_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.LAYER_HIDDEN_SIZE,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )
        self.shop_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.LAYER_HIDDEN_SIZE,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )
        self.item_bench_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.LAYER_HIDDEN_SIZE,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )

        # Combine encoded features from all components
        self.feature_combiner = mlp(
            (BOARD_SIZE + BENCH_SIZE + SHOP_SIZE + ITEM_BENCH_SIZE) *
            model_config.CHAMPION_EMBEDDING_DIM +
            7 + len(config.TEAM_TIERS_VECTOR),
            [model_config.LAYER_HIDDEN_SIZE] *
            model_config.N_HEAD_HIDDEN_LAYERS,
            model_config.HIDDEN_STATE_SIZE
        )

        self.feature_processor = mlp(model_config.HIDDEN_STATE_SIZE,
                                     [model_config.LAYER_HIDDEN_SIZE] *
                                     model_config.N_HEAD_HIDDEN_LAYERS,
                                     model_config.HIDDEN_STATE_SIZE)

        # Optional: Add residual connections around encoders and combiner for better gradient flow
        self.board_residual = ResidualBlock(self.board_encoder, 28)
        self.bench_residual = ResidualBlock(self.bench_encoder, 9)
        self.shop_residual = ResidualBlock(self.shop_encoder, 5)
        self.item_bench_residual = ResidualBlock(self.item_bench_encoder, 10)
        self.combiner_residual = ResidualBlock(
            self.feature_processor, model_config.HIDDEN_STATE_SIZE, local_norm=False)

    def forward(self, observation):
        # --- Encode Individual Components --- #
        board_emb = self.champion_embedding(observation['board'])
        board_enc = self.board_residual(board_emb)

        bench_emb = self.champion_embedding(observation['bench'])
        bench_enc = self.bench_residual(bench_emb)

        shop_emb = self.shop_embedding(observation['shop'])
        shop_enc = self.shop_residual(shop_emb)

        item_bench_emb = self.item_embedding(observation['items'])
        item_bench_enc = self.item_bench_residual(item_bench_emb)

        # --- Combine Encoded Features --- #
        # Flatten and concatenate encoded features
        combined_features = torch.cat([
            board_enc.flatten(start_dim=1),
            bench_enc.flatten(start_dim=1),
            shop_enc.flatten(start_dim=1),
            item_bench_enc.flatten(start_dim=1),
            observation['traits'],
            observation['scalars']  # Add scalar features directly
        ], dim=1)

        # Pass through final combiner network
        hidden_state = self.feature_combiner(combined_features)
        hidden_state = self.feature_processor(hidden_state)

        return hidden_state


class TraitPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(26, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 26*5)
        )

    def forward(self, x):
        logits = self.mlp(x)
        # reshape into 26, 5
        logits = logits.view(-1, 26, 5)
        return logits
