import config
import numpy as np

from Simulator.battle_generator import BattleGenerator
from Simulator import pool
from Simulator.observation.vector.observation import ObservationVector
from Simulator.observation.token.observation import ObservationToken
from Simulator.origin_class_stats import tiers
from Simulator.observation.vector.gemini_observation import GeminiObservation
from Simulator.player_manager import PlayerManager
from Simulator.tft_simulator import TFTConfig


class BatchGenerator:
    def __init__(self):
        self.battle_generator = BattleGenerator()
        # self.observation_class = ObservationVector
        self.observation_class = GeminiObservation

    # So this needs to take in a batch size then generate the necessary number of positions
    # It will need to create the observations as the x and the y as the labels
    def generate_batch(self, batch_size):
        input_batch = []
        labels = []
        for _ in range(batch_size):
            starting_level = np.random.randint(1, 6)
            item_count = np.random.randint(0, 3)
            [player, opponent, other_players] = self.battle_generator.generate_battle(
                starting_level=starting_level, item_count=item_count, scenario_info=False, extra_randomness=False
            )

            pool_obj = pool.pool()
            player.opponent = opponent
            opponent.opponent = player

            player.shop = pool_obj.sample(player, 5)
            player.shop_champions = player.create_shop_champions()

            player.gold = np.random.randint(0, 102)
            player.exp = np.random.randint(0, player.level_costs[player.level])
            player.health = np.random.randint(1, 101)

            player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj,
                                           TFTConfig(observation_class=self.observation_class))
            player_manager.reinit_player_set(
                [player] + list(other_players.values()))

            initial_observation = player_manager.fetch_observation(
                f"player_{player.player_num}")
            observation = self.observation_class.observation_to_input(
                initial_observation)

            input_batch.append(observation)

            comp = player.get_tier_labels()
            champ = player.get_champion_labels()
            shop = player.get_shop_labels()
            item = player.get_item_labels()
            scalar = player.get_scalar_labels()

            labels.append([comp, champ, shop, item, scalar])
        input_batch = self.observation_class.observation_to_dictionary(
            input_batch)
        return input_batch, labels


class TokenBatchGenerator:
    def __init__(self):
        self.battle_generator = BattleGenerator()
        self.trait_mapping = {i: tier for i, tier in enumerate(tiers)}

    def get_random_player(self):
        starting_level = np.random.randint(1, 6)
        item_count = np.random.randint(0, 3)
        player, _, _ = self.battle_generator.generate_battle(
            starting_level, item_count, scenario_info=False, extra_randomness=False
        )

        return player

    def generate_tier_labels(self, trait_vector):
        # Max levels is 4 + 1 for no level
        tier_label = np.zeros((len(trait_vector), 5))
        for id, level in enumerate(trait_vector):
            trait = self.trait_mapping[id]
            tier_list = tiers[trait]
            index = len(tier_list)
            for tier in reversed(tier_list):
                if level >= tier:
                    break
                index -= 1
            tier_label[id, index] = 1
        return tier_label

    def generate_label(self):
        player = self.get_random_player()
        observation = ObservationToken(player).fetch_player_observation()
        trait_vector = observation["traits"]
        tier_label = self.generate_tier_labels(trait_vector)
        return trait_vector, tier_label

    def generate_batch(self, batch_size):
        batch = [self.generate_label() for _ in range(batch_size)]
        # transpose the list of tuples
        traits, labels = [np.stack(x, axis=0) for x in zip(*batch)]
        return (traits, labels)
