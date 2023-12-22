from Simulator.porox.observation import ObservationBase, ActionBase
from Simulator.porox.player import Player as player_class

class PlayerManager:
    def __init__(self, 
                 num_players,
                 pool_obj,
                 config
                 ):
        
        self.pool_obj = pool_obj
        self.config = config
        
        self.players = {
            "player_" + str(player_id) for player_id in range(num_players)
        }
        
        # Ensure that the opponent obs are always in the same order
        self.player_ids = sorted(list(self.players))

        self.terminations = {player: False for player in self.players}
        
        self.player_states = {
            player: player_class(self.pool_obj, player_id)
            for player_id, player in enumerate(self.players)
        }
        
        # In order to support multiple observation types, 
        # we need to store all observation types for EACH player
        self.observation_mapping = self.create_observation_mapping(config.observation_class)
        self.observation_states = self.create_observations(list(self.observation_mapping.values()))
        
        self.action_handlers = self.create_actions(config.action_class)
        
        self.opponent_observations = self.create_opponent_observations()

    # -- Player Management -- #

    def kill_player(self, player_id):
        """TODO: Change how this works... it is like this to be compatible with the old sim"""
        self.terminations[player_id] = True
        
        player = self.player_states[player_id]
        self.pool_obj.return_hero(player)

        self.player_states[player_id] = None
        
    # -- Support for multiple observation types -- #
        
    def create_observation_mapping(self, observation_class: ObservationBase | dict[ObservationBase]):
        if observation_class is not dict:
            observation_class = {player: observation_class for player in self.players}
        
        return observation_class
        
    def create_observations(self, observation_class: list[ObservationBase]):
        """
        Creates a dict of dicts of observation classes

        Each observation class gets its own dictionary of observations
        """
        unique_observation_classes = set(observation_class)
        
        def create_observation_dict(observation_class):
            return {
                player_id: observation_class(player_state)
                for player_id, player_state in self.player_states.items()
            }
        
        return {
            observation_class: create_observation_dict(observation_class)
            for observation_class in unique_observation_classes
        }
        
    def create_actions(self, action_class: ActionBase | dict[ActionBase]):
        """
        Creates a dict of action classes
        """
        if action_class is not dict:
            return {
                player_id: action_class(player_state)
                for player_id, player_state in self.player_states.items()
            }
            
        return {
            player_id: action_class[player_id](player_state)
            for player_id, player_state in self.player_states.items()
        }
        
    def create_opponent_observations(self):
        """
        Because the observation states may be different for each player,
        we stored all the observation_states for each player in a dict.

        We can then fetch the opponent observations for each player regardless of the observation state.
        """
        
        return {
            player: self.fetch_opponent_observations(player)
            for player in self.players
        }
        
    def get_observation_state(self, player_id):
        obs_class = self.observation_mapping[player_id]
        return self.observation_states[obs_class][player_id]
    
    def get_action_handler(self, player_id):
        return self.action_handlers[player_id]
    
    def update_all_observation_states(self, player_id, action):
        """
        We need to update each observation state for each player
        """
        for obs_class, obs_dict in self.observation_states.items():
            obs_dict[player_id].update_observation(action)
        
    # -- fetch observations -- #

    def fetch_opponent_observations(self, player_id):
        """Fetches the opponent observations for the given player.

        Args:
            player_id (int): Player id to fetch opponent observations for.

        Returns:
            list: List of observations for the given player.
        """
        observation_class = self.observation_mapping[player_id]
        observation_states = self.observation_states[observation_class]
        
        observations = [
            observation_states[player].fetch_public_observation()
            if not self.terminations[player]
            else observation_states[player].fetch_dead_observation()
            for player in self.player_ids
            if player != player_id
        ]

        return observations

    def fetch_observation(self, player_id):
        """Creates the observation for the given player.

        Format:
        {
            "player": PlayerObservation
            "action_mask": (5, 11, 38)  # Same as action space
            "opponents": [PlayerPublicObservation, ...]
        }
        """
        observation_state = self.get_observation_state(player_id)
        action_handler = self.get_action_handler(player_id)

        return {
            "player": observation_state.fetch_player_observation(),
            "action_mask": action_handler.fetch_action_mask(),
            "opponents": self.opponent_observations[player_id],
        }

    def update_game_round(self):
        for player in self.players:
            if not self.terminations[player]:
                self.player_states[player].actions_remaining = self.config.max_actions_per_round
                
                self.get_observation_state(player).update_game_round()
                self.get_action_handler(player).update_game_round()

    def refresh_all_shops(self):
        for player in self.players:
            if not self.terminations[player]:
                self.player_states[player].refresh_shop()
                self.get_observation_state(player).update_observation([2, 0, 0])
                self.get_action_handler(player).update_action_mask([2, 0, 0])

    # - Used so I don't have to change the Game_Round class -
    # TODO: Refactor Game_Round class to use refresh_all_shops
    def generate_shops(self, players):
        self.refresh_all_shops()

    def generate_shop_vectors(self, players):
        pass
    # -----
        
    # --- Main Action Function ---
    def perform_action(self, player_id, action):
        """Performs the action given by the agent.

        7 Types of actions:
        [0, 0, 0] - Pass action
        [1, 0, 0] - Level up action
        [2, 0, 0] - Refresh action
        [3, X1, 0] - Buy action; X1 is an index from 0 to 4 for the shop locations
        [4, X1, 0] - Sell Action; X1 is the index of the champion to sell (0 to 36)
        [5, X1, X2] - Move Action; X1 is the index of the champion to move (0 to 36), X2 is the index of the location to move to (0 to 36)
        [6, X1, X2] - Item Action; X1 is the index of the item to move (0 to 9), X2 is the index of the champion to move to (0 to 36)

        Args:
            action (list): Action to perform. Must be of length 3.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        player = self.player_states[player_id]
        action_handler = self.get_action_handler(player_id)

        action = action_handler.action_space_to_action(action)

        if type(action) is not list and len(action) != 3:
            print(f"Action is not a list of length 3: {action}")
            return

        action_type, x1, x2 = action

        # Pass Action
        if action_type == 0:
            player.pass_action()
            # Update opponent observations on pass action
            self.opponent_observations[player_id] = self.fetch_opponent_observations(player_id)

        # Level Action
        elif action_type == 1:
            player.buy_exp_action()

        # Refresh Action
        elif action_type == 2:
            player.refresh_shop_action()

        # Buy Action
        elif action_type == 3:
            player.buy_shop_action(x1)

        # Sell Action
        elif action_type == 4:
            player.sell_action(x1)

        # Move/Sell Action
        elif action_type == 5:
            player.move_champ_action(x1, x2)

        # Item action
        elif action_type == 6:
            player.move_item_action(x1, x2)

        else:
            player.print(f"Action Type is invalid: {action}")
            
        player.actions_remaining -= 1
        
        self.update_all_observation_states(player_id, action)
        action_handler.update_action_mask(action)