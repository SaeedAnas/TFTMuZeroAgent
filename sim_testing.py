import config
import numpy as np
import time

from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
 
from Simulator.tft_ui_simulator import env
from Simulator import utils
import queue


def main():
    global_agent = MCTS(TFTNetwork())
    players = {f"player_{i}": global_agent for i in range(config.NUM_PLAYERS)}

    game = env()
    
    player_observation = game.reset()

    player_observation = separate_observation_to_input(player_observation)
    
    terminated = {player_id: False for player_id in game.possible_agents}

    # action_queue = asyncio.Queue()
    action_queue = queue.Queue()
    actions_taken = 0

    while not all (terminated.values()):
        for key, agent in players.items():
            if not terminated[key]:
                action, _, _ = agent.policy(player_observation[key])
                action = decode_action_to_one_hot(action[0], key)
                action = {"type": "player", "player_id": key, "action": action}
                action_queue.put(action)
            # await action_queue.put(action)
        
        print(action_queue.qsize())
        while not action_queue.empty():
           action = action_queue.get()
           print(action["player_id"])
           next_observation, _, terminated, _, info = game.step(action)
           player_observation = separate_observation_to_input(next_observation)
           # Broadcast changes
           
        actions_taken += 1
        if actions_taken == 3:
           actions_taken = 0
           action = {"type": "env"}
           print(action)
           next_observation, _, terminated, _, info = game.step(action)
           player_observation = separate_observation_to_input(next_observation)
  
    for key, value in info.items():
      print(value)
      if value["player_won"]:
        print(f"{key} has won!")



      
def observation_to_input(observation):
    tensors = []
    masks = []
    for obs in observation.values():
        tensors.append(obs["tensor"])
        masks.append(obs["mask"])
    return [np.asarray(tensors), masks]

def separate_observation_to_input(observation):
  observations = {}
  for key, value in observation.items():
    tensors = np.asarray([value["tensor"]])
    masks = [value["mask"]]
    observations[key] = [tensors, masks]
  return observations


def decode_action_to_one_hot(str_action, key):
    # if key == "player_0":
    #     print(str_action)
    num_items = str_action.count("_")
    split_action = str_action.split("_")
    element_list = [0, 0, 0]
    for i in range(num_items + 1):
        element_list[i] = int(split_action[i])

    decoded_action = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])
    decoded_action[0:6] = utils.one_hot_encode_number(element_list[0], 6)

    if element_list[0] == 1:
        decoded_action[6:11] = utils.one_hot_encode_number(element_list[1], 5)

    if element_list[0] == 2:
        decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38) + \
                               utils.one_hot_encode_number(element_list[2], 38)

    if element_list[0] == 3:
        decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38)
        decoded_action[44:54] = utils.one_hot_encode_number(element_list[2], 10)
    return decoded_action

def getStepActions(terminated, actions):
    step_actions = {}
    i = 0
    for player_id, terminate in terminated.items():
        if not terminate:
            step_actions[player_id] = decode_action_to_one_hot(actions[i], player_id)
            i += 1
    return step_actions


if __name__ == "__main__":
    main()
    
