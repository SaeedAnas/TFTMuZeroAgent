import asyncio
import json
import secrets
import websockets
import logging

import config
import numpy as np
import queue

from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Simulator.tft_simulator import raw_env
from Simulator import utils

logging.basicConfig(filename='server.log', level=logging.DEBUG)

async def error(websocket, message):
    event = {
        "type": "error",
        "message": message,
    }

    await websocket.send(json.dumps(event))

LOBBY = {}


class Player:
    def __init__(self, id, socket):
        self.id = id
        self.socket = socket

    def send(self, message):
        self.socket.send(f"{message}")

    async def handle_message(self, q):
        async for message in self.socket:
            action = {"type": "player", "player_id": self.id, "action": message}
            await q.put(action)

async def wait_for_start(websocket, connected):
    should_start = False
    async for message in websocket:
        if message == "start":
            should_start = True
            websockets.broadcast(connected, "game is starting...")
            break
    return should_start

def init_players(connected):
    players = {f"player_{i}": None for i in range(config.NUM_PLAYERS)}
    humans = {}
    bots = {}

    # Create ids for real players
    for i, socket in enumerate(connected):
        id = f"player_{i}"
        players[id] = Player(id, socket)
        humans[id] = players[id]
        
    for id in players.keys():
        if players[id] is None:
            agent = TFTNetwork()
            agent.tft_load_model(1000)
            players[id] = MCTS(agent)
            bots[id] = players[id]
    
    return players, humans, bots
            
def init_game():
    game = raw_env()

    first_observation, first_state = game.reset()
    
    first_observation = separate_observation_to_input(first_observation)
    
    return game, first_observation, first_state

async def game_loop(q, c, game, obs_context, act_context):
    terminated = await obs_context.access('terminated')
    while not all(terminated.values()):
        item = await q.get()
        if item["type"] == "player":
            item["action"] = decode_action_to_one_hot(item["action"])

        next_observation, _, terminated, _, info = game.step(item)
        next_observation = separate_observation_to_input(next_observation)

        await obs_context.update(obs=next_observation, terminated=terminated)
        
        if item["type"] == "env":
            await act_context.update(action_count=0)
            
        if item["type"] == "player":
            await c.broadcast(f"{item['player_id']} has made a move")
        elif item["type"] == "env":
            await c.broadcast("round changed")
            
    for key, value in info.items():
        if value["player_won"]:
            print(f"{key} has won!")
            await c.broadcast(f"{key} has won!")
        
    # Broadcast changes
    # Update observation
        
class Context:
    """
    A context class for safely sharing variables across multiple async tasks.

    Example usage:
    # Create a context object with some initial values
    context = Context(shared_var=0, other_var=0)

    # Access variables
    shared_var, other_var = await context.access('shared_var', 'other_var')

    # Update variables
    await context.update(shared_var=1, other_var=2)
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.lock = asyncio.Lock()
        
    async def access(self, *attr_names):
        async with self.lock:
            if len(attr_names) == 1:
                return getattr(self, attr_names[0])

            return [getattr(self, name) for name in attr_names]
        
    async def update(self, **kwargs):
        async with self.lock:
            for name, value in kwargs.items():
                setattr(self, name, value)
                

class ConnectionContext(Context):
    def __init__(self, connected):
        super().__init__(connected=connected)
    
    async def broadcast(self, message):
        async with self.lock:
            websockets.broadcast(self.connected, message)
           
        
async def bot_loop(q, c, bots, obs_context, act_context):
    terminated = await obs_context.access('terminated')

    while not all(terminated.values()):
        action_count = await act_context.access('action_count')
        if action_count != config.ACTIONS_PER_TURN:
            for key, agent in bots.items():
                player_observation, terminated = await obs_context.access('obs', 'terminated')
                if not terminated[key]:
                    action, _, _ = agent.policy(player_observation[key])
                    action = {"type": "player", "player_id": key, "action": action[0]}
                    await q.put(action)
            await act_context.update(action_count=action_count + 1)
        else:
            await asyncio.sleep(2)


async def round_timer(q, c):
    while True:
        await asyncio.sleep(30)
        await q.put({"type": "env"})


async def start_game(websocket, connected):
    players, humans, bots = init_players(connected)

    game, player_observation, state = init_game()
    
    terminated = {player_id: False for player_id in players.keys()}
    
    obs_context = Context(obs=player_observation, terminated=terminated)
    act_context = Context(action_count=0)
    q = asyncio.Queue()
    c = ConnectionContext(connected)
    
    # game_task = asyncio.create_task(game_loop(q, connected, game, obs_context, act_context))
    # bot_task = asyncio.create_task(bot_loop(q, connected, bots, obs_context, act_context))
    # round_task = asyncio.create_task(round_timer(q, connected))

    # player_tasks = [asyncio.create_task(humans[k].handle_message(q)) for k in humans.keys()]
    
    await c.broadcast("starting game!")
    done, pending = await asyncio.wait(
        [
            game_loop(q, c, game, obs_context, act_context),
            bot_loop(q, c, bots, obs_context, act_context),
            round_timer(q, c),
             *[humans[k].handle_message(q) for k in humans.keys()]
        ],
        # [game_task, bot_task, round_task, *player_tasks],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


async def create(websocket):
    connected = {websocket}
    lobby_key = secrets.token_urlsafe(12)
    LOBBY[lobby_key] = connected
    
    try:
        await websocket.send(lobby_key)

        start = await wait_for_start(websocket, connected)
        if start:
            await start_game(websocket, connected)
    
    finally:
        del LOBBY[lobby_key]
        
async def join(websocket, lobby_key):
    try:
        connected = LOBBY[lobby_key]
    except KeyError:
        await error(websocket, "Game not found.")
        return
    
    connected.add(websocket)
    
    try:
        websockets.broadcast(connected, "a new player has joined")
        
        await play_handler(websocket)
    finally:
        connected.remove(websocket)
    
async def handler(websocket):
    # First message will decide which type of connection this socket is
    message = await websocket.recv()
    # event = json.loads(message)
    event = message.split(" ")
    if event[0] == "create":
        await create(websocket)
    elif event[0] == "join":
        await join(websocket, event[1])

async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()

def separate_observation_to_input(observation):
  observations = {}
  for key, value in observation.items():
    tensors = np.asarray([value["tensor"]])
    masks = [value["mask"]]
    observations[key] = [tensors, masks]
  return observations

def decode_action_to_one_hot(str_action):
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

        
if __name__ == "__main__":
    asyncio.run(main())
