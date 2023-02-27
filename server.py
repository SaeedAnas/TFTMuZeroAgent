import asyncio
import json
import secrets
import websockets
import logging

import config
import numpy as np

if config.ARCHITECTURE == 'Pytorch':
    from Models.MCTS_torch import MCTS
    from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
else:
    from Models.MCTS import MCTS
    from Models.MuZero_keras_agent import TFTNetwork
 
from Simulator.tft_simulator import parallel_env
from Simulator import utils

logging.basicConfig(filename='server.log', level=logging.DEBUG)

async def error(websocket, message):
    event = {
        "type": "error",
        "message": message,
    }

    await websocket.send(json.dumps(event))

LOBBY = {}

async def handle_action(message):
    print(message)
        
async def play_handler(websocket):
    async for message in websocket:
        await handle_action(message)
        
class Player:
    def __init__(self):
        ...
        
# Put game logic here
async def game_handler(connected):
    players = {f"player_{i}": None for i in range(config.NUM_PLAYERS)}

    # Create ids for real players
    for i, connection in enumerate(connected):
        id = f"player_{i}"
        players[id] = Player()
        connection.send(f"Your id is {id}")
    
    # Rest of the players are agents
    for i in range(len(connected), config.NUM_PLAYERS):
        id = f"player_{i}"
        agent = TFTNetwork()
        agent.tft_load_model(1000)
        players[id] = MCTS(agent)
        
    env = parallel_env()

    player_observation = env.reset()

    player_observation = observation_to_input(player_observation)
    
    # Used to know when players die and which agent is currently acting
    terminated = {player_id: False for player_id in env.possible_agents}
    
    while not all(terminated.values()):
        actions = {agent: None for agent in players.keys()}
        for i, [key, agent] in enumerate(players.items()):
            action, _ = agent.policy(np.expand_dims(player_observation[i], axis=0))
            actions[key] = action

        # actions, policy = agent.policy(player_observation)
        
    while True:
        await asyncio.sleep(2)
        websockets.broadcast(connected, "game is happening...")

def observation_to_input(observation):
    tensors = []
    masks = []
    for obs in observation.values():
        tensors.append(obs["tensor"])
        masks.append(obs["mask"])
    return [np.asarray(tensors), masks]

async def wait_for_start(websocket, connected):
    should_start = False
    async for message in websocket:
        if message == "start":
            should_start = True
            websockets.broadcast(connected, "game is starting...")
            break
    return should_start

async def start_game(websocket, connected):
    player_task = asyncio.create_task(play_handler(websocket))
    game_task = asyncio.create_task(game_handler(connected))
    websockets.broadcast(connected, "starting game!")
    done, pending = await asyncio.wait(
        [player_task, game_task],
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
        
if __name__ == "__main__":
    asyncio.run(main())