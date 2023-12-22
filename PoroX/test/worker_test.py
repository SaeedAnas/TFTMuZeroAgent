import time

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

from PoroX.modules.observation import PoroXObservation
from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.config import porox_config
import PoroX.modules.batch_utils as batch_utils

def test_worker(env, key):
    obs, infos = env.reset()
    
    batched_obs = batch_utils.collect_obs(obs)
    agent = PoroXV1(porox_config, key, batched_obs)
    
    terminated = {agent: False for agent in env.agents}
    truncated  = {agent: False for agent in env.agents}
    
    while not all(terminated.values()):
        start = time.time()
        batched_obs = batch_utils.collect_obs(obs)
        actions, policy, values = agent.act(batched_obs)
        
        step_actions = batch_utils.map_actions(
            actions,
            batched_obs.player_ids,
            batched_obs.player_len
        )
        
        obs, rewards, terminated, truncated, infos = env.step(step_actions)
        print(f"Step time: {time.time() - start}")

def test_batched_workers(key):
    N = 5
    config = TFTConfig(observation_class=PoroXObservation)
    
    envs = [
        parallel_env(config)
        for _ in range(N)
    ]
    
    obs = []
    infos = []
    terminated = []
    truncated = []
    
    for env in envs:
        o, i = env.reset()
        obs.append(o)
        infos.append(i)
        terminated.append({agent: False for agent in env.agents})
        truncated.append({agent: False for agent in env.agents})
        
    batched_obs = batch_utils.collect_multi_game_obs(obs)
    agent = PoroXV1(porox_config, key, batched_obs)
        
    def all_games_terminated():
        return all([
            all(t.values())
            for t in terminated
        ])
        
    while not all_games_terminated():
        start = time.time()
        batched_obs = batch_utils.collect_multi_game_obs(obs)
        actions, policy, values = agent.act(batched_obs, game_batched=True)
    
        step_actions = batch_utils.batch_map_actions(
            actions,
            batched_obs
        )
        
        obs = []
        infos = []
        terminated = []
        truncated = []
        
        for env, actions in zip(envs, step_actions):
            o, r, t, tr, i = env.step(actions)
            obs.append(o)
            infos.append(i)
            terminated.append(t)
            truncated.append(tr)
            
        print(f"Step time: {time.time() - start}")