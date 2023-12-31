import ray
import PoroX.modules.replay_buffer as rpb
from dataclasses import dataclass

"""
Utility Classes for interacting with the game in a batched manner.
Contains:
- BatchedEnv: A class to interact with multiple games at once.
- EnvState:  A class to hold the state of an environment as well as the buffer.
- GameBuffer:  Class to store game trajectories.
"""
    
# --- Env State --- #
@dataclass
class WorkerConfig:
    env_fn: callable
    unroll_steps: int
    num_games: int
    
class BatchedEnv:
    """
    Allow for batched data collection from multiple games.
    """
    def __init__(self, config: WorkerConfig):
        self.envs = [
            EnvState.remote(config)
            for _ in range(config.num_games)
        ]
        
    def reset(self):
        reset_results = ray.get([
            env.reset.remote()
            for env in self.envs
        ])
        
        # Goes from [(obs, infos), (obs, infos)] to (obs, ...), (infos, ...)
        obs, infos = zip(*reset_results)
        
        return obs, infos
        
    def step(self, batch_actions):
        step_results = ray.get([
            env.step.remote(batch_actions[i])
            for i, env in enumerate(self.envs)
        ])
        
        # Goes from [(obs, reward, ...), (obs, reward, ...)] to (obs, ...), (reward, ...)
        obs, reward, terminated, truncated, infos = zip(*step_results)
        
        return obs, reward, terminated, truncated, infos
    
    def add_experience(self, batch_experiences):
        ray.get([
            env.add_experience.remote(batch_experiences[i])
            for i, env in enumerate(self.envs)
        ])
            
    def is_terminated(self):
        return all(ray.get([
            env.is_terminated.remote()
            for env in self.envs
        ]))

@ray.remote(num_cpus=0.25)
class EnvState:
    """
    A dataclass to hold the state of an environment as well as the buffer.
    """
    def __init__(self, config: WorkerConfig):
        self.env = config.env_fn()
        self.buffer = GameBuffer(config.unroll_steps)
        
    def reset(self):
        self.obs, self.infos = self.env.reset()
        self.terminated = {agent: False for agent in self.env.agents}
        self.truncated = {agent: False for agent in self.env.agents}
        self.rewards = {agent: 0 for agent in self.env.agents}
        
        self.buffer.clear()
        
        return self.obs, self.infos
        
    def step(self, actions):
        self.obs, self.rewards, self.terminated, self.truncated, self.infos = self.env.step(actions)
        
        return self.obs, self.rewards, self.terminated, self.truncated, self.infos
        
    def add_experience(self, experiences):
        self.buffer.store_batch_experience(experiences)
        
    def is_terminated(self):
        return all(self.terminated.values())

class GameBuffer:
    """
    Store game trajectories for each agent in a game.
    """
    player_buffers: dict[int, rpb.ReplayBufferState]
    
    def __init__(self, unroll_steps=6):
        self.player_buffers = {}
        self.unroll_steps = unroll_steps
        
    def store_experience(self, player_id, experience):
        if self.player_buffers.get(player_id) is None:
            self.player_buffers[player_id] = rpb.init(experience, self.unroll_steps)
            
        self.player_buffers[player_id] = \
            rpb.add(self.player_buffers[player_id], experience)
            
    def store_batch_experience(self, experiences):
        for player_id, experience in experiences.items():
            self.store_experience(player_id, experience)
            
    def get_buffers(self):
        return self.player_buffers
    
    def clear(self):
        del self.player_buffers

        self.player_buffers = {}