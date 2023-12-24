import ray
import time

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.observation import PoroXObservation

def collect_gameplay_experience(agent, num_games=5):
    """Collects gameplay experience from the environment using the agent.
    Currently very minimal to be used for testing purposes.
    """
    config = TFTConfig(observation_class=PoroXObservation)
    envs = BatchedEnv(lambda: parallel_env(config), num_games)
    
    obs = envs.get_obs()
    
    start_game = time.time()
    while not envs.is_terminated():
        start_action = time.time()
        batched_obs = batch_utils.collect_multi_game_obs(obs)
        output = agent.act(batched_obs, game_batched=True)
        
        step_actions = batch_utils.batch_map_actions(
            output.action,
            batched_obs
        )
        
        envs.step(step_actions)
        obs = envs.get_obs()
        print(f"Action time: {time.time() - start_action}")
    print(f"Game time: {time.time() - start_game}")
    
def collect_gameplay_experience_ray(agent, num_games=5):
    config = TFTConfig(observation_class=PoroXObservation)
    envs = BatchedEnv(lambda: parallel_env(config), num_games)
    
    obs = envs.get_obs()
    
    start_game = time.time()

    while not envs.is_terminated():
        start_action = time.time()
        actions = agent.act(obs)
        envs.step(actions)
        obs = envs.get_obs()
        print(f"Action time: {time.time() - start_action}")
    print(f"Game time: {time.time() - start_game}")

@ray.remote(num_cpus=0.5)
class EnvWrapper:
    """A dataclass to hold the state of an environment."""
    def __init__(self, env):
        self.env = env
        self.obs, self.infos = env.reset()
        
        self.terminated = {agent: False for agent in env.agents}
        self.truncated = {agent: False for agent in env.agents}
        
    def step(self, actions):
        if self.is_terminated():
            # Just don't update the state if the game is over.
            return

        self.obs, rewards, self.terminated, self.truncated, self.infos = self.env.step(actions)
        
    def is_terminated(self):
        return all(self.terminated.values())
    
    def get_obs(self):
        return self.obs
        
class BatchedEnv:
    """
    Allow for batched data collection from multiple games.
    """
    def __init__(self, env_fn, num_games):
        self.envs = [
            EnvWrapper.remote(env_fn())
            for _ in range(num_games)
        ]
        
    def step(self, batch_actions):
        ray.get([
            env.step.remote(batch_actions[i])
            for i, env in enumerate(self.envs)
        ])
            
    def get_obs(self):
        return ray.get([
            env.get_obs.remote()
            for env in self.envs
        ])
        
    def is_terminated(self):
        return all(ray.get([
            env.is_terminated.remote()
            for env in self.envs
        ]))