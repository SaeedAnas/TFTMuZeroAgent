import pytest
import jax
import random

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

from PoroX.modules.observation import PoroXObservation

from Simulator.porox.player import Player
from Simulator import pool

from PoroX.test.utils import sample_action

@pytest.fixture(scope='session', autouse=True)
def player():
    pool_pointer = pool.pool()
    player_id = 0
    player = Player(pool_pointer=pool_pointer, player_num=player_id)
    
    # TODO: Add some champions, items, traits, etc.

    return player

@pytest.fixture(scope='session', autouse=True)
def obs(player):
    return PoroXObservation(player)

@pytest.fixture(scope='session', autouse=True)
def env():
    config = TFTConfig(observation_class=PoroXObservation)
    return parallel_env(config)
    
@pytest.fixture(scope='session', autouse=True)
def key():
    return jax.random.PRNGKey(10)

@pytest.fixture(scope='session', autouse=True)
def first_obs(env):
    """Gets the first observation after a random action is taken."""
    obs, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    actions = {
        agent: sample_action(env, obs, agent)
        for agent in env.agents
        if (
            (agent in terminated and not terminated[agent])
            or (agent in truncated and not truncated[agent])
        )
    }
    obs,rew,terminated,truncated,info = env.step(actions)
    
    return obs

@pytest.fixture(scope='session', autouse=True)
def first_batched_obs(first_obs):
    """
    Clone the first observation into a list of observations.
    Randomly remove some agents to make the batches varying sizes
    to test if batch_utils will still work.
    """
    N = 5
    
    obs_list = []
    
    for _ in range(N):
        obs = first_obs.copy()
        
        # num removed (between 0 and 5)
        num_removed = random.randint(0, 3)
        
        # remove num_removed agents
        for _ in range(num_removed):
            agent = random.choice(list(obs.keys()))
            obs.pop(agent)
            
        obs_list.append(obs)
        
    return obs_list

@pytest.fixture(scope='session', autouse=True)
def nth_obs(env):
    """Gets the nth observation after n random actions are taken."""
    N = 10

    obs, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    
    for _ in range(N):
        actions = {
            agent: sample_action(env, obs, agent)
            for agent in env.agents
            if (
                (agent in terminated and not terminated[agent])
                or (agent in truncated and not truncated[agent])
            )
        }
        obs,rew,terminated,truncated,info = env.step(actions)
    
    return obs

@pytest.fixture(scope='session', autouse=True)
def test_agent(env):
    class TestAgent:
        def __init__(self, env):
            self.env = env

        def act(self, batch_obs):
            batch_actions = []
            for obs in batch_obs:
                actions = {
                    agent: sample_action(self.env, obs, agent)
                    for agent in obs.keys()
                }
                batch_actions.append(actions)
            return batch_actions
        
    return TestAgent(env)
        