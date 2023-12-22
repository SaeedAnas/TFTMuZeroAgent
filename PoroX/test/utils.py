import pytest
import time
import jax
import jax.numpy as jnp
import random
import numpy as np

from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

from PoroX.modules.observation import PoroXObservation

from Simulator.porox.player import Player
from Simulator import pool

# --- Utils ---
def sample_action(
    env,
    obs,
    agent,
    invert = False,
) -> ActionType:
    agent_obs = obs[agent]
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        action_mask = agent_obs["action_mask"]

        if invert: # MCTX uses 0 for valid actions, 1 for invalid actions
            action_mask = 1 - action_mask

        legal_actions = np.flatnonzero(action_mask)
        if len(legal_actions) == 0:
            return 0
        return random.choice(legal_actions)
    return env.action_space(agent).sample()

def softmax_action(
        policy_logits,
        action_mask,
):
    # Policy logits are (..., 55*38)
    # Mask out illegal actions
    # Apply softmax
    # Select action using argmax
    
    # Apply softmax
    policy_probs = jax.nn.softmax(policy_logits, axis=-1)
    
    # Masked argmax
    to_argmax = jnp.where(action_mask, -jnp.inf, policy_probs)

    # Select action using argmax
    action = jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)
    
    return action
    

def profile(N, f, *params):
    total = 0
    for _ in range(N):
        start = time.time()
        x = f(*params)
        total += time.time() - start
    avg = total / N
    print(f'{N} loops, {avg} per loop')


def batched_env_obs(N):
    config = TFTConfig(observation_class=PoroXObservation)
    
    obs_list = []
    
    for _ in range(N):
        env = parallel_env(config)
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
        
        obs_list.append(obs)

    return obs_list