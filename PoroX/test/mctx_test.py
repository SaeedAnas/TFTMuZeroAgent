import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

# from PoroX.models.mctx_agent import MuZeroAgent, RepresentationNetwork, PredictionNetwork, DynamicsNetwork
from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.components.embedding.dynamics import action_space_to_action
from PoroX.models.config import muzero_config, mctx_config, PoroXConfig, MCTXConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

@pytest.mark.skip(reason="Too long")
def test_muzero_network(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    
    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="muzero",
        )
    )
    
    agent = PoroXV1(config, key, obs)
    
    @jax.jit
    def apply(obs):
        return agent.act(obs)
    
    output = apply(obs)
    print(output.action)
    
    N=5
    profile(N, apply, obs)
    
def test_gumbel_muzero_network(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    
    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
        )
    )
    
    agent = PoroXV1(config, key, obs)


    @jax.jit
    def apply(obs):
        return agent.act(obs)
    
    output = apply(obs)
    print(output.action)

    N=5
    print(f"Profiling 1 game, {mctx_config.num_simulations} simulations, {mctx_config.max_num_considered_actions} sampled actions.")
    profile(N, apply, obs)
    
def test_batched_gumbel_muzero_network(first_batched_obs, key):
    obs = batch_utils.collect_multi_game_obs(first_batched_obs)

    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
        )
    )
    
    agent = PoroXV1(config, key, obs)

    @jax.jit
    def apply(obs):
        return agent.act(obs, game_batched=True)
    
    output = apply(obs)
    print(output.action)

    N=3
    print(f"Profiling {output.action.shape[0]} batched games, {mctx_config.num_simulations} simulations, {mctx_config.max_num_considered_actions} sampled actions.")
    profile(N, apply, obs)

