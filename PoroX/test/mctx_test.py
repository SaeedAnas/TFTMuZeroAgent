import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

from PoroX.models.mctx_agent import MCTXAgent, RepresentationNetwork, PredictionNetwork, DynamicsNetwork
from PoroX.models.components.embedding.dynamics import action_space_to_action
from PoroX.models.config import muzero_config, mctx_config

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_muzero_network(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    
    repr_nn = RepresentationNetwork(muzero_config)
    pred_nn = PredictionNetwork(muzero_config)
    dyna_nn = DynamicsNetwork(muzero_config)

    muzero = MCTXAgent(
        representation_nn=repr_nn,
        prediction_nn=pred_nn,
        dynamics_nn=dyna_nn,
        config=mctx_config
    )
    
    variables = muzero.init(key, obs)
    
    @jax.jit
    def apply(variables, key, obs):
        return muzero.policy(variables, key, obs)
    
    policy_output, root = apply(variables, key, obs)
    print(policy_output.action)
    
    N=5
    profile(N, apply, variables, key, obs)
    
def test_gumbel_muzero_network(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)

    repr_nn = RepresentationNetwork(muzero_config)
    pred_nn = PredictionNetwork(muzero_config)
    dyna_nn = DynamicsNetwork(muzero_config)

    muzero = MCTXAgent(
        representation_nn=repr_nn,
        prediction_nn=pred_nn,
        dynamics_nn=dyna_nn,
        config=mctx_config
    )

    variables = muzero.init(key, obs)

    @jax.jit
    def apply(variables, key, obs):
        return muzero.policy_gumbel(variables, key, obs)
    
    policy_output, root = apply(variables, key, obs)
    actions = jax.vmap(action_space_to_action)(policy_output.action)
    print(actions)

    N=5
    print(f"Profiling 1 game, {mctx_config.num_simulations} simulations, {mctx_config.max_num_considered_actions} sampled actions.")
    profile(N, apply, variables, key, obs)
    
def test_batched_gumbel_muzero_network(first_batched_obs, key):
    obs = batch_utils.collect_multi_game_obs(first_batched_obs)
    obs, original_shape = batch_utils.flatten_multi_game_obs(obs)

    repr_nn = RepresentationNetwork(muzero_config)
    pred_nn = PredictionNetwork(muzero_config)
    dyna_nn = DynamicsNetwork(muzero_config)
    
    muzero = MCTXAgent(
        representation_nn=repr_nn,
        prediction_nn=pred_nn,
        dynamics_nn=dyna_nn,
        config=mctx_config
    )

    variables = muzero.init(key, obs)

    @jax.jit
    def apply(variables, key, obs):
        return muzero.policy_gumbel(variables, key, obs)
    
    policy_output, root = apply(variables, key, obs)
    actions = batch_utils.unflatten(policy_output.action, original_shape)
    print(actions)

    N=3
    print(f"Profiling {actions.shape[0]} batched games, {mctx_config.num_simulations} simulations, {mctx_config.max_num_considered_actions} sampled actions.")
    profile(N, apply, variables, key, obs)

