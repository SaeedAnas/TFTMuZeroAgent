import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

from PoroX.models.mctx_agent import RepresentationNetwork, PredictionNetworkV2, DynamicsNetworkV2
from PoroX.models.config import muzero_config as test_config

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile, softmax_action

@pytest.fixture
def actions(first_obs, key):
    obs, _ = batch_utils.collect_obs(first_obs)

    action_mask = obs.action_mask
    
    repr_network = RepresentationNetwork(test_config)
    variables = repr_network.init(key, obs)
    
    hidden_state = repr_network.apply(variables, obs)
    
    prediction_network = PredictionNetworkV2(test_config)
    variables = prediction_network.init(key, hidden_state)
    
    policy_logits, value_logits = prediction_network.apply(variables, hidden_state)
    
    action_idxs = softmax_action(policy_logits, action_mask)
    
    return hidden_state, action_idxs

def test_dynamics_network(actions, key):
    hidden_state, action_idxs = actions

    dynamics_network = DynamicsNetworkV2(test_config)
    variables = dynamics_network.init(key, hidden_state, action_idxs)
    
    @jax.jit
    def apply(variables, hidden_state, action_idxs):
        return dynamics_network.apply(variables, hidden_state, action_idxs)
    
    next_hidden_state, reward = apply(variables, hidden_state, action_idxs)
    print(next_hidden_state.shape, reward.shape, next_hidden_state.dtype)
    
    N=100
    profile(N, apply, variables, hidden_state, action_idxs)
    
def test_params_dynamics_network(actions, key):
    hidden_state, action_idxs = actions

    dynamics_network = DynamicsNetworkV2(test_config)
    variables = dynamics_network.init(key, hidden_state, action_idxs)
    print(parameter_overview.get_parameter_overview(variables))