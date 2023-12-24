import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

from PoroX.models.mctx_agent import RepresentationNetwork, PredictionNetworkV3
from PoroX.models.config import muzero_config as test_config

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

@pytest.fixture
def hidden_state(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    
    repr_network = RepresentationNetwork(test_config)
    variables = repr_network.init(key, obs)
    
    hidden_state = repr_network.apply(variables, obs)
    
    return hidden_state

def test_policy_network(hidden_state, key):
    policy_network = PredictionNetworkV3(test_config)
    variables = policy_network.init(key, hidden_state)
    
    @jax.jit
    def apply(variables, hidden_state):
        return policy_network.apply(variables, hidden_state)
    
    policy, value, reward = apply(variables, hidden_state)
    print(policy.shape, value.shape, reward.shape)

    N=100
    profile(N, apply, variables, hidden_state)
    
def test_params_policy_network(hidden_state, key):
    policy_network = PredictionNetworkV3(test_config)
    variables = policy_network.init(key, hidden_state)
    print(parameter_overview.get_parameter_overview(variables))
