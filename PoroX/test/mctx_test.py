import pytest
import jax
import jax.numpy as jnp
from clu import parameter_overview

from PoroX.models.mctx_agent import MCTXAgent, RepresentationNetwork, PredictionNetwork, DynamicsNetwork
from PoroX.models.config import test_config

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_muzero_network(first_obs, key):
    obs = batch_utils.collect_shared_obs(first_obs)
    
    repr_nn = RepresentationNetwork(test_config)
    pred_nn = PredictionNetwork(test_config)
    dyna_nn = DynamicsNetwork(test_config)

    muzero = MCTXAgent(
        representation_nn=repr_nn,
        prediction_nn=pred_nn,
        dynamics_nn=dyna_nn,
        config=test_config.mctx_config
    )
    
    variables = muzero.init(key, obs)
    
    @jax.jit
    def apply(variables, key, obs):
        return muzero.policy(variables, key, obs)
    
    policy_output, root = apply(variables, key, obs)
    print(policy_output.action)
    
    N=10
    profile(N, apply, variables, key, obs)
