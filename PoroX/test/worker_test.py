import ray
import jax
import time

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.worker import collect_gameplay_experience, collect_gameplay_experience_ray
from PoroX.modules.observation import PoroXObservation

from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.config import muzero_config, PoroXConfig, MCTXConfig

def test_batched_workers(key, first_obs):
    N = 5
    batched_obs = batch_utils.collect_obs(first_obs)
    
    config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
            num_simulations=4,
            max_num_considered_actions=2,
        )
    )

    # Initialize PoroXV1
    start_init = time.time()
    agent = PoroXV1(config, key, batched_obs)
    print(f"Initialization time: {time.time() - start_init}")
    
    collect_gameplay_experience(agent, num_games=N)
    
def test_random_workers(test_agent):
    N = 5
    collect_gameplay_experience_ray(test_agent, num_games=N)
    print('here')