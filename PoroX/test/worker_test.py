import ray
import jax
import time

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.worker import run_worker
from PoroX.modules.observation import PoroXObservation

from PoroX.models.mctx_agent import PoroXV1
from PoroX.models.config import muzero_config, PoroXConfig, MCTXConfig

def test_worker():
    run_worker()