import ray
import jax
import time

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

import PoroX.modules.batch_utils as batch_utils
from PoroX.modules.game import BatchedEnv, WorkerConfig
from PoroX.models.config import muzero_config, PoroXConfig, MCTXConfig
from PoroX.models.mctx_agent import PoroXV1
from PoroX.modules.observation import PoroXObservation
import PoroX.modules.trajectory as trajectory

def run_worker():
    # Test function to make sure worker runs properly
    
    # --- Initialize Config --- #
    tft_config = TFTConfig(observation_class=PoroXObservation)
    
    worker_config = WorkerConfig(
        env_fn=lambda: parallel_env(tft_config),
        unroll_steps=6,
        num_games=5,
    )
    
    porox_config = PoroXConfig(
        muzero=muzero_config,
        mctx=MCTXConfig(
            policy_type="gumbel",
            num_simulations=4,
            max_num_considered_actions=2,
        )
    )
    
    # --- Initialize Agent --- #
    def get_init_obs():
        """
        Get a dummy observation to initialize the agent.
        """
        dummy_env = parallel_env(tft_config)
        dummy_obs, _ = dummy_env.reset()
        batched_obs, _ = batch_utils.collect_obs(dummy_obs)
        return batched_obs

    def init_porox():
        key = jax.random.PRNGKey(10)
        obs = get_init_obs()
        agent = PoroXV1(porox_config, key, obs)
        
        return agent
    
    init_time = time.time()
    print("Initializing...")
    # --- Create Agent --- #
    agent = init_porox()
    
    # --- Initialize Environments --- #
    envs = BatchedEnv(worker_config)
    
    # --- Reset envs --- #
    obs, infos = envs.reset()

    print(f"Init time: {time.time() - init_time}")
    
    # --- Game Loop --- #
    game_time = time.time()
    print("Starting game loop...")
    while not envs.is_terminated():
        step_time = time.time()
        # --- Convert Obs into usable format --- #
        act_time = time.time()
        batched_obs, mapping = batch_utils.collect_multi_game_obs(obs)
        
        # --- Agent Step --- #
        output = agent.act(batched_obs, game_batched=True)
        
        # --- Convert Actions into usable format --- #
        step_actions = batch_utils.batch_map_actions(
            output.action,
            mapping
        )
        print(f"Act time: {time.time() - act_time}")
        
        env_time = time.time()
        # --- Step Envs --- #
        obs, reward, terminated, truncated, infos = envs.step(step_actions)
        print(f"Env time: {time.time() - env_time}")

        # --- Create Trajectories to add to buffer --- #
        buffer_time = time.time()
        trajectories = trajectory.create_batch_trajectories(output, batched_obs, reward, mapping)
        envs.add_experience(trajectories)
        print(f"Buffer time: {time.time() - buffer_time}")
        
        print(f"Step time: {time.time() - step_time}")
        
    # --- End of Game Loop --- #
    print(f"Game time: {time.time() - game_time}")