import config
import time
import numpy as np
from collections import deque


class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=25000)
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        obs_tensor_batch, obs_image_batch, action_history_batch, target_value_batch,  = [], [], [], [],
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch,  = [], [], [], []
        policy_mask_batch = []
        for gameplay_experience in range(self.batch_size):
            observation, action_history, value_mask, reward_mask, policy_mask,\
                value, reward, policy = self.gameplay_experiences.popleft()
            obs_tensor_batch.append(observation[0])
            obs_image_batch.append(observation[1])
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)

        obs_tensor_batch = np.squeeze(np.asarray(obs_tensor_batch))
        obs_image_batch = np.squeeze(np.asarray(obs_image_batch))
        observation_batch = [obs_tensor_batch, obs_image_batch]
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        target_policy_batch = np.asarray(target_policy_batch).astype('float32')

        return [observation_batch, action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch]

    def store_replay_sequence(self, sample):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(sample)

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size:
            time.sleep(5)
            return True
        time.sleep(20)
        return False

    # Leaving this transpose method here in case some model other than
    # MuZero requires this in the future.
    def transpose(self, matrix):
        rows = len(matrix)
        columns = len(matrix[0])

        matrix_T = []
        for j in range(columns):
            row = []
            for i in range(rows):
                row.append(matrix[i][j])
            matrix_T.append(row)

        return matrix_T
