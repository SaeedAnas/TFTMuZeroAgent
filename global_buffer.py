import ray
import config
import time
import numpy as np
from queue import PriorityQueue
from collections import deque


@ray.remote
class GlobalBuffer:
    def __init__(self, storage_ptr):
        self.gameplay_experiences = PriorityQueue(maxsize=50000)
        self.batch_size = config.BATCH_SIZE
        self.storage_ptr = storage_ptr
        self.average_position = deque(maxlen=50000)

    # Might be a bug with the action_batch not always having correct dims
    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch = [], [], [], []
        sample_set_batch = []
        for _ in range(self.batch_size):
            observation, action_history, value_mask, reward_mask, policy_mask, \
                value, reward, policy, sample_set = self.gameplay_experiences.get()[1]
            obs_tensor_batch.append(observation)
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)
            sample_set_batch.append(sample_set)

        observation_batch = np.squeeze(np.asarray(obs_tensor_batch))
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')

        position_batch = []
        for _ in range(self.batch_size):
            position_batch.append(self.average_position.pop())

        return [observation_batch, action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch, sample_set_batch], np.mean(position_batch)

    def store_replay_sequence(self, sample, position):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        print(self.gameplay_experiences.qsize())
        try:
            self.gameplay_experiences.put((sample[0], sample[1]))
        except ValueError:
            print('VALUE ERROR HERE ---------------------------')
            print(sample[0])
            print(sample[1])
            print('---------------------------')

        self.average_position.append(position)

    def available_batch(self):
        queue_length = self.gameplay_experiences.qsize()
        print(queue_length)
        if queue_length >= self.batch_size and not ray.get(self.storage_ptr.get_trainer_busy.remote()):
            self.storage_ptr.set_trainer_busy.remote(True)
            print(queue_length)
            return True
        time.sleep(5)
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
