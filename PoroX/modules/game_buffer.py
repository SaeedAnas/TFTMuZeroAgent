import chex
import jax
import jax.numpy as jnp
import PoroX.modules.replay_buffer as rpb


class GameBuffer:
    """
    Store game trajectories for each agent in a game.
    """
    player_buffers: dict[int, rpb.ReplayBufferState]
    
    def __init__(self, unroll_steps=6):
        self.player_buffers = {}
        self.unroll_steps = unroll_steps
        
    def store_experience(self, player_id, experience):
        if self.player_buffers.get(player_id) is None:
            self.player_buffers[player_id] = rpb.init(experience, self.unroll_steps)
            
        self.player_buffers[player_id] = \
            rpb.add(self.player_buffers[player_id], experience)
            
    def store_batch_experience(self, experiences):
        for player_id, experience in experiences.items():
            self.store_experience(player_id, experience)
            
    def get_buffers(self):
        return self.player_buffers
    
    def clear(self):
        del self.player_buffers

        self.player_buffers = {}