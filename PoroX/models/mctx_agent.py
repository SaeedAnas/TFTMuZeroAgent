from flax import linen as nn
from jax import numpy as jnp
import jax

from PoroX.modules.observation import BatchedObservation
import PoroX.modules.batch_utils as batch_utils

from PoroX.models.player_encoder import PlayerEncoder, CrossPlayerEncoder, GlobalPlayerEncoder
from PoroX.models.components.transformer import EncoderConfig, CrossAttentionEncoder, Encoder
from PoroX.models.components.embedding import PolicyActionTokens, ValueToken
from PoroX.models.components.fc import FFNSwiGLU
from PoroX.models.config import MuZeroConfig
    
class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Player: PlayerObservation
    - Opponents: [PlayerObservation...]
    """
    config: MuZeroConfig
    
    # Don't ask what's going on here, I have no idea
    @nn.compact
    def __call__(self, obs: BatchedObservation):
        states = GlobalPlayerEncoder(self.config.player_encoder)(obs)
        
        # Split states into player and opponents
        player_shape = obs.players.champions.shape[-3]
        
        player_states = states[..., :player_shape, :, :]
        opponent_states = states[..., player_shape:, :, :]
        
        expanded_opponent_state = jnp.expand_dims(opponent_states, axis=-4)
        broadcast_shape = opponent_states.shape[:-4] + player_states.shape[-3:-2] + opponent_states.shape[-3:]
        broadcasted_opponent = jnp.broadcast_to(expanded_opponent_state, broadcast_shape)
        
        player_ids = obs.players.scalars[..., 0].astype(jnp.int32)
        
        # Create a mask that masks out the player's own state
        mask = jnp.arange(player_states.shape[-3]) == player_ids[..., None]
        masked_opponent_states = broadcasted_opponent * mask[..., None, None]
        # I have no fucking clue what I'm doing but it jit compiles...
        
        cross_states = CrossPlayerEncoder(self.config.cross_encoder)(player_states, masked_opponent_states)
        
        merged_states = CrossAttentionEncoder(self.config.merge_encoder)(player_states, context=cross_states)
        
        return merged_states
    

class PredictionNetwork(nn.Module):
    """
    Prediction Network to return a policy and value for a given hidden state

    Hidden State:
    - Sequence of tokens of size N
    - Comes from the Representation Network/Dynamics Network
    
    Prediction Network is essentially BERT
    Value Network is BERT followed by an MLP
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(self, hidden_state):
        
        # --- Policy Network --- #
        # Policy is of shape (55, 38)
        # Action tokens are <Pass>, <Level>, and <Shop>
        action_tokens_with_hidden_state = PolicyActionTokens()(hidden_state)

        policy_logits = Encoder(self.config.policy_head)(action_tokens_with_hidden_state)
        
        # Ignore tokens after first 55
        policy_logits = policy_logits[..., :55, :]
        
        # Now optionally we could apply one final MLP to get the logits
        policy_shape = policy_logits.shape
        hidden_dim = policy_shape[-1] * policy_shape[-2]
        # Flatten the policy
        policy_logits_flattened = jnp.reshape(policy_logits, policy_shape[:-2] + (hidden_dim,))
        # Apply MLP
        policy_logits_flattened = FFNSwiGLU(hidden_dim=hidden_dim)(policy_logits_flattened)
        # Reshape back to original shape
        policy_logits = jnp.reshape(policy_logits_flattened, policy_shape)
        
        # --- Value Network --- #
        value_token_with_hidden_state = ValueToken()(hidden_state)
        
        value_logits = Encoder(self.config.value_head)(value_token_with_hidden_state)
        
        # Only return the value token
        value_logits = value_logits[..., 0, :]
        
        # Now optionally we could apply one final MLP to get the logits
        value_logits = FFNSwiGLU()(value_logits)
        
        return policy_logits, value_logits
    
class DynamicsNetwork(nn.Module):
    """
    Dynamics Network to return the next hidden state and reward
    given an action and previous hidden state
    
    Embed the action as a vector, and concatenate it to the hidden state
    at the embedding dimension (..., sequence, embedding)
    
    Then pass through an Encoder with a projection of the original hidden state
    """


"""
Stochastic MuZero Network:

Representation Network: hidden_state = R(observation)
Prediction Network: policy_logits, value = P(hidden_state)
Afterstate Dynamics Network: afterstate = AD(hidden_state, action)
Afterstate Prediction Network: chance_outcomes, afterstate_value = AP(afterstate)
Dynamics Network: hidden_state, reward = D(afterstate, chance_outcomes)

"""

class MCTSAgent:
    def act(self, obs):
        actions = {
            player_id: 0 for player_id in obs.keys()
        }

        return actions