from functools import partial
import chex
from flax import linen as nn
from jax import numpy as jnp
import jax
import mctx

from PoroX.modules.observation import BatchedObservation

from PoroX.models.player_encoder import CrossPlayerEncoder, GlobalPlayerEncoder
from PoroX.models.components.transformer import CrossAttentionEncoder, Encoder
from PoroX.models.components.value_encoder import ValueEncoder
from PoroX.models.components.scalar_encoder import ScalarEncoder
from PoroX.models.components.embedding import PolicyActionTokens, ActionEmbedding
from PoroX.models.components.fc import FFNSwiGLU
from PoroX.models.config import MuZeroConfig, MCTXConfig
    
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
        policy_logits = FFNSwiGLU(hidden_dim=hidden_dim)(policy_logits_flattened)

        # Reshape back to original shape
        # policy_logits = jnp.reshape(policy_logits_flattened, policy_shape)
        # The final reshape is unnecessary because we need to reshape it again to apply softmax
        
        # --- Value Network --- #
        value_logits = ValueEncoder(self.config.value_head)(hidden_state)
        
        return policy_logits, value_logits
    
class DynamicsNetwork(nn.Module):
    """
    Dynamics Network to return the next hidden state and reward
    given an action and previous hidden state
    
    Embed the action as a vector, and concatenate it to the hidden state
    at the embedding dimension (..., sequence, embedding)
    
    Then pass through an Encoder with a projection of the original hidden state
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(self, hidden_state, action):
        # --- Next Hidden State --- #
        action_embeddings = ActionEmbedding()(action)
        
        # Expand dims from (..., embedding) to (..., 1, embedding)
        action_embeddings = jnp.expand_dims(action_embeddings, axis=-2)
        # Broadcast to (..., sequence, embedding)
        action_embeddings = jnp.broadcast_to(action_embeddings, hidden_state.shape[:-1] + (action_embeddings.shape[-1],))
        
        # Concatenate the action embeddings to the hidden state
        hidden_state_with_action = jnp.concatenate([
            hidden_state,
            action_embeddings
        ], axis=-1)
        
        next_hidden_state = Encoder(self.config.dynamics_head)(hidden_state_with_action)
        
        # --- Reward --- #
        reward_logits = ValueEncoder(self.config.reward_head)(next_hidden_state)
        
        return next_hidden_state, reward_logits


"""
MuZero Network:

Representation Network: hidden_state = R(observation)
Prediction Network: policy_logits, value = P(hidden_state)
Dynamics Network: hidden_state, reward = D(hidden_state, action)

"""

# Based heavily on https://github.com/bwfbowen/muax/
@chex.dataclass(frozen=True)
class MuZeroParams:
    represnentation: nn.Module
    prediction: nn.Module
    dynamics: nn.Module

class MCTXAgent:
    def __init__(self,
                 representation_nn: nn.Module,
                 prediction_nn: nn.Module,
                 dynamics_nn: nn.Module,
                 config: MCTXConfig,
                 ):
        

        self.represnetation_nn = representation_nn
        self.prediction_nn = prediction_nn
        self.dynamics_nn = dynamics_nn
        
        # TODO: Make this configurable
        self.scalar_encoder = ScalarEncoder(
            min_value=-999,
            max_value=999,
            num_steps=192
        )

        self.mc = config
        
        
    def init(self, key: jax.random.PRNGKey, sample_obs: BatchedObservation):
        repr_variables = self.represnetation_nn.init(key, sample_obs)
        
        hidden_state = self.represnetation_nn.apply(repr_variables, sample_obs)

        prediction_variables = self.prediction_nn.init(key, hidden_state)

        dynamics_variables = self.dynamics_nn.init(key, hidden_state, jnp.array([0]))
        
        params = MuZeroParams(
            represnentation=repr_variables,
            prediction=prediction_variables,
            dynamics=dynamics_variables
        )
        
        self.params = params
        return self.params
    
    def policy(self, params: MuZeroParams, key: jax.random.PRNGKey, obs: BatchedObservation):
        root = self.root_fn(params, key, obs)
        invalid_actions = obs.action_mask
        
        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=self.recurrent_fn,
            invalid_actions=invalid_actions,
            num_simulations=self.mc.num_simulations,
            max_depth=self.mc.max_depth,
            dirichlet_fraction=self.mc.dirichlet_fraction,
            dirichlet_alpha=self.mc.dirichlet_alpha,
            pb_c_init=self.mc.pb_c_init,
            pb_c_base=self.mc.pb_c_base,
            temperature=self.mc.temperature,
        )
        
        return policy_output, root
    
    def policy_gumbel(self, params: MuZeroParams, key: jax.random.PRNGKey, obs: BatchedObservation):
        root = self.root_fn(params, key, obs)
        invalid_actions = obs.action_mask
        policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=key,
                root=root,
                recurrent_fn=self.recurrent_fn,
                invalid_actions=invalid_actions,
                num_simulations=self.mc.num_simulations,
                max_depth=self.mc.max_depth,
                max_num_considered_actions=self.mc.max_num_considered_actions,
                gumbel_scale=self.mc.gumbel_scale,
        )
        
        return policy_output, root
        
    
    @partial(jax.jit, static_argnums=(0,))
    def root_fn(self, params: MuZeroParams, key: jax.random.PRNGKey, obs: BatchedObservation) -> mctx.RootFnOutput:
        del key

        hidden_state = self.represnetation_nn.apply(params.represnentation, obs)
        policy_logits, value_logits = self.prediction_nn.apply(params.prediction, hidden_state)
        value = self.scalar_encoder.decode_softmax(value_logits)
        
        return mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=value,
            embedding=hidden_state
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def recurrent_fn(self, params: MuZeroParams, key: jax.random.PRNGKey, action, embedding) -> (mctx.RecurrentFnOutput, jnp.ndarray):
        del key

        next_hidden_state, reward_logits = self.dynamics_nn.apply(params.dynamics, embedding, action)
        policy_logits, value_logits = self.prediction_nn.apply(params.prediction, next_hidden_state)

        reward = self.scalar_encoder.decode_softmax(reward_logits)
        value = self.scalar_encoder.decode_softmax(value_logits)
        
        discount = jnp.ones_like(reward) * self.mc.discount
        
        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=policy_logits,
            value=value,
        )

        return recurrent_output, next_hidden_state

class MCTSAgent:
    def act(self, obs):
        actions = {
            player_id: 0 for player_id in obs.keys()
        }

        return actions