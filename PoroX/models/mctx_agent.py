from functools import partial
import chex
from flax import linen as nn
from jax import numpy as jnp
import jax
import mctx

from PoroX.modules.observation import BatchedObservation
import PoroX.modules.batch_utils as batch_utils

from PoroX.models.player_encoder import CrossPlayerEncoder, GlobalPlayerEncoder
from PoroX.models.components.transformer import CrossAttentionEncoder, Encoder
from PoroX.models.components.value_encoder import ValueEncoder
from PoroX.models.components.scalar_encoder import ScalarEncoder
from PoroX.models.components.embedding import PolicyActionTokens, ActionEmbedding
from PoroX.models.components.fc import FFNSwiGLU
from PoroX.models.config import MuZeroConfig, MCTXConfig, PoroXConfig
    
class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Player: PlayerObservation
    - Opponents: [PlayerObservation...]
    """
    config: MuZeroConfig
    
    @nn.compact
    def __call__(self, obs: BatchedObservation):
        # Encode the player and opponent observations using the same encoder
        states = GlobalPlayerEncoder(self.config.player_encoder)(obs)
        
        # First index is player, rest are opponents
        
        # (Game Batch, Player Batch, ...)
        player_embedding = states[..., 0, :, :]
        # (Game Batch, Player Batch, 7, ...)
        opponent_embeddings = states[..., 1:, :, :]
        
        # This will be (Game Batch, Player Batch, ...)
        # Essentially it performs cross attention between the player and each opponent
        # and then sums the results
        cross_states = CrossPlayerEncoder(self.config.cross_encoder)(player_embedding, opponent_embeddings)
        
        # Now we perform cross attention on the player and the sum of the opponents to get a final state
        merged_states = CrossAttentionEncoder(self.config.merge_encoder)(player_embedding, context=cross_states)
        
        # Just a couple more self attention layers to get a really good representation
        global_state = Encoder(self.config.global_encoder)(merged_states)
        
        return global_state

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
        root, original_shape = self.root_fn(params, key, obs)
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
    
class PoroXV1:
    def __init__(self, config: PoroXConfig, key: jax.random.PRNGKey, obs: BatchedObservation):
        self.config = config
        
        repr_nn = RepresentationNetwork(config.muzero)
        pred_nn = PredictionNetwork(config.muzero)
        dyna_nn = DynamicsNetwork(config.muzero)
        
        self.muzero = MCTXAgent(
            representation_nn=repr_nn,
            prediction_nn=pred_nn,
            dynamics_nn=dyna_nn,
            config=config.mctx
        )
        
        self.variables = self.muzero.init(key, obs)
        self.key = key
        
    # TODO: Save and load checkpoints
    def save(self):
        pass
    def load(self):
        pass
        
    @partial(jax.jit, static_argnums=(0,))
    def policy(self, obs: BatchedObservation):
        return self.muzero.policy_gumbel(self.variables, self.key, obs)
    
    # @partial(jax.jit, static_argnums=(0,))
    def unflatten(self, policy_output, root, original_shape):
        actions = batch_utils.unflatten(policy_output.action, original_shape)
        action_weights = batch_utils.unflatten(policy_output.action_weights, original_shape)
        values = batch_utils.unflatten(root.value, original_shape)
        
        return actions, action_weights, values
        
    def act(self, obs: BatchedObservation, game_batched=False):
        if game_batched:
            obs, original_shape = batch_utils.flatten_multi_game_obs(obs)
            policy_output, root = self.policy(obs)
            return self.unflatten(policy_output, root, original_shape)
        else:
            policy_output, root = self.policy(obs)
            return policy_output.action, policy_output.action_weights, root.value
