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
from PoroX.models.components.embedding import PolicyActionTokens, ActionEmbedding, ActionGlobalToken, ValueToken, PrependTokens
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
    
class PredictionNetworkV2(nn.Module):
    """
    Only use a single policy token and run MHSA to get the policy logits
    """
    config: MuZeroConfig
    
    @nn.compact
    def __call__(self, hidden_state):
        # --- Policy Network --- #
        # Create one embedding token for the entire policy 
        # and concatenate it to the hidden state
        policy_token_with_hidden_state = ValueToken()(hidden_state)
        
        policy_logits = Encoder(self.config.policy_head)(policy_token_with_hidden_state)
        
        # Only take the first token
        policy_tokens = policy_logits[..., 0, :]
        
        # Apply MLP
        policy_logits = nn.Sequential([
            FFNSwiGLU(),
            FFNSwiGLU(),
            FFNSwiGLU(out_dim=55 * 38),
            FFNSwiGLU(hidden_dim=(55 * 38) // 2, out_dim=55 * 38),
        ])(policy_tokens)
        
        # --- Value Network --- #
        value_logits = ValueEncoder(self.config.value_head)(hidden_state)
        
        return policy_logits, value_logits
    
class PredictionNetworkV3(nn.Module):
    """
    Use the same encoder for policy, value, and reward
    """
    config: MuZeroConfig
    
    @nn.compact
    def __call__(self, hidden_state):
        # --- Policy Network --- #
        policy_value_reward_with_hidden_state = PrependTokens(num_tokens=3)(hidden_state)
        
        policy_value_reward_logits = Encoder(self.config.policy_head)(policy_value_reward_with_hidden_state)
        
        policy_token = policy_value_reward_logits[..., 0, :]
        value_token = policy_value_reward_logits[..., 1, :]
        reward_token = policy_value_reward_logits[..., 2, :]
        
        # Policy MLP
        policy_logits = nn.Sequential([
            FFNSwiGLU(),
            FFNSwiGLU(),
            FFNSwiGLU(out_dim=55 * 38),
            FFNSwiGLU(hidden_dim=(55 * 38) // 2, out_dim=55 * 38),
        ])(policy_token)
        
        # Value MLP
        value_logits = nn.Sequential([
            FFNSwiGLU(),
            FFNSwiGLU(),
        ])(value_token)
        
        # Reward MLP
        reward_logits = nn.Sequential([
            FFNSwiGLU(),
            FFNSwiGLU(),
        ])(reward_token)
        
        return policy_logits, value_logits, reward_logits

    
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
    
class DynamicsNetworkV2(nn.Module):
    """
    Only use a single action token and run MHSA to get the next hidden state
    """
    config: MuZeroConfig
    
    @nn.compact
    def __call__(self, hidden_state, action):
        action_tokens_with_hidden_state = ActionGlobalToken()(hidden_state, action)
        
        next_hidden_state = Encoder(self.config.dynamics_head)(action_tokens_with_hidden_state)
        
        # Exclude the action token
        next_hidden_state = next_hidden_state[..., 1:, :]
        
        # --- Reward --- #
        reward_logits = ValueEncoder(self.config.reward_head)(next_hidden_state)
        
        return next_hidden_state, reward_logits
    
class DynamicsNetworkV3(nn.Module):
    """
    Move reward to the prediction network
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(self, hidden_state, action):
        action_tokens_with_hidden_state = ActionGlobalToken()(hidden_state, action)
        next_hidden_state = Encoder(self.config.dynamics_head)(action_tokens_with_hidden_state)
        
        # Exclude the action token
        next_hidden_state = next_hidden_state[..., 1:, :]
        
        return next_hidden_state


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

class MuZeroBase:
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
        
        return params
    
    def policy_muzero(self, params: MuZeroParams, key: jax.random.PRNGKey, obs: BatchedObservation):
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
    
    def init_policy(self):
        if self.mc.policy_type == "muzero":
            return self.policy_muzero
        elif self.mc.policy_type == "gumbel":
            return self.policy_gumbel
        else:
            raise ValueError(f"Unknown policy type {self.mc.policy_type}")
        
    @partial(jax.jit, static_argnums=(0,))
    def root_fn(self, params: MuZeroParams, key: jax.random.PRNGKey, obs: BatchedObservation) -> mctx.RootFnOutput:
        raise NotImplementedError
        
    @partial(jax.jit, static_argnums=(0,))
    def recurrent_fn(self, params: MuZeroParams, key: jax.random.PRNGKey, action, embedding) -> (mctx.RecurrentFnOutput, jnp.ndarray):
        raise NotImplementedError
    
    
    
class MuZeroAgent(MuZeroBase):
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
    
class MuZeroAgentV2(MuZeroBase):
    @partial(jax.jit, static_argnums=(0,))
    def root_fn(self, params: MuZeroParams, key: jax.random.PRNGKey, obs: BatchedObservation) -> mctx.RootFnOutput:
        del key

        hidden_state = self.represnetation_nn.apply(params.represnentation, obs)
        policy_logits, value_logits, _ = self.prediction_nn.apply(params.prediction, hidden_state)

        value = self.scalar_encoder.decode_softmax(value_logits)
        
        return mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=value,
            embedding=hidden_state
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def recurrent_fn(self, params: MuZeroParams, key: jax.random.PRNGKey, action, embedding) -> (mctx.RecurrentFnOutput, jnp.ndarray):
        del key

        next_hidden_state = self.dynamics_nn.apply(params.dynamics, embedding, action)
        policy_logits, value_logits, reward_logits = self.prediction_nn.apply(params.prediction, next_hidden_state)

        value = self.scalar_encoder.decode_softmax(value_logits)
        reward = self.scalar_encoder.decode_softmax(reward_logits)
        
        discount = jnp.ones_like(value) * self.mc.discount
        
        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=policy_logits,
            value=value,
        )

        return recurrent_output, next_hidden_state

    
@chex.dataclass(frozen=True)
class PoroXOutput:
    action: jnp.ndarray
    action_weights: jnp.ndarray
    value: jnp.ndarray
    
class PoroXV1:
    def __init__(self, config: PoroXConfig, key: jax.random.PRNGKey, obs: BatchedObservation):
        self.config = config
        
        repr_nn = RepresentationNetwork(config.muzero)
        pred_nn = PredictionNetworkV3(config.muzero)
        dyna_nn = DynamicsNetworkV3(config.muzero)
        
        self.muzero = MuZeroAgentV2(
            representation_nn=repr_nn,
            prediction_nn=pred_nn,
            dynamics_nn=dyna_nn,
            config=config.mctx
        )
        
        self.variables = self.muzero.init(key, obs)
        self.key = key
        
        self._policy = self.muzero.init_policy()
        
    # TODO: Save and load checkpoints
    def save(self):
        pass
    def load(self):
        pass
        
    @partial(jax.jit, static_argnums=(0,))
    def policy(self, obs: BatchedObservation):
        return self._policy(self.variables, self.key, obs)
    
    def unflatten(self, policy_output, root, original_shape):
        actions = batch_utils.unflatten(policy_output.action, original_shape)
        action_weights = batch_utils.unflatten(policy_output.action_weights, original_shape)
        values = batch_utils.unflatten(root.value, original_shape)
        
        return actions, action_weights, values
        
    def act(self, obs: BatchedObservation, game_batched=False):
        if game_batched:
            obs, original_shape = batch_utils.flatten_multi_game_obs(obs)
            policy_output, root = self.policy(obs)
            actions, action_weights, values =  self.unflatten(policy_output, root, original_shape)
        else:
            policy_output, root = self.policy(obs)
            actions, action_weights, values = policy_output.action, policy_output.action_weights, root.value

        return PoroXOutput(
            action=actions,
            action_weights=action_weights,
            value=values,
        )
