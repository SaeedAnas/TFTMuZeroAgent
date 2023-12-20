import flax.linen as nn

from PoroX.models.components.transformer import EncoderConfig, Encoder
from PoroX.models.components.fc import FFNSwiGLU
from PoroX.models.components.embedding import ValueToken

class ValueEncoder(nn.Module):
    """
    Vit type encoder where we use a value token to pay attention
    to all other tokens in the hidden state and predict the value
    
    Can be used for both value and reward predictions
    """

    config: EncoderConfig
    
    @nn.compact
    def __call__(self, hidden_state):
        value_token_with_hidden_state = ValueToken()(hidden_state)
        
        value_logits = Encoder(self.config)(value_token_with_hidden_state)

        # Only return the value token
        value_logits = value_logits[..., 0, :]
        
        # Now optionally we could apply one final MLP to get the logits
        value = FFNSwiGLU()(value_logits)
        
        return value