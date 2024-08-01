import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward_block import FeedForwardBlock
from residual_connection import ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Initializes a transformer block.

        Arguments:
            self_attention_block: The multi head attention mechanism that captures complexities from the query key and value matrixes.
            feed_forward_block: The Feed Forward neural network block.
            dropout (float): The dropout rate used in residual connections.
        """
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Initialize a residual connection for the two sub layers(multi head attention, feed forward network)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        """
        Arguments:
            self (Encoder Block Instance): EncoderBlock
        """
        # Inputs xxx refers to the query key and value matrices
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = nn.LayerNorm()
        
    def forward(self, x, mask):
        for layer in range(self.layers):
            x = layer(x, mask)
        return self.norm(x)