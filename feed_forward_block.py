import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        """
        Arguments:
            d_model: dimension of embeddings
            d_ffn: dimension of feed-forward network
            dropout: probability of dropout occuring
        """
        super().__init__()
        """
        - First layer (linear_1) accepts an input dimension of d_model and outputs d_ffn.
        - Allows to explore better feature interactions through a large dimension
        - Second layer (linear_2) accepts an input dimension of d_ffn which is the output dimension from the first layer and outputs a dimension of d_model
        - Compresses all the important feature interactions from a larger dimension back into d_model
        """
        self.linear_1 = nn.Linear(d_model, d_ffn) #w1 and b1
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ffn, d_model) # w2 and b2
    
    def forward(self, x):
        """
        Arguments:
            x: output from attention (batch_size, seq_length, d_model)
        Returns:
            expanded-and-contracted representation
        """
        # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))