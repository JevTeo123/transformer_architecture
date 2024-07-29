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
        self.linear_1 = nn.Linear(d_model, d_ffn) #w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ffn, d_model)
    
    def forward(self, x):
        """
        Arguments:
            x: output from attention (batch_size, seq_length, d_model)
        Returns:
            expanded-and-contracted representation
        """
        # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))