import torch
import torch.nn as nn

class ProjectionLayer(nn.module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # This multiplies the randomly initialized weights with each token in the matrix and adds the bias to get vocab size 
        self.proj = nn.Linear(d_model, vocab_size) 
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        # log softmax is being used as it is just computationally faster
        return torch.log_softmax(self.proj(x), dim = -1)