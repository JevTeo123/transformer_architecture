import torch
import math
import numpy as np
import torch.nn as nn

class MultiHeadAttention(nn.module):
    def __init__(self, d_model: int, dropout: float, h: int = 8) -> None:
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert d_model % h == 0
        self.d_k == d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(
        query, 
        key, 
        value, 
        mask = None, 
        dropout = nn.Dropout
    ):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            # Insert a very low value to 
            attention_scores.masked_fill_(mask == 0, 1e-9)
        attention_scores = torch.softmax(attention_scores, dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

        

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) ==> (Batch, h, seq_len, d_k)
        # .view function to add in number of heads(h) and d_k which is the embedding dimension divided by number of heads
        # .transpose function to reshape matrix ordered by the number of heads as the attention mechanism operates separately on each head
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0]. value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.scaled_dot_product_attention(
            q = query,
            k = key,
            v = value
        )
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.d_k * self.h)

        # Multiply the matrices by weight matrix to get output
        return self.w_o(x)


