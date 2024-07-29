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
    def scaled_dot_product_attention(self, q, k ,v, mask = None):
        dot_prod = torch.matmul(q, k) / math.sqrt(self.d_k)
        attn = (np.exp(dot_prod - np.max(dot_prod))) / (np.exp(dot_prod - np.max(dot_prod))).sum()
        output = torch.matmul(attn, v)
        return output

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) ==> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0]. value.shape[1], self.h, self.d_k).transpose(1, 2)
