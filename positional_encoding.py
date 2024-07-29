import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Arguments:
            d_model: dimension of embedding
            seq_len: Number of tokens (eg. words, subwords)
            dropout: probability of dropout occurring
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float(), step = 2)

        div_term = (1 / torch.tensor(10000 ** (position / d_model)))

        # Odd Position
        pe[:, 0::2] = torch.sin(position * div_term)
        # Even Position
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout