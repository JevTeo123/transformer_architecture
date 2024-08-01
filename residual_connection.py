import torch.nn as nn

class ResidualConnection(nn.module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = nn.LayerNorm()
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    