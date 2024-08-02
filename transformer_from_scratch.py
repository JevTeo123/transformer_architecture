import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.input_embeddings = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: nn.Dropout):
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype = torch.float32).unsqueeze(0)
        div_term = torch.tensor(10000 ** (2(position) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer()

    def forward(self, x):
        # Ensures that the shape of the positional encoder matches up with the shape of the input tensor x
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)
        return (self.dropout(x))
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdims = True)
        std = x.std(dim = -1, keepdims = True)
        return self.alpha * (x - mean / std + self.eps) + self.bias

class FeedForwardBlock():
    def __init__(self, d_model: int, d_ffn: int, dropout: nn.Dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ffn, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: nn.Module, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.d_k = d_model // num_heads
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
            self, 
            query, 
            key, 
            value, 
            dropout,
            mask
    ):
        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, 1e-9)
        attention_scores = torch.softmax(attention_scores, dim = -1)
        if dropout is not None:
            dropout(attention_scores)
        return (attention_scores @ value)
    
    def forward(self, q, k, v, mask):
        d_k = q.shape[-1]
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, d_k).tranpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, d_k).tranpose(1, 2)

        x = MultiHeadAttentionBlock.attention(query, key, value, mask)

        # Reshape (batch, h, seq_len, d_k) --> (batch, seq_len, dmodel)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.h * d_k)

        return self.w_o(x)
    
# Residual Connections are there for allow weight and biases to skip different layers thereby mitigating long compute times related to vanishing gradients and etc
# Provides a shortcut for gradients to backpropagate enabling more effective training
# It is normally implemented in a normalization layer after each sublayer    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: nn.Dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return (x + self.dropout(sublayer(self.norm(x))))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: nn.Dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers = nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization
    def forward(self, x, mask):
        for layer in range(self.layers):
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:  MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: nn.Dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self, encoder_output, x, tgt_mask):
        x = self.residual_connections[0](lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(encoder_output, encoder_output, x, tgt_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers = nn.ModuleList):
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in range(self.layers):
            x = layer(x, mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        return torch.log_softmax(x, dim = -1)

class Transformer(nn.Module):
    def __init__(
            self, 
            encoder: Encoder, 
            decoder: Decoder, 
            src_embed: InputEmbeddings, 
            tgt_embed: InputEmbeddings, 
            src_pos: PositionalEncoding, 
            tgt_pos: PositionalEncoding, 
            projection_layer: ProjectionLayer
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encoder(self, src, src_mask):
        src = self.src_embed
        src = self.src_pos
        return self.encoder(src, src_mask)
    
    def decoder(self, tgt, tgt_mask):
        tgt = self.tgt_embed
        tgt = self.tgt_pos
        return self.decoder(tgt, tgt_mask)
    
    def project(self, x):
        self.projection_layer(x)
    
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    N: int = 8,
    d_model: int = 512,
    dropout: float = 0.1,
    d_ff: int = 2048
):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, dropout, N)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(encoder_blocks)

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, dropout, N)
        cross_attention_block = MultiHeadAttentionBlock(d_model, dropout, N)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(decoder_blocks)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)



    


        



