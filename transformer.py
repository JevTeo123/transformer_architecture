import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        """
        Arguments:
            eps = epsilon is added to offset any infinity occured by dividing values by 0
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

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

        x, self.attention_scores = MultiHeadAttention.attention(
            q = query,
            k = key,
            v = value
        )
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.d_k * self.h)

        # Multiply the matrices by weight matrix to get output
        return self.w_o(x)
    
class ResidualConnection(nn.module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        """
        Arguments:
            x: 
            sublayer: 
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
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
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in range(self.layers):
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: nn.Dropout) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Arguments:
            x: Inputs of the decoder
            encoder_output: Outputs of the encoder
            src_mask: Mask applied to the encoder
            tgt_mask: Mask applied to the decoder
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class ProjectionLayer(nn.module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # This multiplies the randomly initialized weights with each token in the matrix and adds the bias to get vocab size 
        self.proj = nn.Linear(d_model, vocab_size) 
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        # log softmax is being used as it is just computationally faster
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(
            self, 
            encoder: Encoder, 
            decoder: Decoder, 
            src_embed: InputEmbedding, 
            tgt_embed: InputEmbedding, 
            src_pos: PositionalEncoding, 
            tgt_pos: PositionalEncoding, 
            projection_layer: ProjectionLayer
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed
        src = self.src_pos
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed
        tgt = self.tgt_pos
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_seq_len: int, 
        tgt_seq_len: int, 
        d_model:int = 512, 
        N: int = 6, 
        num_heads = 8, 
        dropout: float = 0.1, 
        d_ff: int = 2048
):
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)

    # Create the Positional Encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_blocks = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_blocks = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_blocks, feed_forward_blocks)
        encoder_blocks.append(encoder_block)
    
    # Create the Decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_blocks = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_cross_attention_blocks = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_blocks = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_blocks, decoder_cross_attention_blocks, feed_forward_blocks)
        decoder_blocks.append(decoder_block)

    # Create the Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
