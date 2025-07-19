import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # we multiply sqrt of d_model because with out it the magnitude of the embedding would be too small compare to the positional embeddings
        return self.embeddings(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.empty(seq_len, d_model)

        # vector of shape(seq_len, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)) / d_model)

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self, x, dropout: float = 0.1):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by heads"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model// h
        self.dropout = nn.Dropout(dropout)

        self.w_q = torch.rand(d_model, d_model, bias= False)
        self.w_k = torch.rand(d_model, d_model, bias= False)
        self.w_v = torch.rand(d_model, d_model, bias= False)

        self.w_o = torch.rand(d_model, d_model, bias= False)

        self.attention_scores= None

    def project(self, x: torch.tensor, w: torch.tensor):
        batch_size, seq_len, _ = x.size()

        x = w(x).view(batch_size, seq_len, self.h, self.d_k)

        return x.transpose(1, 2)
    
    def compute(self, q, k, v, mask=0):

        scores = q @ k.transpose(-2, -1)
        scores = scores/ math.sqrt(self.d_k)

        if mask is not None:
            
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = scores.softmax(dim=-1)
        
        self.attention_scores = attn.detach()

        attn = self.dropout(attn)

        output = attn @ v

        return output
    
    def forward(self, q, k, v, mask= None):

        q = self.project(q, self.w_q)
        k = self.project(k, self.w_k)
        v = self.project(v, self.w_v)

        x = self.compute(q,k,v, mask)

        x = x.transpose(1,2).contiguous()
        x = x.view(x.size(0), x.size(1), self.d_model)

        return self.w_o(x)
    
class FeedFoward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, layer_norm, dropout: float= 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.layer_norm(x)))
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, mh_attn: MultiHeadAttention, feed_forward: FeedFoward, residual_connection: ResidualConnection, dropout: float = 0.1):
        super().__init__()
        self.mh_attn = mh_attn
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, dropout: float = 0.1):
        x = self.residual_connection[0](x, lambda x: self.mh_attn(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)

        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, mh_attn: MultiHeadAttention, cross_attn: MultiHeadAttention, feed_forward: FeedFoward, residual_connection: ResidualConnection, dropout: float = 0.1):
        super().__init__()
        self.mh_attn = mh_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask, dropout: float = 0.1):
        x = self.residual_connection[0](x, lambda x: self.mh_attn(x,x,x,src_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)) # this is because the query comes form decoder's mha output and key and value comes form encoder output and also in the decoder's mha we already used tgt mask so in here we are using the src maks for avoid padding
        x = self.residual_connection[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class project(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)