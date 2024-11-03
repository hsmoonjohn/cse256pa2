# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import math

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embed size needs to be divisible by num_heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask, return_attention=False):
        N = query.shape[0] #Size of the batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)

        if return_attention:
            return out, attention
        else:
            return out
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiheadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, return_attention=False):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and then apply dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        if return_attention:
            return out, attention
        else:
            return out


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_hidden_dim, dropout, max_length):
        super(TransformerEncoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_size, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, return_attention=False):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        attention_maps = []
        # Pass `out` as value, key, and query for self-attention in each layer
        for layer in self.layers:
            if return_attention:
                out, attention = layer(out, out, out, mask, return_attention)
                attention_maps.append(attention)
            else:
                out = layer(out, out, out, mask)
        
        if return_attention:
            return out, attention_maps
        else:
            return out.mean(dim=1)  # Mean pooling over sequence dimension


