import math
import torch
from torch import nn


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, -1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class PositionalEncoding(nn.Embedding):
    """Positional Encoding"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__(max_len, num_hiddens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, num_inputs, num_hiddens, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(num_inputs, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_inputs, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_inputs, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_inputs, bias=bias)

    def forward(self, X):
        queries = self.transpose_qkv(self.W_q(X), self.num_heads)
        keys = self.transpose_qkv(self.W_k(X), self.num_heads)
        values = self.transpose_qkv(self.W_v(X), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """Transform shape for parallel computation of multiple attention heads
        Input shape:(batch_size，num of queries/keys/values，num_hiddens)
        Output shape:(batch_size，num of queries/keys/values，num_heads，num_hiddens/num_heads)"""
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        """Reverses the operation of transpose_qkv"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    """Location-based Feedforward Networks"""
    def __init__(self, ffn_num_input, ffn_num_hiddens):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_input)

    def forward(self, X):
        return self.dense2(torch.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """Layer Norm After Residual"""
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, num_inputs, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_inputs, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, n_features, num_hiddens, norm_shape,
                 num_layers=4, num_heads=4, dropout=0.5, use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(n_features, num_hiddens, bias=use_bias)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(num_hiddens, num_hiddens, norm_shape,
                             num_hiddens, num_hiddens * 2, num_heads, dropout))
        self.out = nn.Linear(num_hiddens, n_features, bias=use_bias)

    def forward(self, X):
        X = self.embedding(X)
        X = self.pos_encoding(X)
        for i, blk in enumerate(self.blks):
            X = blk(X)
        return self.out(X)
