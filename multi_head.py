import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model //n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

        print("\n q_linear : ", self.q_linear)
        print("\n k_linear : ", self.k_linear)
        print("\n v_linear : ", self.v_linear)
        print("\n out_linear : ", self.out_linear)
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        print("Q: \n", Q)
        print("K: \n", K)
        print("V: \n", V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        print("scores : \n", scores)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.out_linear(context)

mha = MultiHeadAttention(d_model=4, n_heads=2)

dummy_input = torch.randn(2, 5, 4)
output = mha(dummy_input)

print("\n\n Multi-Head Attention Output Shape: ", output.shape)


