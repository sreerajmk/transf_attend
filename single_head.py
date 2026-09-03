import torch
import torch.nn as nn
import math

batch_size = 2
sequence_len = 5
dimension = 4

torch.manual_seed(42)
Q = torch.randn(batch_size, sequence_len, dimension)
K = torch.randn(batch_size, sequence_len, dimension)
V = torch.randn(batch_size, sequence_len, dimension)

print("Input Shape (Q, K, V): ", Q.shape)

scores = torch.matmul(Q, K.transpose(-2, -1))

print("Raw Scores Shape: ", scores.shape)

d_k = Q.size(-1)
scaled_scores = scores/math.sqrt(d_k)

print(scaled_scores)

mask = torch.triu(torch.ones(sequence_len, sequence_len), diagonal=1).bool()

masked_scores = scaled_scores.masked_fill(mask, -1e9)

print("First batch raw masked matrix:\n", masked_scores[0])

attention_weights = torch.softmax(masked_scores, dim=-1)

print("Attention weights (Row 1 sums to 1):\n", attention_weights[0])

output = torch.matmul(attention_weights, V)

print("Final Context-Aware Output Shape: ", output.shape)
