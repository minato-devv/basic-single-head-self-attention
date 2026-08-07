import torch
import math
import torch.nn.functional as F
import torch.nn as nn

sent = 'Because MLX was designed to feel very similar to PyTorch, everything you learn right now will easily transfer over if you decide to try MLX later'
print("======== Input ========")
print(sent)

dc = {w:i for i, w in enumerate(sorted(sent.replace(',', '').split()))}
print("\n======== Synthetic vocabulary from input ========")
print(dc)

r = [dc[i] for i in sent.replace(',', '').split()]
tok_sent = torch.tensor(r)
print("\n======== Input mapped to a tensor of integer indices ========")
print(tok_sent)

vocab_size = 50000
torch.manual_seed(123)
embed = nn.Embedding(vocab_size, 3)
embedded_sent = embed(tok_sent).detach()
print("\n======== Embedding layer ========")
print(embed)
print("\n======== Embedded input ========")
print(embedded_sent)
print(f"Tokens = {embedded_sent.shape[0]}")
print(f"Embedding dimension = {embedded_sent.shape[1]}")

torch.manual_seed(123)
d = embedded_sent.shape[1]
d_q, d_k, d_v = 2, 2, 4
W_q = torch.nn.Parameter(torch.rand(d, d_q))
W_k = torch.nn.Parameter(torch.rand(d, d_k))
W_v = torch.nn.Parameter(torch.rand(d, d_v))
queries = embedded_sent @ W_q
keys = embedded_sent @ W_k
values = embedded_sent @ W_v
print("\n======== Query, key, and value weight matrices ========")
print(W_q.shape)
print(W_k.shape)
print(W_v.shape)
print("\n======== Querys, keys, and values ========")
print(queries.data)
print(keys.data)
print(values.data)

att_scores = queries @ keys.T / math.sqrt(d_k)
print("\n======== Attention scores ========")
print(att_scores.shape)
# print(att_scores.data)
att_weights = F.softmax(att_scores, dim=-1)
print("\n======== Attention weights (Softmax function) ========")
print(att_weights.shape)
# print(att_weights.data)

context_vector = att_weights @ values
print("\n======== Context vectors ========")
print(context_vector.shape)
print(context_vector.data)
