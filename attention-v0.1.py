import math
import torch
import torch.nn

input = "The quick brown dog jumped over the lazy fox"
tokens = input.replace(',', '').split()
seq_len = len(tokens)
vocab = {t: i for i, t in enumerate(sorted(set(tokens)))}
tokenized_input = torch.tensor([vocab[t] for t in tokens])
print("\n============ Tokenized input ============")
print(tokenized_input)

embed_dim = 5
embedding_weights = torch.nn.Embedding(seq_len, embed_dim)
embeddings = embedding_weights(tokenized_input)
print("\n============ Embeddings (1, seq_len, embed_dim) ============")
print(embeddings)

head_dim = 4
W_q = torch.nn.Parameter(torch.randn(embed_dim, head_dim))
W_k = torch.nn.Parameter(torch.randn(embed_dim, head_dim))
W_v = torch.nn.Parameter(torch.randn(embed_dim, head_dim))
print("\n============ Q/K/V Weights (embed_dim, head_dim) ============")
print(W_q)

Q = embeddings @ W_q
K = embeddings @ W_k
V = embeddings @ W_v
print("\n============ Q/K/V Tensors (seq_len, head_dim) ============")
print(Q)

att_scores = Q @ K.transpose(-2,-1)
att_weights = torch.nn.functional.softmax(att_scores / math.sqrt(embed_dim), dim=-1)
print("\n============ Attention weights (seq_len, seq_len) ============")
print(att_weights)

context_vectors = att_weights @ V
print("\n============ Context vectors (seq_len, head_dim) ============")
print(context_vectors)
