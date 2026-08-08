import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Weights for projecting input to Q, K, and V
        # Shape: (embed_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        x: Tensor of shape (Batch_Size, Seq_Len, Embed_Dim)
        mask: Optional boolean mask of shape (1, 1, Seq_Len, Seq_Len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # 1. Project to Q, K, V
        # Shape: (Batch, Seq_Len, Embed_Dim)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2. Calculate Scaled Dot Product Attention
        # Step A: Compute Q @ K^T
        # Result Shape: (Batch, Seq_Len, Embed_Dim) @ (Batch, Embed_Dim, Seq_Len) -> (Batch, Seq_Len, Seq_Len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))

        # Step B: Scale by sqrt(d_k)
        # d_k is the dimension of the key vector (which is embed_dim here)
        scale = math.sqrt(self.embed_dim)
        attention_scores = attention_scores / scale

        # Step C: Apply Mask (if provided) and Softmax
        if mask is not None:
            # Add a very small negative number to masked positions to make them 0 after softmax
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Step D: Apply Weights to Values (V)
        # Result Shape: (Batch, Seq_Len, Seq_Len) @ (Batch, Seq_Len, Embed_Dim) -> (Batch, Seq_Len, Embed_Dim)
        out = torch.matmul(attention_weights, v)

        # 3. Project Output
        # Shape: (Batch, Seq_Len, Embed_Dim)
        return self.out_proj(out)

# --- Usage Example ---

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 1  # Single Head

    # Create dummy input: (Batch, Sequence_Length, Embed_Dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Initialize the layer
    attention_layer = SingleHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Forward pass
    output = attention_layer(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")