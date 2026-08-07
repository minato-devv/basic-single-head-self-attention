# Basic implementation of a single head self attention mechanism

`attention.py` is the single entry point. It uses a sample sentence, maps tokens to indices, creates an nn.Embedding, makes explicit Q/K/V weight matrices (nn.Parameter), computes attention scores (queries @ keys.T scaled by sqrt(d_k)), applies softmax to get attention weights, and multiplies by values to produce context vectors. The repo is a didactic script (no package structure, no tests).

```sh
brew install uv
uv sync
uv run python -m attention-v0.1
```

Sources:

https://github.com/Apoorva-Udupa/Single_head_selfAttention_Transformer.git

https://mohdfaraaz.medium.com/implementing-self-attention-from-scratch-in-pytorch-776ef7b8f13e

https://youtu.be/ZPLym9rJtM8?si=1NqpW02Bt5_e6341
