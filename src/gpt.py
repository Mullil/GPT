from tensor import Tensor
from embedding import Embedding

class GPT:
    def __init__(self, n_layers, d_model, n_heads, vocab_size):
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.embedding = Embedding(d_model, vocab_size)
