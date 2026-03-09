from tensor import Tensor
from embedding import Embedding

class GPT:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        tok_embedding = Embedding()