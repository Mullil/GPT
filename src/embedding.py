import numpy as np
from tensor import Tensor

class Embedding:
    def __init__(self, embedding_dim, num_embeddings):
        """
        Params:
            embedding_dim: input size
            num_embeddings: Vocab size
        """
        self.w = Tensor(np.random.normal(0, 1, (num_embeddings, embedding_dim)))


    def __call__(self, in_idx: Tensor):
        """
        in_idx: (batch, embedding_dim)

        Returns:
            embeddings: (batch, embedding_dim, d_model)
        """
        embeddings = self.w.data[in_idx]
        print(embeddings)
        return embeddings
    
e = Embedding(10,2)
in_idx = np.array([[0,1], [1,0], [1,1]])
e(in_idx)


    
