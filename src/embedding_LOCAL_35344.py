import numpy as np
from tensor import Tensor

class Embedding:
    def __init__(self, seq_len, num_embeddings):
        """
        Params:
            seq_len: input size
            num_embeddings: Vocab size
        """
        self.w = Tensor(np.random.normal(0, 1, (num_embeddings, seq_len)))


    def __call__(self, in_idx: Tensor):
        """
        in_idx: (batch, seq_len)

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        embeddings = self.w[in_idx]
        return embeddings
    
class PositionalEncoding:
    def __init__(self, embeddings: Tensor):
        """
        embeddings: (batch, seq_len, d_model)
        """
        self.shape = embeddings.data.shape
            
    def __call__(self):
        _, seq_len, d_model = self.shape
        pos = np.arange(seq_len)[:, None] #(seq_len, 1)
        i = np.arange(d_model)[None, :] #(1, d_model)
        angles = pos / np.power(10000, 2 * (i // 2) / d_model) # (seq_len, d_model)

        pe = np.zeros((1, seq_len, d_model))
        pe[0][:, 0::2] = np.sin(angles[:, 0::2])
        pe[0][:, 1::2] = np.cos(angles[:, 1::2])
        
        return Tensor(pe)