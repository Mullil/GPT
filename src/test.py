import numpy as np
from embedding import Embedding, PositionalEncoding

e = Embedding(3,2)
in_idx = np.array([[1, 0], [0, 1]])
em = e(in_idx)

pe = PositionalEncoding(em)
pos = pe()
print(pos, em)