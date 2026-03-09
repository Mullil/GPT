import numpy as np

x = np.array([[1,2], [3,4]])
idx = np.array([[1,1,0], [0,0,0]])
print(x[idx].shape)