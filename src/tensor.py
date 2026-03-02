import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.children = []
        self.grad = np.zeros(data.shape)
        self.backward_fn = None

    def __add__(self, x):
        result = Tensor(self.data + x.data)
        result.children.append(self)
        result.children.append(x)

        def backward_fn(parent):
            parent.children[0].grad += parent.grad
            parent.children[1].grad += parent.grad

        result.backward_fn = backward_fn
        return result
    
    def ffnn_hidden(self, w, b):
        pass

    def softmax(self, data):
        result = np.exp(data - np.max(data)) / \
            np.sum(np.exp(data - np.max(data)))
        return result
    
    def cross_entropy(self, y):
        pass

    def layer_norm(self):
        pass