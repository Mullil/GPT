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
    
    def __mul__(self, x):
        result = Tensor(self.data * x.data)
        result.children.append(self)

        def backward_fn(parent):
            parent.children[0].grad += parent.grad * x

        result.backward_fn = backward_fn
        return result
    
    def __getitem__(self, idx: 'Tensor'):
        return self.data[idx.data]
    
    def __matmul__(self, x):
        result = Tensor(self.data @ x.data)
        result.children.append(self)
        result.children.append(x)

        def backward_fn(parent):
            parent.children[0].grad += parent.grad @ x.data.T
            parent.children[1].grad += self.data.T @ parent.grad

        result.backward_fn = backward_fn
        return result

    def ffnn_hidden(self, w, b):
        result = (self @ w) + b
        return result.relu()
    
    def relu(self):
        return self * (self.data > 0)
    
    def _tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def gelu(self):
        result = Tensor(0.5 * self.data * (1 + self._tanh( np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data**3) )))
        result.children.append(self)

        def backward_fn(parent: Tensor):
            tanh_term = self._tanh( np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data**3) )
            parent.children[0].grad += 0.5 * parent.grad * (1 + tanh_term) + \
                                        0.5 * parent.grad * self.data * (1 - tanh_term**2) * \
                                        (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.data**2))

        result.backward_fn = backward_fn
        return result


    def dropout(self, p):
        mask = (np.random.rand(*self.shape) > p).astype(np.float32)
        result = self * mask
        return result

    def softmax(self, data):
        result = np.exp(data - np.max(data)) / \
            np.sum(np.exp(data - np.max(data)))
        return result
    
    def cross_entropy(self, y):
        pass

    def layer_norm(self):
        pass

    def self_attention(self):
        pass

    def masked_attention(self):
        pass