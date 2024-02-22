import numpy as np

class Model:
    def init(self, x_dim, dim):
        self.w1 = np.random.randn(x_dim, dim)
        self.w2 = np.random.randn(dim, x_dim)

    def update_model(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

def train(X, y, iterations, alpha, hidden_dimensions):
    model = init(X, hidden_dimensions)
    for _ in range(0, iterations):
        a1, a2, z, embeddings = forward(model, X)
        backward(model, a1, z, X, y, alpha)
    return embeddings, model

def predict(model, X):
    a1 = np.dot(X, model.w1)
    a2 = np.dot(a1, model.w2)
    z = softmax(a2)

    return z

def encode(model, v):
    a1 = np.dot(v, model.w1)
    a2 = np.dot(v, model.w2)

    return a2

def init(X, hidden_dimensions):
    model = Model()
    model.init(len(X[0]), hidden_dimensions)
    return model

def forward(model, X):
    a1 = np.dot(X, model.w1)
    a2 = np.dot(a1, model.w2)
    z = softmax(a2)
    return np.array(a1), np.array(a2), np.array(z), a2

def backward(model, a1, z, X, y, alpha):
    da2 = z - y
    dw2 = np.dot(a1.T, da2)
    da1 = np.dot(da2, model.w2.T)
    dw1 = np.dot(X.T, da1)

    model.w1 -= dw1 * alpha
    model.w2 -= dw2 * alpha

    print(f"Cross entropy loss: {cross_entropy(z, y)}")

def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)


def softmax(U):
    ret = []
    for u in U:
        max = np.max(u)
        ret.append(np.exp(u - max) / np.sum(np.exp(u - max)))
    return ret
