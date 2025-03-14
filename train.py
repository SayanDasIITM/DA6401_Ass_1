import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import fashion_mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(-1, 28*28), x_test.reshape(-1, 28*28)
    encoder = OneHotEncoder(sparse_output=False)
    y_train, y_test = encoder.fit_transform(y_train.reshape(-1, 1)), encoder.transform(y_test.reshape(-1, 1))
    return x_train, y_train, x_test, y_test

class Activation:
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_derivative(x): return (x > 0).astype(float)
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def initialize_weights(layer_sizes):
    weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i]) for i in range(len(layer_sizes)-1)]
    biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
    return weights, biases

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights, self.biases = initialize_weights(layer_sizes)
    
    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            X = Activation.relu(X @ self.weights[i] + self.biases[i])
            activations.append(X)
        X = Activation.softmax(X @ self.weights[-1] + self.biases[-1])
        activations.append(X)
        return activations
    
    def backward(self, activations, y_true):
        deltas = [activations[-1] - y_true]
        for i in reversed(range(len(self.weights)-1)):
            delta = (deltas[-1] @ self.weights[i+1].T) * Activation.relu_derivative(activations[i+1])
            deltas.append(delta)
        deltas.reverse()
        grad_w = [activations[i].T @ deltas[i] / y_true.shape[0] for i in range(len(self.weights))]
        grad_b = [np.sum(deltas[i], axis=0, keepdims=True) / y_true.shape[0] for i in range(len(self.biases))]
        return grad_w, grad_b

class Optimizer:
    def __init__(self, method='adam', lr=0.001, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method, self.lr, self.momentum, self.beta1, self.beta2, self.epsilon = method, lr, momentum, beta1, beta2, epsilon
        self.v_w, self.v_b, self.m_w, self.m_b = None, None, None, None
    
    def initialize(self, weights, biases):
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]
        self.m_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
    
    def update(self, weights, biases, grad_w, grad_b, t):
        if self.v_w is None: self.initialize(weights, biases)
        for i in range(len(weights)):
            if self.method == 'sgd':
                weights[i] -= self.lr * grad_w[i]
                biases[i] -= self.lr * grad_b[i]
            elif self.method == 'momentum':
                self.v_w[i] = self.momentum * self.v_w[i] - self.lr * grad_w[i]
                self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grad_b[i]
                weights[i] += self.v_w[i]
                biases[i] += self.v_b[i]
            elif self.method == 'adam':
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w[i]**2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b[i]**2)
                m_hat_w = self.m_w[i] / (1 - self.beta1 ** t)
                m_hat_b = self.m_b[i] / (1 - self.beta1 ** t)
                v_hat_w = self.v_w[i] / (1 - self.beta2 ** t)
                v_hat_b = self.v_b[i] / (1 - self.beta2 ** t)
                weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

def train(model, optimizer, X_train, y_train, epochs=10, batch_size=64):
    num_samples, t = X_train.shape[0], 0
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_train, y_train = X_train[indices], y_train[indices]
        for i in range(0, num_samples, batch_size):
            X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
            activations = model.forward(X_batch)
            grad_w, grad_b = model.backward(activations, y_batch)
            t += 1
            optimizer.update(model.weights, model.biases, grad_w, grad_b, t)
        print(f"Epoch {epoch+1}/{epochs} completed.")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = NeuralNetwork([784, 128, 64, 10])
    optimizer = Optimizer(method='adam', lr=0.001)
    train(model, optimizer, X_train, y_train, epochs=10, batch_size=64)
