#Author : Youness Landa
import numpy as np

from utils import softmax

class ClassificationLinear:
    def __init__(self, input_size, output_size):
        rnd = np.random.RandomState(seed=1)

        self.W = rnd.randn(input_size, output_size) * 0.1
        self.b = rnd.randn(output_size) * 0.1

    def get_params(self):
        return (self.W, self.b) 

    def forward(self, x):
        W, b = self.get_params()
        O = W.shape[1]
        T, _ = x.shape
        y_pred = np.zeros([T, O])

        loss = 0
        for t in range(T):
          y_pred[t] = softmax(np.dot(x[t], W) + b)

        fwd_x = x
        return y_pred, fwd_x
    
    def backward(self, deltas, fwd_x):
        W, b = self.get_params()
        x = fwd_x

        I = W.shape[0]
        T, O = deltas.shape

        dx = np.zeros([T, I])

        gradients = [np.zeros_like(w) for w in self.get_params()]
        dW, db = gradients

        for t in reversed(range(T)):
            dx[t] = np.dot(deltas[t], W.T)

            dW += np.outer(x[t], deltas[t])
            db += deltas[t]

        gradients = dW, db

        return dx, gradients