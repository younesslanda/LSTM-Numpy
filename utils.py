#Author Youness Landa
import numpy as np

#Helper functions
sigma = lambda x: 1./(1 + np.exp(-x))
dsigma = lambda x: sigma(x) * (1 - sigma(x))

g = lambda x: np.tanh(x)
dg = lambda x: 1 - g(x)**2

h = lambda x: np.tanh(x)
dh = lambda x: 1 - h(x)**2

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))