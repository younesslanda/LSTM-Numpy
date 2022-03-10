#Author : Youness Landa
import numpy as np

from utils import sigma, dsigma, g, dg, h, dh, softmax

# The CIFG LSTM Layer
class CifgLSTM:
    def __init__(self, M, N):
        # M : dimensionality of inputs as in the paper
        # N : dimensionality of hidden as in the paper

        rnd = np.random.RandomState(seed=1)

        self.Wz = rnd.randn(N, M) * 0.1
        self.Wi = rnd.randn(N, M) * 0.1
        self.Wo = rnd.randn(N, M) * 0.1

        self.Rz = rnd.randn(N, N) * 0.1
        self.Ri = rnd.randn(N, N) * 0.1
        self.Ro = rnd.randn(N, N) * 0.1

        self.pi = rnd.randn(N) * 0.1
        self.po = rnd.randn(N) * 0.1

        self.bz = rnd.randn(N) * 0.1
        self.bi = rnd.randn(N) * 0.1
        self.bo = rnd.randn(N) * 0.1
    
    def get_params(self):
        return (self.Wz, self.Wi, self.Wo, self.Rz, self.Ri, self.Ro, self.pi, self.po, self.bz, self.bi, self.bo)

    def forward(self, x):
        """
        x is the input : (length of sequence, input_size),  input_size = input dimension
        """
        Wz, Wi, Wo, Rz, Ri, Ro, pi, po, bz, bi, bo = self.get_params()

        N = Wz.shape[0]
        T, M = x.shape

        z_ = np.zeros([T, N])
        z  = np.zeros([T, N])
        i_ = np.zeros([T, N])
        i  = np.zeros([T, N])
        f  = np.zeros([T, N])
        c  = np.zeros([T, N])
        o_ = np.zeros([T, N])
        o  = np.zeros([T, N])
        y  = np.zeros([T, N])

        for t in range(T):
            z_[t] = np.dot(Wz, x[t]) + np.dot(Rz, y[t-1]) + bz
            z[t] = g(z_[t])

            i_[t] = np.dot(Wi, x[t]) + np.dot(Ri, y[t-1]) + pi*c[t-1] + bi
            i[t] = sigma(i_[t])

            f[t] = 1 - i[t] # CIFG

            c[t] = i[t] * z[t] + f[t] * c[t-1]

            o_[t] = np.dot(Wo, x[t]) + np.dot(Ro, y[t-1]) + po*c[t] + bo
            o[t] = sigma(o_[t])

            y[t] = o[t] * h(c[t])

        fwd_state = (x, z_, z, i_, i, f, o_, o, c, y)
        return y, fwd_state

    def backward(self, deltas, fwd_state):
        """
            computes the backward pass, calculates the gradients
        """
        Wz, Wi, Wo, Rz, Ri, Ro, pi, po, bz, bi, bo = self.get_params()
        x, z_, z, i_, i, f, o_, o, c, y = fwd_state

        T, N = deltas.shape
        M = Wz.shape[1]
        
        # the derivative arrays are longer so t+1 is automatically zero at the ends
        dz = np.zeros([T+1, N])
        di = np.zeros([T+1, N])
        dc = np.zeros([T+1, N])
        do = np.zeros([T+1, N])
        dy = np.zeros([T, N])
        dx = np.zeros([T, M])
        
        # initialize gradients
        gradients = [np.zeros_like(w) for w in self.get_params()]
        dWz, dWi, dWo, dRz, dRi, dRo, dpi, dpo, dbz, dbi, dbo = gradients
        
        for t in reversed(range(T)):
            dy[t] = deltas[t] + np.dot(Rz.T, dz[t+1]) + np.dot(Ri.T, di[t+1]) + np.dot(Ro.T, do[t+1])
            do[t] = dy[t] * h(c[t]) * dsigma(o_[t])
            dc[t] = dy[t] * o[t] * dh(c[t]) + po * do[t]
            if t < T-1:
                dc[t] += pi * di[t+1]  + dc[t+1] * f[t+1]
            di[t] = dc[t] * z[t] * dsigma(i_[t])
            dz[t] = dc[t] * i[t] * dg(z_[t])
            
            # Input Deltas
            dx[t] = np.dot(Wz.T, dz[t]) + np.dot(Wi.T, di[t]) + np.dot(Wo.T, do[t])
            
            # Gradients for the weights
            dWz += np.outer(dz[t], x[t])
            dWi += np.outer(di[t], x[t])
            dWo += np.outer(do[t], x[t])
            dRz += np.outer(dz[t+1], y[t])
            dRi += np.outer(di[t+1], y[t])
            dRo += np.outer(do[t+1], y[t])        
            dpi += c[t] * di[t+1]
            dpo += c[t] * do[t]
            dbz += dz[t]
            dbi += di[t]
            dbo += do[t]

        bwd_state = (dz, di, dc, do, dy)
        gradients = dWz, dWi, dWo, dRz, dRi, dRo, dpi, dpo, dbz, dbi, dbo 

        return dx, gradients, bwd_state