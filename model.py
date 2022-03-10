#Author : Youness Landa
import numpy as np

from lstm import CifgLSTM
from linearLayer import ClassificationLinear

#The Char model composed of two stacked LSTMs and a decoder (classification layer)
class TwoLayersLSTM:
    def __init__(self, input_dim, hidden_dim=100):
        self.lstm1   = CifgLSTM(M=input_dim, N=hidden_dim)
        self.lstm2   = CifgLSTM(M=hidden_dim, N=hidden_dim)
        self.decoder = ClassificationLinear(input_size=hidden_dim, output_size=input_dim)

        self.velocities  = None
        self.velocities1 = None
        self.velocities2 = None

    def forward(self, x):
        y1, fwd_state1 = self.lstm1.forward(x)
        y2, fwd_state2 = self.lstm2.forward(y1)

        out, fwd_x = self.decoder.forward(y2)

        return out, fwd_x, fwd_state1, fwd_state2
    
    def backward(self, deltas, fwd_x, fwd_state1, fwd_state2):
        dx, gradients = self.decoder.backward(deltas, fwd_x) 
        dx2, gradients2, bwd_state2 = self.lstm2.backward(dx, fwd_state2)
        dx1, gradients1, bwd_state1 = self.lstm1.backward(dx2, fwd_state1)

        return dx1, gradients, gradients1, gradients2

    def optimize(self, method, gradients, gradients1, gradients2):
        """
        Updates the parameters of the model using a given optimize 
        method.
        """
        weights  = self.decoder.get_params()
        weights1 = self.lstm1.get_params()
        weights2 = self.lstm2.get_params()
        
        new_weights, self.velocities   = method.optim(weights, gradients, self.velocities)
        new_weights1, self.velocities1 = method.optim(weights1, gradients1, self.velocities1)
        new_weights2, self.velocities2 = method.optim(weights2, gradients2, self.velocities2)

        self.decoder.W, self.decoder.b = new_weights
        self.lstm1.Wz, self.lstm1.Wi, self.lstm1.Wo, self.lstm1.Rz, self.lstm1.Ri, self.lstm1.Ro, self.lstm1.pi, self.lstm1.po, self.lstm1.bz, self.lstm1.bi, self.lstm1.bo = new_weights1
        self.lstm2.Wz, self.lstm2.Wi, self.lstm2.Wo, self.lstm2.Rz, self.lstm2.Ri, self.lstm2.Ro, self.lstm2.pi, self.lstm2.po, self.lstm2.bz, self.lstm2.bi, self.lstm2.bo = new_weights2
    
    def clip(self, gradients, gradients1, gradients2, clip_value):
        """
        Clips the gradients in order to avoid the problem of 
        exploding gradient.
        """
        for grad in gradients:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        for grad in gradients1:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        for grad in gradients2:
            np.clip(grad, -clip_value, clip_value, out=grad)

        return gradients, gradients1, gradients2

    def loss(self, y, targets):
        """
        Computes the Cross Entropy Loss for the predicted y values.
        Parameters
        --
        y       : the output of the model
        targets : list of int containing indexes of taget sequennce
        """
        T, N = y.shape
        loss = 0
        
        for t in range(T):
            loss += - np.log(y[t][targets[t]])
        
        return loss
