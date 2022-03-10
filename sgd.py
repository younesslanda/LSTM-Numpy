# Author : Youness Landa

# The SGD w/ momentum algorithm for network optimization
class SGD:
    def __init__(self, lr, beta=0.9):
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        if velocities is None: velocities = [0 for weight in weights]

        velocities = self.update_velocities(gradients, velocities)
        new_weights = []

        for weight, velocity in zip(weights, velocities):
            weight +=  velocity
            new_weights.append(weight)

        return new_weights, velocities

    def update_velocities(self, gradients, velocities):
        """
        Updates the velocities of the derivatives of the params.
        """
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):

            new_velocity = self.beta * velocity - self.lr * gradient
            new_velocities.append(new_velocity)

        return new_velocities