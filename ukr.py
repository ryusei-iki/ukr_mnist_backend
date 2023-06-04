# import torch
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
# torch.set_default_tensor_type(torch.cuda.FloatTensor)


class ukr:
    def __init__(self, x, labels, z, eta, ramuda, sigma, epochs):
        self.x = x
        self.z = z
        self.labels = labels / 9
        self.eta = eta
        self.ramuda = ramuda
        self.sigma = sigma
        self.epochs = epochs

    def train(self, ):

        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
        rgb = plt.get_cmap('viridis')
        rgb = plt.get_cmap('tab10')
        for i in range(10):
            ax1.scatter(i, i, c=rgb(i), s=1000)
        plt.savefig('outputs/a.png')
        fig = plt.figure(figsize=(21, 10))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0], aspect='equal')

        for epoch in range(self.epochs):
            ax1.cla()
            grad = jax.grad(self.E, argnums=0)(self.z)
            self.z = self.z - self.eta * grad
            # x_estimate = self.f(self.z)
            loss = self.E(self.z)
            # self.history['y'][epoch]
            print('{}epoch:{}'.format(epoch, loss))
            if (epoch % 1 == 0):
                ax1.scatter(self.z[:, 0], self.z[:, 1], c=rgb(self.labels))
                plt.pause(0.1)
        return self.z

    def f(self, z):
        d = (z[:, None, :] - self.z[None, :, :])**2
        d = jnp.sum(d, axis=2)
        d = jnp.exp(-1 / (2 * self.sigma**2) * d)
        x = jnp.einsum('ij,jd->id', d, self.x)
        x = x / jnp.sum(d, axis=1, keepdims=True)
        return x

    def E(self, z):
        loss = (self.x - self.f(z))**2
        loss = self.ramuda[0] * jnp.sum(loss) / self.x.shape[0] + self.ramuda[1] * jnp.sum(z**10) / self.x.shape[0]
        return loss
