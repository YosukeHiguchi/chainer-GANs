import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, n_in, n_latent, n_h):
        self.n_latent = n_latent

        super(Generator, self).__init__(
            lg1 = L.Linear(n_latent, n_h),
            lg2 = L.Linear(n_h, n_in)
        )

    def make_hidden(self, batchsize):
        return np.random.normal(0, 1, (batchsize, self.n_latent)).astype(np.float32)

    def forward(self, z, sigmoid=True):
        h1 = F.tanh(self.lg1(z))
        h2 = self.lg2(h1)

        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

class Discriminator(chainer.Chain):
    def __init__(self, n_in, n_h):
        super(Discriminator, self).__init__(
            ld1 = L.Linear(n_in, n_h),
            ld2 = L.Linear(n_h, n_h),
            ld3 = L.Linear(n_h, 1)
        )

    def forward(self, x):
        h1 = F.tanh(self.ld1(x))
        h2 = F.tanh(self.ld2(h1))
        p = self.ld3(h2)

        return p
