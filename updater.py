import numpy as np

import chainer
from chainer import cuda
from chainer import Variable
import chainer.functions as F
import chainer.links as L

class GANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs,
                 converter=chainer.convert.concat_examples, device=None):
        self.gen, self.dis = kwargs.pop('models')
        self._iterators = kwargs.pop('iterator')
        self._optimizers = kwargs.pop('optimizer')
        self.device = kwargs.pop('device')
        params = kwargs.pop('params')
        self.batchsize = params['batchsize']
        self.n_latent = params['n_latent']

        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        x_real = Variable(self.converter(batch, self.device))
        xp = cuda.get_rray_module(x_real.data)

        gen = self.gen
        opt_gen= self._optimizers['gen']
        dis = self.dis
        opt_dis = self._optimizers['dis']

        y_real = dis(x_real)
        z = Variable(xp.random.normal(0, 1, (self.batchsize, self.n_latent)).astype(np.float32))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        t_real = Variable(xp.ones(len(x_real), 1), dtype=np.int32)
        t_fake = Variable(xp.zeros(len(x_real), 1), dtype=np.int32)

        dis_loss = F.sigmoid_cross_entropy(y_real, t_real)
        dis_loss += F.sigmoid_cross_entropy(y_fake, t_fake)

        gen_loss = F.sigmoid_cross_entropy(y_fake, t_real)

