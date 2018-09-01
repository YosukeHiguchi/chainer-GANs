import numpy as np

import chainer
from chainer import cuda
from chainer import Variable
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L

class GANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, converter=convert.concat_examples, device=None, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iterators = kwargs.pop('iterators')
        self._optimizers = kwargs.pop('optimizers')
        self.converter = converter
        self.device = device
        self.iteration = 0

        params = kwargs.pop('params')
        self.batchsize = params['batchsize']
        self.n_latent = params['n_latent']

    def update_core(self):
        batch = self._iterators['main'].next()
        x_real = Variable(self.converter(batch, self.device)[0])
        xp = cuda.get_array_module(x_real.data)

        gen = self.gen
        opt_gen= self._optimizers['gen']
        dis = self.dis
        opt_dis = self._optimizers['dis']

        y_real = dis(x_real)
        z = Variable(xp.asarray(self.gen.make_hidden(len(batch))))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        t_real = Variable(xp.ones((len(x_real), 1), dtype=np.int32))
        t_fake = Variable(xp.zeros((len(x_real), 1), dtype=np.int32))

        dis_loss = F.sigmoid_cross_entropy(y_real, t_real)
        dis_loss += F.sigmoid_cross_entropy(y_fake, t_fake)
        gen_loss = F.sigmoid_cross_entropy(y_fake, t_real)

        dis.cleargrads()
        dis_loss.backward()
        opt_dis.update()

        gen.cleargrads()
        gen_loss.backward()
        opt_gen.update()

        chainer.report({'loss': gen_loss}, gen)
        chainer.report({'loss': dis_loss}, dis)
