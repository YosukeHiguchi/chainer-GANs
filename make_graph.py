import numpy as np
from chainer import Variable
import chainer.computational_graph as c

import net


gen = net.Generator(784, 20, 500)
dis = net.Discriminator(784, 500)

x_real = np.empty((1, 784), dtype=np.float32)
z = Variable(np.asarray(gen.make_hidden(1)))

y_real = dis(x_real)
x_fake = gen(z)
y_fake = dis(x_fake)

g = c.build_computational_graph([y_real, x_fake, y_fake])
with open('graph.dot', 'w') as o:
    o.write(g.dump())
