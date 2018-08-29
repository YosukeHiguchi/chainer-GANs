import os
import math

import numpy as np
from PIL import Image
import chainer
from chainer import Variable

def generate_image(gen, rows, cols, seed, image_path):
    np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp
    z = Variable(xp.asarray(gen.make_hidden(n_images)))

    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.backends.cuda.to_cpu(x.data)

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)

    _, H = x.shape
    H = int(math.sqrt(H))
    W = H

    x = np.reshape(x, (rows, cols, H, W))
    x = np.transpose(x, (0, 2, 1, 3))
    x = np.reshape(x, (rows * H, cols * W))

    Image.fromarray(x).save(image_path)

def out_generated_image(gen, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        preview_path = dst + '/preview'
        if not os.path.exists(preview_path):
            os.makedirs(preview_path)

        image_path = preview_path + '/image{:0>4}.png'.format(trainer.updater.epoch)
        generate_image(gen, rows, cols, seed, image_path)

    return make_image
