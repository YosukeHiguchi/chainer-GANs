import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import training
from chainer import cuda, optimizers, serializers

import net


def main():
    parser = argparse.ArgumentParser(description='GAN practice on MNIST')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--snapepoch', '-s', default=20, type=int,
                        help='number of epochs to snapshot')
    parser.add_argument('--outdir', '-o', default='data',
                        help='path to the output directory')
    parser.add_argument('--load_gen_model', default='',
                        help='load generator model')
    parser.add_argument('--load_dis_model', default='',
                        help='load generator model')
    args = parser.parse_args()

    if not os.path.exists(outdir)
        os.makedirs

    print(args)


    gen = net.Generator(784, n_latent, 500)
    dis = net.Discriminator(784, 500)

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print('Generator model loaded successfully!')
    if args.load_dis_model != '':
        serializers.load_npz(args.load_dis_model, dis)
        print('Discriminator model loaded successfully!')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        print('use gpu {}'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

if __name__ == '__main__':
    main()
