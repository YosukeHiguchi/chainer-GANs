import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import training
from chainer import cuda
from chainer.training import extensions

import net
from data import MnistDataset
from updater import GANUpdater
from visualize import out_generated_image

def main():
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser(description='GAN practice on MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                    help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--out', '-o', type=str, default='model',
                        help='path to the output directory')
    parser.add_argument('--dimz', '-z', type=int, default=20,
                        help='dimention of encoded vector')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapepoch', '-s', type=int, default=10,
                        help='number of epochs to snapshot')
    parser.add_argument('--load_gen_model', type=str, default='',
                        help='load generator model')
    parser.add_argument('--load_dis_model', type=str, default='',
                        help='load generator model')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(args)


    gen = net.Generator(784, args.dimz, 500)
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
        print('GPU {}'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

    opt_gen = chainer.optimizers.Adam()
    opt_dis = chainer.optimizers.Adam()
    opt_gen.setup(gen)
    opt_dis.setup(dis)

    dataset = MnistDataset('./data')
    # train, val = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.9))

    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize, shuffle=True)
    # val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

    updater = GANUpdater(
        models=(gen, dis),
        iterators={
            'main': train_iter
        },
        optimizers={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=args.gpu,
        params={
            'batchsize': args.batchsize,
            'n_latent': args.dimz
        })
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapepoch, 'epoch')
    display_interval = (100, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis{.updater.epoch}.npz'), trigger=snapshot_interval)

    log_keys = ['epoch', 'iteration', 'gen/loss', 'dis/loss']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=display_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(out_generated_image(gen, 10, 10, args.seed, args.out),
        trigger=(1, 'epoch'))

    trainer.run()

if __name__ == '__main__':
    main()
