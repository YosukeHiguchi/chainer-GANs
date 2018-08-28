import argparse
import os

import numpy as np

import chainer

import net
from visualize import out_generated_image


def main():
    parser = argparse.ArgumentParser(description='Generate mnist image')
    parser.add_argument('--genpath', type=str,
                        help='path to a trained generator')
    parser.add_argument('--dimz', '-z', type=int, default=20,
                        help='dimention of encoded vector')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(args)


    gen = net.Generator(784, args.dimz, 500)
    chainer.serializers.load_npz(args.genpath, gen)
    print('Generator model loaded successfully from {}'.format(args.genpath))

    out_generated_image(gen, 10, 10, 0, args.out)

if __name__ == '__main__':
    main()
