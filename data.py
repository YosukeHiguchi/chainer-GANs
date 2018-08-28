import os
import gzip

import numpy as np
import pickle
from urllib import request

from chainer import dataset

url = 'http://yann.lecun.com/exdb/mnist'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'
n_train = 60000
n_test = 10000
dim = 784


def load_mnist(images, labels, num):
    data = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
    target = np.zeros(num, dtype=np.uint8).reshape((num, ))

    with gzip.open(images, 'rb') as fi, gzip.open(labels, 'rb') as fl:
        fi.read(16)
        fl.read(8)
        for i in range(num):
            target[i] = ord(fl.read(1))
            for j in range(dim):
                data[i][j] = ord(fi.read(1))

    data = data.astype(np.float32)
    data /= 255.
    return data, target

def download_mnist_data(local_parent):
    def _download(file_name):
        print('Downloading {}...'.format(file_name))
        file_path = os.path.join(local_parent, file_name)
        request.urlretrieve('{}/{}'.format(url, file_name), file_path)
        print('Done')
        return file_path

    if not os.path.exists(local_parent):
        os.makedirs(local_parent)

    train_images_path = _download(train_images)
    train_labels_path = _download(train_labels)
    test_images_path = _download(test_images)
    test_labels_path = _download(test_labels)

    print('Converting train data...')
    data_train, target_train = load_mnist(
        train_images_path, train_labels_path, n_train)
    print('Done')
    print('Converting test data...')
    data_test, target_test = load_mnist(
        test_images_path, test_labels_path, n_test)
    mnist = {
        'data': np.append(data_train, data_test, axis=0),
        'target': np.append(target_train, target_test, axis=0)}
    print('Done')
    print('Saving output...')
    mnist_pickle_path = os.path.join(local_parent, 'mnist.pickle')
    with open(mnist_pickle_path, 'wb') as f:
        pickle.dump(mnist, f)
    print('Done')
    print('Conversion completed')

def load_mnist_data(local_parent='./'):
    mnist_pickle_path = os.path.join(local_parent, 'mnist.pickle')
    if not os.path.exists(mnist_pickle_path):
        download_mnist_data(local_parent)
    with open(mnist_pickle_path, 'rb') as f:
        mnist = pickle.load(f)

    return mnist

class MnistDataset(dataset.DatasetMixin):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = load_mnist_data(dataset_path)
        self.data_size = len(self.data['data'])

    def __len__(self):
        return self.data_size

    def get_example(self, i):
        return self.data['data'][i], self.data['target'][i]

if __name__ == '__main__':
    data = MnistDataset('./')
