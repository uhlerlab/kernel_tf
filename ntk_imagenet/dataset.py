import pickle
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import flowers102, dtd
from torchvision import models
from numpy.linalg import norm
import time
import random

    
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def get_train_data(path=None):
    X = []
    Y = []

    # Path to Imagenet32 data 
    assert path is not None
        
    for i in range(1, 11):
        out = unpickle(path + 'train_data_batch_' + str(i))
        x = out['data']
        mean_image = out['mean']
        
        y = out['labels']

        y = [i-1 for i in y]
        data_size = x.shape[0]

        img_size = 32
        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2],
                       x[:, img_size2:2*img_size2],
                       x[:, 2*img_size2:]))

        x = x.reshape((x.shape[0], img_size, img_size, 3))
        x = np.rollaxis(x, -1, 1)

        X.append(x)
        Y.extend(y)

    X = np.concatenate(X, axis=0)

    one_hot = np.zeros((len(Y), 1000))
    one_hot[np.arange(len(Y)), Y] = 1.
    return X, one_hot


def get_test_data(path=None):
    assert path is not None
    out = unpickle(path + 'val_data')
    x = out['data']
    y = out['labels']
    y = [i-1 for i in y]
    data_size = x.shape[0]

    img_size = 32
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2],
                   x[:, img_size2:2*img_size2],
                   x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    x = np.rollaxis(x, -1, 1)

    one_hot = np.zeros((len(y), 1000))
    one_hot[np.arange(len(y)), y] = 1.

    return x, one_hot


def group_by_class(X_train, y_train):
    num_classes = np.sum(y_train, axis=0)
    print(num_classes)
    class_lookup = {}
    labels = y_train.nonzero()[1]

    for idx, label in enumerate(labels):
        x = X_train[idx]
        if label in class_lookup:
            class_lookup[label].append(x)
        else:
            class_lookup[label] = [x]
    return class_lookup


def remap_classes(class_list):
    count = 0
    label_map = {}
    for l in class_list:
        label_map[l] = count
        count += 1
    return label_map


def select_classes(class_lookup, class_list):
    X_train = []
    y_train = []

    label_map = remap_classes(class_list)

    for l in class_list:
        X = class_lookup[l]
        X_train.extend(X)
        ys = np.zeros((len(X), len(class_list)))
        ys[:, label_map[l]] = 1.
        y_train.extend(ys)

    X_train = np.array(X_train).astype('float32')
    y_train = np.array(y_train).astype('float32')
    return X_train, y_train


def get_imagenet_data(classlist=None):
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    X_train = X_train.reshape(-1, 3072)
    X_train /= norm(X_train, axis=-1).reshape(-1, 1)

    X_test = X_test.reshape(-1, 3072)
    X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    if classlist is not None:
        class_lookup = group_by_class(X_train, y_train)
        X_train, y_train = select_classes(class_lookup, classlist)

    print("Train set: ", X_train.shape, y_train.shape)
    print("Test set: ", X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)


def get_cifar_data(path=None):
    cifar_path = path
    assert path is not None

    loader = transforms.Compose([
        transforms.ToTensor(),        
    ])
        
    training_data = torchvision.datasets.CIFAR10(
        root=cifar_path,
        train=True,
        download=False,
        transform=loader
    )
    
    test_data = torchvision.datasets.CIFAR10(
        root=cifar_path,
        train=False,
        download=False,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader


def get_flowers102_data(path=None):
    assert path is not None
    loader = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),        
    ])

    training_data = flowers102.Flowers102(
        root=path,
        split='train',
        download=False,
        transform=loader
    )
    
    test_data = flowers102.Flowers102(
        root=path,
        split='test',
        download=False,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader


def get_svhn_data(path=None):
    assert path is not None
    loader = transforms.Compose([
        transforms.ToTensor(),        
    ])

    training_data = torchvision.datasets.SVHN(
        root=path,
        split='train',
        download=False,
        transform=loader
    )
    
    test_data = torchvision.datasets.SVHN(
        root=path,
        split='test',
        download=False,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    return train_loader, test_loader


def get_dtd_data(path=None):
    assert path is not None
    loader = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),        
    ])

    training_data = dtd.DTD(
        root=path,
        split='train',
        download=False,
        transform=loader
    )
    
    test_data = dtd.DTD(
        root=path,
        split='test',
        download=False,
        transform=loader
    )
    
    train_loader = DataLoader(training_data, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    return train_loader, test_loader


def get_cifar_c_data(shift_name='fog', path=None, flatten=False):
    assert path is not None
    prefix = path
    data = np.load(prefix + shift_name + '.npy')
    labels = np.load(prefix + 'labels.npy')

    print("Data shape: ", data.shape, labels.shape)

    X = data
    X = np.rollaxis(X, -1, 1)
    
    y = []
    NUM_CLASSES = 10
    y = np.zeros((len(labels), NUM_CLASSES))
    y[np.arange(len(labels)), labels] = 1.

    X = X.reshape(-1, 3072)
    norms = norm(X, axis=-1).reshape(-1, 1)
    X = X / norms
        
    train_X = X[-10000:-1000]
    test_X = X[-1000:]
    train_y = y[-10000:-1000]
    test_y = y[-1000:]
    return (train_X, train_y), (test_X, test_y)

    
def normalized_numpy_data(loader, NUM_CLASSES, size=3072, shift=False):
    X, Y = [], []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        x = inputs.numpy().reshape(-1, 3072)
        x /= norm(x, axis=-1).reshape(-1, 1)
        y = np.zeros((len(labels), NUM_CLASSES))
        if shift:
            y[np.arange(len(labels)), labels-1] = 1.
        else:
            y[np.arange(len(labels)), labels] = 1.
        X.append(x)
        Y.append(y)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y
    
