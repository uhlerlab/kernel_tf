import torchvision.datasets as datasets
import numpy as np
from numpy.linalg import norm
import visdom


def load_cifar_c(num_train=1000, shift_name='fog',
                 flatten=False, label_set=None):

    prefix = '../cifar-c_data/'

    data = np.load(prefix + shift_name + '.npy')
    labels = np.load(prefix + 'labels.npy')
    
    print(data.shape, labels.shape)

    dataset = list(zip(data, labels))

    images, targets = load_data(dataset, num=num_train, label_set=label_set,
                                flatten=flatten)
    num_classes = 10
    if label_set is not None:
        num_classes = len(label_set)

    one_hot = make_one_hot(targets, num_classes)

    return images, one_hot


def load_cifar(num_train=1000, num_test=10000, flatten=False, label_set=None):
    cifar_path = '~/NeuralNetworks/expressivity_through_depth/cifar_experiments/data/'
    
    cifar_trainset = datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=None)
    cifar_testset = datasets.CIFAR10(root=cifar_path, train=False, download=True, transform=None)

    train_images, train_targets = load_data(cifar_trainset, num=num_train,
                                            label_set=label_set,
                                            flatten=flatten)
    test_images, test_targets = load_data(cifar_testset, num=num_test,
                                          label_set=label_set,
                                          flatten=flatten)

    num_classes = 10
    if label_set is not None:
        num_classes = len(label_set)

    one_hot_train = make_one_hot(train_targets, num_classes)
    one_hot_test = make_one_hot(test_targets, num_classes)

    return train_images, one_hot_train, test_images, one_hot_test


def load_data(dataset, num, label_set=None, flatten=False):
    labels = {}
    targets = []
    images = []

    if label_set is not None:
        label_set_map = {}
        for idx, l in enumerate(sorted(label_set)):
            label_set_map[l] = idx

    for i in range(min(num, len(dataset))):
        img, label = dataset[i]
        if label_set is not None:
            if label not in label_set:
                continue
            label = label_set_map[label]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
        targets.append(label)
        img = np.array(img) / 255.

        if flatten:
            images.append(img.reshape(1, -1))
        else:
            img = img.reshape(1, 32, 32, 3)
            img = np.rollaxis(img, -1, 1)
            images.append(img)
    images = np.concatenate(images, axis=0)
    return images, np.array(targets)


def make_one_hot(targets, num_classes):
    one_hot = np.zeros((len(targets), num_classes))
    for i in range(len(targets)):
        one_hot[i, targets[i]] = 1
    return one_hot
